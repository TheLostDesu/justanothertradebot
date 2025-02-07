#!/usr/bin/env python3
"""
Построение датасета фич из архивов ордербуков.

— Скачивание архивов (если их ещё нет) происходит последовательно.
— Обработка каждого архива (потоковая генерация обучающих примеров) выполняется в отдельном потоке.
"""

import os
import re
import io
import json
import time
import zipfile
import requests
import argparse
import logging
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures

# Импортируем настройки и класс Orderbook
from config import (
    TRAINING_DATE_RANGES,   # например, ["2025-01-01,2025-01-07"]
    URL_TEMPLATE,           # "https://quote-saver.bycsi.com/orderbook/linear/{pair}/{date}_{pair}_ob500.data.zip"
    SYMBOLS,                # список, например, ["BTC/USDT", "ETH/USDT", ...]
    SEQUENCE_LENGTH,        # длина окна (например, 3)
    HORIZON_MS              # горизонт в секундах (например, 30)
)
from orderbook import Orderbook

# Интервалы (в секундах)
SNAPSHOT_INTERVAL = 10    # делаем snapshot раз в 10 секунд
TRAINING_HORIZON  = 30    # через 30 секунд после появления первого snapshot-а, считается исход

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =========================================================
# Формирование URL архивов для заданного диапазона и торговой пары
# =========================================================
def generate_date_urls(date_range: str, pair: str) -> list:
    """
    Для диапазона дат (строка "YYYY-MM-DD,YYYY-MM-DD") и торговой пары (без символа "/")
    формирует список URL архивов.
    """
    start_str, end_str = date_range.split(',')
    start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
    end_date   = datetime.strptime(end_str.strip(), "%Y-%m-%d")
    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = URL_TEMPLATE.format(pair=pair, date=date_str)
        urls.append(url)
        current_date += timedelta(days=1)
    return urls

# =========================================================
# Скачивание архива (если его ещё нет)
# =========================================================
def download_archive(url: str, zips_dir: str) -> str:
    """
    Скачивает архив по URL в папку zips_dir.
    Если архив уже существует, возвращает путь к нему.
    """
    filename = url.split("/")[-1]
    filepath = os.path.join(zips_dir, filename)
    if os.path.exists(filepath):
        logging.info(f"Архив уже существует: {filepath}")
        return filepath

    logging.info(f"Скачиваем {url} ...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        logging.info(f"Скачано: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Ошибка скачивания {url}: {e}")
        return None

# =========================================================
# Обработка архива (потоковая генерация обучающих примеров)
# Многопоточность применяется только на этом этапе.
# =========================================================
def process_archive_streaming(filepath: str, timeout: float = 30) -> list:
    """
    Обрабатывает архив (zip‑файл) потоково.

    Алгоритм:
      1. Читаем записи из архива и обновляем Orderbook.
      2. Раз в SNAPSHOT_INTERVAL (10 сек) извлекается snapshot (фича) и добавляется в очередь.
      3. Если время текущей записи (ts) >= времени первого snapshot-а + TRAINING_HORIZON (30 сек)
         и в очереди накоплено не менее SEQUENCE_LENGTH snapshot‑ов, формируется обучающий пример:
            – признаки: объединяем (flatten) первые SEQUENCE_LENGTH snapshot‑ов ([bid, ask, mid])
            – цель: относительное изменение mid‑цены, вычисленное как (current_mid – candidate_mid) / candidate_mid.
         После формирования примера первый snapshot удаляется из очереди.
      4. Если в течение timeout (30 сек) реального времени не появляется новый пример, возвращаются накопленные данные.
    """
    training_examples = []
    feature_queue = []      # очередь snapshot‑ов (каждый snapshot – dict с 'bid', 'ask', 'mid', 'ts')
    last_snapshot_time = None  # для контроля интервала между snapshot‑ами
    last_generated_time = time.time()
    ob = Orderbook()

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            for filename in zf.namelist():
                file_info = zf.getinfo(filename)
                total_size = file_info.file_size
                bytes_read = 0
                logged_thresholds = set()

                logging.info(f"Обработка файла {filename} ({total_size} байт)...")
                with zf.open(filename) as f:
                    line_count = 0
                    for line in f:
                        line_count += 1
                        bytes_read += len(line)
                        # Логирование прогресса по процентам
                        current_percentage = int((bytes_read / total_size) * 100)
                        for threshold in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                            if current_percentage >= threshold and threshold not in logged_thresholds:
                                logging.info(f"Файл {filename}: обработано {threshold}%")
                                logged_thresholds.add(threshold)

                        try:
                            record = json.loads(line.decode("utf-8").strip())
                        except Exception:
                            continue
                        data = record.get("data", {})
                        r_type = record.get("type")
                        ts = record.get("ts")
                        try:
                            ts = float(ts)
                        except Exception:
                            ts = time.time()

                        # Обновляем Orderbook; предполагается, что Orderbook.update() раз в секунду
                        # сохраняет snapshot, а get_snapshot_features() возвращает словарь с [bid, ask, mid]
                        ob.update(data, r_type, timestamp=ts)

                        # Получаем snapshot из Orderbook и добавляем метку времени, если её нет
                        snap = ob.get_snapshot_features()
                        if snap is None:
                            continue
                        if "ts" not in snap:
                            snap["ts"] = ts

                        # Добавляем snapshot в очередь, если прошло не менее SNAPSHOT_INTERVAL секунд
                        if last_snapshot_time is None or (ts - last_snapshot_time) >= SNAPSHOT_INTERVAL:
                            feature_queue.append(snap)
                            last_snapshot_time = ts
                            logging.debug(f"Добавлен snapshot: ts={ts}")

                        # Формируем обучающие примеры, если для первого snapshot-а прошёл TRAINING_HORIZON секунд
                        while feature_queue:
                            candidate = feature_queue[0]
                            if ts >= candidate["ts"] + TRAINING_HORIZON:
                                if len(feature_queue) >= SEQUENCE_LENGTH:
                                    window = feature_queue[:SEQUENCE_LENGTH]
                                    lob_features = []
                                    for snap_in_window in window:
                                        lob_features.extend([
                                            float(snap_in_window.get("bid", 0.0)),
                                            float(snap_in_window.get("ask", 0.0)),
                                            float(snap_in_window.get("mid", 0.0))
                                        ])
                                    features_vec = np.array(lob_features, dtype=np.float32)
                                    candidate_mid = float(candidate.get("mid", 0))
                                    current_mid = float(snap.get("mid", 0))
                                    if candidate_mid != 0:
                                        target_delta = (current_mid - candidate_mid) / candidate_mid
                                    else:
                                        target_delta = 0.0
                                    training_examples.append({
                                        "features": features_vec,
                                        "target": np.float32(target_delta)
                                    })
                                    logging.info(f"Сформирован обучающий пример (всего примеров: {len(training_examples)})")
                                    # Удаляем первый snapshot из очереди
                                    feature_queue.pop(0)
                                    last_generated_time = time.time()
                                else:
                                    break
                            else:
                                break

                        # Если с момента формирования последнего примера прошло больше timeout секунд, возвращаем накопленные примеры.
                        if time.time() - last_generated_time > timeout:
                            logging.info("Достигнут timeout (30 сек) – возвращаю накопленные примеры.")
                            return training_examples
                logging.info(f"Файл {filename}: обработано {line_count} строк.")
    except Exception as e:
        logging.error(f"Ошибка при обработке архива {filepath}: {e}")
        return training_examples

    return training_examples

# =========================================================
# Главная функция: скачивание архивов и их многопоточная обработка
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Построение датасета фич из архивов ордербуков (обработка архивов многопоточна)."
    )
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Индекс торговой пары из SYMBOLS (0-based). Если не указан – обрабатываются все пары.")
    parser.add_argument("--download", action="store_true",
                        help="Скачивать архивы, если их нет в папке zips.")
    args = parser.parse_args()

    # Определяем, какие торговые пары обрабатывать
    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

    # Формируем список URL архивов для выбранных пар и диапазонов дат
    all_urls = []
    for dr in TRAINING_DATE_RANGES:
        for sym in selected_pairs:
            pair_formatted = sym.replace("/", "")
            urls = generate_date_urls(dr, pair_formatted)
            all_urls.extend(urls)
    logging.info(f"Найдено архивов: {len(all_urls)}")

    zips_dir = "zips"
    data_dir = "data"
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Скачивание архивов (последовательно)
    archive_paths = []
    for url in all_urls:
        archive_filename = url.split("/")[-1]
        archive_path = os.path.join(zips_dir, archive_filename)
        if not os.path.exists(archive_path):
            if args.download:
                archive_path = download_archive(url, zips_dir)
            else:
                logging.warning(f"Архив не найден: {archive_path}. Для скачивания используйте флаг --download")
                continue
        if archive_path is not None:
            archive_paths.append(archive_path)

    if not archive_paths:
        logging.info("Нет архивов для обработки. Завершаем работу.")
        return

    # Обработка архивов с использованием многопоточности
    num_workers = min(len(archive_paths), os.cpu_count() or 4)
    logging.info(f"Запуск обработки архивов в {num_workers} потоках.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_archive = {
            executor.submit(process_archive_streaming, path, timeout=30): path
            for path in archive_paths
        }
        for future in concurrent.futures.as_completed(future_to_archive):
            archive_path = future_to_archive[future]
            try:
                examples = future.result()
            except Exception as exc:
                logging.error(f"Ошибка при обработке {archive_path}: {exc}")
                continue

            archive_filename = os.path.basename(archive_path)
            if examples:
                X = np.stack([ex["features"] for ex in examples])
                Y = np.array([ex["target"] for ex in examples], dtype=np.float32)
                output_filename = os.path.splitext(archive_filename)[0] + ".npz"
                output_path = os.path.join(data_dir, output_filename)
                np.savez_compressed(output_path, X=X, Y=Y)
                logging.info(f"Сохранён датасет: {output_path} (образцов: {X.shape[0]})")
            else:
                logging.info(f"Для архива {archive_path} не сформировано обучающих примеров.")

if __name__ == "__main__":
    main()
