#!/usr/bin/env python3
"""
Построение датасета фич из архивов ордербуков с использованием ProcessPoolExecutor.
Каждый URL обрабатывается в отдельном процессе:
  - Если архив отсутствует и передан флаг --download, архив скачивается.
  - Затем архив обрабатывается (парсинг, сбор snapshot-ов, формирование обучающих примеров).
"""

import os
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
import traceback

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
SNAPSHOT_INTERVAL = 10    # собираем snapshot раз в 10 секунд
TRAINING_HORIZON  = 30    # через 30 секунд после появления первого snapshot-а считается исход

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

def process_archive_streaming(filepath: str, timeout: float = 30) -> list:
    """
    Обрабатывает архив (zip‑файл) «на лету».

    Алгоритм:
      1. Из записей архива обновляется Orderbook.
      2. Раз в SNAPSHOT_INTERVAL (10 сек) извлекается snapshot (фича) и добавляется в очередь.
      3. Если время текущей записи (ts) >= времени первого snapshot-а + TRAINING_HORIZON (30 сек)
         и в очереди накоплено не менее SEQUENCE_LENGTH snapshot‑ов, формируется обучающий пример:
            – признаки: объединяются (flatten) первые SEQUENCE_LENGTH snapshot‑ов ([bid, ask, mid])
            – цель: относительное изменение mid‑цены, вычисляемое как (current_mid – candidate_mid) / candidate_mid.
         После формирования примера первый snapshot удаляется из очереди.
      4. Если в течение timeout (30 сек) реального времени не появляется новый пример,
         возвращаются накопленные данные.
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
                    for line in f:
                        bytes_read += len(line)
                        # Логирование процента обработанных байт
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

                        # Обновляем Orderbook (предполагается, что Orderbook.update() раз в секунду сохраняет snapshot)
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

                        # Формируем обучающие примеры из очереди, если для первого snapshot-а прошло TRAINING_HORIZON секунд
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
                                    target_delta = ((current_mid - candidate_mid) / candidate_mid
                                                    if candidate_mid != 0 else 0.0)
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

                        # Если с момента формирования последнего примера прошло больше timeout секунд, выходим.
                        if time.time() - last_generated_time > timeout:
                            logging.info("Достигнут timeout (30 сек) – возвращаю накопленные примеры.")
                            return training_examples
    except Exception as e:
        logging.error(f"Ошибка при обработке архива {filepath}: {traceback.format_exc()}")
        return training_examples

    return training_examples

def download_and_process(url: str, zips_dir: str, timeout: float, download: bool) -> tuple:
    """
    Для данного URL:
      - Если архив отсутствует и download=True, скачивает архив.
      - Если архив найден (или уже скачан), запускает его обработку.
      - Возвращает (archive_filename, training_examples) или None, если архив недоступен.
    """
    try:
        filename = url.split("/")[-1]
        filepath = os.path.join(zips_dir, filename)
        if not os.path.exists(filepath):
            if download:
                filepath = download_archive(url, zips_dir)
                if filepath is None:
                    logging.error(f"Не удалось скачать архив {url}")
                    return None
            else:
                logging.warning(f"Архив {filepath} не найден. Для скачивания используйте флаг --download")
                return None

        examples = process_archive_streaming(filepath, timeout=timeout)
        return (filename, examples)
    except Exception:
        logging.error(f"Ошибка в download_and_process для {url}: {traceback.format_exc()}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Построение датасета фич из архивов ордербуков (обработка архивов в процессах)."
    )
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Индекс торговой пары из SYMBOLS (0-based). Если не указан – обрабатываются все пары.")
    parser.add_argument("--download", action="store_true",
                        help="Скачивать архивы, если их нет в папке zips.")
    args = parser.parse_args()

    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

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

    # Используем ProcessPoolExecutor для CPU-интенсивной обработки
    num_workers = os.cpu_count() or 4
    logging.info(f"Запуск обработки архивов в {num_workers} процессах.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = {
            executor.submit(download_and_process, url, zips_dir, 30, args.download): url
            for url in all_urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            result = future.result()
            if result is None:
                continue
            archive_filename, examples = result
            if examples:
                X = np.stack([ex["features"] for ex in examples])
                Y = np.array([ex["target"] for ex in examples], dtype=np.float32)
                output_filename = os.path.splitext(archive_filename)[0] + ".npz"
                output_path = os.path.join(data_dir, output_filename)
                np.savez_compressed(output_path, X=X, Y=Y)
                logging.info(f"Сохранён датасет: {output_path} (образцов: {X.shape[0]})")
            else:
                logging.info(f"Для архива {archive_filename} не сформировано обучающих примеров.")

if __name__ == "__main__":
    main()
