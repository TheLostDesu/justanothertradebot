#!/usr/bin/env python3
"""
Построение датасета фич из архивов ордербуков с использованием ProcessPoolExecutor на UNIX.
Каждый URL обрабатывается в отдельном процессе:
  - Если архив отсутствует и передан флаг --download, архив скачивается.
  - Далее архив обрабатывается «на лету»: из записей формируются snapshot‑ы с интервалом 10 сек,
    и когда для первого snapshot‑а проходит 30 сек, формируется обучающий пример.
  - Если обучающие примеры сформированы, .npz‑файл сохраняется в папку data.
  - Рабочий процесс возвращает простую строку с результатом.
  
Обратите внимание:
  • Все функции определены на глобальном уровне (это необходимо для pickle).
  • Используется метод запуска «spawn», что позволяет избежать проблем с наследованием состояния в UNIX.
  • При получении результатов из пула добавлены try/except, чтобы ошибки (например, BrokenProcessPool)
    не приводили к аварийному завершению всего скрипта.
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
import multiprocessing

# Импортируем настройки и класс Orderbook.
# Убедитесь, что модуль config и класс Orderbook определены на уровне модуля.
from config import (
    TRAINING_DATE_RANGES,   # например, ["2025-01-01,2025-01-07"]
    URL_TEMPLATE,           # "https://quote-saver.bycsi.com/orderbook/linear/{pair}/{date}_{pair}_ob500.data.zip"
    SYMBOLS,                # список, например, ["BTC/USDT", "ETH/USDT", …]
    SEQUENCE_LENGTH,        # длина окна (например, 3)
    HORIZON_MS              # горизонт в секундах (например, 30)
)
from orderbook import Orderbook

# Параметры интервалов (в секундах)
SNAPSHOT_INTERVAL = 10    # собираем snapshot раз в 10 сек
TRAINING_HORIZON  = 30    # через 30 сек после появления первого snapshot‑а считается исход

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def generate_date_urls(date_range: str, pair: str) -> list:
    """
    Для заданного диапазона дат (строка "YYYY-MM-DD,YYYY-MM-DD") и торговой пары (без символа "/")
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
    Обрабатывает архив (zip‑файл) «на лету» с использованием Orderbook.get_features().

    Алгоритм:
      1. Для каждой записи в архиве обновляется ордербук.
      2. В ордербуке накапливается последовательность snapshot‑ов (через update и internal sequence_history).
      3. Когда длина последовательности достигает SEQUENCE_LENGTH, формируется обучающий пример:
            – признаки: сохраняется окно из SEQUENCE_LENGTH snapshot‑ов (без flattening),
            – цель: вычисляется как относительное изменение mid‑цены между первым и последним snapshot‑ом.
         После формирования примера окно сдвигается (удаляя первый snapshot).
      4. Если в течение timeout (30 сек) не появляется новый пример, возвращаются накопленные данные.
    """
    training_examples = []
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

                        # Обновляем ордербук; внутри update() происходит накопление snapshot-ов
                        ob.update(data, r_type, timestamp=ts)

                        # Получаем накопленные фичи (включая sequence и candles)
                        features = ob.get_features()
                        sequence = features.get("sequence", [])

                        # Если накоплено достаточно snapshot-ов, формируем обучающий пример
                        if len(sequence) >= SEQUENCE_LENGTH:
                            window = sequence[-SEQUENCE_LENGTH:]
                            first_mid = float(window[0].get("mid", 0))
                            last_mid = float(window[-1].get("mid", 0))
                            target_delta = ((last_mid - first_mid) / first_mid) if first_mid != 0 else 0.0

                            # В обучающем примере просто сохраняем список snapshot-ов
                            training_examples.append({
                                "features": window,
                                "target": np.float32(target_delta)
                            })
                            last_generated_time = time.time()

                            # Сдвигаем окно, удаляя первый snapshot, чтобы сформировать новое окно для будущего примера
                            if ob.sequence_history:
                                ob.sequence_history.popleft()

                        if time.time() - last_generated_time > timeout:
                            logging.info("Достигнут timeout (30 сек) – возвращаю накопленные примеры.")
                            return training_examples
    except Exception as e:
        logging.error(f"Ошибка при обработке архива {filepath}: {traceback.format_exc()}")
        return training_examples

    return training_examples

def download_and_process(url: str, zips_dir: str, data_dir: str, timeout: float, download: bool) -> str:
    """
    Для данного URL:
      - Если архив отсутствует и download=True, архив скачивается.
      - Если архив найден (или уже скачан), запускается его обработка.
      - Если сформированы обучающие примеры, .npz‑файл записывается в data_dir.
      - Возвращается строка с результатом.
    """
    try:
        filename = url.split("/")[-1]
        filepath = os.path.join(zips_dir, filename)
        if not os.path.exists(filepath):
            if download:
                filepath = download_archive(url, zips_dir)
                if filepath is None:
                    logging.error(f"Не удалось скачать архив {url}")
                    return f"Archive {url} failed download"
            else:
                logging.warning(f"Архив {filepath} не найден. Используйте флаг --download")
                return f"Archive {filepath} not found"
        examples = process_archive_streaming(filepath, timeout=timeout)
        if examples:
            X = np.stack([ex["features"] for ex in examples])
            Y = np.array([ex["target"] for ex in examples], dtype=np.float32)
            output_filename = os.path.splitext(filename)[0] + ".npz"
            output_path = os.path.join(data_dir, output_filename)
            np.savez_compressed(output_path, X=X, Y=Y)
            logging.info(f"Сохранён датасет: {output_path} (образцов: {X.shape[0]})")
            return f"Processed: {output_path}"
        else:
            logging.info(f"Для архива {filename} не сформировано обучающих примеров.")
            return f"No examples for {filename}"
    except Exception:
        logging.error(f"Ошибка в download_and_process для {url}: {traceback.format_exc()}")
        return f"Error processing {url}"

def main():
    parser = argparse.ArgumentParser(
        description="Построение датасета из архивов ордербуков (ProcessPoolExecutor, UNIX)"
    )
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Индекс торговой пары из SYMBOLS (0-based). Если не указан, обрабатываются все пары.")
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

    num_workers = 12
    logging.info(f"Запуск обработки архивов в {num_workers} процессах.")
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(download_and_process, url, zips_dir, data_dir, 30, args.download)
            for url in all_urls
        ]
        for future in concurrent.futures.as_completed(futures):
            exc = future.exception()
            if exc is not None:
                logging.error("Future raised an exception:")
                logging.error(traceback.format_exception(type(exc), exc, exc.__traceback__))
                continue
            result = future.result()
            logging.info(f"Результат: {result}")
    logging.info("Обработка завершена.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
