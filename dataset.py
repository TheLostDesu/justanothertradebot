#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import re
import threading
import queue
import time
from datetime import datetime, timedelta
import numpy as np
import requests
import io
import json
import zipfile
from tqdm import tqdm
from sortedcontainers import SortedList
from orderbook import OrderBookSide, get_features_from_orderbook, get_mid_price
from config import (TRAINING_DATE_RANGES, URL_TEMPLATE, SYMBOLS, NUM_LEVELS,
                    SEQUENCE_LENGTH, HORIZON_MS, MAX_TARGET_CHANGE_PERCENT,
                    CANDLE_INTERVAL_MIN, CANDLE_TOTAL_HOURS)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# Функции расчёта признаков
# -----------------------------------------------------------------------------
def generate_date_urls(date_range, template):
    """Принимает строку 'YYYY-MM-DD,YYYY-MM-DD' и возвращает список URL-ов."""
    start_str, end_str = date_range.split(',')
    start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
    end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        urls.append(template.format(pair="BTCUSDT", date=date_str))
        current_date += timedelta(days=1)
    return urls

def open_zip(source):
    """Открывает zip-архив по URL или локальному пути."""
    if isinstance(source, str) and source.startswith("http"):
        try:
            r = requests.get(source)
            r.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            logging.error(f"Error downloading {source}: {e}")
            return None
    else:
        try:
            with open(source, "rb") as f:
                content = f.read()
            return zipfile.ZipFile(io.BytesIO(content))
        except Exception as e:
            logging.error(f"Error opening zip file {source}: {e}")
            return None

def compute_candle_features(records, end_time):
    """
    Группирует записи по 5-минутным интервалам за последние 5 часов (60 свечей)
    и для каждого интервала вычисляет два признака: return и range.
    """
    interval_ms = CANDLE_INTERVAL_MIN * 60 * 1000
    candle_count = int((CANDLE_TOTAL_HOURS * 60) / CANDLE_INTERVAL_MIN)
    start_time = end_time - (CANDLE_TOTAL_HOURS * 60 * 1000)
    buckets = [[] for _ in range(candle_count)]
    for rec in records:
        ts = rec["ts"]
        if start_time <= ts < end_time:
            bucket_index = int((ts - start_time) // interval_ms)
            if bucket_index < candle_count:
                buckets[bucket_index].append(rec["mid_price"])
    features = []
    for bucket in buckets:
        if bucket:
            open_price = bucket[0]
            close_price = bucket[-1]
            high = max(bucket)
            low = min(bucket)
            ret = (close_price - open_price) / open_price if open_price != 0 else 0
            rng = (high - low) / open_price if open_price != 0 else 0
        else:
            ret, rng = 0, 0
        features.extend([ret, rng])
    return features

# -----------------------------------------------------------------------------
# Обработка архивного файла
# -----------------------------------------------------------------------------
def process_archive_file(filepath):
    logging.info(f"Processing file: {filepath}")
    try:
        with open(filepath, "rb") as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return []
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except Exception as e:
        logging.error(f"Error opening zip file {filepath}: {e}")
        return []
    records = []
    current_ob = None
    for file_name in zf.namelist():
        with zf.open(file_name) as f:
            for line in f:
                try:
                    record = json.loads(line.decode('utf-8').strip())
                except Exception:
                    continue
                data = record.get('data', {})
                r_type = record.get('type')
                if r_type == 'snapshot' or data.get('u') == 1:
                    current_ob = {
                        'a': OrderBookSide(is_bid=False),
                        'b': OrderBookSide(is_bid=True)
                    }
                    if 'a' in data:
                        for level in data['a']:
                            try:
                                price = float(level[0])
                                size = float(level[1])
                                current_ob['a'].update(price, size)
                            except Exception:
                                continue
                    if 'b' in data:
                        for level in data['b']:
                            try:
                                price = float(level[0])
                                size = float(level[1])
                                current_ob['b'].update(price, size)
                            except Exception:
                                continue
                elif r_type == 'delta':
                    if current_ob is None:
                        continue
                    for side in ['a', 'b']:
                        if side in data:
                            for update in data[side]:
                                try:
                                    price = float(update[0])
                                    size = float(update[1])
                                except Exception:
                                    continue
                                current_ob[side].update(price, size)
                else:
                    continue
                if current_ob is None:
                    continue
                ob_lists = {
                    'a': current_ob['a'].get_list(),
                    'b': current_ob['b'].get_list()
                }
                mid = get_mid_price(ob_lists)
                if mid is None:
                    continue
                feats = get_features_from_orderbook(ob_lists, NUM_LEVELS)
                ts = record.get('ts')
                try:
                    ts = int(ts)
                except:
                    continue
                records.append({"ts": ts, "features": feats, "mid_price": mid})
    zf.close()
    records.sort(key=lambda x: x["ts"])
    samples = []
    n = len(records)
    for i in range(n):
        start_ts = records[i]["ts"]
        target_index = None
        for j in range(i+1, n):
            if records[j]["ts"] >= start_ts + HORIZON_MS:
                target_index = j
                break
        if target_index is None:
            break
        start_mid = records[i]["mid_price"]
        target_mid = records[target_index]["mid_price"]
        target_delta = (target_mid - start_mid) / start_mid
        if abs(target_delta) > MAX_TARGET_CHANGE_PERCENT:
            continue
        if i - SEQUENCE_LENGTH + 1 < 0:
            continue
        lob_seq = []
        for k in range(i - SEQUENCE_LENGTH + 1, i + 1):
            lob_seq.extend(records[k]["features"])
        candle_feats = compute_candle_features(records, start_ts)
        combined = lob_seq + candle_feats
        samples.append({
            "features": np.array(combined, dtype=np.float32),
            "target": np.float32(target_delta)
        })
    logging.info(f"Finished processing {filepath}. Generated {len(samples)} samples.")
    return samples

# -----------------------------------------------------------------------------
# Продюсер и потребитель: скачивание архивов
# -----------------------------------------------------------------------------
def download_archive(url, zips_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()
        filename = url.split("/")[-1]
        filepath = os.path.join(zips_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return None

def processor_worker(filepath):
    logging.info(f"Started processing archive {os.path.basename(filepath)}")
    return process_archive_file(filepath)

def build_dataset_for_pairs(selected_pairs):
    all_urls = []
    for dr in TRAINING_DATE_RANGES:
        for sym in selected_pairs:
            pair = sym.replace("/", "")
            urls = generate_date_urls(dr, URL_TEMPLATE.replace("{pair}", pair))
            all_urls.extend(urls)
    return all_urls

# -----------------------------------------------------------------------------
# Отдельный поток для записи файлов (одновременно только один)
# -----------------------------------------------------------------------------
def writer_thread_func(write_queue, stop_event):
    logging.info("Writer thread started.")
    while not stop_event.is_set() or not write_queue.empty():
        try:
            output_path, X, Y = write_queue.get(timeout=1)
            np.savez_compressed(output_path, X=X, Y=Y)
            logging.info(f"Written file: {output_path} (samples: {X.shape[0]})")
        except queue.Empty:
            continue
    logging.info("Writer thread stopped.")

# -----------------------------------------------------------------------------
# Основная функция
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dataset builder for trade bot.")
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Номер пары из списка SYMBOLS (0-based). Если не указан, обрабатываются все пары.")
    args = parser.parse_args()

    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

    all_urls = build_dataset_for_pairs(selected_pairs)
    logging.info(f"Total archives to process: {len(all_urls)}")
    os.makedirs("zips", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    download_queue = queue.Queue(maxsize=100)
    write_queue = queue.Queue()
    stop_event = threading.Event()

    # Запуск потока для скачивания архивов
    def downloader():
        for url in tqdm(all_urls, desc="Downloading archives"):
            if stop_event.is_set():
                break
            while download_queue.full():
                if stop_event.is_set():
                    return
                time.sleep(0.1)
            filepath = download_archive(url, "zips")
            if filepath:
                download_queue.put(filepath)

    downloader_thread = threading.Thread(target=downloader, name="Downloader")
    downloader_thread.start()

    # Запуск отдельного потока для записи файлов (одновременно только один)
    writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, stop_event), name="Writer")
    writer_thread.start()

    regex = re.compile(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_ob500\.data\.zip")
    futures = []
    num_workers = os.cpu_count() or 4

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Отправляем задачи в пул, пока скачиватель работает или очередь не пуста
            while downloader_thread.is_alive() or not download_queue.empty():
                if stop_event.is_set():
                    break
                try:
                    filepath = download_queue.get(timeout=1)
                except queue.Empty:
                    continue
                future = executor.submit(processor_worker, filepath)
                future.filepath = filepath
                futures.append(future)
            # Обрабатываем результаты по мере завершения
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing archives"):
                filepath = getattr(future, "filepath", "unknown")
                try:
                    samples = future.result(timeout=300)
                    m = regex.search(os.path.basename(filepath))
                    if m:
                        date_str = m.group(1)
                        pair = m.group(2)
                        key = f"{pair}_{date_str}"
                    else:
                        key = "unknown"
                    if samples:
                        try:
                            features_list = [s["features"] for s in samples]
                            targets_list = [s["target"] for s in samples]
                            X = np.stack(features_list)
                            Y = np.array(targets_list)
                            archive_id = os.path.splitext(os.path.basename(filepath))[0]
                            output_path = os.path.join("data", f"{key}_{archive_id}.npz")
                            # Вместо записи здесь, кладем данные в очередь для writer-а
                            write_queue.put((output_path, X, Y))
                            logging.info(f"Queued file for writing: {output_path}")
                        except Exception as e:
                            logging.error(f"Error preparing output for {filepath}: {e}")
                except TimeoutError:
                    logging.error(f"Timeout processing {filepath}")
                    future.cancel()
                except Exception as e:
                    logging.error(f"Error processing {filepath}: {e}")
                finally:
                    try:
                        os.remove(filepath)
                        logging.info(f"Deleted archive: {filepath}")
                    except Exception as e:
                        logging.error(f"Failed to delete file {filepath}: {e}")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, stopping...")
        stop_event.set()
    finally:
        downloader_thread.join()
        # Ждем, пока все задачи записи будут обработаны
        while not write_queue.empty():
            time.sleep(0.5)
        stop_event.set()
        writer_thread.join()
        logging.info("Processing finished.")

if __name__ == '__main__':
    main()
