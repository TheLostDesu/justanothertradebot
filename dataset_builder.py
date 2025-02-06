#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pickle
import re
import threading
import queue
from datetime import datetime, timedelta
import numpy as np
import requests
import io
import json
import zipfile
from tqdm import tqdm

from config import (
    TRAINING_DATE_RANGES, URL_TEMPLATE, NUM_LEVELS, SEQUENCE_LENGTH,
    HORIZON_MS, SYMBOLS, MAX_TARGET_CHANGE_PERCENT, CANDLE_INTERVAL_MIN, CANDLE_TOTAL_HOURS
)

def generate_date_urls(date_range, template):
    """Принимает строку "YYYY-MM-DD,YYYY-MM-DD" и возвращает список URL-ов."""
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
    """Открывает zip-архив по URL или по локальному пути."""
    if isinstance(source, str) and source.startswith("http"):
        try:
            r = requests.get(source)
            r.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"Error downloading {source}: {e}")
            return None
    else:
        try:
            with open(source, "rb") as f:
                content = f.read()
            return zipfile.ZipFile(io.BytesIO(content))
        except Exception as e:
            print(f"Error opening zip file {source}: {e}")
            return None

def get_features_from_orderbook(ob, num_levels=NUM_LEVELS):
    """Извлекает признаки из словаря orderbook (bid и ask уровни)."""
    features = []
    bids = sorted(ob.get('b', {}).items(), key=lambda x: x[0], reverse=True)
    for i in range(num_levels):
        if i < len(bids):
            price, size = bids[i]
            features.extend([price, size])
        else:
            features.extend([0.0, 0.0])
    asks = sorted(ob.get('a', {}).items(), key=lambda x: x[0])
    for i in range(num_levels):
        if i < len(asks):
            price, size = asks[i]
            features.extend([price, size])
        else:
            features.extend([0.0, 0.0])
    return features

def get_mid_price(ob):
    """Возвращает среднюю цену между лучшим бидом и лучшим аском."""
    bids = sorted(ob.get('b', {}).items(), key=lambda x: x[0], reverse=True)
    asks = sorted(ob.get('a', {}).items(), key=lambda x: x[0])
    if not bids or not asks:
        return None
    return (bids[0][0] + asks[0][0]) / 2.0

def compute_candle_features(records, end_time):
    """
    Группирует записи по 5-минутным интервалам за 5 часов (60 свечей)
    и вычисляет для каждого интервала два признака: return и range.
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

def process_archive_file(filepath):
    """
    Обрабатывает локальный zip-файл (архив) и возвращает список sample-словарей,
    где каждый sample имеет ключи "features" (np.array) и "target" (np.float32).
    """
    try:
        with open(filepath, "rb") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return []
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except Exception as e:
        print(f"Error opening zip file {filepath}: {e}")
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
                    current_ob = {'a': {}, 'b': {}}
                    if 'a' in data:
                        for level in data['a']:
                            try:
                                price = float(level[0])
                                size = float(level[1])
                                current_ob['a'][price] = size
                            except Exception:
                                continue
                    if 'b' in data:
                        for level in data['b']:
                            try:
                                price = float(level[0])
                                size = float(level[1])
                                current_ob['b'][price] = size
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
                                if size == 0.0:
                                    if price in current_ob[side]:
                                        del current_ob[side][price]
                                else:
                                    current_ob[side][price] = size
                else:
                    continue
                if current_ob is None:
                    continue
                mid = get_mid_price(current_ob)
                if mid is None:
                    continue
                feats = get_features_from_orderbook(current_ob, NUM_LEVELS)
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
    return samples

# ===============================
# Продюсер и консьюмер
# ===============================
def download_archive(url, zips_dir):
    """
    Скачивает архив по URL и сохраняет его в папку zips.
    Возвращает путь к файлу или None при ошибке.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        filename = url.split("/")[-1]
        filepath = os.path.join(zips_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# ===============================
# Основной блок
# ===============================
def build_dataset_for_pairs(selected_pairs):
    all_urls = []
    for dr in TRAINING_DATE_RANGES:
        for sym in selected_pairs:
            pair = sym.replace("/", "")
            urls = generate_date_urls(dr, URL_TEMPLATE.replace("{pair}", pair))
            all_urls.extend(urls)
    return all_urls

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
    print(f"Total archives to process: {len(all_urls)}")
    os.makedirs("zips", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    download_queue = queue.Queue()

    # Продюсер: скачивает архивы и кладёт их в очередь
    def downloader():
        for url in tqdm(all_urls, desc="Downloading archives"):
            filepath = download_archive(url, "zips")
            if filepath:
                download_queue.put(filepath)

    # Консьюмер: обрабатывает архивы из очереди
    def processor_worker(filepath):
        print(f"Начинаю обрабатывать архив {os.path.basename(filepath)}")
        return process_archive_file(filepath)

    # Запускаем продюсера в отдельном потоке
    downloader_thread = threading.Thread(target=downloader)
    downloader_thread.start()

    # Используем ProcessPoolExecutor для обработки архивов
    processed_results = []
    num_workers = os.cpu_count() or 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        # Пока продюсер работает или очередь не пуста, извлекаем файлы и отправляем задачи
        while downloader_thread.is_alive() or not download_queue.empty():
            try:
                filepath = download_queue.get(timeout=1)
                futures.append(executor.submit(processor_worker, filepath))
                futures[-1].filepath = filepath
            except queue.Empty:
                continue

        # Обрабатываем результаты задач
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing downloaded archives"):
            filepath = getattr(future, "filepath", "unknown")
            try:
                samples = future.result()
                m = re.search(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_ob500\.data\.zip", os.path.basename(filepath))
                if m:
                    date_str = m.group(1)
                    pair = m.group(2)
                else:
                    date_str = "unknown"
                    pair = "unknown"
                out_filename = f"data/{pair}_{date_str}.pkl"
                with open(out_filename, "wb") as f:
                    pickle.dump({"samples": samples}, f)
                processed_results.append((filepath, len(samples)))
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    downloader_thread.join()
    total_samples = sum(s for _, s in processed_results)
    print(f"Dataset built and saved. Total samples: {total_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
