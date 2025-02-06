#!/usr/bin/env python3
import argparse
import threading
import queue
import os
import pickle
import re
from tqdm import tqdm
import requests
import io
import json
import zipfile
import numpy as np

# Импортируем функцию генерации URL-ов из диапазона дат
from dataset import generate_date_urls
from config import TRAINING_DATE_RANGES, URL_TEMPLATE, NUM_LEVELS, SEQUENCE_LENGTH, HORIZON_MS, SYMBOLS

# --- Функция скачивания архива ---
def download_archive(url, zips_dir):
    """
    Скачивает архив по URL и сохраняет его в папку zips.
    Возвращает полный путь к сохранённому файлу или None при ошибке.
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

# --- Функция обработки локального архива ---
def process_archive_file(filepath):
    """
    Обрабатывает локальный zip-файл и возвращает список sample-словарей.
    Каждый sample имеет ключи:
      "features": np.array(...), "target": np.float32(...)
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
    # Импортируем необходимые функции из dataset.py
    from dataset import get_features_from_orderbook, get_mid_price, compute_candle_features
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
        from config import MAX_TARGET_CHANGE_PERCENT
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

# --- Продюсер: скачивает архивы и кладёт их в очередь ---
def downloader(all_urls, zips_dir, download_queue):
    for url in tqdm(all_urls, desc="Downloading archives"):
        filepath = download_archive(url, zips_dir)
        if filepath:
            download_queue.put(filepath)

# --- Консьюмер: обрабатывает архивы из очереди и сохраняет результат ---
def processor(download_queue):
    while True:
        try:
            filepath = download_queue.get(timeout=10)
        except queue.Empty:
            break
        samples = process_archive_file(filepath)
        # Извлекаем дату и пару из имени файла
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
        download_queue.task_done()

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

    # Запускаем продюсер, который скачивает архивы
    downloader_thread = threading.Thread(target=downloader, args=(all_urls, "zips", download_queue))
    downloader_thread.start()

    # Запускаем несколько воркеров для обработки скачанных архивов
    processor_threads = []
    for _ in range(4):
        t = threading.Thread(target=processor, args=(download_queue,))
        t.start()
        processor_threads.append(t)

    downloader_thread.join()
    download_queue.join()
    for t in processor_threads:
        t.join()

    print("All archives processed.")

if __name__ == '__main__':
    main()
