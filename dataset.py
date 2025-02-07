#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import re
import threading
import queue
import time
import json
import zipfile
import io
import requests
import numpy as np
from tqdm import tqdm
from config import TRAINING_DATE_RANGES, URL_TEMPLATE, SYMBOLS, SEQUENCE_LENGTH, HORIZON_MS, MAX_TARGET_CHANGE_PERCENT, CANDLE_INTERVAL_MIN, CANDLE_TOTAL_HOURS
from orderbook import Orderbook

def generate_date_urls(date_range, template):
    start_str, end_str = date_range.split(',')
    from datetime import datetime, timedelta
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

def compute_candle_features_from_candles(candles, end_ts):
    """
    Для свечей, завершившихся до end_ts, вычисляет для каждой свечи:
      - return = (close - open)/open
      - range = (high - low)/open
    Возвращает одномерный numpy-массив.
    """
    feats = []
    for candle in candles:
        if candle.get('end', 0) <= end_ts:
            o = candle['open']
            c = candle['close']
            h = candle['high']
            l = candle['low']
            if o != 0:
                ret = (c - o) / o
                rng = (h - l) / o
            else:
                ret, rng = 0, 0
            feats.extend([ret, rng])
    return np.array(feats, dtype=np.float32)

def process_archive_file(filepath):
    print(f"Processing archive: {filepath}")
    try:
        with open(filepath, "rb") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except Exception as e:
        print(f"Error opening zip file {filepath}: {e}")
        return []
    # Создаем новый экземпляр Orderbook
    ob = Orderbook()
    # Для каждого файла внутри архива
    for file_name in zf.namelist():
        with zf.open(file_name) as f:
            for line in f:
                try:
                    record = json.loads(line.decode('utf-8').strip())
                except Exception:
                    continue
                data = record.get('data', {})
                r_type = record.get('type')
                ts = record.get('ts')
                try:
                    ts = int(ts)
                except:
                    ts = int(time.time())
                ob.update(data, r_type, timestamp=ts)
    zf.close()
    # Извлекаем накопленные фичи: sequence (snapshot-ы) и свечи
    features = ob.get_features()  # {'sequence': [...], 'candles': [...]}
    sequence = features.get('sequence', [])
    candles = features.get('candles', [])
    samples = []
    n = len(sequence)
    # Генерируем обучающие примеры: для каждого snapshot, ищем будущий snapshot, чтобы вычислить target_delta
    for i in range(n):
        current_snap = sequence[i]
        current_ts = current_snap.get('ts')
        if current_ts is None:
            continue
        target_index = None
        for j in range(i+1, n):
            if sequence[j].get('ts', 0) >= current_ts + HORIZON_MS:
                target_index = j
                break
        if target_index is None:
            break
        start_mid = current_snap.get('mid')
        target_mid = sequence[target_index].get('mid')
        if start_mid is None or target_mid is None or start_mid == 0:
            continue
        target_delta = (target_mid - start_mid) / start_mid
        if abs(target_delta) > MAX_TARGET_CHANGE_PERCENT:
            continue
        if i - SEQUENCE_LENGTH + 1 < 0:
            continue
        seq_window = sequence[i - SEQUENCE_LENGTH + 1 : i+1]
        # Преобразуем окно snapshot-ов: для каждого берём [bid, ask, mid]
        lob_features = []
        for snap in seq_window:
            lob_features.extend([snap.get('bid', 0.0), snap.get('ask', 0.0), snap.get('mid', 0.0)])
        # Используем свечи, завершившиеся до current_ts
        candle_feats = compute_candle_features_from_candles(candles, current_ts)
        combined = np.array(lob_features + list(candle_feats), dtype=np.float32)
        samples.append({"features": combined, "target": np.float32(target_delta)})
    return samples

def download_archive(url, zips_dir):
    try:
        r = requests.get(url)
        r.raise_for_status()
        filename = url.split("/")[-1]
        filepath = os.path.join(zips_dir, filename)
        with open(filepath, "wb") as f:
            f.write(r.content)
        print(f"Downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def processor_worker(filepath):
    print(f"Started processing {filepath}")
    return process_archive_file(filepath)

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
                        help="Index of pair from SYMBOLS (0-based). If not specified, process all pairs.")
    args = parser.parse_args()
    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

    all_urls = build_dataset_for_pairs(selected_pairs)
    print(f"Total archives to process: {len(all_urls)}")
    os.makedirs("zips", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    download_queue = queue.Queue(maxsize=100)
    write_queue = queue.Queue()
    stop_event = threading.Event()

    def downloader():
        from tqdm import tqdm
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

    def writer_thread_func():
        while not stop_event.is_set() or not write_queue.empty():
            try:
                output_path, X, Y = write_queue.get(timeout=1)
                np.savez_compressed(output_path, X=X, Y=Y)
                print(f"Written: {output_path} (samples: {X.shape[0]})")
            except queue.Empty:
                continue

    downloader_thread = threading.Thread(target=downloader)
    writer_thread = threading.Thread(target=writer_thread_func)
    downloader_thread.start()
    writer_thread.start()

    regex = re.compile(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)_ob500\.data\.zip")
    futures = []
    num_workers = os.cpu_count() or 4

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
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
            for future in as_completed(futures, timeout=300):
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
                        features_list = [s["features"] for s in samples]
                        targets_list = [s["target"] for s in samples]
                        X = np.stack(features_list)
                        Y = np.array(targets_list)
                        archive_id = os.path.splitext(os.path.basename(filepath))[0]
                        output_path = os.path.join("data", f"{key}_{archive_id}.npz")
                        write_queue.put((output_path, X, Y))
                        print(f"Queued {output_path} for writing")
                except TimeoutError:
                    print(f"Timeout processing {filepath}")
                    future.cancel()
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                finally:
                    try:
                        os.remove(filepath)
                        print(f"Deleted {filepath}")
                    except Exception as e:
                        print(f"Failed to delete {filepath}: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping...")
        stop_event.set()
    finally:
        downloader_thread.join()
        while not write_queue.empty():
            time.sleep(0.5)
        stop_event.set()
        writer_thread.join()
        print("Processing finished.")

if __name__ == '__main__':
    main()
