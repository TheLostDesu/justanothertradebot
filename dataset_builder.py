#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pickle
import re
from datetime import datetime, timedelta

import numpy as np
import requests
from tqdm import tqdm

from config import TRAINING_DATE_RANGES, URL_TEMPLATE, NUM_LEVELS, SEQUENCE_LENGTH, HORIZON_MS, SYMBOLS, MAX_TARGET_CHANGE_PERCENT

# Функция для обработки одного архива (см. dataset.py для аналогичной логики)
def process_archive(url):
    import io, json, zipfile
    def open_zip(url):
        try:
            r = requests.get(url)
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    from dataset import get_features_from_orderbook, get_mid_price, compute_candle_features
    content = open_zip(url)
    if content is None:
        return []
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except Exception as e:
        print(f"Error opening zip from {url}: {e}")
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

def main():
    parser = argparse.ArgumentParser(description="Dataset builder for trade bot.")
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Номер пары из списка SYMBOLS (0-based). Если не указан, обрабатываются все пары.")
    args = parser.parse_args()

    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

    all_urls = []
    for dr in TRAINING_DATE_RANGES:
        for sym in selected_pairs:
            pair = sym.replace("/", "")
            from dataset import generate_date_urls
            urls = generate_date_urls(dr, URL_TEMPLATE.replace("{pair}", pair))
            all_urls.extend(urls)
    print(f"Total archives to process: {len(all_urls)}")
    os.makedirs("data", exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_archive, url): url for url in all_urls}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing archives"):
            url = futures[future]
            try:
                samples = future.result()
                if samples:
                    m = re.search(r"/(\d{4}-\d{2}-\d{2})_([A-Z]+)_ob500\.data\.zip", url)
                    if m:
                        date_str = m.group(1)
                        pair = m.group(2)
                    else:
                        date_str = "unknown"
                        pair = "unknown"
                    filename = f"data/{pair}_{date_str}.pkl"
                    with open(filename, "wb") as f:
                        pickle.dump({"samples": samples}, f)
                    results.append((url, len(samples)))
            except Exception as e:
                print(f"Error processing {url}: {e}")
    total_samples = sum(s for _, s in results)
    print(f"Dataset built and saved. Total samples: {total_samples}")

if __name__ == '__main__':
    import argparse, concurrent.futures, re, os
    main()
