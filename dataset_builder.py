#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pickle
import re
from tqdm import tqdm

from dataset import process_archive, generate_date_urls
from config import TRAINING_DATE_RANGES, URL_TEMPLATE, NUM_LEVELS, SEQUENCE_LENGTH, HORIZON_MS, SYMBOLS

def build_dataset_for_pairs(selected_pairs):
    all_urls = []
    # Для каждой даты и для каждой выбранной пары формируем URL
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
