#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import pickle
import re
from tqdm import tqdm

from dataset import process_zipfile, generate_date_urls
from config import TRAINING_DATE_RANGES, URL_TEMPLATE, NUM_LEVELS, SEQUENCE_LENGTH, HORIZON_MS, SYMBOLS

def download_archive(url, out_dir="zips"):
    """Скачивает архив по URL и сохраняет его в папку out_dir."""
    try:
        import requests
        r = requests.get(url)
        r.raise_for_status()
        # Извлекаем имя файла из URL
        filename = url.split("/")[-1]
        file_path = os.path.join(out_dir, filename)
        with open(file_path, "wb") as f:
            f.write(r.content)
        return file_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def build_url_list(selected_pairs):
    """Формирует список URL архивов для выбранных пар по всем диапазонам."""
    all_urls = []
    for dr in TRAINING_DATE_RANGES:
        for sym in selected_pairs:
            pair = sym.replace("/", "")
            urls = generate_date_urls(dr, URL_TEMPLATE.replace("{pair}", pair))
            all_urls.extend(urls)
    return all_urls

def download_phase(urls, out_dir="zips"):
    os.makedirs(out_dir, exist_ok=True)
    downloaded = []
    # Скачиваем архивы последовательно (один поток)
    for url in tqdm(urls, desc="Downloading archives"):
        file_path = download_archive(url, out_dir)
        if file_path:
            downloaded.append(file_path)
    return downloaded

def process_phase(file_paths, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_zip_file, fp): fp for fp in file_paths}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing downloaded zips"):
            fp = futures[future]
            try:
                samples = future.result()
                if samples:
                    # Из имени файла извлекаем дату и пару
                    # Ожидаем имя вида <pair>_<date>_ob500.data.zip
                    m = re.search(r"([A-Z]+)_(\d{4}-\d{2}-\d{2})", os.path.basename(fp))
                    if m:
                        pair = m.group(1)
                        date_str = m.group(2)
                    else:
                        pair, date_str = "unknown", "unknown"
                    filename = os.path.join(out_dir, f"{pair}_{date_str}.pkl")
                    with open(filename, "wb") as f:
                        pickle.dump({"samples": samples}, f)
                    results.append((fp, len(samples)))
            except Exception as e:
                print(f"Error processing {fp}: {e}")
    total_samples = sum(s for _, s in results)
    print(f"Processing complete. Total samples: {total_samples}")

def process_zip_file(file_path):
    """Открывает локальный zip-файл и обрабатывает его, возвращая список samples."""
    try:
        zf = open(file_path, "rb")
        import zipfile
        zf = zipfile.ZipFile(file_path, "r")
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return []
    samples = process_zipfile(zf)
    return samples

def main():
    parser = argparse.ArgumentParser(description="Dataset builder for trade bot (two-phase: download then process).")
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Номер пары из списка SYMBOLS (0-based). Если не указан, обрабатываются все пары.")
    args = parser.parse_args()

    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

    urls = build_url_list(selected_pairs)
    print(f"Total archives to process: {len(urls)}")
    downloaded_files = download_phase(urls, out_dir="zips")
    process_phase(downloaded_files, out_dir="data")

if __name__ == '__main__':
    main()
