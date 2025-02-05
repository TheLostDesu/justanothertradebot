# dataset.py
import json
import zipfile
import io
import os
import requests
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from config import SEQUENCE_LENGTH, HORIZON_MS, NUM_LEVELS, TRAINING_DATE_RANGES, URL_TEMPLATE, MAX_TARGET_CHANGE_PERCENT, CANDLE_INTERVAL_MIN, CANDLE_TOTAL_HOURS, CANDLE_FEATURES_PER_CANDLE

def generate_date_urls(date_range, template):
    start_str, end_str = date_range.split(',')
    start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
    end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        # Для обучения подбираем данные по каждой паре; здесь для упрощения используем BTCUSDT, но в дальнейшем можно делать цикл по TRAINING_PAIRS
        urls.append(template.format(pair="BTCUSDT", date=date_str))
        current_date += timedelta(days=1)
    return urls

def open_zip(zip_path_or_url):
    if zip_path_or_url.startswith("http"):
        try:
            r = requests.get(zip_path_or_url)
            r.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"Error downloading {zip_path_or_url}: {e}")
            return None
    else:
        if not os.path.exists(zip_path_or_url):
            print(f"File {zip_path_or_url} not found.")
            return None
        return zipfile.ZipFile(zip_path_or_url, 'r')

def get_features_from_orderbook(ob, num_levels=NUM_LEVELS):
    features = []
    bids = sorted(ob['b'].items(), key=lambda x: x[0], reverse=True)
    for i in range(num_levels):
        if i < len(bids):
            price, size = bids[i]
            features.extend([price, size])
        else:
            features.extend([0.0, 0.0])
    asks = sorted(ob['a'].items(), key=lambda x: x[0])
    for i in range(num_levels):
        if i < len(asks):
            price, size = asks[i]
            features.extend([price, size])
        else:
            features.extend([0.0, 0.0])
    return features

def get_mid_price(ob):
    bids = sorted(ob['b'].items(), key=lambda x: x[0], reverse=True)
    asks = sorted(ob['a'].items(), key=lambda x: x[0])
    if not bids or not asks:
        return None
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    return (best_bid + best_ask) / 2.0

def compute_candle_features(records, end_time):
    """
    Вычисляет свечные признаки за 5 часов до end_time.
    Для каждого 5-минутного интервала вычисляется:
      - return = (close - open) / open
      - range = (high - low) / open
    Если данных нет, используются 0.
    Возвращает список признаков длиной: (CANDLE_TOTAL_HOURS*60/CANDLE_INTERVAL_MIN)*CANDLE_FEATURES_PER_CANDLE
    """
    interval_ms = CANDLE_INTERVAL_MIN * 60 * 1000
    candle_count = int((CANDLE_TOTAL_HOURS * 60) / CANDLE_INTERVAL_MIN)
    start_time = end_time - (CANDLE_TOTAL_HOURS * 60 * 1000)
    bucket_data = [[] for _ in range(candle_count)]
    for rec in records:
        ts = rec["ts"]
        if start_time <= ts < end_time:
            bucket_index = int((ts - start_time) // interval_ms)
            bucket_data[bucket_index].append(rec["mid_price"])
    features = []
    for bucket in bucket_data:
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

class LOBDataset(Dataset):
    def __init__(self, zip_sources, sequence_length=SEQUENCE_LENGTH, horizon_ms=HORIZON_MS, num_levels=NUM_LEVELS):
        self.sequence_length = sequence_length
        self.horizon_ms = horizon_ms
        self.num_levels = num_levels
        self.samples = []
        self.records = []
        all_urls = []
        # Объединяем URL из всех диапазонов
        if isinstance(zip_sources, list):
            for dr in zip_sources:
                all_urls.extend(generate_date_urls(dr, URL_TEMPLATE))
        else:
            all_urls = generate_date_urls(zip_sources, URL_TEMPLATE)
        for src in all_urls:
            zf = open_zip(src)
            if zf is None:
                continue
            for file_name in zf.namelist():
                with zf.open(file_name) as f:
                    for line in f:
                        try:
                            record = json.loads(line.decode('utf-8').strip())
                        except Exception:
                            continue
                        data = record.get('data', {})
                        if record.get('type') == 'snapshot' or data.get('u') == 1:
                            ob = {'a': {}, 'b': {}}
                            if 'a' in data:
                                for level in data['a']:
                                    try:
                                        price = float(level[0])
                                        size = float(level[1])
                                        ob['a'][price] = size
                                    except Exception:
                                        continue
                            if 'b' in data:
                                for level in data['b']:
                                    try:
                                        price = float(level[0])
                                        size = float(level[1])
                                        ob['b'][price] = size
                                    except Exception:
                                        continue
                        elif record.get('type') == 'delta':
                            # Для обучения пропускаем delta
                            continue
                        else:
                            continue
                        mid = get_mid_price(ob)
                        if mid is None:
                            continue
                        feats = get_features_from_orderbook(ob, self.num_levels)
                        ts = record.get('ts')
                        try:
                            ts = int(ts)
                        except:
                            continue
                        rec = {"ts": ts, "features": feats, "mid_price": mid}
                        self.records.append(rec)
            zf.close()
        self.records.sort(key=lambda x: x["ts"])
        n = len(self.records)
        for i in range(n):
            start_ts = self.records[i]["ts"]
            target_index = None
            for j in range(i+1, n):
                if self.records[j]["ts"] >= start_ts + self.horizon_ms:
                    target_index = j
                    break
            if target_index is None:
                break
            start_mid = self.records[i]["mid_price"]
            target_mid = self.records[target_index]["mid_price"]
            target_delta = (target_mid - start_mid) / start_mid
            if abs(target_delta) > MAX_TARGET_CHANGE_PERCENT:
                continue
            if i - self.sequence_length + 1 < 0:
                continue
            lob_seq = []
            for k in range(i - self.sequence_length + 1, i + 1):
                lob_seq.extend(self.records[k]["features"])
            candle_feats = compute_candle_features(self.records, start_ts)
            combined = lob_seq + candle_feats
            self.samples.append({
                "features": np.array(combined, dtype=np.float32),
                "target": np.float32(target_delta)
            })
        print(f"Total records: {n}. Total samples: {len(self.samples)}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["features"], sample["target"]
