# dataset.py
import json
import zipfile
import io
import os
import requests
import numpy as np
from torch.utils.data import Dataset
from config import SEQUENCE_LENGTH, HORIZON_MS, NUM_LEVELS

def open_zip(zip_path_or_url):
    """Открывает zip‑архив из URL или локального пути."""
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
    """
    Формирует вектор признаков из текущего состояния стакана.
    Входной формат: ob = { 'a': {price: size, ...}, 'b': {price: size, ...} }
    Сортировка: bids – по убыванию, asks – по возрастанию.
    Если уровней меньше num_levels, дополняем нулями.
    Возвращает список длины num_levels*4.
    """
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
    """
    Вычисляет mid‑price как среднее между лучшим bid и лучшей ask.
    """
    bids = sorted(ob['b'].items(), key=lambda x: x[0], reverse=True)
    asks = sorted(ob['a'].items(), key=lambda x: x[0])
    if not bids or not asks:
        return None
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    return (best_bid + best_ask) / 2.0

class LOBDataset(Dataset):
    def __init__(self, zip_sources, sequence_length=SEQUENCE_LENGTH, horizon_ms=HORIZON_MS, num_levels=NUM_LEVELS):
        """
        Аргументы:
          zip_sources: строка с URL/путями к zip‑архивам, разделёнными запятой, или список строк.
          sequence_length: число последовательных snapshot'ов.
          horizon_ms: горизонт для расчёта целевой дельты (будет переведён в процентное изменение).
          num_levels: число уровней с каждой стороны стакана.
        """
        self.sequence_length = sequence_length
        self.horizon_ms = horizon_ms
        self.num_levels = num_levels
        self.samples = []

        if isinstance(zip_sources, str):
            zip_sources = [s.strip() for s in zip_sources.split(',')]
        records = []
        current_ob = None

        for src in zip_sources:
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
                        elif record.get('type') == 'delta':
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
                                            current_ob[side].pop(price, None)
                                        else:
                                            current_ob[side][price] = size
                        else:
                            continue
                        if current_ob is None:
                            continue
                        mid = get_mid_price(current_ob)
                        if mid is None:
                            continue
                        feats = get_features_from_orderbook(current_ob, self.num_levels)
                        records.append({
                            'ts': record.get('ts'),
                            'features': feats,
                            'mid_price': mid
                        })
            zf.close()

        records.sort(key=lambda x: x['ts'])
        n = len(records)
        for i in range(n):
            start_ts = records[i]['ts']
            target_index = None
            for j in range(i+1, n):
                if records[j]['ts'] >= start_ts + self.horizon_ms:
                    target_index = j
                    break
            if target_index is None:
                break
            # Вычисляем процентное изменение: (future - current) / current
            start_mid = records[i]['mid_price']
            target_mid = records[target_index]['mid_price']
            target_delta = (target_mid - start_mid) / start_mid
            if i - self.sequence_length + 1 < 0:
                continue
            seq_features = []
            for k in range(i - self.sequence_length + 1, i + 1):
                seq_features.append(records[k]['features'])
            self.samples.append({
                'sequence': np.array(seq_features, dtype=np.float32),
                'target': np.float32(target_delta)
            })
        print(f"Total records: {len(records)}. Total samples: {len(self.samples)}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['sequence'], sample['target']
