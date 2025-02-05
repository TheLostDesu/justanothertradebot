# dataset_builder.py
import pickle
from dataset import LOBDataset, generate_date_urls
from config import TRAINING_DATE_RANGES, URL_TEMPLATE, NUM_LEVELS, SEQUENCE_LENGTH, HORIZON_MS, SYMBOLS
import numpy as np
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

def build_dataset():
    all_urls = []
    # Для каждой даты и для каждой торговой пары из SYMBOLS формируем URL
    for dr in TRAINING_DATE_RANGES:
        for sym in SYMBOLS:
            pair = sym.replace("/", "")
            # Подставляем пару в шаблон URL
            urls = generate_date_urls(dr, URL_TEMPLATE.replace("{pair}", pair))
            all_urls.extend(urls)
    print(f"Total archives to process: {len(all_urls)}")
    # Преобразуем итератор tqdm в список, чтобы передать его в конструктор датасета
    dataset = LOBDataset(list(tqdm(all_urls, desc="Processing archives")), 
                         sequence_length=SEQUENCE_LENGTH, 
                         horizon_ms=HORIZON_MS, 
                         num_levels=NUM_LEVELS)
    features = []
    targets = []
    # Оборачиваем итерацию по датасету в tqdm, если набор данных большой
    for feat, target in tqdm(dataset, desc="Building dataset samples"):
        features.append(feat)
        targets.append(target)
    features = np.array(features)
    targets = np.array(targets)
    return {"features": features, "targets": targets}

if __name__ == '__main__':
    dataset_data = build_dataset()
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset_data, f)
    print(f"Dataset built and saved. Samples: {len(dataset_data['features'])}")
