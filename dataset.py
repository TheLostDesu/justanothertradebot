#!/usr/bin/env python3
"""
Реализация потоковой генерации обучающих примеров из архивов ордербуков.

Основные моменты:
  - Снимок (snapshot) фич берётся не чаще, чем раз в 10 секунд.
  - Через 30 секунд после создания первого snapshot‑а (из очереди) считается его исход,
    и на его основе формируется обучающий пример:
      • Вход (features): окно из SEQUENCE_LENGTH snapshot‑ов, где каждый snapshot задаёт [bid, ask, mid]
      • Цель (target): относительное изменение mid‑цены, вычисленное как (current_mid - candidate_mid) / candidate_mid
  - После генерации примера первый snapshot удаляется из очереди.
  - Если в течение 30 секунд (реального времени) не появляется новый пример, функция возвращает уже накопленные данные.
  - Архивы скачиваются по URL, если их ещё нет в папке "zips".
"""

import os
import re
import io
import json
import time
import zipfile
import requests
import argparse
import numpy as np
from datetime import datetime, timedelta

# Импортируем настройки и класс Orderbook
from config import (
    TRAINING_DATE_RANGES,   # например, ["2025-01-01,2025-01-07"]
    URL_TEMPLATE,           # "https://quote-saver.bycsi.com/orderbook/linear/{pair}/{date}_{pair}_ob500.data.zip"
    SYMBOLS,                # список, например, ["BTC/USDT", "ETH/USDT", …]
    SEQUENCE_LENGTH,        # длина окна (например, 3 или 4)
    HORIZON_MS              # ожидаемый горизонт в секундах (будет использоваться как 30)
)
from orderbook import Orderbook

# Интервалы (в секундах)
SNAPSHOT_INTERVAL = 10   # снимок каждые 10 секунд
TRAINING_HORIZON  = 30   # через 30 секунд исход известен

def generate_date_urls(date_range: str, pair: str) -> list:
    """
    Для заданного диапазона дат (строка "YYYY-MM-DD,YYYY-MM-DD") и торговой пары (без "/")
    формирует список URL архивов.
    """
    start_str, end_str = date_range.split(',')
    start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
    end_date   = datetime.strptime(end_str.strip(), "%Y-%m-%d")
    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = URL_TEMPLATE.format(pair=pair, date=date_str)
        urls.append(url)
        current_date += timedelta(days=1)
    return urls

def download_archive(url: str, zips_dir: str) -> str:
    """
    Скачивает архив по указанному URL в папку zips_dir.
    Если архив уже существует, возвращает путь к нему.
    """
    filename = url.split("/")[-1]
    filepath = os.path.join(zips_dir, filename)
    if os.path.exists(filepath):
        print(f"Архив уже существует: {filepath}")
        return filepath

    print(f"Скачиваем {url} ...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"Скачано: {filepath}")
        return filepath
    except Exception as e:
        print(f"Ошибка скачивания {url}: {e}")
        return None

def process_archive_streaming(filepath: str, timeout: float = 30) -> list:
    """
    Обрабатывает архив (zip‑файл) потоково.

    Алгоритм:
      1. Обновляем Orderbook из записей архива.
      2. Каждый раз, когда Orderbook генерирует snapshot (метод get_snapshot_features),
         если с предыдущего snapshot прошло не менее 10 секунд, добавляем его в очередь.
      3. Если время текущей записи (ts) >= времени первого snapshot-а + 30 секунд,
         и если в очереди накоплено не менее SEQUENCE_LENGTH snapshot‑ов,
         формируем обучающий пример:
             - Признаки: объединяем (flatten) первые SEQUENCE_LENGTH snapshot‑ов (для каждого [bid, ask, mid]).
             - Цель: (current_mid - candidate_mid) / candidate_mid, где candidate_mid – mid первого snapshot-а.
         После этого удаляем первый snapshot из очереди.
      4. Если в течение timeout (30 секунд) в реальном времени не появляется новый пример,
         функция сразу возвращает уже накопленные результаты.
    """
    training_examples = []
    feature_queue = []      # очередь snapshot‑ов; каждый snapshot – dict с ключами: 'bid', 'ask', 'mid', 'ts'
    last_snapshot_time = None  # для контроля интервала 10 сек между snapshot‑ами
    last_generated_time = time.time()
    ob = Orderbook()

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            for filename in zf.namelist():
                with zf.open(filename) as f:
                    # Читаем архив построчно; каждая строка – JSON с данными апдейта
                    for line in f:
                        try:
                            record = json.loads(line.decode("utf-8").strip())
                        except Exception:
                            continue
                        data = record.get("data", {})
                        r_type = record.get("type")
                        ts = record.get("ts")
                        try:
                            ts = float(ts)
                        except Exception:
                            ts = time.time()
                        
                        # Обновляем ордербук; предполагается, что метод update()
                        # (при вызове с timestamp) обновляет внутреннее состояние и раз в секунду
                        # сохраняет snapshot (но мы сами фильтруем – берем только раз в 10 секунд).
                        ob.update(data, r_type, timestamp=ts)
                        
                        # Получаем snapshot из Orderbook.
                        # Важно: snapshot должен содержать помимо bid, ask, mid также метку времени.
                        snap = ob.get_snapshot_features()
                        if snap is None:
                            continue
                        # Добавляем ts в snapshot (если его там ещё нет)
                        snap["ts"] = ts
                        
                        # Добавляем snapshot в очередь, если прошло не менее 10 секунд с предыдущего
                        if last_snapshot_time is None or (ts - last_snapshot_time) >= SNAPSHOT_INTERVAL:
                            feature_queue.append(snap)
                            last_snapshot_time = ts
                            # Можно добавить отладочный вывод:
                            # print(f"Добавлен snapshot: ts={ts}")
                        
                        # Проверяем, готов ли первый snapshot в очереди (его исход известен)
                        while feature_queue:
                            candidate = feature_queue[0]
                            if ts >= candidate["ts"] + TRAINING_HORIZON:
                                # Если в очереди накоплено не менее SEQUENCE_LENGTH snapshot‑ов,
                                # формируем обучающий пример.
                                if len(feature_queue) >= SEQUENCE_LENGTH:
                                    window = feature_queue[:SEQUENCE_LENGTH]
                                    lob_features = []
                                    for snap_in_window in window:
                                        lob_features.extend([
                                            float(snap_in_window.get("bid", 0.0)),
                                            float(snap_in_window.get("ask", 0.0)),
                                            float(snap_in_window.get("mid", 0.0))
                                        ])
                                    features_vec = np.array(lob_features, dtype=np.float32)
                                    candidate_mid = float(candidate.get("mid", 0))
                                    current_mid = float(snap.get("mid", 0))
                                    if candidate_mid != 0:
                                        target_delta = (current_mid - candidate_mid) / candidate_mid
                                    else:
                                        target_delta = 0.0
                                    training_examples.append({
                                        "features": features_vec,
                                        "target": np.float32(target_delta)
                                    })
                                    # После формирования примера удаляем первый snapshot из очереди.
                                    feature_queue.pop(0)
                                    last_generated_time = time.time()
                                else:
                                    # Если в очереди недостаточно snapshot‑ов, ждем появления следующего.
                                    break
                            else:
                                break
                        
                        # Если с момента формирования последнего примера прошло более timeout секунд (реального времени),
                        # возвращаем накопленные результаты.
                        if time.time() - last_generated_time > timeout:
                            print("Достигнут timeout (30 сек) – возвращаю накопленные примеры.")
                            return training_examples
    except Exception as e:
        print(f"Ошибка при обработке архива {filepath}: {e}")
        return training_examples

    return training_examples

def main():
    parser = argparse.ArgumentParser(
        description="Построение датасета фич из архивов ордербуков (потоковая генерация примеров)."
    )
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Индекс торговой пары из SYMBOLS (0-based). Если не указан – обрабатываются все пары.")
    parser.add_argument("--download", action="store_true",
                        help="Скачивать архивы, если их нет в папке zips.")
    args = parser.parse_args()

    # Определяем, какие торговые пары обрабатывать.
    if args.pair_index is not None:
        selected_pairs = [SYMBOLS[args.pair_index]]
    else:
        selected_pairs = SYMBOLS

    # Формируем список URL архивов для выбранных пар и диапазонов дат.
    all_urls = []
    for dr in TRAINING_DATE_RANGES:
        for sym in selected_pairs:
            pair_formatted = sym.replace("/", "")
            urls = generate_date_urls(dr, pair_formatted)
            all_urls.extend(urls)
    print(f"Найдено архивов: {len(all_urls)}")

    zips_dir = "zips"
    data_dir = "data"
    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Обрабатываем каждый архив по очереди.
    for url in all_urls:
        archive_filename = url.split("/")[-1]
        archive_path = os.path.join(zips_dir, archive_filename)
        if not os.path.exists(archive_path):
            if args.download:
                archive_path = download_archive(url, zips_dir)
            else:
                print(f"Архив не найден: {archive_path}. Для скачивания используйте флаг --download")
                continue
        if archive_path is None:
            continue

        print(f"Обрабатываю архив: {archive_path}")
        examples = process_archive_streaming(archive_path, timeout=30)
        if examples:
            X = np.stack([ex["features"] for ex in examples])
            Y = np.array([ex["target"] for ex in examples], dtype=np.float32)
            output_filename = os.path.splitext(archive_filename)[0] + ".npz"
            output_path = os.path.join(data_dir, output_filename)
            np.savez_compressed(output_path, X=X, Y=Y)
            print(f"Сохранён датасет: {output_path} (образцов: {X.shape[0]})")
        else:
            print(f"Для архива {archive_path} не сформировано ни одного обучающего примера.")

if __name__ == "__main__":
    main()
