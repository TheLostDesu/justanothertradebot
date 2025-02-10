#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import optuna
import numpy as np
from tqdm import tqdm
import os

# Импортируем модель и необходимые константы из config.py.
from model import CombinedModel
from config import (
    TRAIN_NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    MIN_SIGNAL_PERCENT,
    PENALTY_FACTOR,
    TOTAL_INPUT_DIM,       # Новый входной размер (например, 2520)
    SEQUENCE_LENGTH,
    MODEL_DIM,
    NUM_LAYERS,
    NHEAD,
    DROPOUT
)

print(f"TOTAL_INPUT_DIM = {TOTAL_INPUT_DIM}")

def process_input(data):
    processed_data = []

    for snapshot in data:
        processed_snapshot = []
        for entry in snapshot:
            if 'bid' in entry.keys():
                processed_snapshot.extend([entry['bid'], entry['ask'], entry['mid']])
            else:
                processed_snapshot.extend([entry['open'], entry['high'], entry['low'], entry['close']])
        processed_data.append(np.array(processed_snapshot))
    
    return processed_data

############################################
# 1. Определяем кастомный лосс
############################################
def custom_loss(predictions: torch.Tensor,
                targets: torch.Tensor,
                min_signal_percent: float = MIN_SIGNAL_PERCENT,
                penalty_factor: float = PENALTY_FACTOR) -> torch.Tensor:
    mse = (predictions - targets) ** 2

    long_mask = predictions > min_signal_percent
    long_penalty = torch.where(long_mask & (targets < 0),
                               (-targets) ** 2,
                               torch.zeros_like(targets))

    short_mask = predictions < -min_signal_percent
    short_penalty = torch.where(short_mask & (targets > 0),
                                (targets) ** 2,
                                torch.zeros_like(targets))
    penalty = long_penalty + short_penalty
    return (mse + penalty_factor * penalty).mean()

############################################
# 2. Загружаем датасет из NPZ-файла
############################################
def load_dataset(npz_path: str = "data/2024-05-02_BTCUSDT_ob500.data.npz") -> TensorDataset:
    """
    Ожидается, что в NPZ-файле содержатся массивы:
      - "X": numpy-массив с признаками размерности [N, SEQUENCE_LENGTH, SNAPSHOT_SIZE]
      - "Y": numpy-массив с целевыми значениями размерности [N] или [N, 1]
    """
    data = np.load(npz_path, allow_pickle=True)
    # Преобразуем признаки в новый формат
    features = np.array(process_input(data["X"]), dtype=np.float32)
    
    # Проверяем размерность данных
    # actual_seq_length = features.shape[1]
    # if actual_seq_length != SEQUENCE_LENGTH:
    #     raise ValueError(f"Размерность последовательности в датасете ({actual_seq_length}) не совпадает с SEQUENCE_LENGTH ({SEQUENCE_LENGTH}).")
    
    # Обрабатываем целевые значения (Y)
    targets = np.array(data["Y"], dtype=np.float32)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(axis=1)
    
    # Создаём TensorDataset для обучения
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    return dataset


############################################
# 3. Настраиваем Optuna (objective + обучение)
############################################
def objective(trial: optuna.trial.Trial,
              train_dataset: TensorDataset,
              val_dataset: TensorDataset,
              device: torch.device) -> float:
    # Сначала выбираем nhead, затем model_dim так, чтобы model_dim был кратен nhead.
    nhead = trial.suggest_int("nhead", 2, 8)
    model_dim = trial.suggest_int("model_dim", nhead, 256, step=nhead)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    model = CombinedModel(
        input_dim=TOTAL_INPUT_DIM,
        model_dim=model_dim,
        num_layers=num_layers,
        nhead=nhead,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    
    for epoch in range(TRAIN_NUM_EPOCHS):
        print(f'Эпоха {epoch} оптимайза пошла')
        model.train()
        print(f'Трейн {epoch} оптимайза закончился')
        train_loss_accum = 0.0
        c = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = custom_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * X_batch.size(0)
            print(c)
            c += 1
        train_loss = train_loss_accum / len(train_dataset)
        print(f'eval {epoch}')
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                preds_val = model(X_val)
                loss_val = custom_loss(preds_val, y_val)
                val_loss_accum += loss_val.item() * X_val.size(0)
        val_loss = val_loss_accum / len(val_dataset)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

############################################
# 4. Основной блок обучения
############################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print("Using device:", device)
    
    full_dataset = load_dataset("data/2024-05-02_BTCUSDT_ob500.data.npz")
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    print('Живой я !')
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device),
                   n_trials=20)
    
    print("Best trial:")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print("\nRetraining on full dataset with best parameters...")
    best_params = best_trial.params
    
    model = CombinedModel(
        input_dim=TOTAL_INPUT_DIM,
        model_dim=best_params["model_dim"],
        num_layers=best_params["num_layers"],
        nhead=best_params["nhead"],
        dropout=best_params["dropout"]
    ).to(device)
    
    # Если доступно больше одного GPU, оборачиваем модель в DataParallel.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    full_loader = DataLoader(full_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    
    for epoch in range(TRAIN_NUM_EPOCHS):
        print(f"Эпоха {epoch} почалась")
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(full_loader, desc=f"Epoch {epoch+1}/{TRAIN_NUM_EPOCHS}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = custom_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(full_dataset)
        print(f"Epoch {epoch+1}/{TRAIN_NUM_EPOCHS}, Loss: {epoch_loss:.6f}")
    
    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved as final_model.pth")
