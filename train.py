import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import optuna
import numpy as np
from tqdm import tqdm
import pickle

# Импортируем вашу модель и конфиг:
from model import CombinedModel, TOTAL_INPUT_DIM
from config import (
    TRAIN_NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    MIN_SIGNAL_PERCENT,
    PENALTY_FACTOR
)

############################################
# 1. Определяем функцию кастомного лосса.
############################################
def custom_loss(predictions: torch.Tensor,
                targets: torch.Tensor,
                min_signal_percent: float = MIN_SIGNAL_PERCENT,
                penalty_factor: float = PENALTY_FACTOR) -> torch.Tensor:
    """
    MSE + дополнительный штраф за «сильные» лонг/шорт‑сигналы, ошибочные по направлению.
    - Если outputs > min_signal_percent, значит модель рекомендует «лонг».
      Если при этом real < 0, штрафуем дополнительно.
    - Если outputs < -min_signal_percent, значит «шорт».
      Если при этом real > 0, штрафуем дополнительно.
    """
    # Основная MSE-составляющая
    mse = (predictions - targets) ** 2

    # Штраф за неправильный лонг
    long_mask = predictions > min_signal_percent
    # Если модель «сильно» советует лонг, а реальный таргет < 0, штрафуем сильнее
    long_penalty = torch.where(long_mask & (targets < 0),
                               (-targets) ** 2,
                               torch.zeros_like(targets))

    # Штраф за неправильный шорт
    short_mask = predictions < -min_signal_percent
    # Если модель «сильно» советует шорт, а реальный таргет > 0, штрафуем сильнее
    short_penalty = torch.where(short_mask & (targets > 0),
                                (targets) ** 2,
                                torch.zeros_like(targets))

    penalty = long_penalty + short_penalty
    return (mse + penalty_factor * penalty).mean()


############################################
# 2. Загружаем датасет
############################################
def load_dataset(npz_path: str = "dataset.npz") -> TensorDataset:
    """
    Ожидается, что в `dataset.npz` есть массивы:
      - features (X) размерностью [N, feature_dim]
      - targets (Y) размерностью [N] или [N, 1]
    """
    data = np.load(npz_path)
    features = data["features"].astype(np.float32)  # приведение к float32
    targets = data["targets"].astype(np.float32)
    
    # Если targets имеют форму [N, 1], можно преобразовать к [N]
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(axis=1)
    
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets)
    )
    return dataset


############################################
# 3. Настраиваем Optuna (objective + обучение)
############################################
def objective(trial: optuna.trial.Trial,
              train_dataset: TensorDataset,
              val_dataset: TensorDataset,
              device: torch.device) -> float:
    """
    Функция objective для Optuna:
      - Получаем гиперпараметры из trial
      - Создаём модель и оптимизатор
      - Обучаем несколько эпох
      - Возвращаем val_loss
    """
    # Выбираем гиперпараметры
    model_dim = trial.suggest_int("model_dim", 64, 256)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    nhead = trial.suggest_int("nhead", 2, 8)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    
    # Создаём модель
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
    
    # Обучаем несколько эпох, отслеживая val_loss
    for epoch in range(TRAIN_NUM_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = custom_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item() * X_batch.size(0)
        train_loss = train_loss_accum / len(train_dataset)
        
        # Валидация
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
        
        # Сообщаем Optuna о текущем значении лосса на валидации
        trial.report(val_loss, epoch)
        
        # Проверка на условие прерывания (pruning)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss


############################################
# 4. Основной блок скрипта
############################################
if __name__ == '__main__':
    # Определяем устройство (GPU / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Загружаем полный датасет
    full_dataset = load_dataset("dataset.npz")

    # Разделяем на train/val
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)  # 80% на обучение
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Создаём Optuna-исследование
    study = optuna.create_study(direction="minimize")
    
    # Запускаем процесс оптимизации (количество проб – n_trials=20)
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, device),
        n_trials=20
    )

    print("Best trial:")
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # После нахождения лучших гиперпараметров — переобучаем на ВСЁМ датасете
    best_params = best_trial.params
    print("\nПерезапуск обучения на всём датасете с лучшими параметрами...")

    model = CombinedModel(
        input_dim=TOTAL_INPUT_DIM,
        model_dim=best_params["model_dim"],
        num_layers=best_params["num_layers"],
        nhead=best_params["nhead"],
        dropout=best_params["dropout"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    train_loader = DataLoader(full_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    for epoch in range(TRAIN_NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_NUM_EPOCHS}"):
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

    # Сохраняем финальную модель
    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved as final_model.pth")
