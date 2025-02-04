# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
from dataset import LOBDataset
from model import OrderBookTransformer
from config import (
    TRAIN_NUM_EPOCHS, TRAIN_BATCH_SIZE, TRAINING_DATE_RANGE, SEQUENCE_LENGTH,
    HORIZON_MS, NUM_LEVELS, PENALTY_FACTOR, MIN_SIGNAL_PERCENT, URL_TEMPLATE
)
import numpy as np
from tqdm import tqdm
from dataset import generate_date_urls  # импортируем функцию генерации URL

# Для обучения используем список URL, сгенерированный по диапазону дат
TRAINING_DATA_URLS = ",".join(generate_date_urls(TRAINING_DATE_RANGE, URL_TEMPLATE))

def custom_loss(predictions, targets, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR):
    """
    Функция потерь, состоящая из MSE и дополнительного штрафа:
      - Если для лонга (prediction > min_signal_percent) целевое изменение оказывается отрицательным,
      - Если для шорта (prediction < -min_signal_percent) целевое изменение оказывается положительным.
    Штраф пропорционален квадрату размера убытка.
    """
    mse = (predictions - targets) ** 2
    long_mask = predictions > min_signal_percent
    long_penalty = torch.where(long_mask & (targets < 0), (-targets) ** 2, torch.zeros_like(targets))
    short_mask = predictions < -min_signal_percent
    short_penalty = torch.where(short_mask & (targets > 0), (targets) ** 2, torch.zeros_like(targets))
    penalty = long_penalty + short_penalty
    total_loss = mse + penalty_factor * penalty
    return total_loss.mean()

def objective(trial):
    model_dim = trial.suggest_int("model_dim", 64, 256)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    nhead = trial.suggest_int("nhead", 2, 8)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    
    dataset = LOBDataset(TRAINING_DATA_URLS, sequence_length=SEQUENCE_LENGTH, horizon_ms=HORIZON_MS, num_levels=NUM_LEVELS)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    
    input_dim = NUM_LEVELS * 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrderBookTransformer(input_dim=input_dim, model_dim=model_dim, num_layers=num_layers, nhead=nhead, dropout=dropout)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_epochs = TRAIN_NUM_EPOCHS
    pbar_epoch = tqdm(range(num_epochs), desc="Epochs")
    for epoch in pbar_epoch:
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for sequences, targets in pbar:
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = custom_loss(outputs, targets, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = custom_loss(outputs, targets, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR)
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_dataset)
        pbar_epoch.set_postfix(val_loss=val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = trial.params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = NUM_LEVELS * 4
    final_model = OrderBookTransformer(
        input_dim=input_dim,
        model_dim=best_params["model_dim"],
        num_layers=best_params["num_layers"],
        nhead=best_params["nhead"],
        dropout=best_params["dropout"]
    )
    final_model.to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
    
    dataset = LOBDataset(TRAINING_DATA_URLS, sequence_length=SEQUENCE_LENGTH, horizon_ms=HORIZON_MS, num_levels=NUM_LEVELS)
    train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    
    pbar_epoch = tqdm(range(TRAIN_NUM_EPOCHS), desc="Final Model Epochs")
    for epoch in pbar_epoch:
        final_model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for sequences, targets in pbar:
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = final_model(sequences)
            loss = custom_loss(outputs, targets, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * sequences.size(0)
            pbar.set_postfix(loss=loss.item())
        epoch_loss /= len(dataset)
        pbar_epoch.set_postfix(epoch_loss=epoch_loss)
        print(f"Final Model Epoch {epoch+1}/{TRAIN_NUM_EPOCHS}, Loss: {epoch_loss:.6f}")
    
    torch.save(final_model.state_dict(), "final_model.pth")
    print("Final model saved as final_model.pth")
