import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import optuna
from model import CombinedModel, TOTAL_INPUT_DIM
from config import TRAIN_NUM_EPOCHS, TRAIN_BATCH_SIZE, MIN_SIGNAL_PERCENT, PENALTY_FACTOR
import numpy as np
from tqdm import tqdm
import pickle

# Загружаем датасет
with open("dataset.pkl", "rb") as f:
    dataset_data = pickle.load(f)
features = dataset_data["features"]
targets = dataset_data["targets"]
full_dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))

n_total = len(full_dataset)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_loss(predictions, targets, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR):
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
    
    model = CombinedModel(input_dim=TOTAL_INPUT_DIM, model_dim=model_dim, num_layers=num_layers, nhead=nhead, dropout=dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    
    num_epochs = TRAIN_NUM_EPOCHS
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = custom_loss(outputs, targets_batch, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features_batch.size(0)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features_batch, targets_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)
                outputs = model(features_batch)
                loss = custom_loss(outputs, targets_batch, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR)
                val_loss += loss.item() * features_batch.size(0)
        val_loss /= len(val_dataset)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_loss

if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial), n_trials=20)
    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    best_params = trial.params
    model = CombinedModel(input_dim=TOTAL_INPUT_DIM,
                          model_dim=best_params["model_dim"],
                          num_layers=best_params["num_layers"],
                          nhead=best_params["nhead"],
                          dropout=best_params["dropout"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    
    train_loader = DataLoader(full_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    for epoch in range(TRAIN_NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for features_batch, targets_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = custom_loss(outputs, targets_batch, min_signal_percent=MIN_SIGNAL_PERCENT, penalty_factor=PENALTY_FACTOR)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features_batch.size(0)
        epoch_loss /= len(full_dataset)
        print(f"Epoch {epoch+1}/{TRAIN_NUM_EPOCHS}, Loss: {epoch_loss:.6f}")
    
    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved as final_model.pth")
