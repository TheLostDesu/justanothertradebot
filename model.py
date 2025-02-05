# model.py
import torch
import torch.nn as nn
import math
from config import MODEL_DIM, DROPOUT, NUM_LAYERS, NHEAD, TOTAL_INPUT_DIM

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CombinedModel(nn.Module):
    def __init__(self, input_dim=TOTAL_INPUT_DIM, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, nhead=NHEAD, dropout=DROPOUT):
        super(CombinedModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x[-1]
        out = self.fc_out(x)
        return out.squeeze(-1)
