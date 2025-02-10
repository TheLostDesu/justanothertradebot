import torch
import torch.nn as nn
import math
from config import (
    MODEL_DIM,
    DROPOUT,
    NUM_LAYERS,
    NHEAD,
    LOB_INPUT_DIM,
    CANDLE_INPUT_DIM,
    SEQUENCE_LENGTH,
    CANDLE_INTERVAL_MIN,
    CANDLE_TOTAL_HOURS,
    CANDLE_FEATURES_PER_CANDLE
)

# Вычисляем количество свечей из конфигурации:
CANDLE_COUNT = int((CANDLE_TOTAL_HOURS * 60) / CANDLE_INTERVAL_MIN)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        pe_slice = self.pe[:, :x.size(1), :]  # (1, seq_len, d_model)
        return self.dropout(x + pe_slice)

class CombinedModel(nn.Module):
    def __init__(self, input_dim, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, nhead=NHEAD, dropout=DROPOUT):
        """
        :param input_dim: размер входного вектора, равный TOTAL_INPUT_DIM, то есть
                          LOB_INPUT_DIM + CANDLE_INPUT_DIM.
        :param model_dim: размерность эмбеддингов Transformer-а.
        :param num_layers: число слоев TransformerEncoder.
        :param nhead: число голов self-attention.
        :param dropout: dropout для Transformer и positional encoding.
        """
        super(CombinedModel, self).__init__()
        self.input_dim = input_dim  # Ожидается, что input_dim == TOTAL_INPUT_DIM из config.
        
        # Для LOB-данных:
        # LOB_INPUT_DIM = SEQUENCE_LENGTH * (NUM_LEVELS * 4)
        self.lob_seq_length = SEQUENCE_LENGTH
        self.lob_token_dim = LOB_INPUT_DIM // SEQUENCE_LENGTH  # Ожидается, что это NUM_LEVELS*4
        
        # Для свечных данных:
        # CANDLE_INPUT_DIM = CANDLE_COUNT * CANDLE_FEATURES_PER_CANDLE
        self.candle_seq_length = CANDLE_COUNT
        self.candle_token_dim = CANDLE_INPUT_DIM // CANDLE_COUNT  # Должно быть равно CANDLE_FEATURES_PER_CANDLE
        
        self.total_seq_length = self.lob_seq_length + self.candle_seq_length
        
        # Определяем два отдельных линейных слоя для проекции токенов в embedding пространство model_dim.
        self.lob_embedding = nn.Linear(self.lob_token_dim, model_dim)
        self.candle_embedding = nn.Linear(self.candle_token_dim, model_dim)
        
        # Позиционное кодирование для объединенной последовательности длины total_seq_length.
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=self.total_seq_length)
        
        # TransformerEncoderLayer с batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Финальный линейный слой
        self.fc_out = nn.Linear(model_dim, 1)
        
    def forward(self, x):
        """
        Принимает x размерности (batch, input_dim), где input_dim == TOTAL_INPUT_DIM.
        Разбивает его на две части:
          - первые LOB_INPUT_DIM элементов → LOB данные,
          - оставшиеся CANDLE_INPUT_DIM элементов → свечные данные.
        Затем каждую часть преобразует в последовательность токенов, эмбеддит, объединяет и обрабатывает через Transformer.
        """
        batch_size = x.size(0)
        # Извлекаем LOB и свечную части:
        from config import LOB_INPUT_DIM, CANDLE_INPUT_DIM
        lob_part = x[:, :LOB_INPUT_DIM]             # (batch, LOB_INPUT_DIM)
        candle_part = x[:, LOB_INPUT_DIM:]            # (batch, CANDLE_INPUT_DIM)
        
        # Преобразуем LOB часть в последовательность: (batch, SEQUENCE_LENGTH, lob_token_dim)
        lob_tokens = lob_part.view(batch_size, self.lob_seq_length, self.lob_token_dim)
        # Преобразуем свечную часть в последовательность: (batch, CANDLE_COUNT, candle_token_dim)
        candle_tokens = candle_part.view(batch_size, self.candle_seq_length, self.candle_token_dim)
        
        # Применяем линейные слои (embedding) для каждой части:
        lob_emb = self.lob_embedding(lob_tokens)       # (batch, SEQUENCE_LENGTH, model_dim)
        candle_emb = self.candle_embedding(candle_tokens)  # (batch, CANDLE_COUNT, model_dim)
        
        # Объединяем последовательности по временной оси:
        x_seq = torch.cat([lob_emb, candle_emb], dim=1)   # (batch, total_seq_length, model_dim)
        
        # Добавляем позиционное кодирование:
        x_seq = self.pos_encoder(x_seq)                   # (batch, total_seq_length, model_dim)
        
        # Проходим через TransformerEncoder:
        x_seq = self.transformer_encoder(x_seq)           # (batch, total_seq_length, model_dim)
        
        # Применяем, например, mean pooling по временной оси:
        x_pool = x_seq.mean(dim=1)                         # (batch, model_dim)
        out = self.fc_out(x_pool)                          # (batch, 1)
        return out.squeeze(-1)                           # (batch,)
