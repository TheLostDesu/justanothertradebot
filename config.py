# config.py
#############################################
# CONFIGURATION CONSTANTS
#############################################

# --- Data settings ---
# Задаём диапазон дат для загрузки данных (формат: YYYY-MM-DD,YYYY-MM-DD)
TRAINING_DATE_RANGE = "2024-01-01,2024-01-31"
URL_TEMPLATE = "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/{}_BTCUSDT_ob500.data.zip"

# --- Training parameters (под мощный GPU 8×4090) ---
TRAIN_NUM_EPOCHS    = 50       # Количество эпох обучения
TRAIN_BATCH_SIZE    = 256      # Размер батча
TRAIN_LEARNING_RATE = 1e-4     # Начальная скорость обучения

# --- LOB processing parameters ---
SEQUENCE_LENGTH = 10          # Количество последовательных snapshot'ов для формирования примера
HORIZON_MS      = 10000       # Горизонт для расчёта целевой дельты (10 сек = 10000 мс)
NUM_LEVELS      = 5           # Количество уровней с каждой стороны стакана

# --- Model parameters (default, для тюнинга optuna можно менять) ---
MODEL_DIM   = 128             # Размерность эмбеддинга
NUM_LAYERS  = 4               # Количество слоёв Transformer
NHEAD       = 8               # Количество голов внимания
DROPOUT     = 0.1             # Доля dropout

# --- Adaptive TP and SL parameters ---
# TP рассчитывается как: 
#   для лонга: TP = entry × (1 + (ADAPTIVE_TP_MULTIPLIER × volatility_pct × (1 + CONFIDENCE_WEIGHT × confidence)))
#   для шорта: TP = entry × (1 - (ADAPTIVE_TP_MULTIPLIER × volatility_pct × (1 + CONFIDENCE_WEIGHT × confidence)))
# SL рассчитывается как: SL = entry ∓ (TP_distance_pct × SL_FACTOR)
ADAPTIVE_TP_MULTIPLIER = 2.0  # Множитель для расчёта TP, базируясь на волатильности
SL_FACTOR            = 0.5  # Фактор для расчёта SL от TP

# --- Параметры для учёта уверенности сети при расчёте TP ---
CONFIDENCE_SCALE  = 0.05    # Если |predicted_delta_pct| >= 5% (0.05), уверенность считается равной 1
CONFIDENCE_WEIGHT = 1.0     # Вес, с которым уверенность сети увеличивает TP
MIN_CONFIDENCE    = 0.5     # Минимальное значение уверенности для открытия сделки

# --- Trading signal thresholds in percentage ---
# Здесь торговый сигнал считается, если предсказанное изменение (%) превышает MIN_SIGNAL_PERCENT
MIN_SIGNAL_PERCENT = 0.0005   # 0.05%

# --- Risk management ---
RISK_PERCENTAGE = 0.05  # Максимальный риск на сделку – 5% от свободного баланса
PENALTY_FACTOR  = 10.0  # Коэффициент штрафа за торговые убытки

# --- Live trading settings ---
TIMEFRAME_SECONDS  = 10    # Интервал между запросами (сек)
TRADE_MAX_DURATION = 60    # Если позиция открыта более 60 секунд – принудительно закрываем
TRADE_COOLDOWN     = 10    # После закрытия позиции ждём 10 секунд, прежде чем генерировать новый сигнал

# --- API and exchange settings ---
EXCHANGE_ID = 'binance'
SYMBOL      = 'BTC/USDT'
API_KEY     = "YOUR_API_KEY"
API_SECRET  = "YOUR_API_SECRET"

# --- Paper trading flag ---
PAPER_TRADING = True  # Если True – ордера не отправляются, а только логируются

#############################################
# End of configuration constants
