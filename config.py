# config.py
#############################################
# CONFIGURATION CONSTANTS
#############################################

# --- Data settings ---
TRAINING_DATE_RANGE = "2024-01-01,2024-01-31"  # Диапазон дат для исторических данных
URL_TEMPLATE = "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/{}_BTCUSDT_ob500.data.zip"

# --- Trading symbols (Bybit instruments) ---
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# --- Trading order parameters for broker integration ---
DEFAULT_STOP_LOSS_FACTOR = 0.005      # 0.5% от цены
DEFAULT_TAKE_PROFIT_FACTOR = 0.01       # 1% от цены
DEFAULT_TRAILING_STOP = 0.005           # 0.5% trailing stop
DEFAULT_LEVERAGE = 5                  # Используем максимум 5x

# --- Risk settings ---
MIN_RISK_PERCENT = 0.01               # Минимум 1% риска
MAX_RISK_PERCENT = 0.05               # Максимум 5% риска
RISK_PERCENTAGE = MAX_RISK_PERCENT    # Риск на сделку (максимум)
PENALTY_FACTOR  = 10.0

# --- REST and WebSocket endpoints (Bybit v5) ---
REST_API_BASE = "https://api.bybit.com"
ORDER_CREATE_ENDPOINT = "/v5/order/create"
BALANCE_ENDPOINT = "/v5/account/wallet-balance"
WS_API_URL = "wss://stream.bybit.com/realtime_public"

# --- Training parameters (под мощный GPU 8×4090) ---
TRAIN_NUM_EPOCHS    = 50
TRAIN_BATCH_SIZE    = 256
TRAIN_LEARNING_RATE = 1e-4

# --- LOB processing parameters ---
SEQUENCE_LENGTH = 10
HORIZON_MS      = 10000
NUM_LEVELS      = 5

# --- Model parameters ---
MODEL_DIM   = 128
NUM_LAYERS  = 4
NHEAD       = 8
DROPOUT     = 0.1

# --- Adaptive TP and SL parameters ---
ADAPTIVE_TP_MULTIPLIER = 2.0
SL_FACTOR            = 0.5

# --- Confidence and signal thresholds ---
CONFIDENCE_SCALE  = 0.05    # Если |predicted_delta_pct| >= 5% (0.05), уверенность = 1
CONFIDENCE_WEIGHT = 1.0
MIN_CONFIDENCE    = 0.5     # Минимальная уверенность для открытия сделки
MIN_SIGNAL_PERCENT = 0.0005  # 0.05%

# --- Anomaly filtering settings ---
MAX_TARGET_CHANGE_PERCENT = 0.2  # 20%

# --- Market conditions ---
MAX_VOLATILITY_THRESHOLD = 0.05  # Если волатильность > 5%, торговля приостанавливается

# --- Live trading settings ---
TIMEFRAME_SECONDS  = 10
TRADE_MAX_DURATION = 60
TRADE_COOLDOWN     = 10

# --- Broker API credentials (Bybit v5) ---
EXCHANGE_ID = 'bybit'
API_KEY     = "YOUR_BYBIT_API_KEY"
API_SECRET  = "YOUR_BYBIT_API_SECRET"

# --- Paper trading flag ---
PAPER_TRADING = True

#############################################
# End of configuration constants
