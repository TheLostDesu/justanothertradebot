# config.py
#############################################
# CONFIGURATION CONSTANTS
#############################################

# --- Data settings ---
TRAINING_DATE_RANGES = [
    "2024-05-01,2024-12-31",   # Bullish период
    "2024-01-01,2024-04-30"    # Bearish период
]
# Шаблон URL для архивов LOB-данных; {pair} заменяется на торговую пару (без слэша), {date} – на дату
URL_TEMPLATE = "https://quote-saver.bycsi.com/orderbook/linear/{pair}/{date}_{pair}_ob500.data.zip"

# --- Trading pairs (единый список для обучения и live‑трейдинга) ---
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "SOL/USDT", "DOT/USDT", "DOGE/USDT", "LTC/USDT", "MATIC/USDT"
]

# --- Order parameters ---
DEFAULT_STOP_LOSS_FACTOR = 0.005      # 0.5% от цены
DEFAULT_TAKE_PROFIT_FACTOR = 0.01       # 1% от цены
DEFAULT_TRAILING_STOP = 0.005           # 0.5% от цены
DEFAULT_LEVERAGE = 5

# --- Risk settings ---
MIN_RISK_PERCENT = 0.01
MAX_RISK_PERCENT = 0.05
RISK_PERCENTAGE = MAX_RISK_PERCENT
PENALTY_FACTOR  = 10.0

# --- API endpoints (Bybit v5) ---
REST_API_BASE = "https://api.bybit.com"
ORDER_CREATE_ENDPOINT = "/v5/order/create"
BALANCE_ENDPOINT = "/v5/account/wallet-balance"
KLINE_ENDPOINT = "/v5/market/kline"
WS_API_URL = "wss://stream.bybit.com/realtime_public"

# --- Training parameters ---
TRAIN_NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 256
TRAIN_LEARNING_RATE = 1e-4

# --- LOB and Candle parameters ---
SEQUENCE_LENGTH = 120          # Количество LOB-снимков
HORIZON_MS = 10000            # Горизонт для расчёта target (10 сек)
NUM_LEVELS = 5

CANDLE_INTERVAL_MIN = 5       # 5-минутные свечи
CANDLE_TOTAL_HOURS = 5        # За 5 часов → 60 свечей
CANDLE_FEATURES_PER_CANDLE = 2  # (return, range) → 120 признаков

# --- Derived model parameters ---
LOB_INPUT_DIM = SEQUENCE_LENGTH * (NUM_LEVELS * 4)
CANDLE_COUNT = int((CANDLE_TOTAL_HOURS * 60) / CANDLE_INTERVAL_MIN)
CANDLE_INPUT_DIM = CANDLE_COUNT * CANDLE_FEATURES_PER_CANDLE
TOTAL_INPUT_DIM = LOB_INPUT_DIM + CANDLE_INPUT_DIM

MODEL_DIM = 128
NUM_LAYERS = 4
NHEAD = 8
DROPOUT = 0.1

# --- Adaptive TP/SL, Confidence ---
ADAPTIVE_TP_MULTIPLIER = 2.0
SL_FACTOR = 0.5

CONFIDENCE_SCALE = 0.05
CONFIDENCE_WEIGHT = 1.0
MIN_CONFIDENCE = 0.5
MIN_SIGNAL_PERCENT = 0.0005

# --- Anomaly filtering & Market conditions ---
MAX_TARGET_CHANGE_PERCENT = 0.2
MAX_VOLATILITY_THRESHOLD = 0.1

# --- Live trading settings ---
TIMEFRAME_SECONDS = 10
TRADE_MAX_DURATION = 60
TRADE_COOLDOWN = 10

ERROR_COOLDOWN_SECONDS = 60
NEGATIVE_PERFORMANCE_COOLDOWN_SECONDS = 30
MIN_AVG_PROFIT_THRESHOLD = -0.005

# --- Broker credentials ---
EXCHANGE_ID = 'bybit'
API_KEY = "YOUR_BYBIT_API_KEY"
API_SECRET = "YOUR_BYBIT_API_SECRET"

# --- Paper trading flag ---
PAPER_TRADING = True

#############################################
# End of configuration constants
