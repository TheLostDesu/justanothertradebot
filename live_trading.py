#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import numpy as np
import torch
import logging
from model import CombinedModel
from config import (TIMEFRAME_SECONDS, SEQUENCE_LENGTH, MIN_SIGNAL_PERCENT,
                    MIN_CONFIDENCE, SYMBOLS, TRADE_MAX_DURATION, TRADE_COOLDOWN,
                    DEFAULT_LEVERAGE, DEFAULT_STOP_LOSS_FACTOR, DEFAULT_TAKE_PROFIT_FACTOR,
                    DEFAULT_TRAILING_STOP, ERROR_COOLDOWN_SECONDS, NEGATIVE_PERFORMANCE_COOLDOWN_SECONDS,
                    MIN_AVG_PROFIT_THRESHOLD, TOTAL_INPUT_DIM, RISK_PERCENTAGE, CONFIDENCE_SCALE,
                    MAX_VOLATILITY_THRESHOLD, HORIZON_MS)
from bybit_api import fetch_balance, create_order, fetch_candles
from orderbook import Orderbook  # новый класс Orderbook
from bybit_ws import get_orderbook  # предполагается, что он возвращает экземпляр Orderbook для символа

logging.basicConfig(level=logging.INFO)

error_cooldown_until = 0
position_history = []

def flatten_sequence(sequence):
    """
    Преобразует последовательность snapshot-ов (каждый – словарь с ключами 'bid', 'ask', 'mid')
    в одномерный numpy-массив: [bid_0, ask_0, mid_0, bid_1, ask_1, mid_1, ...]
    """
    flat = []
    for snap in sequence:
        flat.extend([snap.get('bid', 0.0), snap.get('ask', 0.0), snap.get('mid', 0.0)])
    return np.array(flat, dtype=np.float32)

def compute_candle_features(candles):
    """
    Для каждого свечного интервала (словарь с ключами 'open', 'high', 'low', 'close')
    вычисляет два признака: return = (close - open)/open и range = (high - low)/open.
    Возвращает одномерный numpy-массив.
    """
    feats = []
    for candle in candles:
        o = candle['open']
        c = candle['close']
        h = candle['high']
        l = candle['low']
        if o != 0:
            ret = (c - o) / o
            rng = (h - l) / o
        else:
            ret, rng = 0, 0
        feats.extend([ret, rng])
    return np.array(feats, dtype=np.float32)

async def calculate_position_size(session, symbol, side, entry_price, stoploss_price):
    try:
        balance_data = await fetch_balance(session)
        available_usdt = float(balance_data.get("result", {}).get("USDT", 0))
    except Exception as e:
        logging.error(f"[{symbol}] Error fetching balance: {e}")
        return None
    risk_amount = available_usdt * RISK_PERCENTAGE
    position_size = (risk_amount * DEFAULT_LEVERAGE) / abs(entry_price - stoploss_price)
    logging.info(f"[{symbol}] Available USDT: {available_usdt:.2f}, risk: {risk_amount:.2f} USDT, position size: {position_size:.6f}")
    return position_size

async def compute_candle_features_from_api(session, symbol):
    data = await fetch_candles(session, symbol, interval="5", limit=60)
    candles = data.get("result", {}).get("list", [])
    feats = []
    for candle in candles:
        try:
            o = float(candle["open"])
            c = float(candle["close"])
            h = float(candle["high"])
            l = float(candle["low"])
            ret = (c - o) / o if o != 0 else 0
            rng = (h - l) / o if o != 0 else 0
        except Exception:
            ret, rng = 0, 0
        feats.extend([ret, rng])
    expected = 60 * 2
    if len(feats) < expected:
        feats.extend([0] * (expected - len(feats)))
    return np.array(feats, dtype=np.float32)

async def trade_symbol(symbol, model, device):
    global error_cooldown_until, position_history
    position = None
    last_trade_time = 0
    daily_profit = 0.0
    current_day = time.strftime("%Y-%m-%d", time.localtime())

    def check_performance():
        if len(position_history) >= 3:
            return sum(position_history[-3:]) / 3.0
        return 0.0

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                now = time.time()
                if now < error_cooldown_until:
                    logging.info(f"[{symbol}] In error cooldown until {error_cooldown_until}")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                now_day = time.strftime("%Y-%m-%d", time.localtime())
                if now_day != current_day:
                    logging.info(f"[{symbol}] Profit for {current_day}: {daily_profit:.2f} USDT")
                    daily_profit = 0.0
                    current_day = now_day

                if check_performance() < MIN_AVG_PROFIT_THRESHOLD:
                    logging.info(f"[{symbol}] Recent avg profit below threshold; performance cooldown.")
                    await asyncio.sleep(NEGATIVE_PERFORMANCE_COOLDOWN_SECONDS)
                    continue

                # Получаем текущий Orderbook для символа
                ob = get_orderbook(symbol)  # ожидается, что это экземпляр Orderbook
                if ob is None:
                    logging.info(f"[{symbol}] Orderbook not available.")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                features_dict = ob.get_features()  # {'sequence': [...], 'candles': [...]}
                sequence = features_dict.get('sequence', [])
                candles = features_dict.get('candles', [])
                if len(sequence) < SEQUENCE_LENGTH:
                    logging.info(f"[{symbol}] Accumulating orderbook snapshots...")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                # Используем последние SEQUENCE_LENGTH snapshot-ов
                seq_window = sequence[-SEQUENCE_LENGTH:]
                lob_flat = flatten_sequence(seq_window)
                candle_feats = compute_candle_features(candles)
                combined_features = np.concatenate((lob_flat, candle_feats))
                inp = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted_delta_pct = model(inp).item()
                logging.info(f"[{symbol}] Predicted percentage change: {predicted_delta_pct*100:.2f}%")

                current_mid = sequence[-1].get('mid', None)
                if current_mid is None:
                    logging.info(f"[{symbol}] Current mid price not available.")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                volatility_pct = np.std(lob_flat) / (np.mean(lob_flat) + 1e-6)
                if volatility_pct > MAX_VOLATILITY_THRESHOLD:
                    logging.info(f"[{symbol}] Volatility {volatility_pct*100:.2f}% exceeds threshold; skipping trade.")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                confidence = min(1.0, abs(predicted_delta_pct) / CONFIDENCE_SCALE)
                if confidence < MIN_CONFIDENCE:
                    logging.info(f"[{symbol}] Confidence {confidence:.2f} below minimum; skipping trade.")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                if predicted_delta_pct > MIN_SIGNAL_PERCENT:
                    entry_price = current_mid
                    stoploss = entry_price * (1 - DEFAULT_STOP_LOSS_FACTOR)
                    takeprofit = entry_price * (1 + DEFAULT_TAKE_PROFIT_FACTOR)
                    trailing_stop = entry_price * DEFAULT_TRAILING_STOP
                    qty = await calculate_position_size(session, symbol, 'long', entry_price, stoploss)
                    if qty is None or qty <= 0:
                        logging.error(f"[{symbol}] Unable to calculate position size for long trade.")
                    else:
                        logging.info(f"[{symbol}] Long signal: entry {entry_price:.2f}, SL {stoploss:.2f}, TP {takeprofit:.2f}, TS {trailing_stop:.2f}, Confidence {confidence:.2f}")
                        order_resp = await create_order(session, symbol, 'buy', qty, DEFAULT_LEVERAGE, stoploss, takeprofit, trailing_stop)
                        logging.info(f"[{symbol}] Order response: {order_resp}")
                        position = {'side': 'long', 'entry_price': entry_price, 'qty': qty, 'entry_time': time.time()}
                        last_trade_time = time.time()
                elif predicted_delta_pct < -MIN_SIGNAL_PERCENT:
                    entry_price = current_mid
                    stoploss = entry_price * (1 + DEFAULT_STOP_LOSS_FACTOR)
                    takeprofit = entry_price * (1 - DEFAULT_TAKE_PROFIT_FACTOR)
                    trailing_stop = entry_price * DEFAULT_TRAILING_STOP
                    qty = await calculate_position_size(session, symbol, 'short', entry_price, stoploss)
                    if qty is None or qty <= 0:
                        logging.error(f"[{symbol}] Unable to calculate position size for short trade.")
                    else:
                        logging.info(f"[{symbol}] Short signal: entry {entry_price:.2f}, SL {stoploss:.2f}, TP {takeprofit:.2f}, TS {trailing_stop:.2f}, Confidence {confidence:.2f}")
                        order_resp = await create_order(session, symbol, 'sell', qty, DEFAULT_LEVERAGE, stoploss, takeprofit, trailing_stop)
                        logging.info(f"[{symbol}] Order response: {order_resp}")
                        position = {'side': 'short', 'entry_price': entry_price, 'qty': qty, 'entry_time': time.time()}
                        last_trade_time = time.time()
                else:
                    logging.info(f"[{symbol}] No clear trading signal.")
                
                await asyncio.sleep(TIMEFRAME_SECONDS)
            except Exception as e:
                logging.error(f"[{symbol}] Error in main loop: {e}")
                error_cooldown_until = time.time() + ERROR_COOLDOWN_SECONDS
                await asyncio.sleep(TIMEFRAME_SECONDS)

async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel(input_dim=TOTAL_INPUT_DIM)
    model_path = "final_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")
    
    tasks = [trade_symbol(symbol, model, device) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
