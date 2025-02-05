# live_trading.py
import asyncio
import aiohttp
import time
import numpy as np
import torch
import logging
from model import CombinedModel
from config import (
    TIMEFRAME_SECONDS, NUM_LEVELS, SEQUENCE_LENGTH, MIN_SIGNAL_PERCENT, MIN_CONFIDENCE,
    SYMBOLS, TRADE_MAX_DURATION, TRADE_COOLDOWN, DEFAULT_LEVERAGE,
    ADAPTIVE_TP_MULTIPLIER, SL_FACTOR, RISK_PERCENTAGE,
    CONFIDENCE_SCALE, CONFIDENCE_WEIGHT, MAX_VOLATILITY_THRESHOLD,
    DEFAULT_STOP_LOSS_FACTOR, DEFAULT_TAKE_PROFIT_FACTOR, DEFAULT_TRAILING_STOP,
    ERROR_COOLDOWN_SECONDS, NEGATIVE_PERFORMANCE_COOLDOWN_SECONDS, MIN_AVG_PROFIT_THRESHOLD, TOTAL_INPUT_DIM
)
from bybit_ws import start_bybit_ws, get_orderbook
from dataset import get_features_from_orderbook, get_mid_price
from bybit_api import fetch_balance, create_order, fetch_candles

logging.basicConfig(level=logging.INFO)
error_cooldown_until = 0
position_history = []

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
    features = []
    for candle in candles:
        try:
            open_price = float(candle["open"])
            close_price = float(candle["close"])
            high = float(candle["high"])
            low = float(candle["low"])
            ret = (close_price - open_price) / open_price if open_price != 0 else 0
            rng = (high - low) / open_price if open_price != 0 else 0
        except Exception:
            ret, rng = 0, 0
        features.extend([ret, rng])
    expected = 60 * 2
    if len(features) < expected:
        features.extend([0] * (expected - len(features)))
    return features

async def trade_symbol(symbol, model, device):
    global error_cooldown_until, position_history
    position = None
    last_trade_time = 0
    lob_buffer = []  # Flattened LOB data
    daily_profit = 0.0
    current_day = time.strftime("%Y-%m-%d", time.localtime())
    
    def check_performance():
        if len(position_history) >= 3:
            recent = position_history[-3:]
            avg_profit = sum(recent) / len(recent)
            return avg_profit
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

                avg_profit = check_performance()
                if avg_profit < MIN_AVG_PROFIT_THRESHOLD:
                    logging.info(f"[{symbol}] Recent avg profit {avg_profit:.4f} below threshold; performance cooldown.")
                    await asyncio.sleep(NEGATIVE_PERFORMANCE_COOLDOWN_SECONDS)
                    continue

                current_time = time.time()
                if position is not None:
                    orderbook = get_orderbook(symbol)
                    if orderbook is None:
                        logging.warning(f"[{symbol}] Orderbook not available.")
                    else:
                        current_mid = get_mid_price(orderbook)
                        if current_mid is None:
                            logging.warning(f"[{symbol}] Could not compute mid price.")
                        else:
                            if (current_time - position['entry_time']) > TRADE_MAX_DURATION:
                                logging.info(f"[{symbol}] Closing position due to duration.")
                                if position['side'] == 'long':
                                    profit = (current_mid - position['entry_price']) * position['qty']
                                else:
                                    profit = (position['entry_price'] - current_mid) * position['qty']
                                daily_profit += profit
                                position_history.append(profit)
                                logging.info(f"[{symbol}] Trade profit: {profit:.2f} USDT, Daily profit: {daily_profit:.2f} USDT")
                                position = None
                                last_trade_time = current_time
                    if position is not None:
                        await asyncio.sleep(TIMEFRAME_SECONDS)
                        continue

                if current_time - last_trade_time < TRADE_COOLDOWN:
                    logging.info(f"[{symbol}] Cooldown period after trade...")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                orderbook = get_orderbook(symbol)
                if orderbook is None:
                    logging.info(f"[{symbol}] Waiting for orderbook update...")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                lob_features = get_features_from_orderbook(orderbook, NUM_LEVELS)
                lob_buffer.extend(lob_features)
                if len(lob_buffer) > SEQUENCE_LENGTH * (NUM_LEVELS * 4):
                    lob_buffer = lob_buffer[-(SEQUENCE_LENGTH * (NUM_LEVELS * 4)):]
                if len(lob_buffer) < SEQUENCE_LENGTH * (NUM_LEVELS * 4):
                    logging.info(f"[{symbol}] Accumulating LOB data...")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                candle_features = await compute_candle_features_from_api(session, symbol)
                combined_features = np.array(lob_buffer + candle_features, dtype=np.float32)
                inp = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted_delta_pct = model(inp).item()
                logging.info(f"[{symbol}] Predicted percentage change: {predicted_delta_pct*100:.2f}%")
                
                orderbook = get_orderbook(symbol)
                if orderbook is None:
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue
                current_mid = get_mid_price(orderbook)
                if current_mid is None:
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                volatility_pct = np.std(lob_buffer) / np.mean(lob_buffer)
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
                global error_cooldown_until
                error_cooldown_until = time.time() + ERROR_COOLDOWN_SECONDS
                await asyncio.sleep(TIMEFRAME_SECONDS)
                
async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = TOTAL_INPUT_DIM
    model = CombinedModel(input_dim=input_dim)
    model_path = "final_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")
    
    async with aiohttp.ClientSession() as session:
        ws_task = asyncio.create_task(start_bybit_ws())
        tasks = []
        for symbol in SYMBOLS:
            tasks.append(trade_symbol(symbol, model, device))
        await asyncio.gather(*tasks)
        await ws_task

if __name__ == '__main__':
    """
    IMPORTANT: Test thoroughly in paper trading mode before using real funds.
    """
    asyncio.run(main())
