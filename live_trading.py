# live_trading.py
import asyncio
import aiohttp
import time
import numpy as np
import torch
import logging
from model import OrderBookTransformer
from config import (
    TIMEFRAME_SECONDS, NUM_LEVELS, SEQUENCE_LENGTH, MIN_SIGNAL_PERCENT, MIN_CONFIDENCE,
    SYMBOLS, TRADE_MAX_DURATION, TRADE_COOLDOWN, DEFAULT_LEVERAGE,
    ADAPTIVE_TP_MULTIPLIER, SL_FACTOR, RISK_PERCENTAGE,
    CONFIDENCE_SCALE, CONFIDENCE_WEIGHT, MAX_VOLATILITY_THRESHOLD,
    DEFAULT_STOP_LOSS_FACTOR, DEFAULT_TAKE_PROFIT_FACTOR, DEFAULT_TRAILING_STOP
)
from bybit_ws import start_bybit_ws, get_orderbook
from dataset import get_features_from_orderbook, get_mid_price
from bybit_api import fetch_balance, create_order

logging.basicConfig(level=logging.INFO)

async def calculate_position_size(session, symbol, side, entry_price, stoploss_price):
    try:
        balance_data = await fetch_balance(session)
        available_usdt = float(balance_data.get("result", {}).get("USDT", 0))
    except Exception as e:
        logging.error(f"[{symbol}] Error fetching balance: {e}")
        return None
    risk_amount = available_usdt * RISK_PERCENTAGE
    # С учетом leverage: effective risk = risk_amount * leverage
    position_size = (risk_amount * DEFAULT_LEVERAGE) / abs(entry_price - stoploss_price)
    logging.info(f"[{symbol}] Available USDT: {available_usdt:.2f}, risk: {risk_amount:.2f} USDT, position size: {position_size:.6f}")
    return position_size

async def trade_symbol(symbol, model, device):
    position = None
    last_trade_time = 0
    sequence_buffer = []
    mid_buffer = []
    daily_profit = 0.0
    current_day = time.strftime("%Y-%m-%d", time.localtime())

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                now_day = time.strftime("%Y-%m-%d", time.localtime())
                if now_day != current_day:
                    logging.info(f"[{symbol}] Profit for {current_day}: {daily_profit:.2f} USDT")
                    daily_profit = 0.0
                    current_day = now_day

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
                            if position['side'] == 'long' and (current_time - position['entry_time'] > TRADE_MAX_DURATION):
                                logging.info(f"[{symbol}] Closing long position due to duration.")
                                await session.post(f"{DEFAULT_LEVERAGE}")  # Здесь предполагается, что брокер управляет ордером
                                profit = (current_mid - position['entry_price']) * position['qty']
                                daily_profit += profit
                                logging.info(f"[{symbol}] Trade profit: {profit:.2f} USDT, Daily profit: {daily_profit:.2f} USDT")
                                position = None
                                last_trade_time = time.time()
                            elif position['side'] == 'short' and (current_time - position['entry_time'] > TRADE_MAX_DURATION):
                                logging.info(f"[{symbol}] Closing short position due to duration.")
                                await session.post(f"{DEFAULT_LEVERAGE}")  # Аналогично для шорта
                                profit = (position['entry_price'] - current_mid) * position['qty']
                                daily_profit += profit
                                logging.info(f"[{symbol}] Trade profit: {profit:.2f} USDT, Daily profit: {daily_profit:.2f} USDT")
                                position = None
                                last_trade_time = time.time()
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

                features = get_features_from_orderbook(orderbook, NUM_LEVELS)
                mid = get_mid_price(orderbook)
                if mid is None:
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue
                sequence_buffer.append(features)
                mid_buffer.append(mid)
                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)
                    mid_buffer.pop(0)
                if len(sequence_buffer) < SEQUENCE_LENGTH:
                    logging.info(f"[{symbol}] Accumulating data: {len(sequence_buffer)}/{SEQUENCE_LENGTH}")
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                import numpy as np
                seq_array = np.array(sequence_buffer, dtype=np.float32)
                seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted_delta_pct = model(seq_tensor).item()
                logging.info(f"[{symbol}] Predicted percentage change: {predicted_delta_pct*100:.2f}%")
                
                orderbook = get_orderbook(symbol)
                if orderbook is None:
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue
                current_mid = get_mid_price(orderbook)
                if current_mid is None:
                    await asyncio.sleep(TIMEFRAME_SECONDS)
                    continue

                volatility_pct = np.std(mid_buffer) / np.mean(mid_buffer)
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
                await asyncio.sleep(TIMEFRAME_SECONDS)

async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = NUM_LEVELS * 4
    model = OrderBookTransformer(input_dim=input_dim)
    model_path = "final_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")

    # Инициализируем REST-сессию
    async with aiohttp.ClientSession() as session:
        # Запускаем WebSocket клиент для получения orderbook данных
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
