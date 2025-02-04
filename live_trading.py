# live_trading.py
import ccxt
import time
import numpy as np
import torch
import logging
from model import OrderBookTransformer
from config import (
    TIMEFRAME_SECONDS, NUM_LEVELS, SEQUENCE_LENGTH, MIN_SIGNAL_PERCENT, MIN_CONFIDENCE,
    API_KEY, API_SECRET, PAPER_TRADING, SYMBOL, EXCHANGE_ID, TRADE_MAX_DURATION,
    TRADE_COOLDOWN, ADAPTIVE_TP_MULTIPLIER, SL_FACTOR, RISK_PERCENTAGE,
    CONFIDENCE_SCALE, CONFIDENCE_WEIGHT
)
from dataset import get_features_from_orderbook, get_mid_price

logging.basicConfig(level=logging.INFO)

def execute_order(exchange, order_func, symbol, qty, side, entry_price):
    if PAPER_TRADING:
        logging.info(f"[PAPER TRADING] {side.upper()} order for {qty:.6f} {symbol} at {entry_price:.2f}")
        return {"info": "paper trading order", "symbol": symbol, "qty": qty, "side": side, "price": entry_price}
    else:
        return order_func(symbol, qty)

def calculate_position_size(exchange, side, entry_price, stoploss_price):
    try:
        balance = exchange.fetch_balance()
        available_usdt = balance.get('free', {}).get('USDT', 0)
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        return None
    risk_amount = available_usdt * RISK_PERCENTAGE
    risk_per_unit = abs(entry_price - stoploss_price)
    if risk_per_unit == 0:
        logging.error("Zero risk per unit, cannot compute position size.")
        return None
    qty = risk_amount / risk_per_unit
    logging.info(f"Available USDT: {available_usdt:.2f}, risk per trade: {risk_amount:.2f} USDT, position size: {qty:.6f}")
    return qty

def main():
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

    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
    })

    sequence_buffer = []
    mid_buffer = []
    position = None
    last_trade_time = 0

    daily_profit = 0.0
    current_day = time.strftime("%Y-%m-%d", time.localtime())

    while True:
        try:
            current_time = time.time()
            now_day = time.strftime("%Y-%m-%d", time.localtime())
            if now_day != current_day:
                logging.info(f"Profit for {current_day}: {daily_profit:.2f} USDT")
                daily_profit = 0.0
                current_day = now_day

            # Если позиция открыта, проверяем условия выхода
            if position is not None:
                orderbook = exchange.fetch_order_book(SYMBOL)
                current_mid = get_mid_price(orderbook)
                if current_mid is None:
                    logging.warning("Could not get current mid price.")
                else:
                    if position['side'] == 'long':
                        if current_mid <= position['stoploss']:
                            logging.info(f"Long SL triggered: {current_mid:.2f} <= SL {position['stoploss']:.2f}")
                            execute_order(exchange, exchange.create_market_sell_order, SYMBOL, position['qty'], 'sell', current_mid)
                            profit = (current_mid - position['entry_price']) * position['qty']
                            daily_profit += profit
                            logging.info(f"Trade profit: {profit:.2f} USDT, Total daily profit: {daily_profit:.2f} USDT")
                            position = None
                            last_trade_time = time.time()
                        elif current_mid >= position['takeprofit']:
                            logging.info(f"Long TP triggered: {current_mid:.2f} >= TP {position['takeprofit']:.2f}")
                            execute_order(exchange, exchange.create_market_sell_order, SYMBOL, position['qty'], 'sell', current_mid)
                            profit = (current_mid - position['entry_price']) * position['qty']
                            daily_profit += profit
                            logging.info(f"Trade profit: {profit:.2f} USDT, Total daily profit: {daily_profit:.2f} USDT")
                            position = None
                            last_trade_time = time.time()
                    elif position['side'] == 'short':
                        if current_mid >= position['stoploss']:
                            logging.info(f"Short SL triggered: {current_mid:.2f} >= SL {position['stoploss']:.2f}")
                            execute_order(exchange, exchange.create_market_buy_order, SYMBOL, position['qty'], 'buy', current_mid)
                            profit = (position['entry_price'] - current_mid) * position['qty']
                            daily_profit += profit
                            logging.info(f"Trade profit: {profit:.2f} USDT, Total daily profit: {daily_profit:.2f} USDT")
                            position = None
                            last_trade_time = time.time()
                        elif current_mid <= position['takeprofit']:
                            logging.info(f"Short TP triggered: {current_mid:.2f} <= TP {position['takeprofit']:.2f}")
                            execute_order(exchange, exchange.create_market_buy_order, SYMBOL, position['qty'], 'buy', current_mid)
                            profit = (position['entry_price'] - current_mid) * position['qty']
                            daily_profit += profit
                            logging.info(f"Trade profit: {profit:.2f} USDT, Total daily profit: {daily_profit:.2f} USDT")
                            position = None
                            last_trade_time = time.time()
                if position is not None and (current_time - position['entry_time'] > TRADE_MAX_DURATION):
                    logging.info(f"Closing position due to max duration exceeded ({TRADE_MAX_DURATION} sec).")
                    if position['side'] == 'long':
                        execute_order(exchange, exchange.create_market_sell_order, SYMBOL, position['qty'], 'sell', current_mid)
                        profit = (current_mid - position['entry_price']) * position['qty']
                    elif position['side'] == 'short':
                        execute_order(exchange, exchange.create_market_buy_order, SYMBOL, position['qty'], 'buy', current_mid)
                        profit = (position['entry_price'] - current_mid) * position['qty']
                    daily_profit += profit
                    logging.info(f"Trade profit: {profit:.2f} USDT, Total daily profit: {daily_profit:.2f} USDT")
                    position = None
                    last_trade_time = time.time()
                if position is not None:
                    time.sleep(TIMEFRAME_SECONDS)
                    continue

            if current_time - last_trade_time < TRADE_COOLDOWN:
                logging.info("Cooldown period after trade...")
                time.sleep(TIMEFRAME_SECONDS)
                continue

            orderbook = exchange.fetch_order_book(SYMBOL)
            features = get_features_from_orderbook(orderbook, NUM_LEVELS)
            mid = get_mid_price(orderbook)
            if mid is None:
                time.sleep(TIMEFRAME_SECONDS)
                continue
            sequence_buffer.append(features)
            mid_buffer.append(mid)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)
                mid_buffer.pop(0)
            if len(sequence_buffer) < SEQUENCE_LENGTH:
                logging.info(f"Accumulating data: {len(sequence_buffer)}/{SEQUENCE_LENGTH}")
                time.sleep(TIMEFRAME_SECONDS)
                continue

            seq_array = np.array(sequence_buffer, dtype=np.float32)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_delta_pct = model(seq_tensor).item()  # в виде доли (например, 0.01 = 1%)
            logging.info(f"Predicted percentage change: {predicted_delta_pct*100:.2f}%")
            
            current_mid = get_mid_price(orderbook)
            if current_mid is None:
                time.sleep(TIMEFRAME_SECONDS)
                continue
            
            volatility_pct = np.std(mid_buffer) / np.mean(mid_buffer)
            confidence = min(1.0, abs(predicted_delta_pct) / CONFIDENCE_SCALE)
            # Торгуем только, если уверенность достаточная
            if confidence < MIN_CONFIDENCE:
                logging.info(f"Confidence {confidence:.2f} below minimum {MIN_CONFIDENCE:.2f}; skipping trade.")
                time.sleep(TIMEFRAME_SECONDS)
                continue
            adaptive_tp_distance_pct = ADAPTIVE_TP_MULTIPLIER * volatility_pct * (1 + CONFIDENCE_WEIGHT * confidence)
            
            if predicted_delta_pct > MIN_SIGNAL_PERCENT:
                entry_price = current_mid
                takeprofit = entry_price * (1 + adaptive_tp_distance_pct)
                stoploss = entry_price * (1 - adaptive_tp_distance_pct * SL_FACTOR)
                qty = calculate_position_size(exchange, 'long', entry_price, stoploss)
                if qty is None or qty <= 0:
                    logging.error("Unable to calculate position size for long trade.")
                else:
                    logging.info(f"Long signal: entry {entry_price:.2f}, TP {takeprofit:.2f}, SL {stoploss:.2f}, Confidence {confidence:.2f}")
                    execute_order(exchange, exchange.create_market_buy_order, SYMBOL, qty, 'buy', entry_price)
                    position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'qty': qty,
                        'stoploss': stoploss,
                        'takeprofit': takeprofit,
                        'entry_time': time.time()
                    }
                    last_trade_time = time.time()
            elif predicted_delta_pct < -MIN_SIGNAL_PERCENT:
                entry_price = current_mid
                takeprofit = entry_price * (1 - adaptive_tp_distance_pct)
                stoploss = entry_price * (1 + adaptive_tp_distance_pct * SL_FACTOR)
                qty = calculate_position_size(exchange, 'short', entry_price, stoploss)
                if qty is None or qty <= 0:
                    logging.error("Unable to calculate position size for short trade.")
                else:
                    logging.info(f"Short signal: entry {entry_price:.2f}, TP {takeprofit:.2f}, SL {stoploss:.2f}, Confidence {confidence:.2f}")
                    execute_order(exchange, exchange.create_market_sell_order, SYMBOL, qty, 'sell', entry_price)
                    position = {
                        'side': 'short',
                        'entry_price': entry_price,
                        'qty': qty,
                        'stoploss': stoploss,
                        'takeprofit': takeprofit,
                        'entry_time': time.time()
                    }
                    last_trade_time = time.time()
            else:
                logging.info("No clear trading signal.")
            
            time.sleep(TIMEFRAME_SECONDS)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(TIMEFRAME_SECONDS)

if __name__ == '__main__':
    """
    IMPORTANT: Test thoroughly in paper trading mode before using real funds.
    """
    main()
