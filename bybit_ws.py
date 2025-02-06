# bybit_ws.py
import asyncio
import json
import websockets
import logging
from config import WS_API_URL, SYMBOLS
from orderbook import OrderBookSide

# Глобальный словарь для хранения orderbook для каждого символа.
# Для каждого символа хранится словарь с двумя сторонами: 'a' (asks) и 'b' (bids).
orderbooks = {}

async def subscribe_orderbook(ws, symbol):
    symbol_formatted = symbol.replace("/", "")
    channel = f"orderBookL2_25.{symbol_formatted}"
    sub_msg = {"op": "subscribe", "args": [channel]}
    await ws.send(json.dumps(sub_msg))
    logging.info(f"Subscribed to {channel}")

async def bybit_ws_client():
    global orderbooks
    async with websockets.connect(WS_API_URL) as ws:
        # Подписываемся на каналы и инициализируем пустые orderbook'и.
        for symbol in SYMBOLS:
            await subscribe_orderbook(ws, symbol)
            symbol_formatted = symbol.replace("/", "")
            orderbooks[symbol_formatted] = {
                "a": OrderBookSide(is_bid=False),
                "b": OrderBookSide(is_bid=True)
            }
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                if "topic" in data and data["topic"].startswith("orderBookL2_25."):
                    topic = data["topic"]
                    symbol_formatted = topic.split(".")[1]
                    if "data" in data:
                        orders = data["data"]
                        ob = orderbooks.get(symbol_formatted)
                        if ob is None:
                            ob = {
                                "a": OrderBookSide(is_bid=False),
                                "b": OrderBookSide(is_bid=True)
                            }
                            orderbooks[symbol_formatted] = ob
                        for order in orders:
                            try:
                                price = float(order["price"])
                                size = float(order["size"])
                            except Exception:
                                continue
                            side = order["side"].lower()
                            if side == "buy":
                                ob["b"].update(price, size)
                            elif side == "sell":
                                ob["a"].update(price, size)
                        logging.info(f"Updated orderbook for {symbol_formatted}")
            except Exception as e:
                logging.error(f"Error in WebSocket client: {e}")
                await asyncio.sleep(1)

async def start_bybit_ws():
    await bybit_ws_client()

def get_orderbook(symbol):
    """
    Возвращает orderbook для символа в виде словаря с ключами 'a' и 'b',
    где каждая сторона — обычный список кортежей (sort_key, price, size).
    """
    symbol_formatted = symbol.replace("/", "")
    ob = orderbooks.get(symbol_formatted)
    if ob is None:
        return None
    return {"a": ob["a"].get_list(), "b": ob["b"].get_list()}
