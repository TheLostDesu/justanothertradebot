# bybit_ws.py
import asyncio
import json
import websockets
import logging
from config import WS_API_URL, SYMBOLS

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
        for symbol in SYMBOLS:
            await subscribe_orderbook(ws, symbol)
            symbol_formatted = symbol.replace("/", "")
            orderbooks[symbol_formatted] = {"a": {}, "b": {}}
        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                if "topic" in data and data["topic"].startswith("orderBookL2_25."):
                    topic = data["topic"]
                    symbol_formatted = topic.split(".")[1]
                    if "data" in data:
                        orders = data["data"]
                        ob = {"a": {}, "b": {}}
                        for order in orders:
                            price = float(order["price"])
                            size = float(order["size"])
                            side = order["side"].lower()
                            if side == "buy":
                                ob["b"][price] = size
                            elif side == "sell":
                                ob["a"][price] = size
                        orderbooks[symbol_formatted] = ob
                        logging.info(f"Updated orderbook for {symbol_formatted}")
            except Exception as e:
                logging.error(f"Error in WebSocket client: {e}")
                await asyncio.sleep(1)

async def start_bybit_ws():
    await bybit_ws_client()

def get_orderbook(symbol):
    symbol_formatted = symbol.replace("/", "")
    return orderbooks.get(symbol_formatted, None)
