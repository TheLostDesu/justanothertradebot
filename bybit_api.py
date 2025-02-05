# bybit_api.py
import time
import hmac
import hashlib
import aiohttp
import json
from config import API_KEY, API_SECRET, REST_API_BASE, ORDER_CREATE_ENDPOINT, BALANCE_ENDPOINT

def sign_request(method, endpoint, params, secret):
    timestamp = str(int(time.time() * 1000))
    # Сортируем параметры и формируем строку для подписи (проверьте согласно документации Bybit v5)
    sorted_params = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sign_str = f"{timestamp}{method.upper()}{endpoint}{sorted_params}"
    signature = hmac.new(secret.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).hexdigest()
    return timestamp, signature

async def fetch_balance(session):
    endpoint = BALANCE_ENDPOINT
    url = REST_API_BASE + endpoint
    params = {}
    timestamp, signature = sign_request("GET", endpoint, params, API_SECRET)
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature
    }
    async with session.get(url, headers=headers, params=params) as response:
        return await response.json()

async def create_order(session, symbol, side, qty, leverage, stop_loss, take_profit, trailing_stop):
    endpoint = ORDER_CREATE_ENDPOINT
    url = REST_API_BASE + endpoint
    params = {
        "category": "linear",
        "symbol": symbol.replace("/", ""),
        "side": side.upper(),
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "GoodTillCancel",
        "leverage": leverage,
        "stopLoss": stop_loss,
        "takeProfit": take_profit,
        "trailingStop": trailing_stop
    }
    timestamp, signature = sign_request("POST", endpoint, params, API_SECRET)
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }
    async with session.post(url, headers=headers, json=params) as response:
        return await response.json()
