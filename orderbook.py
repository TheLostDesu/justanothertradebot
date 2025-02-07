import time
from collections import deque
from config import SEQUENCE_LENGTH, CANDLE_INTERVAL_MIN, CANDLE_TOTAL_HOURS

# Вспомогательный класс для хранения уровней одной стороны ордербука (bid или ask)
class OrderBookSide:
    def __init__(self, is_bid: bool):
        """
        :param is_bid: True – если эта сторона bid, False – если ask
        """
        self.is_bid = is_bid
        self.levels = {}

    def update(self, price: float, size: float):
        """
        Обновление уровня. Если size равен 0, уровень удаляется.
        """
        if size == 0:
            self.levels.pop(price, None)
        else:
            self.levels[price] = size

    def best_price(self):
        """
        Возвращает лучший уровень: для bid – максимальная цена, для ask – минимальная.
        """
        if not self.levels:
            return None
        if self.is_bid:
            return max(self.levels.keys())
        else:
            return min(self.levels.keys())

    def get_levels(self):
        """
        Возвращает уровни, отсортированные по цене:
          - для bid – по убыванию,
          - для ask – по возрастанию.
        """
        return sorted(self.levels.items(), key=lambda x: x[0], reverse=self.is_bid)

# Основной класс Orderbook
class Orderbook:
    def __init__(self):
        self.current_ob = None
        self.sequence_history = deque(maxlen=SEQUENCE_LENGTH)
        num_candles = int((CANDLE_TOTAL_HOURS * 60) / CANDLE_INTERVAL_MIN)
        self.candles = deque(maxlen=num_candles)
        self.current_candle = None 
        self.last_snapshot_time = None

    def update(self, data: dict, r_type: str, timestamp: float = None):
        """
        Принимает апдейт ордербука.
        
        :param data: данные апдейта, содержащие уровни для сторон 'a' (asks) и 'b' (bids)
        :param r_type: тип апдейта: 'snapshot' или 'delta'
        :param timestamp: время получения апдейта (если None – используется time.time())
        """
        if timestamp is None:
            timestamp = time.time()

        # Если snapshot или поле 'u' == 1 – создаём новый ордербук
        if r_type == 'snapshot' or data.get('u') == 1:
            self.current_ob = {
                'a': OrderBookSide(is_bid=False),
                'b': OrderBookSide(is_bid=True)
            }
            if 'a' in data:
                for level in data['a']:
                    try:
                        price = float(level[0])
                        size = float(level[1])
                        self.current_ob['a'].update(price, size)
                    except Exception:
                        continue
            if 'b' in data:
                for level in data['b']:
                    try:
                        price = float(level[0])
                        size = float(level[1])
                        self.current_ob['b'].update(price, size)
                    except Exception:
                        continue
        # Иначе, если апдейт delta
        elif r_type == 'delta':
            if self.current_ob is None:
                # Если ордербук ещё не создан, пропускаем delta-апдейт
                return
            for side in ['a', 'b']:
                if side in data:
                    for update in data[side]:
                        try:
                            price = float(update[0])
                            size = float(update[1])
                        except Exception:
                            continue
                        self.current_ob[side].update(price, size)

        # Сохраняем snapshot фичи ордербука раз в секунду
        if self.last_snapshot_time is None or (timestamp - self.last_snapshot_time) >= 1:
            self.last_snapshot_time = timestamp
            snapshot = self.get_snapshot_features()
            if snapshot:
                self.sequence_history.append(snapshot)
            # Если можно вычислить mid‑цену, обновляем данные для свечи
            mid_price = snapshot.get('mid') if snapshot else None
            if mid_price is not None:
                self._update_candle(mid_price, timestamp)

    def get_snapshot_features(self):
        """
        Извлекает фичи из текущего ордербука – лучшие цены и mid‑цену.
        Возвращает словарь:
            {'bid': лучшая_bid, 'ask': лучшая_ask, 'mid': mid_цена}
        """
        if self.current_ob is None:
            return None
        best_bid = self.current_ob['b'].best_price()
        best_ask = self.current_ob['a'].best_price()
        if best_bid is None or best_ask is None:
            mid = None
        else:
            mid = (best_bid + best_ask) / 2
        return {'bid': best_bid, 'ask': best_ask, 'mid': mid}

    def _update_candle(self, mid_price: float, timestamp: float):
        """
        Обновляет текущую свечу, используя mid‑цену.
        Если интервал свечи (CANDLE_INTERVAL_MIN минут) прошёл, завершает свечу и добавляет её в историю.
        """
        if self.current_candle is None:
            # Инициализируем новую свечу
            self.current_candle = {
                'open': mid_price,
                'high': mid_price,
                'low': mid_price,
                'close': mid_price,
                'start': timestamp
            }
        else:
            # Обновляем high, low и close текущей свечи
            self.current_candle['high'] = max(self.current_candle['high'], mid_price)
            self.current_candle['low'] = min(self.current_candle['low'], mid_price)
            self.current_candle['close'] = mid_price

            # Если прошёл интервал свечи, завершаем её
            if timestamp - self.current_candle['start'] >= CANDLE_INTERVAL_MIN * 60:
                candle = {
                    'open': self.current_candle['open'],
                    'high': self.current_candle['high'],
                    'low': self.current_candle['low'],
                    'close': self.current_candle['close'],
                    'start': self.current_candle['start'],
                    'end': self.current_candle['start'] + CANDLE_INTERVAL_MIN * 60
                }
                self.candles.append(candle)
                # Начинаем новую свечу с текущей mid‑ценой
                self.current_candle = {
                    'open': mid_price,
                    'high': mid_price,
                    'low': mid_price,
                    'close': mid_price,
                    'start': timestamp
                }

    def get_features(self):
        """
        Возвращает накопленные фичи:
         - 'sequence': последовательность snapshot-ов (история за SEQUENCE_LENGTH секунд)
         - 'candles': список сформированных свечей за период CANDLE_TOTAL_HOURS
        """
        return {
            'sequence': list(self.sequence_history),
            'candles': list(self.candles)
        }
