# orderbook.py
from sortedcontainers import SortedList

class OrderBookSide:
    """
    Класс для хранения ордеров одной стороны orderbook с использованием SortedList.
    
    Для бидов (is_bid=True) ключ сортировки = -price (чтобы сортировать по убыванию цены),
    для асков (is_bid=False) ключ = price (сортировка по возрастанию цены).
    
    Каждый элемент имеет вид: (sort_key, price, size).
    """
    def __init__(self, is_bid=False):
        self.is_bid = is_bid
        self.sl = SortedList(key=lambda x: x[0])
    
    def update(self, price, size):
        key = -price if self.is_bid else price
        idx = self.sl.bisect_left((key, price, 0.0))
        if idx < len(self.sl) and self.sl[idx][1] == price:
            if size == 0.0:
                self.sl.pop(idx)
            else:
                self.sl[idx] = (key, price, size)
        else:
            if size != 0.0:
                self.sl.add((key, price, size))
    
    def get_list(self):
        """Возвращает содержимое SortedList как обычный список."""
        return list(self.sl)

def get_features_from_orderbook(ob, num_levels):
    """
    Извлекает признаки из orderbook, представленного в виде словаря с ключами 'a' и 'b',
    где каждая сторона — обычный список кортежей (sort_key, price, size).
    Для каждого уровня возвращает цену и размер.
    """
    features = []
    bids = ob.get('b', [])
    for i in range(num_levels):
        if i < len(bids):
            _, price, size = bids[i]
            features.extend([price, size])
        else:
            features.extend([0.0, 0.0])
    asks = ob.get('a', [])
    for i in range(num_levels):
        if i < len(asks):
            _, price, size = asks[i]
            features.extend([price, size])
        else:
            features.extend([0.0, 0.0])
    return features

def get_mid_price(ob):
    """
    Вычисляет среднюю цену между лучшим бидом и лучшим аском.
    Если хотя бы одна сторона пуста, возвращает None.
    """
    bids = ob.get('b', [])
    asks = ob.get('a', [])
    if not bids or not asks:
        return None
    best_bid = bids[0][1]
    best_ask = asks[0][1]
    return (best_bid + best_ask) / 2.0
