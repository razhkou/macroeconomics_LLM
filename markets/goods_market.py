from collections import defaultdict
from typing import List, Dict, Any, Tuple


class GoodsMarket:
    def __init__(self):
        self.asks = defaultdict(list)  # good -> list of (agent_id, price, quantity)
        self.bids = defaultdict(list)  # good -> list of (agent_id, price, quantity)
        self.prices = {}               # good -> last price

    def reset(self):
        self.asks.clear()
        self.bids.clear()
        self.prices.clear()

    def place_ask(self, agent_id: str, good: str, price: float, quantity: float):
        if quantity > 0 and price >= 0:
            self.asks[good].append((agent_id, price, quantity))

    def place_bid(self, agent_id: str, good: str, price: float, quantity: float):
        if quantity > 0 and price >= 0:
            self.bids[good].append((agent_id, price, quantity))

    def clear(self) -> List[Tuple[str, str, str, float, float]]:
        trades = []
        for good in set(self.asks.keys()) | set(self.bids.keys()):
            asks = sorted(self.asks.get(good, []), key=lambda x: x[1])
            bids = sorted(self.bids.get(good, []), key=lambda x: -x[1])
            i, j = 0, 0
            while i < len(asks) and j < len(bids):
                ask_price = asks[i][1]
                bid_price = bids[j][1]
                if ask_price <= bid_price:
                    trade_price = (ask_price + bid_price) / 2
                    quantity = min(asks[i][2], bids[j][2])
                    trades.append((asks[i][0], bids[j][0], good, trade_price, quantity))
                    asks[i] = (asks[i][0], ask_price, asks[i][2] - quantity)
                    bids[j] = (bids[j][0], bid_price, bids[j][2] - quantity)
                    if asks[i][2] <= 0:
                        i += 1
                    if bids[j][2] <= 0:
                        j += 1
                else:
                    break
            if trades:
                self.prices[good] = trades[-1][3]
        self.asks.clear()
        self.bids.clear()
        return trades

    def get_current_prices(self) -> Dict[str, float]:
        return self.prices.copy()