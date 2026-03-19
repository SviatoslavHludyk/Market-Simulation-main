import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd


# =========================
# CONFIG
# =========================
NUM_TRADERS = 1000
INITIAL_PRICE = 100.0

EVENTS_PER_UPDATE = 250
UPDATE_INTERVAL_SEC = 0.05

MAKER_PROBABILITY = 0.10
MAX_PRICE_DEVIATION = 0.005
MIN_ORDER_SIZE = 1
MAX_ORDER_SIZE = 10
CANCEL_PROBABILITY_PER_EVENT = 0.01

MAX_CANDLES_ON_SCREEN = 120
SIM_SECONDS_PER_EVENT = 0.02

ZOOM = 4
CANDLE_WIDTH = 0.35
Y_RANGE_CANDLES = 60


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Order:
    trader_id: int
    side: str
    price: float
    size: int
    timestamp: datetime


# =========================
# ORDER BOOK
# =========================
class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []

    def add_order(self, order):
        if order.side == "buy":
            self.bids.append(order)
            self.bids.sort(key=lambda o: o.price, reverse=True)
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda o: o.price)

    def cancel_random_order(self):
        side = random.choice(["buy", "sell"])
        if side == "buy" and self.bids:
            self.bids.pop(random.randrange(len(self.bids)))
        elif side == "sell" and self.asks:
            self.asks.pop(random.randrange(len(self.asks)))

    def best_bid(self):
        return self.bids[0] if self.bids else None

    def best_ask(self):
        return self.asks[0] if self.asks else None

    def execute_taker(self, side, size, timestamp):
        trades = []

        if side == "buy":
            while size > 0 and self.asks:
                best = self.asks[0]
                trade_size = min(size, best.size)

                trades.append({
                    "time": timestamp,
                    "price": best.price,
                    "size": trade_size
                })

                best.size -= trade_size
                size -= trade_size

                if best.size == 0:
                    self.asks.pop(0)

        else:
            while size > 0 and self.bids:
                best = self.bids[0]
                trade_size = min(size, best.size)

                trades.append({
                    "time": timestamp,
                    "price": best.price,
                    "size": trade_size
                })

                best.size -= trade_size
                size -= trade_size

                if best.size == 0:
                    self.bids.pop(0)

        return trades


# =========================
# SIMULATION
# =========================
class MarketSimulation:
    def __init__(self):
        self.order_book = OrderBook()
        self.current_price = INITIAL_PRICE
        self.trade_log = []
        self.start_time = datetime.now().replace(microsecond=0)
        self.sim_time_seconds = 0.0
        self._seed_initial_liquidity()

    def _seed_initial_liquidity(self):
        for i in range(300):
            bid_price = round(self.current_price * (1 - random.uniform(0.0002, 0.01)), 4)
            ask_price = round(self.current_price * (1 + random.uniform(0.0002, 0.01)), 4)

            self.order_book.add_order(
                Order(i, "buy", bid_price, random.randint(1, 10), self.start_time)
            )
            self.order_book.add_order(
                Order(i, "sell", ask_price, random.randint(1, 10), self.start_time)
            )

    def _generate_random_order(self, trader_id, timestamp):
        side = random.choice(["buy", "sell"])
        is_maker = random.random() < MAKER_PROBABILITY
        size = random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE)

        price_multiplier = 1 + random.uniform(-MAX_PRICE_DEVIATION, MAX_PRICE_DEVIATION)
        price = round(max(0.01, self.current_price * price_multiplier), 4)

        return {
            "trader_id": trader_id,
            "side": side,
            "is_maker": is_maker,
            "price": price,
            "size": size,
            "timestamp": timestamp
        }

    def step(self, n_events=EVENTS_PER_UPDATE):
        for _ in range(n_events):
            self.sim_time_seconds += SIM_SECONDS_PER_EVENT
            sim_time = self.start_time + timedelta(seconds=self.sim_time_seconds)

            if random.random() < CANCEL_PROBABILITY_PER_EVENT:
                self.order_book.cancel_random_order()

            trader_id = random.randint(0, NUM_TRADERS - 1)
            order = self._generate_random_order(trader_id, sim_time)

            if order["is_maker"]:
                self.order_book.add_order(
                    Order(
                        order["trader_id"],
                        order["side"],
                        order["price"],
                        order["size"],
                        order["timestamp"]
                    )
                )
            else:
                trades = self.order_book.execute_taker(
                    order["side"],
                    order["size"],
                    sim_time
                )

                if trades:
                    self.trade_log.extend(trades)
                    self.current_price = trades[-1]["price"]

            while self.order_book.best_bid() and self.order_book.best_ask():
                if self.order_book.best_bid().price >= self.order_book.best_ask().price:
                    bid = self.order_book.best_bid()
                    ask = self.order_book.best_ask()

                    trade_size = min(bid.size, ask.size)
                    trade_price = ask.price

                    self.trade_log.append({
                        "time": sim_time,
                        "price": trade_price,
                        "size": trade_size
                    })

                    self.current_price = trade_price

                    bid.size -= trade_size
                    ask.size -= trade_size

                    if bid.size == 0:
                        self.order_book.bids.pop(0)
                    if ask.size == 0:
                        self.order_book.asks.pop(0)
                else:
                    break

    def get_ohlc(self):
        if not self.trade_log:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(self.trade_log).sort_values("time")
        df = df.set_index("time")

        ohlc = df["price"].resample("1s").ohlc()
        volume = df["size"].resample("1s").sum()

        ohlc["volume"] = volume
        ohlc = ohlc.dropna().reset_index()

        if len(ohlc) > MAX_CANDLES_ON_SCREEN:
            ohlc = ohlc.iloc[-MAX_CANDLES_ON_SCREEN:].copy()

        return ohlc


# =========================
# DRAWING
# =========================
def draw_candles(ax, ohlc_df, current_price):
    ax.clear()
    ax.set_facecolor("black")

    if ohlc_df.empty:
        ax.set_title("Waiting for trades...", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        return

    for i, row in enumerate(ohlc_df.itertuples(index=False)):
        o = row.open
        h = row.high
        l = row.low
        c = row.close

        color = "lime" if c >= o else "red"

        ax.plot([i, i], [l, h], color="white", linewidth=1.0, zorder=1)

        lower = min(o, c)
        height = abs(c - o)
        if height < 1e-6:
            height = 0.0005

        rect = Rectangle(
            (i - CANDLE_WIDTH / 2, lower),
            CANDLE_WIDTH,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            zorder=2
        )
        ax.add_patch(rect)

    ax.axhline(current_price, color="gray", linestyle="--", linewidth=0.8)

    recent = ohlc_df.tail(Y_RANGE_CANDLES)
    ymin = float(recent["low"].min())
    ymax = float(recent["high"].max())

    mid = (ymin + ymax) / 2
    half_range = (ymax - ymin) / 2
    if half_range < 1e-6:
        half_range = 0.01

    ax.set_ylim(mid - half_range * ZOOM, mid + half_range * ZOOM)
    ax.set_xlim(-2, len(ohlc_df) + 2)

    ax.set_title("Real-Time Market Simulation", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    step = max(1, len(ohlc_df) // 8)
    ticks = list(range(0, len(ohlc_df), step))
    labels = [ohlc_df.iloc[t]["time"].strftime("%H:%M:%S") for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", color="white")
    ax.grid(False)


# =========================
# MAIN DESKTOP LOOP
# =========================
def main():
    sim = MarketSimulation()

    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("black")

    while plt.fignum_exists(fig.number):
        sim.step(EVENTS_PER_UPDATE)
        ohlc = sim.get_ohlc()
        draw_candles(ax, ohlc, sim.current_price)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(UPDATE_INTERVAL_SEC)

    plt.ioff()


if __name__ == "__main__":
    main()
    