# headless_forex_bot.py
import websocket, requests, json, threading, pandas as pd, numpy as np, csv
from collections import deque
from datetime import datetime
import time

# -------------------------------
# CONFIG
# -------------------------------
API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"
REST_URL = "https://api-fxpractice.oanda.com/v3"

PAIRS = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CHF",
         "USD_CAD","EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY"]

RISK_PER_TRADE = 0.005
MAX_TOTAL_RISK = 0.02
MAX_OPEN_TRADES = 4
MAX_DAILY_LOSS = 0.015
MAX_WEEKLY_LOSS = 0.04

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

LOG_FILE = "trade_log.csv"
EQUITY_FILE = "equity_curve.csv"

# -------------------------------
# PORTFOLIO ENGINE
# -------------------------------
class PortfolioEngine:
    def __init__(self):
        self.trade_states = {pair: {"in_position": False, "trade_id": None, "units": 0, "entry": None, "risk": 0, "stop_loss": None} for pair in PAIRS}
        self.price_buffers = {pair: deque(maxlen=300) for pair in PAIRS}
        self.daily_start_balance = None
        self.weekly_start_balance = None
        self.losing_streak = 0

        # CSV logs
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","pair","direction","units","entry_price","exit_price","result","R_multiple"])
        with open(EQUITY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","balance"])

    def get_balance(self):
        r = requests.get(f"{REST_URL}/accounts/{ACCOUNT_ID}", headers=HEADERS).json()
        return float(r["account"]["balance"])

    def total_open_risk(self):
        return sum([v["risk"] for v in self.trade_states.values() if v["in_position"]])

    def open_positions_count(self):
        return sum([1 for v in self.trade_states.values() if v["in_position"]])

    def risk_locks(self):
        balance = self.get_balance()
        if self.daily_start_balance is None:
            self.daily_start_balance = balance
        if self.weekly_start_balance is None:
            self.weekly_start_balance = balance

        daily_dd = (self.daily_start_balance - balance)/self.daily_start_balance
        weekly_dd = (self.weekly_start_balance - balance)/self.weekly_start_balance

        if daily_dd >= MAX_DAILY_LOSS or weekly_dd >= MAX_WEEKLY_LOSS:
            print("Drawdown limit reached. Trading paused.")
            return True
        return False

    # -------------------------------
    # SIGNALS
    # -------------------------------
    def generate_signal(self, pair):
        buf = self.price_buffers[pair]
        if len(buf) < 200: return None

        df = pd.DataFrame(buf, columns=["price"])
        df["ema200"] = df["price"].ewm(span=200).mean()
        df["ema21"] = df["price"].ewm(span=21).mean()
        df["tr"] = df["price"].diff().abs()
        df["atr"] = df["tr"].rolling(14).mean()
        df["atr_med"] = df["atr"].rolling(30).median()
        last = df.iloc[-1]
        prev = df.iloc[-2]

        if last["atr"] < 0.7*last["atr_med"]: return None
        if last["price"] > last["ema200"] and prev["price"] < prev["ema21"]:
            return "buy"
        if last["price"] < last["ema200"] and prev["price"] > prev["ema21"]:
            return "sell"
        return None

    # -------------------------------
    # ATR TRAILING STOP
    # -------------------------------
    def calculate_trailing_stop(self, pair, atr_multiplier=1.5):
        buf = self.price_buffers[pair]
        if len(buf) < 14: return None
        df = pd.DataFrame(buf, columns=["price"])
        df["tr"] = df["price"].diff().abs()
        df["atr"] = df["tr"].rolling(14).mean()
        atr = df["atr"].iloc[-1]

        trade = self.trade_states[pair]
        if not trade["in_position"]: return None

        if trade["units"] > 0:
            return max(trade.get("stop_loss", trade["entry"] - atr*atr_multiplier),
                       buf[-1] - atr*atr_multiplier)
        else:
            return min(trade.get("stop_loss", trade["entry"] + atr*atr_multiplier),
                       buf[-1] + atr*atr_multiplier)

    # -------------------------------
    # EXECUTION
    # -------------------------------
    def place_order(self, pair, direction):
        if self.risk_locks(): return
        if self.open_positions_count() >= MAX_OPEN_TRADES: return
        if self.trade_states[pair]["in_position"]: return
        if self.total_open_risk() >= MAX_TOTAL_RISK: return

        balance = self.get_balance()
        risk_amount = balance * RISK_PER_TRADE
        units = int(risk_amount / 0.001)
        if direction=="sell": units=-units

        order = {"order":{"instrument":pair,"units":str(units),"type":"MARKET","timeInForce":"FOK","positionFill":"DEFAULT"}}
        r = requests.post(f"{REST_URL}/accounts/{ACCOUNT_ID}/orders", headers=HEADERS, json=order)
        print(f"Order sent: {pair} {direction}")

    # -------------------------------
    # STREAM HANDLER
    # -------------------------------
    def on_price(self, ws, message):
        data = json.loads(message)
        if "instrument" in data and "bids" in data:
            pair = data["instrument"]
            price = float(data["bids"][0]["price"])
            self.price_buffers[pair].append(price)
            signal = self.generate_signal(pair)
            if signal: self.place_order(pair, signal)

            if self.trade_states[pair]["in_position"]:
                stop_price = self.calculate_trailing_stop(pair)
                if stop_price and stop_price != self.trade_states[pair].get("stop_loss"):
                    self.trade_states[pair]["stop_loss"] = stop_price
                    print(f"Trailing stop updated for {pair} to {stop_price}")

# -------------------------------
# START BOT
# -------------------------------
engine = PortfolioEngine()

price_stream = websocket.WebSocketApp(
    f"wss://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream?instruments={','.join(PAIRS)}",
    header=[f"Authorization: Bearer {API_KEY}"],
    on_message=engine.on_price
)

print("Headless Forex bot running 24/5...")
price_stream.run_forever()
