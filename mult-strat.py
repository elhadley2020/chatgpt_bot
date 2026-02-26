import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ===== CONFIG =====
API_KEY = "YOUR_OANDA_API_KEY"
ACCOUNT_ID = "YOUR_OANDA_ACCOUNT_ID"
BASE_URL = "https://api-fxpractice.oanda.com/v3"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
RISK_PER_TRADE = 0.01  # 1% of equity per trade
LOG_FILE = "ultimate_all_strategies_bot.csv"

# Indicator periods
EMA_FAST, EMA_SLOW, EMA_LONG = 12, 26, 200
ATR_PERIOD, RSI_PERIOD, BOLL_PERIOD = 14, 14, 20
DONCHIAN_PERIOD, STOCH_K, STOCH_D = 14, 3, 3
VWAP_PERIOD = 20
PIVOT_PERIOD = 14

# ===== UTILITY FUNCTIONS =====
def fetch_candles(pair, count=500, granularity="M5"):
    url = f"{BASE_URL}/instruments/{pair}/candles"
    params = {"count": count, "granularity": granularity, "price": "M"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    candles = r.json()['candles']
    df = pd.DataFrame([{"time": c['time'],
                        "open": float(c['mid']['o']),
                        "high": float(c['mid']['h']),
                        "low": float(c['mid']['l']),
                        "close": float(c['mid']['c'])} for c in candles])
    return df

def get_account_equity():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/summary"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return float(r.json()['account']['NAV'])

def place_order(pair, units, stop_loss=None, take_profit=None):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"
    order = {"order": {"instrument": pair, "units": str(units), "type": "MARKET", "positionFill": "DEFAULT"}}
    if stop_loss: order["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
    if take_profit: order["order"]["takeProfitOnFill"] = {"price": str(take_profit)}
    r = requests.post(url, headers=HEADERS, json=order)
    r.raise_for_status()
    print(f"{datetime.now()} | Order: {pair} {units} units")
    return r.json()

def log_trade(pair, units, strategy, price):
    df = pd.DataFrame([[datetime.now(), pair, units, strategy, price]],
                      columns=["time", "pair", "units", "strategy", "price"])
    df.to_csv(LOG_FILE, mode="a", header=not pd.io.common.file_exists(LOG_FILE), index=False)

# ===== INDICATORS =====
def add_indicators(df):
    # EMA
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    # MACD
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # ATR
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    df['boll_mid'] = df['close'].rolling(BOLL_PERIOD).mean()
    df['boll_std'] = df['close'].rolling(BOLL_PERIOD).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    # Donchian Channel
    df['donchian_high'] = df['close'].rolling(DONCHIAN_PERIOD).max()
    df['donchian_low'] = df['close'].rolling(DONCHIAN_PERIOD).min()
    # Pivot Points
    df['pivot'] = (df['high'].rolling(PIVOT_PERIOD).max() +
                   df['low'].rolling(PIVOT_PERIOD).min() +
                   df['close'].rolling(PIVOT_PERIOD).mean()) / 3
    # VWAP
    df['vwap'] = (df['close'] * df['tr']).rolling(VWAP_PERIOD).sum() / df['tr'].rolling(VWAP_PERIOD).sum()
    # Stochastic
    df['stoch_k'] = ((df['close'] - df['low'].rolling(STOCH_K).min()) /
                     (df['high'].rolling(STOCH_K).max() - df['low'].rolling(STOCH_K).min()) * 100)
    df['stoch_d'] = df['stoch_k'].rolling(STOCH_D).mean()
    # ATR Filter
    df['atr_filter'] = df['atr'] > df['atr'].rolling(50).mean() * 0.5
    return df

# ===== STRATEGY FUNCTIONS =====
def strat_ema_macd(df):
    last = df.iloc[-1]
    trend = 1 if last['close'] > last['ema_long'] else -1
    signal = 0
    if df['macd'].iloc[-2] < df['signal'].iloc[-2] and last['macd'] > last['signal']:
        signal = 1
    elif df['macd'].iloc[-2] > df['signal'].iloc[-2] and last['macd'] < last['signal']:
        signal = -1
    return trend, signal

def strat_rsi(df):
    last = df.iloc[-1]
    signal = 0
    if last['rsi'] < 30:
        signal = 1
    elif last['rsi'] > 70:
        signal = -1
    return 0, signal

def strat_atr_breakout(df):
    last = df.iloc[-1]
    signal = 0
    if last['close'] > df['donchian_high'].iloc[-2]:
        signal = 1
    elif last['close'] < df['donchian_low'].iloc[-2]:
        signal = -1
    return 0, signal

def strat_bollinger(df):
    last = df.iloc[-1]
    signal = 0
    if last['close'] < last['boll_lower']:
        signal = 1
    elif last['close'] > last['boll_upper']:
        signal = -1
    return 0, signal

def strat_pivot(df):
    last = df.iloc[-1]
    signal = 0
    if last['close'] < last['pivot']:
        signal = 1
    elif last['close'] > last['pivot']:
        signal = -1
    return 0, signal

def strat_vwap(df):
    last = df.iloc[-1]
    signal = 0
    if last['close'] < last['vwap'] and last['close'] > df['vwap'].iloc[-2]:
        signal = 1
    elif last['close'] > last['vwap'] and last['close'] < df['vwap'].iloc[-2]:
        signal = -1
    return 0, signal

def strat_stochastic(df):
    last = df.iloc[-1]
    signal = 0
    if last['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d']:
        signal = 1
    elif last['stoch_k'] > 80 and last['stoch_k'] < last['stoch_d']:
        signal = -1
    return 0, signal

# ===== STRATEGY LIST =====
STRATEGIES = [
    ("EMA_MACD", strat_ema_macd),
    ("RSI", strat_rsi),
    ("ATR_Breakout", strat_atr_breakout),
    ("Bollinger", strat_bollinger),
    ("Pivot", strat_pivot),
    ("VWAP", strat_vwap),
    ("Stochastic", strat_stochastic)
]

# ===== MAIN LOOP =====
if __name__ == "__main__":
    while True:
        equity = get_account_equity()
        for pair in INSTRUMENTS:
            df_m5 = fetch_candles(pair, granularity="M5")
            df_h1 = fetch_candles(pair, granularity="H1")
            df_h4 = fetch_candles(pair, granularity="H4")
            df_m5 = add_indicators(df_m5)
            df_h1 = add_indicators(df_h1)
            df_h4 = add_indicators(df_h4)

            if not df_m5['atr_filter'].iloc[-1]:
                print(f"{datetime.now()} | {pair} sideways, skipping")
                continue

            # Multi-timeframe trend confirmation
            h4_trend = 1 if df_h4['close'].iloc[-1] > df_h4['ema_long'].iloc[-1] else -1

            for name, strat in STRATEGIES:
                trend, signal = strat(df_m5)
                if signal != 0 and signal == h4_trend:
                    last_price = df_m5['close'].iloc[-1]
                    atr = df_m5['atr'].iloc[-1]
                    units = int((equity * RISK_PER_TRADE) / atr) * signal
                    stop_loss = last_price - atr if signal > 0 else last_price + atr
                    take_profit = last_price + 2 * atr if signal > 0 else last_price - 2 * atr
                    place_order(pair, units, stop_loss, take_profit)
                    log_trade(pair, units, name, last_price)
                    break
        time.sleep(60*5)
