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

INSTRUMENTS = [
    "EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CHF","USD_CAD","NZD_USD",
    "EUR_GBP","EUR_JPY","GBP_JPY"
]

RISK_PER_TRADE = 0.01
LOG_FILE = "ultimate_adaptive_bot.csv"

# Indicator periods
EMA_FAST, EMA_SLOW, EMA_LONG = 12,26,200
ATR_PERIOD, RSI_PERIOD, BOLL_PERIOD = 14,14,20
DONCHIAN_PERIOD, STOCH_K, STOCH_D = 14,3,3
VWAP_PERIOD, PIVOT_PERIOD, CCI_PERIOD, ADX_PERIOD = 20,14,20,14
ICHIMOKU_CONVERSION, ICHIMOKU_BASE, ICHIMOKU_LAG = 9,26,52
KELTNER_MULT = 2

# Strategy weights and reward/risk
STRATEGY_WEIGHTS = {"EMA_MACD":1.0,"RSI":0.8,"Stochastic":0.7,"CCI":0.6,
                    "VWAP":0.7,"Ichimoku":0.9,"Keltner":0.8,"Candlestick":0.6}
REWARD_RISK = {"EMA_MACD":2.0,"RSI":1.5,"Stochastic":1.5,"CCI":1.5,
               "VWAP":2.0,"Ichimoku":2.0,"Keltner":2.0,"Candlestick":1.2}

TRAIL_MULT = 0.5
CORR_THRESHOLD = 0.8
strategy_perf = {name: [] for name in STRATEGY_WEIGHTS}
MAX_HISTORY = 20

# ===== UTILITIES =====
def format_price(price, pair):
    precision = 3 if "JPY" in pair else 5
    return f"{price:.{precision}f}"

def fetch_candles(pair, count=500, granularity="M5"):
    url = f"{BASE_URL}/instruments/{pair}/candles"
    params = {"count": count, "granularity": granularity, "price": "M"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    candles = r.json()['candles']
    df = pd.DataFrame([{
        "time": c['time'],
        "open": float(c['mid']['o']),
        "high": float(c['mid']['h']),
        "low": float(c['mid']['l']),
        "close": float(c['mid']['c'])
    } for c in candles])
    return df

def get_account_equity():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/summary"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return float(r.json()['account']['NAV'])

def get_open_trades():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/openTrades"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    trades = r.json().get("trades", [])
    open_trades = {}
    for t in trades:
        pair = t['instrument']
        trade_id = t['id']
        units = int(t['currentUnits'])
        stop_loss = float(t['stopLossOrder']['price']) if t.get('stopLossOrder') else None
        take_profit = float(t['takeProfitOrder']['price']) if t.get('takeProfitOrder') else None
        strategy = t.get('clientExtensions', {}).get('comment', 'unknown')
        open_trades[pair] = {"id": trade_id, "units": units,
                             "stop_loss": stop_loss, "take_profit": take_profit,
                             "strategy": strategy}
    return open_trades

def place_order(pair, units, stop_loss=None, take_profit=None, strategy="Bot"):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"
    order = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": {"comment": strategy}
        }
    }
    if stop_loss:
        order["order"]["stopLossOnFill"] = {"price": format_price(stop_loss, pair)}
    if take_profit:
        order["order"]["takeProfitOnFill"] = {"price": format_price(take_profit, pair)}
    r = requests.post(url, headers=HEADERS, json=order)
    r.raise_for_status()
    print(f"{datetime.now()} | Order: {pair} {units} units")
    return r.json()

def update_stop_loss(trade_id, new_sl, pair):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/trades/{trade_id}/orders"
    data = {"stopLoss": {"price": format_price(new_sl, pair)}}
    r = requests.put(url, headers=HEADERS, json=data)
    if r.status_code in [200,201]:
        print(f"{datetime.now()} | Updated SL for trade {trade_id} to {new_sl}")
    else:
        print(f"{datetime.now()} | Failed to update SL for trade {trade_id}: {r.text}")

def log_trade(pair, units, strategy, score, price):
    df = pd.DataFrame([[datetime.now(), pair, units, strategy, score, price]],
                      columns=["time","pair","units","strategy","score","price"])
    df.to_csv(LOG_FILE, mode="a", header=not pd.io.common.file_exists(LOG_FILE), index=False)

# ===== INDICATORS AND STRATEGIES =====
# (Keep add_indicators() and all strat_* functions from previous script)
# ... [Paste your previous add_indicators() and strategy functions here] ...

# ===== PERFORMANCE TRACKING =====
def update_strategy_perf(strategy, profit):
    strategy_perf.setdefault(strategy, []).append(profit)
    if len(strategy_perf[strategy]) > MAX_HISTORY:
        strategy_perf[strategy].pop(0)

def adjust_strategy_weights():
    for name in STRATEGY_WEIGHTS:
        history = strategy_perf[name]
        if history:
            avg_perf = np.mean(history)
            STRATEGY_WEIGHTS[name] = np.clip(1 + avg_perf, 0.5, 1.5)

# ===== MAIN LOOP =====
if __name__=="__main__":
    while True:
        equity = get_account_equity()
        open_trades = get_open_trades()  # Sync live trades

        # Correlation matrix
        price_data = {pair: fetch_candles(pair, count=200)['close'] for pair in INSTRUMENTS}
        prices = pd.DataFrame(price_data)
        returns = prices.pct_change().dropna()
        cor_matrix = returns.tail(50).corr()

        def can_trade(pair):
            for other in open_trades:
                if other == pair: continue
                try:
                    corr = cor_matrix.loc[pair, other]
                except KeyError:
                    continue
                if abs(corr) > CORR_THRESHOLD:
                    print(f"{datetime.now()} | {pair} skipped due to high correlation ({corr:.2f}) with {other}")
                    return False
            return True

        for pair in INSTRUMENTS:
            df_m5 = add_indicators(fetch_candles(pair, granularity="M5"))
            df_h4 = add_indicators(fetch_candles(pair, granularity="H4"))

            if not df_m5['atr_filter'].iloc[-1]:
                print(f"{datetime.now()} | {pair} sideways, skipping")
                continue

            h4_trend = 1 if df_h4['close'].iloc[-1] > df_h4['ema_long'].iloc[-1] else -1

            adjust_strategy_weights()
            total_score = 0
            signals = {}
            for name, strat in STRATEGIES:
                trend, signal = strat(df_m5)
                total_score += signal * STRATEGY_WEIGHTS[name]
                signals[name] = signal

            last_price = df_m5['close'].iloc[-1]
            atr = df_m5['atr'].iloc[-1]
            units = int((equity * RISK_PER_TRADE) / atr)

            # BUY
            if total_score >= 1.0 and h4_trend > 0 and can_trade(pair):
                strategy = max(signals, key=lambda k: abs(signals[k]*STRATEGY_WEIGHTS[k]))
                rr = REWARD_RISK[strategy]
                stop_loss = last_price - atr
                take_profit = last_price + atr*rr
                order_resp = place_order(pair, units, stop_loss, take_profit, strategy)
                trade_id = order_resp['orderFillTransaction']['tradeOpened']['tradeID']
                open_trades[pair] = {"id": trade_id, "units": units,
                                     "stop_loss": stop_loss, "take_profit": take_profit,
                                     "strategy": strategy}
                log_trade(pair, units, "Weighted Buy", total_score, last_price)

            # SELL
            elif total_score <= -1.0 and h4_trend < 0 and can_trade(pair):
                strategy = max(signals, key=lambda k: abs(signals[k]*STRATEGY_WEIGHTS[k]))
                rr = REWARD_RISK[strategy]
                units = -units
                stop_loss = last_price + atr
                take_profit = last_price - atr*rr
                order_resp = place_order(pair, units, stop_loss, take_profit, strategy)
                trade_id = order_resp['orderFillTransaction']['tradeOpened']['tradeID']
                open_trades[pair] = {"id": trade_id, "units": units,
                                     "stop_loss": stop_loss, "take_profit": take_profit,
                                     "strategy": strategy}
                log_trade(pair, units, "Weighted Sell", total_score, last_price)

            # Update trailing stops live
            for trade_pair, trade in open_trades.items():
                current_price = df_m5['close'].iloc[-1]
                if trade['units'] > 0:
                    new_sl = max(trade['stop_loss'], current_price - TRAIL_MULT*atr)
                else:
                    new_sl = min(trade['stop_loss'], current_price + TRAIL_MULT*atr)
                if new_sl != trade['stop_loss']:
                    update_stop_loss(trade['id'], new_sl, trade_pair)
                    trade['stop_loss'] = new_sl
                profit = abs(current_price - trade['stop_loss'])
                update_strategy_perf(trade['strategy'], profit/atr)

        time.sleep(60*5)
