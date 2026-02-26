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

# ===== INDICATORS =====
def add_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain/avg_loss
    df['rsi'] = 100-(100/(1+rs))
    
    df['boll_mid'] = df['close'].rolling(BOLL_PERIOD).mean()
    df['boll_std'] = df['close'].rolling(BOLL_PERIOD).std()
    df['boll_upper'] = df['boll_mid'] + 2*df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2*df['boll_std']
    
    df['donchian_high'] = df['close'].rolling(DONCHIAN_PERIOD).max()
    df['donchian_low'] = df['close'].rolling(DONCHIAN_PERIOD).min()
    
    df['vwap'] = (df['close']*df['tr']).rolling(VWAP_PERIOD).sum()/df['tr'].rolling(VWAP_PERIOD).sum()
    
    df['stoch_k'] = ((df['close'] - df['low'].rolling(STOCH_K).min()) / 
                     (df['high'].rolling(STOCH_K).max() - df['low'].rolling(STOCH_K).min()) * 100)
    df['stoch_d'] = df['stoch_k'].rolling(STOCH_D).mean()
    
    tp = (df['high']+df['low']+df['close'])/3
    ma = tp.rolling(CCI_PERIOD).mean()
    md = tp.rolling(CCI_PERIOD).apply(lambda x: np.mean(np.abs(x-np.mean(x))))
    df['cci'] = (tp-ma)/(0.015*md)
    
    df['tenkan'] = (df['high'].rolling(ICHIMOKU_CONVERSION).max()+df['low'].rolling(ICHIMOKU_CONVERSION).min())/2
    df['kijun'] = (df['high'].rolling(ICHIMOKU_BASE).max()+df['low'].rolling(ICHIMOKU_BASE).min())/2
    df['senkou_a'] = ((df['tenkan']+df['kijun'])/2).shift(ICHIMOKU_BASE)
    df['senkou_b'] = (df['high'].rolling(ICHIMOKU_LAG).max()+df['low'].rolling(ICHIMOKU_LAG).min())/2
    
    df['keltner_upper'] = df['ema_long'] + KELTNER_MULT*df['atr']
    df['keltner_lower'] = df['ema_long'] - KELTNER_MULT*df['atr']
    
    df['atr_filter'] = df['atr'] > df['atr'].rolling(50).mean()*0.5
    return df

# ===== STRATEGIES =====
def strat_ema_macd(df):
    last=df.iloc[-1]; trend=1 if last['close']>last['ema_long'] else -1
    signal=0
    if df['macd'].iloc[-2]<df['signal'].iloc[-2] and last['macd']>last['signal']: signal=1
    elif df['macd'].iloc[-2]>df['signal'].iloc[-2] and last['macd']<last['signal']: signal=-1
    return trend,signal

def strat_rsi(df):
    last=df.iloc[-1]; signal=0
    if last['rsi']<30: signal=1
    elif last['rsi']>70: signal=-1
    return 0,signal

def strat_stoch(df):
    last=df.iloc[-1]; prev=df.iloc[-2]; signal=0
    if last['stoch_k']<20 and last['stoch_k']>last['stoch_d']: signal=1
    elif last['stoch_k']>80 and last['stoch_k']<prev['stoch_d']: signal=-1
    return 0,signal

def strat_cci(df):
    last=df.iloc[-1]; signal=0
    if last['cci']<-100: signal=1
    elif last['cci']>100: signal=-1
    return 0,signal

def strat_vwap(df):
    last=df.iloc[-1]; prev=df.iloc[-2]; signal=0
    if last['close']<last['vwap'] and prev['close']>prev['vwap']: signal=1
    elif last['close']>last['vwap'] and prev['close']<prev['vwap']: signal=-1
    return 0,signal

def strat_ichimoku(df):
    last=df.iloc[-1]; signal=0
    if last['close']>last['senkou_a'] and last['close']>last['senkou_b']: signal=1
    elif last['close']<last['senkou_a'] and last['close']<last['senkou_b']: signal=-1
    return 0,signal

def strat_keltner(df):
    last=df.iloc[-1]; signal=0
    if last['close']>last['keltner_upper']: signal=1
    elif last['close']<last['keltner_lower']: signal=-1
    return 0,signal

def candlestick_signal(df):
    last=df.iloc[-1]; prev=df.iloc[-2]; signal=0
    if prev['close']<prev['open'] and last['close']>last['open'] and last['close']>prev['open'] and last['open']<prev['close']: signal=1
    elif prev['close']>prev['open'] and last['close']<last['open'] and last['open']>prev['close'] and last['close']<prev['open']: signal=-1
    return 0,signal

STRATEGIES=[
    ("EMA_MACD",strat_ema_macd),
    ("RSI",strat_rsi),
    ("Stochastic",strat_stoch),
    ("CCI",strat_cci),
    ("VWAP",strat_vwap),
    ("Ichimoku",strat_ichimoku),
    ("Keltner",strat_keltner),
    ("Candlestick",candlestick_signal)
]

# ===== PERFORMANCE TRACKING =====
def update_strategy_perf(strategy, profit):
    strategy_perf.setdefault(strategy, []).append(profit)
    if len(strategy_perf[strategy])>MAX_HISTORY:
        strategy_perf[strategy].pop(0)

def adjust_strategy_weights():
    for name in STRATEGY_WEIGHTS:
        history = strategy_perf[name]
        if history:
            avg_perf = np.mean(history)
            STRATEGY_WEIGHTS[name] = np.clip(1+avg_perf,0.5,1.5)

# ===== MAIN LOOP =====
if __name__=="__main__":
    while True:
        equity=get_account_equity()
        open_trades=get_open_trades()
        prices=pd.DataFrame({pair: fetch_candles(pair,count=200)['close'] for pair in INSTRUMENTS})
        cor_matrix=prices.pct_change().dropna().tail(50).corr()

        def can_trade(pair):
            for other in open_trades:
                if other==pair: continue
                try:
                    corr = cor_matrix.loc[pair,other]
                except KeyError:
                    continue
                if abs(corr) > CORR_THRESHOLD:
                    print(f"{datetime.now()} | {pair} skipped due to correlation {corr:.2f} with {other}")
                    return False
            return True

        for pair in INSTRUMENTS:
            df_m5 = add_indicators(fetch_candles(pair,granularity="M5"))
            df_h4 = add_indicators(fetch_candles(pair,granularity="H4"))
            if not df_m5['atr_filter'].iloc[-1]: continue
            h4_trend = 1 if df_h4['close'].iloc[-1]>df_h4['ema_long'].iloc[-1] else -1
            adjust_strategy_weights()
            total_score = 0
            signals = {}
            for name,strat in STRATEGIES:
                trend,signal=strat(df_m5)
                total_score += signal*STRATEGY_WEIGHTS[name]
                signals[name]=signal
            last_price=df_m5['close'].iloc[-1]
            atr=df_m5['atr'].iloc[-1]
            units=int((equity*RISK_PER_TRADE)/atr)

            if total_score>=1.0 and h4_trend>0 and can_trade(pair):
                strategy=max(signals,key=lambda k: abs(signals[k]*STRATEGY_WEIGHTS[k]))
                rr=REWARD_RISK[strategy]
                stop_loss=last_price-atr
                take_profit=last_price+atr*rr
                order_resp=place_order(pair,units,stop_loss,take_profit,strategy)
                trade_id=order_resp['orderFillTransaction']['tradeOpened']['tradeID']
                open_trades[pair]={"id":trade_id,"units":units,"stop_loss":stop_loss,"take_profit":take_profit,"strategy":strategy}
                log_trade(pair,units,"Weighted Buy",total_score,last_price)

            elif total_score<=-1.0 and h4_trend<0 and can_trade(pair):
                strategy=max(signals,key=lambda k: abs(signals[k]*STRATEGY_WEIGHTS[k]))
                rr=REWARD_RISK[strategy]
                units=-units
                stop_loss=last_price+atr
                take_profit=last_price-atr*rr
                order_resp=place_order(pair,units,stop_loss,take_profit,strategy)
                trade_id=order_resp['orderFillTransaction']['tradeOpened']['tradeID']
                open_trades[pair]={"id":trade_id,"units":units,"stop_loss":stop_loss,"take_profit":take_profit,"strategy":strategy}
                log_trade(pair,units,"Weighted Sell",total_score,last_price)

            # Trailing stops
            for trade_pair, trade in open_trades.items():
                current_price=df_m5['close'].iloc[-1]
                if trade['units']>0: new_sl=max(trade['stop_loss'],current_price-TRAIL_MULT*atr)
                else: new_sl=min(trade['stop_loss'],current_price+TRAIL_MULT*atr)
                if new_sl!=trade['stop_loss']:
                    update_stop_loss(trade['id'],new_sl,trade_pair)
                    trade['stop_loss']=new_sl
                profit=abs(current_price-trade['stop_loss'])
                update_strategy_perf(trade['strategy'],profit/atr)

        time.sleep(60*5)
