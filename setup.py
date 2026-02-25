# generate_full_forex_bot.py
import os

# -------------------------------
# Folder structure
# -------------------------------
folder = "forex_portfolio_bot"
templates_folder = os.path.join(folder, "templates")
os.makedirs(templates_folder, exist_ok=True)

# -------------------------------
# 1. requirements.txt
# -------------------------------
requirements = """requests
websocket-client
flask
flask-socketio
eventlet
pandas
numpy
matplotlib
"""

with open(os.path.join(folder, "requirements.txt"), "w") as f:
    f.write(requirements)

# -------------------------------
# 2. dashboard_live.html
# -------------------------------
dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>Live Portfolio Dashboard</title>
    <script src="https://cdn.socket.io/4.6.1/socket.io.min.js"></script>
    <style>
        body { font-family: Arial; background: #f7f7f7; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h2>Live Portfolio Dashboard</h2>
    <p><b>Equity:</b> <span id="balance">0</span></p>
    <p id="alert_msg" style="font-weight:bold;"></p>
    <h3>Open Trades</h3>
    <table id="trades_table">
        <tr><th>Pair</th><th>Units</th><th>Entry</th><th>Trailing Stop</th><th>P/L</th></tr>
    </table>

    <h3>Equity Curve</h3>
    <img id="equity_plot" src="" width="600">

<script>
const socket = io();

socket.on("update", function(data){
    document.getElementById("balance").innerText = data.balance.toFixed(2);

    // Update alert
    document.getElementById("alert_msg").innerText = data.alert || "";
    document.getElementById("alert_msg").style.color = data.alert.includes("High") ? "orange" : "red";

    // Update trades table
    const table = document.getElementById("trades_table");
    table.innerHTML = "<tr><th>Pair</th><th>Units</th><th>Entry</th><th>Trailing Stop</th><th>P/L</th></tr>";
    data.trades.forEach(t=>{
        const row = table.insertRow();
        row.insertCell(0).innerText = t.pair;
        row.insertCell(1).innerText = t.units;
        row.insertCell(2).innerText = t.entry;
        row.insertCell(3).innerText = t.stop || "-";
        row.insertCell(4).innerText = t.pl.toFixed(2);
        row.style.color = t.pl >=0 ? "green" : "red";
    });

    // Update equity plot
    fetch("/equity_plot.png").then(res => res.blob()).then(blob=>{
        const url = URL.createObjectURL(blob);
        document.getElementById("equity_plot").src = url;
    });
});
</script>
</body>
</html>
"""

with open(os.path.join(templates_folder, "dashboard_live.html"), "w") as f:
    f.write(dashboard_html)

# -------------------------------
# 3. professional_portfolio_bot.py
# -------------------------------
bot_code = """# professional_portfolio_bot.py
import websocket, requests, json, threading, pandas as pd, numpy as np, csv, io
from collections import deque
from datetime import datetime
from flask import Flask, render_template
from flask_socketio import SocketIO
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG - Replace with your OANDA API credentials
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
        self.socketio = None

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

        if daily_dd >= MAX_DAILY_LOSS:
            print("Daily loss limit reached. Trading paused.")
            return True
        if weekly_dd >= MAX_WEEKLY_LOSS:
            print("Weekly loss limit reached. Trading paused.")
            return True
        return False

    def calculate_unrealized_pl(self, pair):
        trade = self.trade_states[pair]
        if not trade["in_position"]: return 0
        price = self.price_buffers[pair][-1] if self.price_buffers[pair] else trade["entry"]
        units = trade["units"]
        return (price - trade["entry"])*units if units>0 else (trade["entry"]-price)*abs(units)

    def emit_dashboard_update(self):
        if self.socketio:
            trades_data = []
            total_open_risk = 0
            for pair, state in self.trade_states.items():
                if state["in_position"]:
                    unrealized_pl = self.calculate_unrealized_pl(pair)
                    trades_data.append({
                        "pair": pair,
                        "units": state["units"],
                        "entry": state["entry"],
                        "stop": state.get("stop_loss"),
                        "pl": unrealized_pl
                    })
                    total_open_risk += state["risk"]
            balance = self.get_balance()
            alert = ""
            if total_open_risk > MAX_TOTAL_RISK*0.5:
                alert = "High open risk!"
            daily_dd = (self.daily_start_balance - balance)/self.daily_start_balance
            if daily_dd >= MAX_DAILY_LOSS*0.7:
                alert += " Approaching daily loss limit!"
            self.socketio.emit("update", {"trades": trades_data, "balance": balance, "alert": alert})

# -------------------------------
# Flask-SocketIO Dashboard
# -------------------------------
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')
engine = PortfolioEngine()
engine.socketio = socketio

@app.route("/")
def index():
    return render_template("dashboard_live.html")

@app.route("/equity_plot.png")
def equity_plot():
    try:
        df = pd.read_csv(EQUITY_FILE)
        img = io.BytesIO()
        plt.figure(figsize=(6,3))
        plt.plot(df["timestamp"], df["balance"])
        plt.title("Equity Curve")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return img.getvalue(), 200, {'Content-Type':'image/png'}
    except:
        return "", 404

# -------------------------------
# WebSocket placeholders (price & transaction streams)
# -------------------------------
# You can implement OANDA streaming here and call engine.emit_dashboard_update() after updates

# -------------------------------
# Run Dashboard
# -------------------------------
threading.Thread(target=lambda: socketio.run(app, host="0.0.0.0", port=5000), daemon=True).start()
print("Dashboard running on port 5000")
"""

with open(os.path.join(folder, "professional_portfolio_bot.py"), "w") as f:
    f.write(bot_code)

# -------------------------------
# 4. Empty CSV logs
# -------------------------------
open(os.path.join(folder, "trade_log.csv"), "w").close()
open(os.path.join(folder, "equity_curve.csv"), "w").close()

print(f"Folder '{folder}' with full bot code generated successfully!")
print("Now you can zip it manually:")
print(f"zip -r {folder}.zip {folder}")
