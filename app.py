import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import zipfile
import io
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d
from predict import Predictor
from simple_backtest import run_simple_backtest
import json

# --- Configuration ---
DATABASE_URL = "sqlite:///./crypto_data.db"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 96
BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
SIGMA = 4 # This will now be used to calculate the span for the EMA
THRESHOLD = 0.0004
DEBUG_MODE = False # Master debug switch

# --- Database Setup ---
Base = declarative_base()

class Kline(Base):
    __tablename__ = "klines"
    id = Column(Integer, primary_key=True, index=True)
    open_time = Column(DateTime, unique=True, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    quote_vol = Column(Float)
    taker_buy_vol = Column(Float)
    taker_buy_quote_vol = Column(Float)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(title="Bitcoin Price Direction Predictor")
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = Predictor()
        print("Predictor initialized.")
    except Exception as e:
        print(f"Warning: Predictor failed to initialize (models might be missing): {e}")
    backfill_data()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_binance_data(start_time=None, end_time=None, limit=LIMIT):
    if end_time is None: end_time = datetime.now(timezone.utc)
    if start_time is None: start_time = end_time - timedelta(hours=24)
    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": int(start_time.timestamp() * 1000), "endTime": int(end_time.timestamp() * 1000), "limit": limit}
    try:
        r = requests.get("https://api.binance.com/api/v3/klines", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def download_and_import_binance_vision_data(date_str):
    filename = f"{SYMBOL}-{INTERVAL}-{date_str}.zip"
    url = f"{BINANCE_VISION_BASE_URL}/{SYMBOL}/{INTERVAL}/{filename}"
    print(f"Attempting to download {url}...")
    try:
        r = requests.get(url)
        if r.status_code == 404:
            print(f"Data not found for {date_str} on Binance Vision.")
            return False
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                content = f.read().decode('utf-8')
                reader = csv.reader(io.StringIO(content))
                db = SessionLocal()
                count = 0
                try:
                    for row in reader:
                        if not row: continue
                        open_time = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
                        if not db.query(Kline).filter(Kline.open_time == open_time.replace(tzinfo=None)).first():
                            db.add(Kline(open_time=open_time.replace(tzinfo=None), open_price=float(row[1]), high_price=float(row[2]), low_price=float(row[3]), close_price=float(row[4]), volume=float(row[5]), quote_vol=float(row[7]), taker_buy_vol=float(row[9]), taker_buy_quote_vol=float(row[10])))
                            count += 1
                    db.commit()
                    print(f"Imported {count} klines from {date_str}")
                    return True
                except Exception as e:
                    print(f"Error importing CSV data: {e}"); db.rollback(); return False
                finally:
                    db.close()
    except Exception as e:
        print(f"Error downloading/processing {url}: {e}")
        return False

def backfill_data():
    print("Checking data completeness...")
    db = SessionLocal()
    try:
        end_date = datetime.now(timezone.utc).date()
        for i in range(7):
            current_date = end_date - timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            if db.query(Kline).filter(Kline.open_time >= day_start, Kline.open_time <= day_end).count() < 90:
                print(f"Data missing for {date_str}. Attempting backfill...")
                if not download_and_import_binance_vision_data(date_str):
                    print(f"Binance Vision failed for {date_str}. API fallback not implemented in this simplified flow.")
        update_database()
    finally:
        db.close()

def save_klines_to_db(klines_data):
    if not klines_data: return
    db = SessionLocal()
    try:
        for k in klines_data:
            open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            if not db.query(Kline).filter(Kline.open_time == open_time.replace(tzinfo=None)).first():
                db.add(Kline(open_time=open_time.replace(tzinfo=None), open_price=float(k[1]), high_price=float(k[2]), low_price=float(k[3]), close_price=float(k[4]), volume=float(k[5]), quote_vol=float(k[7]), taker_buy_vol=float(k[9]), taker_buy_quote_vol=float(k[10])))
        db.commit()
    except Exception as e:
        print(f"Error saving klines: {e}"); db.rollback()
    finally:
        db.close()

def update_database():
    save_klines_to_db(fetch_binance_data())
    return True

def create_target_labels(df, sigma=SIGMA, threshold=THRESHOLD):
    df_target = df.copy()
    # FIX: Use a non-repainting Exponential Moving Average (EMA) instead of a Gaussian filter.
    # A span of roughly 4*sigma gives a similar smoothing effect.
    span = sigma * 4
    df_target['smoothed_close'] = df_target['close'].ewm(span=span, adjust=False).mean()
    df_target['smooth_slope'] = np.diff(np.log(df_target['smoothed_close']), prepend=np.nan)
    conditions = [df_target['smooth_slope'] > threshold, df_target['smooth_slope'] < -threshold]
    choices = [2, 0]
    df_target['target'] = np.select(conditions, choices, default=1)
    return df_target

def generate_plot(df, prediction, timestamp, start_date=None, end_date=None):
    plt.figure(figsize=(12, 6))
    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df['time'] >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df['time'] <= pd.to_datetime(end_date)]
    
    if plot_df.empty:
        plt.title("No data available for the selected date range")
        plt.grid(True, alpha=0.3)
        os.makedirs("static", exist_ok=True)
        plot_path = "static/prediction_plot.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path, pd.DataFrame(columns=df.columns.tolist() + ['target'])

    plot_df = create_target_labels(plot_df)
    
    plt.plot(plot_df['time'], plot_df['close'], label='Close Price', color='black', linewidth=1)
    colors = {0: '#ffcccc', 1: '#e0e0e0', 2: '#ccffcc'}
    plot_df['group'] = (plot_df['target'] != plot_df['target'].shift()).cumsum()
    for _, group in plot_df.groupby('group'):
        plt.axvspan(group['time'].iloc[0], group['time'].iloc[-1] + timedelta(minutes=15), color=colors.get(group['target'].iloc[0], 'white'), alpha=0.8)
    
    forecast_colors = {0: 'red', 1: 'gray', 2: 'green'}
    last_time = plot_df['time'].iloc[-1]
    plt.axvspan(last_time, last_time + timedelta(minutes=15), color=forecast_colors.get(prediction, 'gray'), alpha=0.8, label='Forecast')
    plt.title(f"BTCUSDT Price & Forecast: {['DOWN', 'SIDEWAYS', 'UP'][prediction]}")
    plt.legend(); plt.grid(True, alpha=0.3)
    os.makedirs("static", exist_ok=True)
    plot_path = "static/prediction_plot.png"
    plt.savefig(plot_path); plt.close()
    return plot_path, plot_df

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>BTC Predictor</title>
            <style>
                body { font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
                .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-top: 20px; }
                button { padding: 10px 20px; font-size: 16px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 4px; }
                button:hover { background-color: #0056b3; }
                #result { margin-top: 20px; }
                #backtest-details { display: none; margin-top: 15px; }
                img { max-width: 100%; height: auto; margin-top: 20px; }
                .controls { margin-bottom: 15px; display: flex; gap: 10px; align-items: center; }
                input[type="datetime-local"] { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>Bitcoin Price Direction Predictor</h1>
            <p>Predicts the direction of the NEXT 15-minute candle.</p>
            
            <div class="card">
                <div class="controls">
                    <label>From:</label>
                    <input type="datetime-local" id="start-date">
                    <label>To:</label>
                    <input type="datetime-local" id="end-date">
                    <button onclick="predict()">Update Data & Predict</button>
                </div>
                <div id="result"></div>
                <div id="plot-container"></div>
            </div>

            <script>
                function toLocalISOString(date) {
                    const pad = (num) => num.toString().padStart(2, '0');
                    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
                }

                const now = new Date();
                const threeDaysAgo = new Date(now);
                threeDaysAgo.setDate(now.getDate() - 3);
                
                document.getElementById('end-date').value = toLocalISOString(now);
                document.getElementById('start-date').value = toLocalISOString(threeDaysAgo);

                function toggleBacktest() {
                    const details = document.getElementById('backtest-details');
                    const button = document.getElementById('toggle-btn');
                    if (details.style.display === 'none') {
                        details.style.display = 'block';
                        button.textContent = 'Hide Backtest';
                    } else {
                        details.style.display = 'none';
                        button.textContent = 'Show Backtest';
                    }
                }

                async function predict() {
                    const resultDiv = document.getElementById('result');
                    const plotDiv = document.getElementById('plot-container');
                    const startDate = document.getElementById('start-date').value;
                    const endDate = document.getElementById('end-date').value;
                    
                    resultDiv.innerHTML = "Fetching data and calculating...";
                    plotDiv.innerHTML = "";
              
                    try {
                        const response = await fetch('/predict', { 
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ start_date: startDate, end_date: endDate })
                        });
                        const data = await response.json();
                        
                        if (data.error) {
                            resultDiv.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
                        } else {
                            let color = 'gray';
                            if (data.prediction === 2) color = 'green';
                            if (data.prediction === 0) color = 'red';
                            
                            const labels = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'};
                            
                            resultDiv.innerHTML = `
                                <h3>Prediction: <span style="color: ${color}">${labels[data.prediction]}</span></h3>
                                <p>Time: ${data.timestamp}</p>
                                <p>Probabilities:</p>
                                <ul>
                                    <li>DOWN: ${(data.probabilities[0]*100).toFixed(1)}%</li>
                                    <li>SIDEWAYS: ${(data.probabilities[1]*100).toFixed(1)}%</li>
                                    <li>UP: ${(data.probabilities[2]*100).toFixed(1)}%</li>
                                </ul>
                                <hr>
                                <button id="toggle-btn" onclick="toggleBacktest()">Show Backtest</button>
                                <div id="backtest-details">
                                    <h4>Backtest Result (Selected Period)</h4>
                                    <p>Total Return: <strong>${data.backtest_return.toFixed(2)}%</strong></p>
                                    <p>Trades Executed: ${data.backtest_trades}</p>
                                    
                                    <div style="max-height: 300px; overflow-y: auto; margin-top: 10px; border: 1px solid #eee;">
                                        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                                            <thead style="position: sticky; top: 0; background: white;">
                                                <tr>
                                                    <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Entry Time</th>
                                                    <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Exit Time</th>
                                                    <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Entry</th>
                                                    <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Exit</th>
                                                    <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Dir</th>
                                                    <th style="text-align: right; padding: 5px; border-bottom: 1px solid #ddd;">Return %</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${data.trades.map(t => `
                                                    <tr>
                                                        <td style="padding: 5px; border-bottom: 1px solid #eee;">${t.entry_time}</td>
                                                        <td style="padding: 5px; border-bottom: 1px solid #eee;">${t.exit_time}</td>
                                                        <td style="padding: 5px; border-bottom: 1px solid #eee;">${t.entry_price.toFixed(2)}</td>
                                                        <td style="padding: 5px; border-bottom: 1px solid #eee;">${t.exit_price.toFixed(2)}</td>
                                                        <td style="padding: 5px; border-bottom: 1px solid #eee; color: ${t.direction === 'LONG' ? 'green' : 'red'}">${t.direction}</td>
                                                        <td style="padding: 5px; border-bottom: 1px solid #eee; text-align: right; color: ${t.return_pct >= 0 ? 'green' : 'red'}">${t.return_pct.toFixed(2)}%</td>
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            `;
                            
                            plotDiv.innerHTML = `<img src="/plot?t=${new Date().getTime()}" alt="Prediction Plot">`;
                        }
                    } catch (e) {
                        resultDiv.innerHTML = `<span style="color: red;">Network Error: ${e}</span>`;
                    }
                }
            </script>
        </body>
    </html>
    """

@app.get("/plot")
async def get_plot(): return FileResponse("static/prediction_plot.png")

class PredictionRequest(BaseModel):
    start_date: str = None
    end_date: str = None

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    if not predictor: return {"error": "Model not loaded"}
    
    db = SessionLocal()
    try:
        query = db.query(Kline).order_by(Kline.open_time.asc())
        if request.start_date:
            # Use fromisoformat for datetime-local strings
            buffer_start_date = datetime.fromisoformat(request.start_date) - timedelta(days=2)
            query = query.filter(Kline.open_time >= buffer_start_date)
        if request.end_date:
            end_dt = datetime.fromisoformat(request.end_date)
            query = query.filter(Kline.open_time <= end_dt)
        if not request.start_date:
            query = query.filter(Kline.open_time >= (datetime.now(timezone.utc) - timedelta(days=5)))
        
        klines = query.all()
        if not klines: return {"error": "Not enough data in database"}
        
        df = pd.DataFrame([
            {'time':k.open_time, 'open':k.open_price, 'high':k.high_price, 'low':k.low_price, 'close':k.close_price, 'volume': k.volume, 'quote_vol': k.quote_vol, 'taker_buy_vol': k.taker_buy_vol, 'taker_buy_quote_vol': k.taker_buy_quote_vol} 
            for k in klines
        ])
    finally:
        db.close()

    pred_class, pred_proba, timestamp = predictor.predict(df)
    plot_path, plot_df = generate_plot(df, int(pred_class), timestamp, request.start_date, request.end_date)

    backtest_df = plot_df.rename(columns={'target': 'prediction'})
    
    backtest_start_date = pd.to_datetime(request.start_date) if request.start_date else df['time'].max() - timedelta(days=3)

    backtest_return, backtest_trades, trades, combined_df = run_simple_backtest(
        backtest_df, start_date=backtest_start_date, debug=DEBUG_MODE
    )

    trades_serialized = [{"entry_time":t["entry_time"].strftime("%Y-%m-%d %H:%M"),"exit_time":t["exit_time"].strftime("%Y-%m-%d %H:%M"),"entry_price":t["entry_price"],"exit_price":t["exit_price"],"direction":t["direction"],"return_pct":t["return_pct"]} for t in trades]

    if DEBUG_MODE:
        print("\n\n" + "="*80)
        print("DEBUGGING REPORT")
        print("="*80)
        log_df = combined_df[combined_df.index >= backtest_start_date]
        print("\n--- CHART SIGNALS (Used for Backtest) ---")
        print(log_df[['time', 'prediction']].rename(columns={'prediction': 'chart_signal'}).to_string())
        print("\n--- FINAL TRADES TABLE (Sent to Webpage) ---")
        print(json.dumps(trades_serialized, indent=2))
        print("="*80 + "\n\n")

    return {
        "timestamp": str(timestamp), "prediction": int(pred_class), "probabilities": [float(p) for p in pred_proba],
        "backtest_return": backtest_return, "backtest_trades": backtest_trades, "trades": trades_serialized
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11111)