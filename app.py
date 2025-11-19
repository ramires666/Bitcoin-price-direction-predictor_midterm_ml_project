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
from predict import Predictor
from simple_backtest import run_simple_backtest
import json

# --- Configuration ---
DATABASE_URL = "sqlite:///./crypto_data.db"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 1000 # Increased limit for robust fetching
BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
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
    # Initial backfill on startup for recent data
    backfill_data_on_startup()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_binance_data(start_time, end_time, limit=LIMIT):
    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": int(start_time.timestamp() * 1000), "endTime": int(end_time.timestamp() * 1000), "limit": limit}
    try:
        r = requests.get("https://api.binance.com/api/v3/klines", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching data from Binance API: {e}")
        return []

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
        print(f"Error saving klines to DB: {e}"); db.rollback()
    finally:
        db.close()

def download_and_import_binance_vision_data(date_str):
    filename = f"{SYMBOL}-{INTERVAL}-{date_str}.zip"
    url = f"{BINANCE_VISION_BASE_URL}/{SYMBOL}/{INTERVAL}/{filename}"
    print(f"Attempting to download from Binance Vision: {url}...")
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
                klines_to_add = []
                for row in reader:
                    if not row: continue
                    open_time = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).replace(tzinfo=None)
                    klines_to_add.append(Kline(open_time=open_time, open_price=float(row[1]), high_price=float(row[2]), low_price=float(row[3]), close_price=float(row[4]), volume=float(row[5]), quote_vol=float(row[7]), taker_buy_vol=float(row[9]), taker_buy_quote_vol=float(row[10])))
                save_klines_to_db_bulk(klines_to_add)
                print(f"Imported {len(klines_to_add)} klines from {date_str}")
                return True
    except Exception as e:
        print(f"Error downloading/processing {url}: {e}")
        return False

def save_klines_to_db_bulk(klines):
    if not klines: return
    db = SessionLocal()
    try:
        existing_times = {t[0] for t in db.query(Kline.open_time).filter(Kline.open_time.in_([k.open_time for k in klines])).all()}
        new_klines = [k for k in klines if k.open_time not in existing_times]
        if new_klines:
            db.bulk_save_objects(new_klines)
            db.commit()
    except Exception as e:
        print(f"Error bulk saving klines: {e}"); db.rollback()
    finally:
        db.close()

def ensure_data_availability(start_dt, end_dt):
    print(f"Ensuring data availability from {start_dt.date()} to {end_dt.date()}...")
    db = SessionLocal()
    try:
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            date_str = current_date.strftime("%Y-%m-%d")
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            count = db.query(Kline).filter(Kline.open_time >= day_start, Kline.open_time <= day_end).count()
            # Expect 96 candles for a full day (24 * 4)
            if count < 96 and current_date < datetime.now(timezone.utc).date():
                print(f"Data for {date_str} is incomplete (found {count}). Attempting backfill...")
                if not download_and_import_binance_vision_data(date_str):
                    print(f"Binance Vision failed for {date_str}. Falling back to API for the whole day.")
                    api_data = fetch_binance_data(day_start, day_end)
                    save_klines_to_db(api_data)
            current_date += timedelta(days=1)
    finally:
        db.close()
    # Always fetch the most recent data
    print("Updating latest data from API...")
    latest_data = fetch_binance_data(datetime.now(timezone.utc) - timedelta(hours=48), datetime.now(timezone.utc))
    save_klines_to_db(latest_data)


def backfill_data_on_startup():
    # Backfill for the last 7 days on startup
    ensure_data_availability(datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc))

def generate_plot(df, start_date=None, end_date=None):
    plt.figure(figsize=(12, 6))
    plot_df = df.copy()
    
    if start_date: plot_df = plot_df[plot_df['time'] >= pd.to_datetime(start_date)]
    if end_date: plot_df = plot_df[plot_df['time'] <= pd.to_datetime(end_date)]
    
    if plot_df.empty:
        plt.title("No data available for the selected date range")
        plt.grid(True, alpha=0.3)
        os.makedirs("static", exist_ok=True)
        plt.savefig("static/prediction_plot.png"); plt.close()
        return "static/prediction_plot.png"

    plt.plot(plot_df['time'], plot_df['close'], label='Close Price', color='black', linewidth=1)
    colors = {0: '#ffcccc', 1: '#e0e0e0', 2: '#ccffcc'}
    
    plot_df['group'] = (plot_df['prediction'] != plot_df['prediction'].shift()).cumsum()
    
    for _, group in plot_df.groupby('group'):
        plt.axvspan(group['time'].iloc[0], group['time'].iloc[-1] + timedelta(minutes=15), color=colors.get(group['prediction'].iloc[0], 'white'), alpha=0.8)
    
    last_prediction = plot_df['prediction'].iloc[-1]
    last_time = plot_df['time'].iloc[-1]
    forecast_colors = {0: 'red', 1: 'gray', 2: 'green'}
    plt.axvspan(last_time, last_time + timedelta(minutes=15), color=forecast_colors.get(last_prediction, 'gray'), alpha=0.8, label='Forecast')
    
    plt.title(f"BTCUSDT Price & Model Signal: {['DOWN', 'SIDEWAYS', 'UP'][int(last_prediction)]}")
    plt.legend(); plt.grid(True, alpha=0.3)
    os.makedirs("static", exist_ok=True)
    plot_path = "static/prediction_plot.png"
    plt.savefig(plot_path); plt.close()
    return plot_path

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
            <p>Backtests the XGBoost model's signals.</p>
            
            <div class="card">
                <div class="controls">
                    <label>From:</label>
                    <input type="datetime-local" id="start-date">
                    <label>To:</label>
                    <input type="datetime-local" id="end-date">
                    <button onclick="predict()">Run Backtest</button>
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
                    
                    resultDiv.innerHTML = "Fetching data and running backtest...";
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
                            resultDiv.innerHTML = `
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
                            
                            plotDiv.innerHTML = `<img src="/plot?t=${new Date().getTime()}" alt="Backtest Plot">`;
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
    
    # --- 1. Ensure data is available for the requested range ---
    start_dt = datetime.fromisoformat(request.start_date) if request.start_date else datetime.now(timezone.utc) - timedelta(days=3)
    end_dt = datetime.fromisoformat(request.end_date) if request.end_date else datetime.now(timezone.utc)
    
    # Add buffer for indicators
    buffer_start_dt = start_dt - timedelta(days=2)
    ensure_data_availability(buffer_start_dt, end_dt)

    # --- 2. Fetch data from DB ---
    db = SessionLocal()
    try:
        query = db.query(Kline).order_by(Kline.open_time.asc()).filter(Kline.open_time >= buffer_start_dt, Kline.open_time <= end_dt)
        klines = query.all()
        if not klines: return {"error": "Not enough data in database for the selected range."}
        
        df = pd.DataFrame([
            {'time':k.open_time, 'open':k.open_price, 'high':k.high_price, 'low':k.low_price, 'close':k.close_price, 'volume': k.volume, 'quote_vol': k.quote_vol, 'taker_buy_vol': k.taker_buy_vol, 'taker_buy_quote_vol': k.taker_buy_quote_vol} 
            for k in klines
        ])
        df = df.set_index('time', drop=False)
    finally:
        db.close()

    # --- 3. Get predictions from the model ---
    predictions_df = predictor.predict_batch(df)
    df_with_preds = df.join(predictions_df[['prediction']], how='inner')

    # --- 4. Generate plot and run backtest ---
    plot_path = generate_plot(df_with_preds, request.start_date, request.end_date)
    
    backtest_return, backtest_trades, trades, combined_df = run_simple_backtest(
        df_with_preds, start_date=start_dt, debug=DEBUG_MODE
    )

    # --- 5. Serialize and respond ---
    trades_serialized = [{"entry_time":t["entry_time"].strftime("%Y-%m-%d %H:%M"),"exit_time":t["exit_time"].strftime("%Y-%m-%d %H:%M"),"entry_price":t["entry_price"],"exit_price":t["exit_price"],"direction":t["direction"],"return_pct":t["return_pct"]} for t in trades]

    return {
        "backtest_return": backtest_return,
        "backtest_trades": backtest_trades,
        "trades": trades_serialized
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11111)