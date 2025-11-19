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
import json

# --- Configuration ---
DATABASE_URL = "sqlite:///./crypto_data.db"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 96  # 24 hours of 15m candles
BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
SIGMA = 4
THRESHOLD = 0.0004

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
    
    # Check and backfill data
    backfill_data()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_binance_data(start_time=None, end_time=None, limit=LIMIT):
    """Fetches data from Binance API."""
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is None:
        start_time = end_time - timedelta(hours=24)
    
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": limit
    }
    
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def download_and_import_binance_vision_data(date_str):
    """Downloads and imports data from Binance Vision for a specific date."""
    # URL format: https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2025-11-17.zip
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
            # There should be one CSV file inside
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                # Read CSV
                # Format: open_time, open, high, low, close, volume, close_time, quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
                # Note: Binance Vision CSVs usually don't have headers, but the user example shows headers.
                # We'll check the first line.
                content = f.read().decode('utf-8')
                reader = csv.reader(io.StringIO(content))
                header = next(reader)
                
                # Check if first row is header
                if header[0] == 'open_time':
                    pass # Header exists, proceed
                else:
                    # Reset reader if no header (re-create StringIO is easiest)
                    reader = csv.reader(io.StringIO(content))
                
                db = SessionLocal()
                count = 0
                try:
                    for row in reader:
                        if not row: continue
                        open_time_ms = int(row[0])
                        open_time = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
                        
                        # Check if exists
                        existing = db.query(Kline).filter(Kline.open_time == open_time.replace(tzinfo=None)).first()
                        if not existing:
                            kline = Kline(
                                open_time=open_time.replace(tzinfo=None),
                                open_price=float(row[1]),
                                high_price=float(row[2]),
                                low_price=float(row[3]),
                                close_price=float(row[4]),
                                volume=float(row[5]),
                                quote_vol=float(row[7]),
                                taker_buy_vol=float(row[9]),
                                taker_buy_quote_vol=float(row[10])
                            )
                            db.add(kline)
                            count += 1
                    db.commit()
                    print(f"Imported {count} klines from {date_str}")
                    return True
                except Exception as e:
                    print(f"Error importing CSV data: {e}")
                    db.rollback()
                    return False
                finally:
                    db.close()
                    
    except Exception as e:
        print(f"Error downloading/processing {url}: {e}")
        return False

def backfill_data():
    """Checks for missing data and backfills from Binance Vision or API."""
    print("Checking data completeness...")
    db = SessionLocal()
    try:
        # Check last 7 days
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=7)
        
        current_date = start_date
        while current_date < end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Check if we have data for this day (simple check: at least one record)
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            count = db.query(Kline).filter(
                Kline.open_time >= day_start,
                Kline.open_time <= day_end
            ).count()
            
            if count < 90: # Expecting 96 candles per day
                print(f"Data missing for {date_str} (found {count}). Attempting backfill...")
                success = download_and_import_binance_vision_data(date_str)
                if not success:
                    print(f"Binance Vision failed for {date_str}. Trying API fallback...")
                    # Fallback to API for this day
                    # Note: API limit is 1000 candles, 15m * 96 = 1 day is fine.
                    start_ts = int(day_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    end_ts = int(day_end.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    
                    klines_data = fetch_binance_data(
                        start_time=day_start.replace(tzinfo=timezone.utc),
                        end_time=day_end.replace(tzinfo=timezone.utc),
                        limit=1000
                    )
                    if klines_data:
                        save_klines_to_db(klines_data)
            
            current_date += timedelta(days=1)
            
        # Always update latest data from API
        update_database()
        
    finally:
        db.close()

def ensure_data_availability(start_date_str, end_date_str):
    """Ensures data exists for the requested range, backfilling if necessary."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    
    # Prevent fetching future data beyond tomorrow (to account for timezone diffs)
    max_allowed_date = datetime.now(timezone.utc).date() + timedelta(days=1)
    if end_date > max_allowed_date:
        end_date = max_allowed_date
    
    db = SessionLocal()
    try:
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            count = db.query(Kline).filter(
                Kline.open_time >= day_start,
                Kline.open_time <= day_end
            ).count()
            
            if count < 90:
                print(f"Data missing for {date_str} (found {count}). Backfilling...")
                success = download_and_import_binance_vision_data(date_str)
                if not success:
                    print(f"Binance Vision failed for {date_str}. Trying API fallback...")
                    start_ts = int(day_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    end_ts = int(day_end.replace(tzinfo=timezone.utc).timestamp() * 1000)
                    
                    klines_data = fetch_binance_data(
                        start_time=day_start.replace(tzinfo=timezone.utc),
                        end_time=day_end.replace(tzinfo=timezone.utc),
                        limit=1000
                    )
                    if klines_data:
                        save_klines_to_db(klines_data)
            
            current_date += timedelta(days=1)
    finally:
        db.close()

def save_klines_to_db(klines_data):
    """Helper to save API klines to DB."""
    if not klines_data: return
    db = SessionLocal()
    try:
        for k in klines_data:
            open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            existing = db.query(Kline).filter(Kline.open_time == open_time.replace(tzinfo=None)).first()
            if not existing:
                kline = Kline(
                    open_time=open_time.replace(tzinfo=None),
                    open_price=float(k[1]),
                    high_price=float(k[2]),
                    low_price=float(k[3]),
                    close_price=float(k[4]),
                    volume=float(k[5]),
                    quote_vol=float(k[7]),
                    taker_buy_vol=float(k[9]),
                    taker_buy_quote_vol=float(k[10])
                )
                db.add(kline)
        db.commit()
    except Exception as e:
        print(f"Error saving klines: {e}")
        db.rollback()
    finally:
        db.close()

def update_database():
    """Fetches latest data and updates SQLite."""
    # Fetch last 24h to be safe
    klines_data = fetch_binance_data()
    save_klines_to_db(klines_data)
    return True

def get_data_for_prediction():
    """Retrieves data from DB for prediction."""
    db = SessionLocal()
    try:
        # Get enough data for rolling windows (96 + buffer)
        # We need at least ~100 bars for indicators
        klines = db.query(Kline).order_by(Kline.open_time.desc()).limit(200).all()
        if not klines:
            return None
            
        # Convert to DataFrame
        data = []
        for k in reversed(klines): # Sort ascending for pandas
            data.append({
                'time': k.open_time,
                'open': k.open_price,
                'high': k.high_price,
                'low': k.low_price,
                'close': k.close_price,
                'volume': k.volume,
                'quote_vol': k.quote_vol,
                'taker_buy_vol': k.taker_buy_vol,
                'taker_buy_quote_vol': k.taker_buy_quote_vol
            })
            
        df = pd.DataFrame(data)
        
        # Calculate derived columns needed for predictor
        # maker_sell_vol = volume - taker_buy_vol
        # We map to what predictor expects. 
        # Predictor expects 'volume' and 'taker_buy_vol' to calculate 'volume_delta'
        # if 'ask_vol'/'bid_vol' are missing.
        
        return df
    finally:
        db.close()

# --- Routes ---

def create_target_labels(df, sigma=SIGMA, threshold=THRESHOLD):
    """Creates target labels using Gaussian smoothing (same as train.py)."""
    df_target = df.copy()
    df_target['smoothed_close'] = gaussian_filter1d(df_target['close'], sigma=sigma)
    df_target['smooth_slope'] = np.diff(np.log(df_target['smoothed_close']), prepend=np.nan)
    
    conditions = [
        df_target['smooth_slope'] > threshold,
        df_target['smooth_slope'] < -threshold
    ]
    choices = [2, 0]
    df_target['target'] = np.select(conditions, choices, default=1)
    return df_target

def generate_plot(df, prediction, timestamp, start_date=None, end_date=None):
    """Generates a plot with prediction highlight."""
    plt.figure(figsize=(12, 6))
    
    # Filter by date range
    if start_date:
        df = df[df['time'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['time'] <= pd.to_datetime(end_date)]
        
    # If no specific range, default to last 3 days
    if start_date is None and end_date is None:
        cutoff_date = df['time'].max() - timedelta(days=3)
        df = df[df['time'] > cutoff_date]
        
    plot_df = df.copy()
    
    # Calculate ground truth labels for history
    plot_df = create_target_labels(plot_df)
    
    plt.plot(plot_df['time'], plot_df['close'], label='Close Price', color='black', linewidth=1)
    # Removed smoothed trend line as requested
    # plt.plot(plot_df['time'], plot_df['smoothed_close'], label='Smoothed Trend', color='blue', linestyle='--', alpha=0.7)
    
    # Highlight historical regions
    # 0: DOWN (Red), 1: SIDEWAYS (Gray), 2: UP (Green)
    # Increased intensity colors
    colors = {0: '#ffcccc', 1: '#e0e0e0', 2: '#ccffcc'}
    
    # Iterate and highlight
    # We group consecutive identical targets to minimize span calls
    plot_df['group'] = (plot_df['target'] != plot_df['target'].shift()).cumsum()
    
    for _, group in plot_df.groupby('group'):
        start_time = group['time'].iloc[0]
        end_time = group['time'].iloc[-1] + timedelta(minutes=15) # Extend to cover the bar
        target = group['target'].iloc[0]
        plt.axvspan(start_time, end_time, color=colors.get(target, 'white'), alpha=0.8) # Increased alpha

    # Highlight prediction (Forecast)
    forecast_colors = {0: 'red', 1: 'gray', 2: 'green'}
    color = forecast_colors.get(prediction, 'gray')
    
    last_time = plot_df['time'].iloc[-1]
    next_time = last_time + timedelta(minutes=15)
    
    plt.axvspan(last_time, next_time, color=color, alpha=0.8, label='Forecast')
    
    plt.title(f"BTCUSDT Price & Forecast: {['DOWN', 'SIDEWAYS', 'UP'][prediction]}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = "static/prediction_plot.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Removed evaluate_model to avoid confusion about training vs inference accuracy.

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
                #result { margin-top: 20px; font-weight: bold; }
                img { max-width: 100%; height: auto; margin-top: 20px; }
                .controls { margin-bottom: 15px; display: flex; gap: 10px; align-items: center; }
                input[type="date"] { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>Bitcoin Price Direction Predictor</h1>
            <p>Predicts the direction of the NEXT 15-minute candle.</p>
            
            <div class="card">
                <div class="controls">
                    <label>From:</label>
                    <input type="date" id="start-date">
                    <label>To:</label>
                    <input type="date" id="end-date">
                    <button onclick="predict()">Update Data & Predict</button>
                </div>
                <div id="result"></div>
                <div id="plot-container"></div>
            </div>

            <script>
                // Set default dates (last 3 days)
                const today = new Date();
                const threeDaysAgo = new Date();
                threeDaysAgo.setDate(today.getDate() - 3);
                
                document.getElementById('end-date').valueAsDate = today;
                document.getElementById('start-date').valueAsDate = threeDaysAgo;

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
                                <p>Model Accuracy: <strong>${data.accuracy}</strong></p>
                                <p>Probabilities:</p>
                                <ul>
                                    <li>DOWN: ${(data.probabilities[0]*100).toFixed(1)}%</li>
                                    <li>SIDEWAYS: ${(data.probabilities[1]*100).toFixed(1)}%</li>
                                    <li>UP: ${(data.probabilities[2]*100).toFixed(1)}%</li>
                                </ul>
                            `;
                            
                            // Load plot
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
async def get_plot():
    return FileResponse("static/prediction_plot.png")

class PredictionRequest(BaseModel):
    start_date: str = None
    end_date: str = None

@app.post("/predict")
async def make_prediction(request: PredictionRequest, background_tasks: BackgroundTasks):
    # 1. Validate Date Range
    if request.start_date and request.end_date:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        if (end - start).days > 14:
            return {"error": "Date range cannot exceed 2 weeks"}
        
        # Ensure data availability for the requested range
        ensure_data_availability(request.start_date, request.end_date)
    
    # 2. Update Data (Latest)
    success = update_database()
    if not success:
        return {"error": "Failed to fetch data from Binance"}
    
    # 3. Get Data
    db = SessionLocal()
    try:
        query = db.query(Kline).order_by(Kline.open_time.desc())
        
        if request.start_date:
            # Add buffer for indicators (e.g. 2 days before start date)
            start_buffer = datetime.strptime(request.start_date, "%Y-%m-%d") - timedelta(days=2)
            query = query.filter(Kline.open_time >= start_buffer)
            
        if request.end_date:
            # End date inclusive (end of day)
            end_dt = datetime.strptime(request.end_date, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(Kline.open_time < end_dt)
            
        # If no dates, limit to recent history
        if not request.start_date:
            query = query.limit(2000)
            
        klines = query.all()
        
        if not klines:
            return {"error": "Not enough data in database"}
        
        data = []
        for k in reversed(klines):
            data.append({
                'time': k.open_time,
                'open': k.open_price,
                'high': k.high_price,
                'low': k.low_price,
                'close': k.close_price,
                'volume': k.volume,
                'quote_vol': k.quote_vol,
                'taker_buy_vol': k.taker_buy_vol,
                'taker_buy_quote_vol': k.taker_buy_quote_vol
            })
        df = pd.DataFrame(data)
    finally:
        db.close()

    if df is None or len(df) < 50:
        return {"error": "Not enough data in database"}
    
    # 4. Predict
    if predictor is None:
        return {"error": "Model not loaded"}
        
    try:
        pred_class, pred_proba, timestamp = predictor.predict(df)
        
        if pred_class is None:
             return {"error": "Prediction failed (insufficient data for features)"}

        # Generate Plot
        generate_plot(df, int(pred_class), timestamp, request.start_date, request.end_date)

        return {
            "timestamp": str(timestamp),
            "prediction": int(pred_class),
            "probabilities": [float(p) for p in pred_proba],
            "accuracy": "N/A (Run training to see)"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11111)