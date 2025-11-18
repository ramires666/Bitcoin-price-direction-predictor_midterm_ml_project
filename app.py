import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from predict import Predictor

# --- Configuration ---
DATABASE_URL = "sqlite:///./crypto_data.db"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 96  # 24 hours of 15m candles

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_binance_data():
    """Fetches last 24h data from Binance."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)
    
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": LIMIT
    }
    
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def update_database():
    """Fetches data and updates SQLite."""
    klines_data = fetch_binance_data()
    if not klines_data:
        return False
        
    db = SessionLocal()
    try:
        for k in klines_data:
            open_time = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            
            # Check if exists
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
        return True
    except Exception as e:
        print(f"Database update error: {e}")
        db.rollback()
        return False
    finally:
        db.close()

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>BTC Predictor</title>
            <style>
                body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-top: 20px; }
                button { padding: 10px 20px; font-size: 16px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 4px; }
                button:hover { background-color: #0056b3; }
                #result { margin-top: 20px; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Bitcoin Price Direction Predictor</h1>
            <p>Predicts the direction of the NEXT 15-minute candle.</p>
            
            <div class="card">
                <button onclick="predict()">Update Data & Predict</button>
                <div id="result"></div>
            </div>

            <script>
                async function predict() {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = "Fetching data and calculating...";
                    
                    try {
                        const response = await fetch('/predict', { method: 'POST' });
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
                            `;
                        }
                    } catch (e) {
                        resultDiv.innerHTML = `<span style="color: red;">Network Error: ${e}</span>`;
                    }
                }
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def make_prediction(background_tasks: BackgroundTasks):
    # 1. Update Data
    success = update_database()
    if not success:
        return {"error": "Failed to fetch data from Binance"}
    
    # 2. Get Data
    df = get_data_for_prediction()
    if df is None or len(df) < 50:
        return {"error": "Not enough data in database"}
    
    # 3. Predict
    if predictor is None:
        return {"error": "Model not loaded"}
        
    try:
        pred_class, pred_proba, timestamp = predictor.predict(df)
        
        if pred_class is None:
             return {"error": "Prediction failed (insufficient data for features)"}

        return {
            "timestamp": str(timestamp),
            "prediction": int(pred_class),
            "probabilities": [float(p) for p in pred_proba]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11111)