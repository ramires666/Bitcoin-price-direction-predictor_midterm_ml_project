import uvicorn
from fastapi import FastAPI
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
LIMIT = 1000
BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
DEBUG_MODE = False

# --- Database Setup ---
Base = declarative_base()
class Kline(Base):
    __tablename__ = "klines"
    id = Column(Integer, primary_key=True, index=True)
    open_time = Column(DateTime, unique=True, index=True)
    open_price = Column(Float); high_price = Column(Float); low_price = Column(Float); close_price = Column(Float)
    volume = Column(Float); quote_vol = Column(Float); taker_buy_vol = Column(Float); taker_buy_quote_vol = Column(Float)

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
    backfill_data_on_startup()

# --- Data Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_binance_data(start_time, end_time, limit=LIMIT):
    start_ts = int(start_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(end_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": start_ts, "endTime": end_ts, "limit": limit}
    try:
        r = requests.get("https://api.binance.com/api/v3/klines", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching data from Binance API: {e}")
        return []

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

def save_klines_to_db(klines_data):
    if not klines_data: return
    klines_to_add = []
    for k in klines_data:
        try:
            open_time = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).replace(tzinfo=None)
            klines_to_add.append(Kline(open_time=open_time, open_price=float(k[1]), high_price=float(k[2]), low_price=float(k[3]), close_price=float(k[4]), volume=float(k[5]), quote_vol=float(k[7]), taker_buy_vol=float(k[9]), taker_buy_quote_vol=float(k[10])))
        except (ValueError, IndexError):
            continue
    save_klines_to_db_bulk(klines_to_add)

def download_and_import_binance_vision_data(date_str):
    filename = f"{SYMBOL}-{INTERVAL}-{date_str}.zip"
    url = f"{BINANCE_VISION_BASE_URL}/{SYMBOL}/{INTERVAL}/{filename}"
    print(f"Attempting to download from Binance Vision: {url}...")
    try:
        r = requests.get(url)
        if r.status_code == 404: return False
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                reader = csv.reader(io.TextIOWrapper(f))
                klines_to_add = []
                for row in reader:
                    try:
                        if not row or not row[0].isdigit(): continue
                        klines_to_add.append(Kline(open_time=datetime.fromtimestamp(int(row[0])/1000, tz=timezone.utc).replace(tzinfo=None), open_price=float(row[1]), high_price=float(row[2]), low_price=float(row[3]), close_price=float(row[4]), volume=float(row[5]), quote_vol=float(row[7]), taker_buy_vol=float(row[9]), taker_buy_quote_vol=float(row[10])))
                    except (ValueError, IndexError): continue
                save_klines_to_db_bulk(klines_to_add)
                print(f"Imported {len(klines_to_add)} klines from {date_str}")
                return True
    except Exception as e:
        print(f"Error downloading/processing {url}: {e}")
        return False

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
            if count < 96 and current_date < datetime.now(timezone.utc).date():
                print(f"Data for {date_str} is incomplete. Attempting backfill...")
                if not download_and_import_binance_vision_data(date_str):
                    print(f"Binance Vision failed for {date_str}. Falling back to API.")
                    save_klines_to_db(fetch_binance_data(day_start, day_end))
            current_date += timedelta(days=1)
    finally:
        db.close()
    print("Updating latest data from API...")
    save_klines_to_db(fetch_binance_data(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48), datetime.now(timezone.utc).replace(tzinfo=None)))

def backfill_data_on_startup():
    now_naive_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    ensure_data_availability(now_naive_utc - timedelta(days=7), now_naive_utc)

def generate_plot(df, start_date=None, end_date=None):
    plt.figure(figsize=(12, 6))
    plot_df = df.copy()
    if start_date: plot_df = plot_df[plot_df.index >= pd.to_datetime(start_date)]
    if end_date: plot_df = plot_df[plot_df.index <= pd.to_datetime(end_date)]
    if plot_df.empty:
        plt.title("No data available for the selected date range")
    else:
        plt.plot(plot_df.index, plot_df['close'], label='Close Price', color='black', linewidth=1)
        colors = {0: '#ffcccc', 1: '#e0e0e0', 2: '#ccffcc'}
        plot_df['group'] = (plot_df['prediction'] != plot_df['prediction'].shift()).cumsum()
        for _, group in plot_df.groupby('group'):
            plt.axvspan(group.index[0], group.index[-1] + timedelta(minutes=15), color=colors.get(group['prediction'].iloc[0], 'white'), alpha=0.8)
        last_prediction = plot_df['prediction'].iloc[-1]
        plt.title(f"BTCUSDT Price & Model Signal: {['DOWN', 'SIDEWAYS', 'UP'][int(last_prediction)]}")
        plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/prediction_plot.png"); plt.close()
    return "static/prediction_plot.png"

# --- HTML & Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html><head><title>BTC Predictor</title>
    <style>
        body{font-family:sans-serif;max-width:1000px;margin:auto;padding:20px}
        .card{border:1px solid #ddd;padding:20px;border-radius:8px;margin-top:20px}
        button{padding:10px 20px;font-size:16px;cursor:pointer;background-color:#007bff;color:white;border:none;border-radius:4px}
        button:hover{background-color:#0056b3}
        #prediction-info, #backtest-results{margin-top:20px}
        img{max-width:100%;height:auto;margin-top:20px}
        .controls{margin-bottom:15px;display:flex;gap:10px;align-items:center}
        input[type="datetime-local"]{padding:8px;border:1px solid #ccc;border-radius:4px}
    </style>
    </head><body onload="loadInitialData()">
        <h1>Bitcoin Price Direction Predictor</h1>
        <div id="prediction-info"><p>Loading initial prediction...</p></div>
        <div id="plot-container"></div>
        <div class="card">
            <div class="controls">
                <label>From:</label><input type="datetime-local" id="start-date">
                <label>To:</label><input type="datetime-local" id="end-date">
                <button onclick="runBacktest()">Run Backtest</button>
            </div>
            <div id="backtest-results"></div>
        </div>
        <script>
            function toLocalISOString(date){const p=(n)=>n.toString().padStart(2,'0');return `${date.getFullYear()}-${p(date.getMonth()+1)}-${p(date.getDate())}T${p(date.getHours())}:${p(date.getMinutes())}`}
            
            const today = new Date();
            today.setHours(23, 59, 0, 0);

            const twoDaysAgo = new Date();
            twoDaysAgo.setDate(today.getDate() - 2);
            twoDaysAgo.setHours(0, 0, 0, 0);
            
            document.getElementById('end-date').value = toLocalISOString(today);
            document.getElementById('start-date').value = toLocalISOString(twoDaysAgo);

            async function loadInitialData(){
                const predDiv=document.getElementById('prediction-info'),plotDiv=document.getElementById('plot-container');
                try{
                    const response=await fetch('/initial_load');
                    const data=await response.json();
                    if(data.error){predDiv.innerHTML=`<span style="color:red">Error: ${data.error}</span>`;return}
                    const color={'2':'green','0':'red'}[data.prediction]||'gray',labels={0:'DOWN',1:'SIDEWAYS',2:'UP'};
                    predDiv.innerHTML=`<h3>Prediction for next candle: <span style="color:${color}">${labels[data.prediction]}</span></h3><p>Time: ${data.timestamp}</p>
                    <p>Probabilities:</p><ul><li>DOWN: ${(data.probabilities[0]*100).toFixed(1)}%</li><li>SIDEWAYS: ${(data.probabilities[1]*100).toFixed(1)}%</li><li>UP: ${(data.probabilities[2]*100).toFixed(1)}%</li></ul><hr>`;
                    plotDiv.innerHTML=`<img src="/plot?t=${new Date().getTime()}" alt="Initial Plot">`;
                }catch(e){predDiv.innerHTML=`<span style="color:red">Network Error: ${e}</span>`}
            }
            
            async function runBacktest(){
                const backtestDiv=document.getElementById('backtest-results'),plotDiv=document.getElementById('plot-container'),startDate=document.getElementById('start-date').value,endDate=document.getElementById('end-date').value;
                backtestDiv.innerHTML="Running backtest...";
                try{
                    const response=await fetch('/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start_date:startDate,end_date:endDate})});
                    const data=await response.json();
                    if(data.error){backtestDiv.innerHTML=`<span style="color:red">Error: ${data.error}</span>`;return}
                    
                    plotDiv.innerHTML = `<img src="/plot?t=${new Date().getTime()}" alt="Backtest Plot">`;

                    backtestDiv.innerHTML=`<h4>Backtest Result</h4>
                    <p>Total Return: <strong>${data.backtest_return.toFixed(2)}%</strong> | Trades: ${data.backtest_trades}</p>
                    <div style="max-height:300px;overflow-y:auto;margin-top:10px;border:1px solid #eee"><table style="width:100%;border-collapse:collapse;font-size:12px">
                    <thead><tr><th>Entry</th><th>Exit</th><th>Price In</th><th>Price Out</th><th>Dir</th><th>Return %</th></tr></thead>
                    <tbody>${data.trades.map(t=>`<tr><td>${t.entry_time}</td><td>${t.exit_time}</td><td>${t.entry_price.toFixed(2)}</td><td>${t.exit_price.toFixed(2)}</td><td style="color:${t.direction==='LONG'?'green':'red'}">${t.direction}</td><td style="color:${t.return_pct>=0?'green':'red'}">${t.return_pct.toFixed(2)}%</td></tr>`).join('')}</tbody>
                    </table></div>`;
                }catch(e){backtestDiv.innerHTML=`<span style="color:red">Network Error: ${e}</span>`}
            }
        </script>
    </body></html>
    """

@app.get("/plot")
async def get_plot(): return FileResponse("static/prediction_plot.png")

class BacktestRequest(BaseModel):
    start_date: str; end_date: str

def get_dataframe(start_dt, end_dt):
    db = SessionLocal()
    try:
        klines = db.query(Kline).order_by(Kline.open_time.asc()).filter(Kline.open_time >= start_dt, Kline.open_time <= end_dt).all()
        if not klines: return None
        df = pd.DataFrame([{'time':k.open_time,'open':k.open_price,'high':k.high_price,'low':k.low_price,'close':k.close_price,'volume':k.volume,'quote_vol':k.quote_vol,'taker_buy_vol':k.taker_buy_vol,'taker_buy_quote_vol':k.taker_buy_quote_vol} for k in klines])
        return df.set_index(pd.to_datetime(df['time'])).drop('time', axis=1)
    finally:
        db.close()

@app.get("/initial_load")
async def initial_load():
    if not predictor: return {"error": "Model not loaded"}
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=3)
    ensure_data_availability(start_dt, end_dt)
    df = get_dataframe(start_dt, end_dt)
    if df is None: return {"error": "Not enough data"}
    
    latest_pred_class, latest_pred_proba, latest_timestamp = predictor.predict(df)
    predictions_df = predictor.predict_batch(df)
    df_with_preds = df.join(predictions_df[['prediction']], how='inner')
    
    plot_start_date = end_dt - timedelta(days=2)
    generate_plot(df_with_preds, start_date=plot_start_date, end_date=end_dt)
    
    return {"prediction":int(latest_pred_class),"probabilities":[float(p) for p in latest_pred_proba],"timestamp":str(latest_timestamp)}

@app.post("/backtest")
async def run_backtest_endpoint(request: BacktestRequest):
    if not predictor: return {"error": "Model not loaded"}
    start_dt = datetime.fromisoformat(request.start_date)
    end_dt = datetime.fromisoformat(request.end_date)
    buffer_start_dt = start_dt - timedelta(days=2)
    
    ensure_data_availability(buffer_start_dt, end_dt)
    df = get_dataframe(buffer_start_dt, end_dt)
    if df is None: return {"error": "Not enough data for backtest range"}

    predictions_df = predictor.predict_batch(df)
    df_with_preds = df.join(predictions_df[['prediction']], how='inner')
    
    generate_plot(df_with_preds, start_date=start_dt, end_date=end_dt)
    
    backtest_return, backtest_trades, trades, _ = run_simple_backtest(df_with_preds, start_date=start_dt, end_date=end_dt, debug=DEBUG_MODE)
    
    trades_serialized = [{"entry_time":t["entry_time"].strftime("%Y-%m-%d %H:%M"),"exit_time":t["exit_time"].strftime("%Y-%m-%d %H:%M"),"entry_price":t["entry_price"],"exit_price":t["exit_price"],"direction":t["direction"],"return_pct":t["return_pct"]} for t in trades]
    return {"backtest_return":backtest_return,"backtest_trades":backtest_trades,"trades":trades_serialized}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11111)