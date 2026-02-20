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
    plt.figure(figsize=(16, 8))
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
    plt.savefig("static/prediction_plot.png", dpi=140, bbox_inches="tight"); plt.close()
    return "static/prediction_plot.png"

# --- HTML & Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>BTC Direction Console</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            :root{
                --bg:#0b111e;
                --bg-soft:#121a2c;
                --card:#111a2e;
                --line:#2a3552;
                --text:#e5ecff;
                --muted:#8ea0c8;
                --accent:#1ac7b4;
                --up:#28d87c;
                --down:#ff6b6b;
                --flat:#f7b955;
                --shadow:0 22px 50px rgba(0, 0, 0, 0.35);
            }
            *{box-sizing:border-box}
            body{
                margin:0;
                font-family:'Space Grotesk',sans-serif;
                color:var(--text);
                background:
                    radial-gradient(900px 560px at -10% -10%, #123053 0%, transparent 62%),
                    radial-gradient(900px 560px at 110% 110%, #43244e 0%, transparent 62%),
                    linear-gradient(135deg, #090e1a, #0d1424 45%, #11172a);
                min-height:100vh;
            }
            .shell{max-width:1150px;margin:34px auto;padding:0 16px}
            .hero,.panel{
                background:linear-gradient(160deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
                border:1px solid rgba(255,255,255,0.12);
                border-radius:22px;
                box-shadow:var(--shadow);
                backdrop-filter:blur(5px);
            }
            .hero{padding:20px 22px;display:flex;justify-content:space-between;gap:12px;align-items:center}
            .hero h1{margin:0;font-size:29px;letter-spacing:.3px}
            .hero p{margin:6px 0 0;color:var(--muted)}
            .brand{display:flex;align-items:center;gap:10px}
            .icon-wrap{width:38px;height:38px;border-radius:10px;background:rgba(26,199,180,.16);display:grid;place-items:center}
            .grid{margin-top:14px;display:grid;grid-template-columns:1.12fr .88fr;gap:14px}
            .panel{padding:18px}
            .title{margin:0 0 12px;font-size:16px;display:flex;align-items:center;gap:8px;color:#cedbff}
            #prediction-info .loading,#backtest-results .loading{color:var(--muted)}
            .signal-pill{
                display:inline-flex;align-items:center;gap:7px;
                padding:7px 12px;border-radius:999px;border:1px solid transparent;font-weight:700;font-size:12px
            }
            .signal-up{color:var(--up);background:rgba(40,216,124,.12);border-color:rgba(40,216,124,.35)}
            .signal-down{color:var(--down);background:rgba(255,107,107,.12);border-color:rgba(255,107,107,.35)}
            .signal-flat{color:var(--flat);background:rgba(247,185,85,.12);border-color:rgba(247,185,85,.35)}
            .metrics{margin-top:14px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}
            .metric{padding:10px;border-radius:12px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08)}
            .metric .name{font-size:12px;color:var(--muted)}
            .metric .bar{margin-top:7px;height:7px;border-radius:999px;background:#232e4a;overflow:hidden}
            .metric .fill{display:block;height:100%}
            .metric .value{margin-top:7px;font-weight:700;font-family:'IBM Plex Mono',monospace}
            .chart-wrap{
                margin-top:10px;
                overflow:hidden;
                border-radius:16px;
                border:1px solid rgba(255,255,255,.1);
                background:rgba(10,15,27,.75);
                min-height:480px;
                display:flex;
                align-items:center;
                justify-content:center
            }
            .plot-image{display:block;width:100%;height:auto}
            .plot-panel{margin-top:14px}
            .controls{
                display:grid;
                grid-template-columns:1fr 1fr auto;
                gap:10px;
                align-items:end;
                margin-bottom:12px
            }
            label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}
            input[type="datetime-local"]{
                width:100%;padding:10px 12px;border-radius:11px;
                border:1px solid var(--line);background:#0f1628;color:var(--text);font-family:'IBM Plex Mono',monospace
            }
            button{
                height:42px;padding:0 15px;border:0;border-radius:11px;cursor:pointer;
                background:linear-gradient(135deg,var(--accent),#39f0d1);color:#041218;font-weight:700;
                display:flex;align-items:center;gap:8px;justify-content:center
            }
            button:hover{filter:brightness(1.05)}
            .bt-head{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
            .bt-main{font-size:20px;font-weight:700}
            .bt-main.up{color:var(--up)}
            .bt-main.down{color:var(--down)}
            .bt-main.flat{color:var(--flat)}
            .bt-sub{color:var(--muted);font-size:13px}
            .table-wrap{margin-top:12px;max-height:290px;overflow:auto;border:1px solid rgba(255,255,255,.1);border-radius:12px}
            table{width:100%;border-collapse:collapse;font-size:12px}
            th,td{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,.08);white-space:nowrap}
            th{position:sticky;top:0;background:#17223b;color:#d3e0ff;text-align:left}
            .dir-long{color:var(--up);font-weight:700}
            .dir-short{color:var(--down);font-weight:700}
            .ret-pos{color:var(--up);font-weight:700}
            .ret-neg{color:var(--down);font-weight:700}
            .error{
                margin-top:6px;padding:10px 12px;border-radius:10px;
                border:1px solid rgba(255,107,107,.35);color:#ffb6b6;background:rgba(255,107,107,.1)
            }
            @media(max-width:980px){
                .grid{grid-template-columns:1fr}
            }
            @media(max-width:640px){
                .hero{padding:16px}
                .hero h1{font-size:22px}
                .metrics{grid-template-columns:1fr}
                .controls{grid-template-columns:1fr}
                .chart-wrap{min-height:300px}
            }
        </style>
    </head>
    <body>
        <main class="shell">
            <section class="hero">
                <div class="brand">
                    <div class="icon-wrap"><i data-lucide="bitcoin"></i></div>
                    <div>
                        <h1>BTC Direction Console</h1>
                        <p>Real-time signal and backtest dashboard</p>
                    </div>
                </div>
                <div class="signal-pill signal-flat"><i data-lucide="activity"></i>Live model</div>
            </section>

            <section class="grid">
                <article class="panel">
                    <h2 class="title"><i data-lucide="trending-up"></i>Latest Prediction</h2>
                    <div id="prediction-info"><p class="loading">Loading prediction...</p></div>
                </article>

                <article class="panel">
                    <h2 class="title"><i data-lucide="flask-conical"></i>Backtest</h2>
                    <div class="controls">
                        <div>
                            <label for="start-date">From</label>
                            <input type="datetime-local" id="start-date">
                        </div>
                        <div>
                            <label for="end-date">To</label>
                            <input type="datetime-local" id="end-date">
                        </div>
                        <button id="run-backtest-btn" onclick="runBacktest()"><i data-lucide="play"></i>Run Backtest</button>
                    </div>
                    <div id="backtest-results"><p class="loading">Select dates and run backtest.</p></div>
                </article>
            </section>

            <section class="panel plot-panel">
                <h2 class="title"><i data-lucide="bar-chart-3"></i>Price Chart</h2>
                <div id="plot-container" class="chart-wrap"></div>
            </section>
        </main>

        <script>
            const labels = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'};
            const tones = {0: 'signal-down', 1: 'signal-flat', 2: 'signal-up'};
            const icons = {0: 'trending-down', 1: 'minus', 2: 'trending-up'};

            function toLocalISOString(date){
                const p=(n)=>n.toString().padStart(2,'0');
                return `${date.getFullYear()}-${p(date.getMonth()+1)}-${p(date.getDate())}T${p(date.getHours())}:${p(date.getMinutes())}`;
            }

            function setDefaultRange(){
                const today = new Date();
                today.setHours(23, 59, 0, 0);
                const twoDaysAgo = new Date(today);
                twoDaysAgo.setDate(today.getDate() - 2);
                twoDaysAgo.setHours(0, 0, 0, 0);
                document.getElementById('end-date').value = toLocalISOString(today);
                document.getElementById('start-date').value = toLocalISOString(twoDaysAgo);
            }

            function refreshIcons(){
                if (window.lucide) window.lucide.createIcons();
            }

            function probabilityMetric(name, value, color){
                return `
                    <div class="metric">
                        <div class="name">${name}</div>
                        <div class="bar"><span class="fill" style="width:${value.toFixed(1)}%;background:${color}"></span></div>
                        <div class="value">${value.toFixed(1)}%</div>
                    </div>
                `;
            }

            function renderPlot(tag){
                const plotDiv = document.getElementById('plot-container');
                plotDiv.innerHTML = `<img class="plot-image" src="/plot?t=${Date.now()}" alt="${tag} plot">`;
            }

            async function loadInitialData(){
                const predDiv = document.getElementById('prediction-info');
                try{
                    const response = await fetch('/initial_load');
                    const data = await response.json();
                    if(data.error){
                        predDiv.innerHTML = `<div class="error"><i data-lucide="alert-triangle"></i> ${data.error}</div>`;
                        refreshIcons();
                        return;
                    }

                    const p = Number(data.prediction);
                    const probs = data.probabilities.map(x => Number(x) * 100);
                    predDiv.innerHTML = `
                        <div class="signal-pill ${tones[p] || 'signal-flat'}"><i data-lucide="${icons[p] || 'activity'}"></i>${labels[p]}</div>
                        <p style="margin:10px 0 0;color:var(--muted);font-family:'IBM Plex Mono',monospace;">${data.timestamp}</p>
                        <div class="metrics">
                            ${probabilityMetric('DOWN', probs[0], 'var(--down)')}
                            ${probabilityMetric('SIDEWAYS', probs[1], 'var(--flat)')}
                            ${probabilityMetric('UP', probs[2], 'var(--up)')}
                        </div>
                    `;
                    renderPlot('Initial');
                }catch(e){
                    predDiv.innerHTML = `<div class="error"><i data-lucide="wifi-off"></i> Network error: ${e}</div>`;
                }
                refreshIcons();
            }

            async function runBacktest(){
                const backtestDiv = document.getElementById('backtest-results');
                const startDate = document.getElementById('start-date').value;
                const endDate = document.getElementById('end-date').value;
                backtestDiv.innerHTML = `<p class="loading"><i data-lucide="loader-circle"></i> Running backtest...</p>`;
                refreshIcons();

                try{
                    const response = await fetch('/backtest', {
                        method:'POST',
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({start_date:startDate,end_date:endDate})
                    });
                    const data = await response.json();
                    if(data.error){
                        backtestDiv.innerHTML = `<div class="error"><i data-lucide="alert-triangle"></i> ${data.error}</div>`;
                        refreshIcons();
                        return;
                    }

                    renderPlot('Backtest');
                    const result = Number(data.backtest_return);
                    const cls = result > 0 ? 'up' : (result < 0 ? 'down' : 'flat');

                    backtestDiv.innerHTML = `
                        <div class="bt-head">
                            <div>
                                <div class="bt-main ${cls}">${result.toFixed(2)}%</div>
                                <div class="bt-sub">Total return</div>
                            </div>
                            <div class="signal-pill signal-flat"><i data-lucide="bar-chart-3"></i>${data.backtest_trades} trades</div>
                        </div>
                        <div class="table-wrap">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Entry</th><th>Exit</th><th>Price In</th><th>Price Out</th><th>Dir</th><th>Return</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.trades.map(t => `
                                        <tr>
                                            <td>${t.entry_time}</td>
                                            <td>${t.exit_time}</td>
                                            <td>${Number(t.entry_price).toFixed(2)}</td>
                                            <td>${Number(t.exit_price).toFixed(2)}</td>
                                            <td class="${t.direction === 'LONG' ? 'dir-long' : 'dir-short'}">${t.direction}</td>
                                            <td class="${Number(t.return_pct) >= 0 ? 'ret-pos' : 'ret-neg'}">${Number(t.return_pct).toFixed(2)}%</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                }catch(e){
                    backtestDiv.innerHTML = `<div class="error"><i data-lucide="wifi-off"></i> Network error: ${e}</div>`;
                }
                refreshIcons();
            }

            document.addEventListener('DOMContentLoaded', () => {
                refreshIcons();
                setDefaultRange();
                loadInitialData();
            });
        </script>
    </body>
    </html>
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
    uvicorn.run(app, host="0.0.0.0", port=8080)
