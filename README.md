# Bitcoin Direction Predictor

Production-ready FastAPI service for BTC direction classification (`UP`, `DOWN`, `SIDEWAYS`) with backtesting UI.


## LIVE DEMO:
https://btc-prediction.grom.world/

## Tech Stack
- Python 3.12
- FastAPI + Uvicorn
- XGBoost + technical indicators (`pandas-ta`)
- SQLite (local market cache)
- Docker / Docker Compose

## Project Layout
- `app.py` - FastAPI app and web UI.
- `predict.py` - feature engineering + model inference.
- `simple_backtest.py` - backtest logic.
- `models/` - trained model artifacts.
- `deploy/prod_server/` - production Dockerfile and deploy script.

## Local Run (Docker Compose)
Prerequisites:
- Docker Engine + Docker Compose

Start locally:

```bash
docker compose up --build -d
```

Open:
- `http://localhost:9743`

Stop:

```bash
docker compose down
```

Notes:
- Local compose uses `deploy/prod_server/Dockerfile`.
- It mounts:
  - `./models -> /app/models` (read-only)
  - `./data/crypto_data.db -> /app/crypto_data.db`
  - `./static -> /app/static`

## Training
Train locally (Conda):

```bash
conda env create -f environment.yml
conda activate xgb_fints_project
python train.py
```

## Production Packaging and Upload
Use the script:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1 -Tag "2026-02-20"
```

Default remote target:
- Host alias: `prod`
- Directory: `/home/user/GROM/bitcoin_direction`

On the server:

```bash
cd /home/user/GROM/bitcoin_direction
docker load -i xgb-bitcoin-direction-<tag>.tar
docker rm -f xgb-bitcoin-direction 2>/dev/null || true
docker run -d --name xgb-bitcoin-direction --restart unless-stopped -p 9743:9743 xgb-bitcoin-direction:latest
```

## Health Check Commands
```bash
docker ps --filter name=xgb-bitcoin-direction
docker logs -n 200 xgb-bitcoin-direction
```

## Disclaimer
This project is for research/engineering purposes and is not financial advice.
