# 🔮 Bitcoin Price Direction Classifier

> "The market is a pendulum that forever swings between unsustainable optimism (which makes stocks too expensive) and unjustified pessimism (which makes them too cheap). The intelligent investor is a realist who sells to optimists and buys from pessimists." - Benjamin Graham

## 📖 Introduction

Trading cryptocurrencies is a high-stakes game that attracts millions with the promise of unlimited upside. But for every winner, there are countless losers. The difference often boils down to one thing: **information**.

Imagine having a "magic ball" that doesn't just guess, but scientifically analyzes the market's hidden states. That's what we've built.

This project isn't just another technical indicator. It's a sophisticated machine learning system that combines:
1.  **"Oracle" Labeling**: We use a zero-lag smoothing technique to define the "true" trend, training our model on what *actually* happened, not just lagging indicators.
2.  **Advanced Feature Engineering**: We generate hundreds of technical indicators across multiple categories (Momentum, Trend, Volatility, Volume, Overlap, Statistics, Candles).
3.  **Automated Feature Selection**: The training pipeline intelligently selects the optimal combination of indicator groups and the top-k most important features to maximize accuracy and reduce noise.
4.  **XGBoost Classification**: We use a powerful gradient boosting model tuned via Randomized Search with Walk-Forward Validation to predict market direction.

---

## 🚀 LIVE DEMO: Try it now!

The service is currently running and accessible here:
# 👉 **[https://webservice-732860276994.europe-central2.run.app/](https://webservice-732860276994.europe-central2.run.app/)**

**How it works:**
The model on the page automatically analyzes the current market state and predicts the likely direction: **UP**, **DOWN**, or **SIDEWAYS**.
*   The **background color** behind the price chart changes to reflect the model's prediction:
    *   🟢 **Green**: Upward trend predicted.
    *   🔴 **Red**: Downward trend predicted.
    *   ⚪ **Gray**: Sideways / Neutral market.

---

## 🔬 Research & Development Journey

This project is the result of a rigorous research process detailed in `midterm_project_bitcoin_price_direction_classificator_final.ipynb`. Here is a summary of the key steps taken:

1.  **Data Analysis & Stationarity**:
    *   We started by analyzing Bitcoin price data and confirmed its non-stationary nature using the **Augmented Dickey-Fuller (ADF)** test.
    *   We demonstrated that simple regression models fail to predict price changes ($R^2 \approx 0$), necessitating a switch to a classification approach (Up/Down/Sideways).

2.  **"Oracle" Target Generation**:
    *   Instead of using simple future returns, we created high-quality "Oracle" labels using a **Centered Moving Average (Gaussian Smoothing)**.
    *   This technique uses future data to draw a smooth trend line, allowing us to train the model on what *actually* happened (the true trend) rather than noisy price fluctuations.
    *   *Note: This smoothed line is used ONLY for creating training labels and is never used as a feature to prevent data leakage.*

3.  **Feature Engineering**:
    *   We generated a massive set of technical indicators across multiple categories:
        *   **Momentum**: RSI, MACD, Stochastic, etc.
        *   **Trend**: ADX, Aroon, Vortex, etc.
        *   **Volatility**: Bollinger Bands, ATR, etc.
        *   **Volume**: OBV, MFI, CMF, etc.
        *   **Candlestick Patterns**: Doji, Hammer, Engulfing, etc.

4.  **Feature Selection (The "Secret Sauce")**:
    *   We didn't just dump all features into the model. We performed an iterative **Brute-Force Search** to find the optimal combination of indicator groups (e.g., "Momentum + Volatility + Trend").
    *   We then refined this further by selecting only the **Top-K best features** based on feature importance, removing noise and improving model generalization.

5.  **Model Training & Tuning**:
    *   We chose **XGBoost** as our classifier due to its performance and ability to handle non-linear relationships.
    *   We used **RandomizedSearchCV** with **TimeSeriesSplit** (Walk-Forward Validation) to tune hyperparameters. This ensures the model is tested on "future" data it hasn't seen, simulating real-world trading conditions.

### 🛡️ Preventing Data Leakage (Crucial!)
In financial time series analysis, preventing data leakage is paramount. We paid extreme attention to this:
*   **Direct Leakage**: We ensured that the "Oracle" target (which uses future data) is strictly separated from the input features.
*   **Indirect Leakage**: When creating features (like moving averages), we ensured they only use past data relative to the prediction point.
*   **Future Information**: Our validation strategy (**TimeSeriesSplit**) strictly respects the temporal order. We never train on future data and test on past data. This "Walk-Forward" approach mimics real-life trading where you only know the past.

## 📂 Project File Structure

```text
.
├── app.py                                      # FastAPI web service for live predictions
├── best_features_list.json                     # List of optimal features selected during training
├── best_xgb_model.json                         # Saved XGBoost model artifact
├── docker-compose.yml                          # Docker Compose configuration
├── Dockerfile                                  # Docker build instructions
├── environment.yml                             # Conda environment dependencies
├── main.py                                     # Entry point for some operations
├── midterm_project_bitcoin_price_direction_classificator_final.ipynb # Main research notebook
├── predict.py                                  # Inference script for generating predictions
├── requirements.txt                            # Python pip dependencies
├── train.py                                    # Main training pipeline script
├── data/                                       # Data directory
│   ├── downloader.py                           # Script to download raw data
│   ├── parsing_fundings.py                     # Script to parse funding rates
│   ├── parsing_klines.py                       # Script to parse candlestick data
│   ├── parsing_trades.py                       # Script to parse trade data
│   └── processed/                              # Stored Parquet data files
│       ├── aggtrades_15min_all.parquet
│       ├── all_merged.parquet
│       ├── fundings.parquet
│       └── klines_15min_all.parquet
└── models/                                     # Directory for saved models
    ├── best_features_list.json
    └── best_xgb_model.joblib
```

## 📊 Backtesting Results
Our research shows that while regression fails miserably ($R^2 \approx 0$), classification is viable. By filtering for high-confidence setups and understanding market regimes, the model demonstrates predictive power significantly better than a coin flip (~69% accuracy on test data).

---

## ⚡ How to Run (Local & Docker)

### Prerequisites

You'll need **Docker** and **Docker Compose** installed. That's it! We've containerized everything for your convenience.

### 1. Get the Data
The project comes pre-loaded with **2 years of historical data** (15-minute timeframe) stored in `data/processed/` as optimized Parquet files. You don't need to download anything to get started!

However, if you need data for a different period or timeframe, we've included powerful scripts in the `data/` folder:
*   **`data/downloader.py`**: Downloads raw data from Binance.
*   **`data/parsing_*.py`**: Processes raw data into clean Parquet files.

**Automatic Updates:**
When you launch the project (via Docker or `app.py`), the service **automatically fetches the latest data** from Binance to ensure predictions are always based on the most current market state. The built-in **Backtester** also allows you to download and test any specific time range with a single click.

### 2. Train the Model

Before you can predict, you must train. You can do this locally or inside the container.

**Option A: Locally (with Conda)**
```bash
conda env create -f environment.yml
conda activate xgb_fints_project
python train.py
```
*Note: If you encounter OpenMP/KMeans crashes on Windows, try Option B.*

**Option B: Inside Docker (Recommended for Windows)**
If you have issues running locally, you can run the training script inside the Docker container:

1. Build the image:
```bash
docker build -t xgb_fints_project .
```

2. Run the training script (choose your OS):

**Windows (PowerShell):**
```powershell
docker run --rm -v ${PWD}:/app xgb_fints_project python train.py
```

**Windows (Command Prompt):**
```cmd
docker run --rm -v %cd%:/app xgb_fints_project python train.py
```

**Linux / macOS:**
```bash
docker run --rm -v $(pwd):/app xgb_fints_project python train.py
```

This will create a `models/` directory with the trained artifacts (`best_xgb_model.joblib`, `best_features_list.json`, etc.).

### 3. Launch the Web Service 🚀
Spin up the prediction service with a single command:

```bash
docker-compose up --build
```

The service will start on **port 8080**.

### 4. Get Predictions!
Open your browser and go to:
👉 **http://localhost:8080**

You'll see a simple interface. Click **"Update Data & Predict"** to fetch the latest market data from Binance and consult the Oracle.

---

## ☁️ Deployment to Google Cloud Run

Want to take it to the cloud? Here is a step-by-step guide to deploying this service to Google Cloud Run.

### 1. One-time Project Setup

Initialize your Google Cloud environment and enable the necessary services:

```bash
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

*Note: You can find your `YOUR_PROJECT_ID` in the Google Cloud Console dashboard.*

### 2. Build the Docker Image

Build the image locally (ensure you are in the directory with the `Dockerfile`):

```bash
docker build -t gcr.io/YOUR_PROJECT_ID/webservice .
```

### 3. Push to Container Registry

Configure Docker to authenticate with GCP and push your image:

```bash
gcloud auth configure-docker
docker push gcr.io/YOUR_PROJECT_ID/webservice
```

### 4. Deploy to Cloud Run

Deploy the service to the cloud:

```bash
gcloud run deploy webservice \
  --image gcr.io/YOUR_PROJECT_ID/webservice \
  --platform managed \
  --region europe-central2 \
  --allow-unauthenticated
```

Once completed, you will see a URL like `URL: https://...run.app`. This is your live application address.

---
*Disclaimer: This is a research project, not financial advice. Trading cryptocurrencies involves significant risk. Use this "magic ball" responsibly!* 😉