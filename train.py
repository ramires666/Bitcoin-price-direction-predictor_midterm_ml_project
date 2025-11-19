import os
# Workaround for Windows OpenMP/scikit-learn conflict
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
from scipy.ndimage import gaussian_filter1d
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH_FUNDINGS = "data/processed/fundings.parquet"
DATA_PATH_KLINES = "data/processed/klines_15min_all.parquet"
DATA_PATH_VOLUMES = "data/processed/aggtrades_15min_all.parquet"
MODEL_DIR = "models"
MODEL_FILENAME = "xgb_hmm_model.joblib"
SCALER_FILENAME = "scaler.joblib"
HMM_FILENAME = "hmm_model.joblib"
HMM_SCALER_FILENAME = "hmm_scaler.joblib"

# Hyperparameters
SIGMA = 4
THRESHOLD = 0.0004

def load_and_merge_data():
    """Loads and merges funding, klines, and volume data."""
    print("Loading data...")
    try:
        fundings = pd.read_parquet(DATA_PATH_FUNDINGS)
        klines = pd.read_parquet(DATA_PATH_KLINES)
        volumes = pd.read_parquet(DATA_PATH_VOLUMES)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    if "datetime" in volumes.columns: volumes = volumes.rename(columns={"datetime": "time"})
    if "calc_time" in fundings.columns: fundings = fundings.rename(columns={"calc_time": "time"})
    if "open_time" in klines.columns: klines = klines.rename(columns={"open_time": "time"})

    for col in ["time"]:
        volumes[col] = pd.to_datetime(volumes[col], utc=True)
        fundings[col] = pd.to_datetime(fundings[col], utc=True)
        klines[col] = pd.to_datetime(klines[col], utc=True)

    df = pd.merge(volumes, klines, on="time", how="inner")
    df = pd.merge(df, fundings, on="time", how="left")

    df = df.sort_values("time").reset_index(drop=True)
    if "funding_rate" in df.columns:
        df["funding_rate"] = df["funding_rate"].ffill()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def create_target_labels(df, sigma=1, threshold=0.0005):
    """Creates target labels using Gaussian smoothing."""
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

def add_features(df):
    """Adds technical indicators and HMM features."""
    df = df.copy()
    
    # Basic Features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Volume Features
    df["volume_delta"] = df["ask_vol"] - df["bid_vol"]
    df["cvd"] = df["volume_delta"].cumsum()
    for window in [4, 12, 24]:
        df[f"vol_delta_rolling_{window}"] = df["volume_delta"].rolling(window).sum()
        
    # Volatility Features
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    bbands = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Handle bb_width column name variation
    try:
        df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    except KeyError:
        bbu = [c for c in df.columns if c.startswith('BBU_')][0]
        bbl = [c for c in df.columns if c.startswith('BBL_')][0]
        bbm = [c for c in df.columns if c.startswith('BBM_')][0]
        df['bb_width'] = (df[bbu] - df[bbl]) / df[bbm]
    
    # Trend Features
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)
    df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
    
    return df

def train_hmm(df):
    """Trains HMM and returns the model and scaler."""
    print("Training HMM...")
    hmm_data = df[['log_return', 'range_pct', 'volume_delta']].copy()
    hmm_data['log_volume'] = np.log(df['total_vol'] + 1)
    hmm_data = hmm_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    scaler_hmm = RobustScaler()
    X_hmm = scaler_hmm.fit_transform(hmm_data)
    
    # Initialize HMM with random parameters to avoid K-Means initialization crash on Windows
    # We use init_params="stc" to skip 'm' (means) initialization which uses KMeans by default
    model_hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42, init_params="stc")
    
    # Initialize means manually to skip KMeans
    # We split the data into 3 chunks and take their means as initial guesses
    means_init = []
    chunk_size = len(X_hmm) // 3
    for i in range(3):
        chunk = X_hmm[i*chunk_size:(i+1)*chunk_size]
        if len(chunk) > 0:
            means_init.append(chunk.mean(axis=0))
        else:
            means_init.append(np.random.rand(X_hmm.shape[1]))
            
    model_hmm.means_ = np.array(means_init)
    
    model_hmm.fit(X_hmm)
    
    return model_hmm, scaler_hmm

def add_hmm_features(df, model_hmm, scaler_hmm):
    """Adds HMM state probabilities to the dataframe."""
    hmm_data = df[['log_return', 'range_pct', 'volume_delta']].copy()
    hmm_data['log_volume'] = np.log(df['total_vol'] + 1)
    
    # Handle NaNs for prediction (fill with 0 or drop) - here we drop for simplicity in training
    # In production, you might want to forward fill or use 0
    # Create a boolean mask to filter out invalid rows (containing Infinity or NaN values)
    # 1. ~hmm_data.isin([np.inf, -np.inf]).any(axis=1):
    #    - hmm_data.isin(...) checks for positive or negative infinity in the data.
    #    - .any(axis=1) returns True if any column in a row contains infinity.
    #    - The tilde (~) negates this, so we get True for rows WITHOUT any infinity.
    # 2. ~hmm_data.isna().any(axis=1):
    #    - hmm_data.isna() checks for NaN (Not a Number) values.
    #    - .any(axis=1) returns True if any column in a row contains NaN.
    #    - The tilde (~) negates this, so we get True for rows WITHOUT any NaN.
    # 3. The '&' operator combines these two conditions, ensuring we keep only rows
    #    that are free from both infinite and missing values.
    mask = ~hmm_data.isin([np.inf, -np.inf]).any(axis=1) & ~hmm_data.isna().any(axis=1)
    valid_data = hmm_data[mask]
    
    if len(valid_data) == 0:
        return df
        
    X_hmm = scaler_hmm.transform(valid_data)
    state_probs = model_hmm.predict_proba(X_hmm)
    
    # Initialize columns with NaN
    for i in range(3):
        df[f'hmm_prob_{i}'] = np.nan
        
    # Fill valid rows
    for i in range(3):
        df.loc[mask, f'hmm_prob_{i}'] = state_probs[:, i]
        
    return df

def main():
    # 1. Load Data
    df = load_and_merge_data()
    if df is None: return

    # 2. Target Generation
    df = create_target_labels(df, sigma=SIGMA, threshold=THRESHOLD)
    
    # 3. Feature Engineering
    df = add_features(df)
    df = df.dropna()
    
    # 4. Train HMM
    model_hmm, scaler_hmm = train_hmm(df)
    df = add_hmm_features(df, model_hmm, scaler_hmm)
    
    # 5. Prepare for XGBoost
    feature_cols = [
        'volume_delta', 'cvd', 'vol_delta_rolling_4', 'vol_delta_rolling_12', 'vol_delta_rolling_24',
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'trend_ema',
        'bb_width', 'BBP_20_2.0', 'range_pct',
        'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'
    ]
    
    # Handle BBP column name variation
    if 'BBP_20_2.0' not in df.columns:
        # Try to find BBP column
        bbp_cols = [c for c in df.columns if c.startswith('BBP_')]
        if bbp_cols:
            df['BBP_20_2.0'] = df[bbp_cols[0]]
        else:
            # Calculate manually if missing: (Close - Lower) / (Upper - Lower)
            try:
                bbu = [c for c in df.columns if c.startswith('BBU_')][0]
                bbl = [c for c in df.columns if c.startswith('BBL_')][0]
                df['BBP_20_2.0'] = (df['close'] - df[bbl]) / (df[bbu] - df[bbl])
            except IndexError:
                print("Warning: Could not calculate BBP. Filling with 0.")
                df['BBP_20_2.0'] = 0

    # Shift features (predict next bar)
    df[feature_cols] = df[feature_cols].shift(1)
    df = df.dropna()
    
    X = df[feature_cols]
    y = df['target']
    
    # Scale Features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. Train XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        objective='multi:softprob', 
        num_class=3, 
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        tree_method='hist'
    )
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }


    """Handling Imbalance: The primary purpose of using the 'f1_weighted' metric is to provide a balanced evaluation when dealing with imbalanced datasets, where some classes have significantly more samples than others. Standard accuracy can be misleading in such cases because a model might perform well on the majority class but poorly on rare, important classes.
Calculation:
The F1 score (harmonic mean of precision and recall) is computed for each individual class.
Each class's F1 score is then multiplied by its "support" (the actual number of samples in that class).
These weighted scores are summed and divided by the total number of instances across all classes to get a single, overall score."""
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=200,
        scoring='f1_weighted',
        cv=TimeSeriesSplit(n_splits=3),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_scaled, y)
    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Score: {random_search.best_score_}")
    
    # 7. Save Models
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Save metrics
    metrics = {
        "best_score": random_search.best_score_,
        "best_params": random_search.best_params_
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    joblib.dump(best_model, os.path.join(MODEL_DIR, MODEL_FILENAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILENAME))
    joblib.dump(model_hmm, os.path.join(MODEL_DIR, HMM_FILENAME))
    joblib.dump(scaler_hmm, os.path.join(MODEL_DIR, HMM_SCALER_FILENAME))
    
    print(f"Models saved to {MODEL_DIR}/")

if __name__ == "__main__":
    main()
