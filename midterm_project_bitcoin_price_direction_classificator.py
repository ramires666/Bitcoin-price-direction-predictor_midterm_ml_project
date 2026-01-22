#%% [markdown]
# # Bitcoin Price Direction Prediction Project
#
# ## Project Overview
# This project aims to build a machine learning model to predict the direction of Bitcoin price movements (Up, Down, Sideways).
# We will use **XGBoost** as the primary classifier and explore the integration of **Hidden Markov Models (HMM)** to capture market regimes.
#
# ## Key Steps:
# 1.  **Data Analysis**: Demonstrate non-stationarity of prices and failure of simple regression.
# 2.  **Target Generation**: Create "Oracle" labels using a zero-lag Centered Moving Average.
# 3.  **HMM Integration**: Model market states to use as features.
# 4.  **Classification**: Train XGBoost with iterative feature engineering and Grid Search.
# 5.  **Evaluation**: rigorous testing and visualization.

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas_ta as ta # Removed due to installation issues
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import adfuller
import warnings
import os

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
plt.style.use('seaborn-v0_8-darkgrid')

#%%
def load_and_merge_data(
    fundings_path: str = "data/processed/fundings.parquet",
    klines_path: str = "data/processed/klines_15min_all.parquet",
    volumes_path: str = "data/processed/aggtrades_15min_all.parquet",
) -> pd.DataFrame:
    """Loads and merges funding, klines, and volume data."""
    print("Loading data...")
    fundings = pd.read_parquet(fundings_path)
    klines = pd.read_parquet(klines_path)
    volumes = pd.read_parquet(volumes_path)

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

#%%
# Load the data
df = load_and_merge_data()
print(df.head())

#%% [markdown]
# ## 1. Data Analysis: Stationarity & Regression Failure
#
# ### 1.1 Stationarity Check
# Financial time series like Bitcoin prices are typically non-stationary (mean and variance change over time), making them unsuitable for direct prediction using many statistical models. Log-returns are usually stationary.
#
# We will use the **Augmented Dickey-Fuller (ADF)** test to statistically prove this.

#%%
# Calculate Log Returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna()

def check_stationarity(series, name="Series"):
    result = adfuller(series)
    print(f"--- ADF Test for {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    if result[1] < 0.05:
        print(f"Result: {name} is STATIONARY (Reject H0)")
    else:
        print(f"Result: {name} is NON-STATIONARY (Fail to reject H0)")
    print("\n")

# Check Close Price
check_stationarity(df['close'].iloc[:10000], "Close Price")

# Check Log Returns
check_stationarity(df['log_return'].iloc[:10000], "Log Returns")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(df['close'].iloc[:1000], label='Close Price')
axes[0].set_title('Bitcoin Close Price (Non-Stationary)')
axes[0].legend()

axes[1].plot(df['log_return'].iloc[:1000], label='Log Returns', color='orange')
axes[1].set_title('Bitcoin Log Returns (Stationary)')
axes[1].legend()

plt.tight_layout()
plt.show()

#%% [markdown]
# ### 1.2 Why Regression Fails
# We will now train a simple XGBoost Regressor to predict the *next bar's log-return* using a small subset of data. 
# We expect the $R^2$ score to be close to zero (or negative), indicating that the model cannot predict the magnitude of price changes better than a simple mean baseline.

#%%
# Prepare data for regression demo (Subset)
subset_size = 5000
df_reg = df.iloc[:subset_size].copy()

# Features: Lagged returns
for i in range(1, 6):
    df_reg[f'lag_{i}'] = df_reg['log_return'].shift(i)

df_reg = df_reg.dropna()

X_reg = df_reg[[f'lag_{i}' for i in range(1, 6)]]
y_reg = df_reg['log_return'] # Predict next return

# Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, shuffle=False)

# Train Simple XGBoost Regressor
regressor = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Predict
y_pred_reg = regressor.predict(X_test_reg)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Regression MSE: {mse:.8f}")
print(f"Regression R^2: {r2:.4f} (Expected to be near 0)")

# Visualization
plt.figure(figsize=(15, 5))
plt.plot(y_test_reg.values[:200], label='Actual Returns', alpha=0.7)
plt.plot(y_pred_reg[:200], label='Predicted Returns', alpha=0.7)
plt.title('Regression: Actual vs Predicted Returns (First 200 Test Samples)')
plt.legend()
plt.show()

#%% [markdown]
# ## 2. Target Generation: The "Oracle" Labeling
#
# Since regression is difficult, we switch to **Classification** (Up/Down/Sideways).
# To create high-quality labels, we use a **Centered Moving Average (Gaussian Smoothing)** on the *entire* dataset. 
#
# **Why?** 
# This creates a smooth trend line that uses future data (zero lag). This is valid for *target generation* because we want to teach the model what *actually happened*. We must ensure we **never** use this smoothed line as a feature.
#
# **Classes:**
# *   **2 (UP):** Slope > Threshold
# *   **0 (DOWN):** Slope < -Threshold
# *   **1 (SIDEWAYS):** |Slope| <= Threshold

#%%
def create_target_labels(df, sigma=1, threshold=0.0005):
    """
    Creates target labels using Gaussian smoothing (Centered MA).
    sigma: Controls smoothness (higher = smoother trend)
    threshold: Slope threshold for Up/Down classification
    """
    df_target = df.copy()
    
    # 1. Apply Gaussian Filter (Centered Smoothing)
    # This uses the entire series (future and past) to smooth
    df_target['smoothed_close'] = gaussian_filter1d(df_target['close'], sigma=sigma)
    
    # 2. Calculate Slope (Derivative) of the smoothed line
    # We use log returns of the smoothed line to be scale-invariant
    df_target['smooth_slope'] = np.diff(np.log(df_target['smoothed_close']), prepend=np.nan)
    
    # 3. Define Classes
    conditions = [
        df_target['smooth_slope'] > threshold,
        df_target['smooth_slope'] < -threshold
    ]
    choices = [2, 0] # 2=UP, 0=DOWN
    df_target['target'] = np.select(conditions, choices, default=1) # 1=SIDEWAYS
    
    return df_target

# Apply Target Generation
# Adjust sigma and threshold to get a balanced distribution
SIGMA = 4  # Smoothing factor
THRESHOLD = 0.0002 # Slope threshold

df_labeled = create_target_labels(df, sigma=SIGMA, threshold=THRESHOLD)

print("Class Distribution:")
print(df_labeled['target'].value_counts(normalize=True))

# Visualization of Labels
def plot_labels(df, start_idx, end_idx):
    subset = df.iloc[start_idx:end_idx]
    plt.figure(figsize=(15, 7))
    
    # Plot Close Price
    plt.plot(subset.index, subset['close'], label='Close Price', color='black', alpha=0.6)
    
    # Plot Smoothed Target Line
    plt.plot(subset.index, subset['smoothed_close'], label='Smoothed Target (Oracle)', color='blue', linestyle='--', linewidth=2)
    
    # Color background based on Target
    # We iterate to fill regions (this is slow for large data, good for zoom)
    y_min, y_max = subset['close'].min(), subset['close'].max()
    
    for i in range(len(subset) - 1):
        idx = subset.index[i]
        next_idx = subset.index[i+1]
        target = subset['target'].iloc[i]
        
        color = 'gray'
        if target == 2: color = 'green'
        elif target == 0: color = 'red'
        
        plt.axvspan(i, i+1, color=color, alpha=0.2, lw=0)

    plt.title(f'Target Labeling (Sigma={SIGMA}, Thresh={THRESHOLD})')
    plt.legend()
    plt.show()

# Plot a zoomed-in section
plot_labels(df_labeled.reset_index(), 1000, 1200)

#%% [markdown]
# ## 3. HMM Integration: Modeling Market Regimes
#
# We use **Hidden Markov Models (HMM)** to identify latent market states (e.g., Low Volatility, Bull Trend, Bear Trend).
# We will train a Gaussian HMM with 3 states.
#
# **Features for HMM:**
# *   Log Returns (Price movement)
# *   Range (High - Low) / Close (Volatility)
# *   Volume (Market Activity)
#
# **Important:** We fit the HMM *only* on the training portion of the data to avoid data leakage, then predict states for the entire dataset.

#%%
# Prepare Data for HMM
df_hmm = df_labeled.copy()

# Feature Engineering for HMM
df_hmm['log_return'] = np.log(df_hmm['close'] / df_hmm['close'].shift(1))
df_hmm['range_volatility'] = (df_hmm['high'] - df_hmm['low']) / df_hmm['close']
df_hmm['log_volume'] = np.log(df_hmm['total_vol'] + 1)

df_hmm = df_hmm.dropna()

# Select features
hmm_features = ['log_return', 'range_volatility', 'log_volume']
X_hmm = df_hmm[hmm_features].values

# Scale features (RobustScaler is good for outliers)
scaler_hmm = RobustScaler()
X_hmm_scaled = scaler_hmm.fit_transform(X_hmm)

# Split for Training (Use first 70% to fit HMM)
train_size = int(len(X_hmm_scaled) * 0.7)
X_train_hmm = X_hmm_scaled[:train_size]

# Train Gaussian HMM
print("Training HMM...")
model_hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model_hmm.fit(X_train_hmm)

# Predict States for ALL data
hidden_states = model_hmm.predict(X_hmm_scaled)
state_probs = model_hmm.predict_proba(X_hmm_scaled)

# Add to DataFrame
df_hmm['hmm_state'] = hidden_states
for i in range(3):
    df_hmm[f'hmm_prob_{i}'] = state_probs[:, i]

print("HMM Training Complete.")
print("State Distribution:")
print(df_hmm['hmm_state'].value_counts(normalize=True))

# Visualization of HMM States
def plot_hmm_states(df, start_idx, end_idx):
    subset = df.iloc[start_idx:end_idx]
    plt.figure(figsize=(15, 7))
    
    # Plot Close Price
    plt.plot(subset.index, subset['close'], color='black', alpha=0.6, label='Close Price')
    
    # Color background by HMM State
    colors = ['red', 'green', 'blue']
    for i in range(len(subset) - 1):
        idx = subset.index[i]
        state = subset['hmm_state'].iloc[i]
        plt.axvspan(i, i+1, color=colors[state], alpha=0.2, lw=0)
        
    plt.title('HMM Market States (Red/Green/Blue)')
    plt.legend()
    plt.show()

# Visualize a section
plot_hmm_states(df_hmm.reset_index(), 1000, 1200)

#%% [markdown]
# ## 4. XGBoost Classifier Development
#
# We will now build the main prediction model.
#
# ### 4.1 Feature Engineering
# We will add a rich set of features:
# *   **Volume:** Delta (Ask - Bid), Cumulative Delta, Moving Averages of Volume.
# *   **Momentum:** RSI, MACD, EMA trends.
# *   **Volatility:** Bollinger Bands (Width, %B).
# *   **HMM Features:** The state probabilities we generated in the previous step.

#%%
def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bbands(series, length=20, std=2):
    ma = series.rolling(window=length).mean()
    std_dev = series.rolling(window=length).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return upper, ma, lower

def add_features(df):
    df = df.copy()
    
    # --- Volume Features ---
    df["volume_delta"] = df["ask_vol"] - df["bid_vol"]
    for window in [4, 12, 24, 96]:
        df[f"vol_delta_rolling_{window}"] = df["volume_delta"].rolling(window).sum()
    
    # --- Momentum Features ---
    df['RSI_14'] = calculate_rsi(df['close'], length=14)
    df['MACD_12_26_9'], df['MACDs_12_26_9'], df['MACDh_12_26_9'] = calculate_macd(df['close'])
    df['ema_12'] = calculate_ema(df['close'], length=12)
    df['ema_26'] = calculate_ema(df['close'], length=26)
    df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
    
    # --- Volatility Features ---
    df['BBU_20_2.0'], df['BBM_20_2.0'], df['BBL_20_2.0'] = calculate_bbands(df['close'])
    df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    df['BBP_20_2.0'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    return df

# Apply Feature Engineering
df_full = add_features(df_hmm)

# Define Feature Columns (Exclude Target and Future Data)
feature_cols = [
    'volume_delta', 'vol_delta_rolling_4', 'vol_delta_rolling_12', 'vol_delta_rolling_24', 'vol_delta_rolling_96',
    'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'trend_ema',
    'bb_width', 'BBP_20_2.0',
    'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2' # HMM Features
]

# Shift features by 1 to prevent look-ahead bias (predicting NEXT bar using CURRENT features)
df_full[feature_cols] = df_full[feature_cols].shift(1)
df_full = df_full.dropna()

print(f"Features: {len(feature_cols)}")
print(feature_cols)

#%% [markdown]
# ### 4.2 Grid Search & Training
# We will use `RandomizedSearchCV` (a more efficient version of Grid Search) to find optimal hyperparameters for XGBoost.
# We will perform 200 iterations.

#%%
# Prepare X and y
X = df_full[feature_cols]
y = df_full['target']

# Split Data (Time Series Split)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

# Scale Features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define Hyperparameter Grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5]
}

# Initialize XGBoost
xgb = XGBClassifier(
    objective='multi:softprob', 
    num_class=3, 
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist' # Faster training
)

# Randomized Search (200 iterations)
print("Starting Grid Search (200 iterations)...")
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

random_search.fit(X_train_scaled, y_train)

print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

#%% [markdown]
# ## 5. Evaluation & Visualization
#
# We evaluate the best model on the Test set and visualize the results.

#%%
# Evaluate on Test Set
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['DOWN', 'SIDEWAYS', 'UP']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'SIDEWAYS', 'UP'], yticklabels=['DOWN', 'SIDEWAYS', 'UP'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
importance = best_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title('Feature Importance')
plt.show()

# Final Prediction Visualization
def plot_predictions(df, start_idx, end_idx, y_true, y_pred):
    subset = df.iloc[start_idx:end_idx]
    subset_true = y_true.iloc[start_idx-len(X_train)-len(X_val):end_idx-len(X_train)-len(X_val)] # Adjust index for test set
    subset_pred = y_pred[start_idx-len(X_train)-len(X_val):end_idx-len(X_train)-len(X_val)]
    
    plt.figure(figsize=(15, 7))
    plt.plot(subset.index, subset['close'], color='black', alpha=0.6, label='Close Price')
    
    # Color background by Predicted Class
    colors = {0: 'red', 1: 'gray', 2: 'green'}
    for i in range(len(subset) - 1):
        pred = subset_pred[i]
        plt.axvspan(i, i+1, color=colors[pred], alpha=0.2, lw=0)
        
    plt.title('Model Predictions (Background Color)')
    plt.legend()
    plt.show()

# Visualize a section of the test set
# Note: Indices need to be aligned with the test set slice
test_start_idx = len(X_train) + len(X_val)
plot_predictions(df_full.reset_index(), test_start_idx, test_start_idx + 200, y_test, y_pred)