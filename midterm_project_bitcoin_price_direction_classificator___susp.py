#%% md
# # # Bitcoin Price Direction Prediction Project
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
# 
#%%
!conda install -c conda-forge hmmlearn -y
#%%
!conda install -c conda-forge statsmodels -y
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
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

#%% md
# # ## 1. Data Analysis: Stationarity & Regression Failure
# #
# # ### 1.1 Stationarity Check
# # Financial time series like Bitcoin prices are typically non-stationary (mean and variance change over time), making them unsuitable for direct prediction using many statistical models. Log-returns are usually stationary.
# #
# # We will use the **Augmented Dickey-Fuller (ADF)** test to statistically prove this.
# 
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

#%% md
# # ### 1.2 Why Regression Fails
# # We will now train a simple XGBoost Regressor to predict the *next bar's log-return* using a small subset of data.
# # We expect the $R^2$ score to be close to zero (or negative), indicating that the model cannot predict the magnitude of price changes better than a simple mean baseline.
# 
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

#%% md
# ## 2. Target Generation: The "Oracle" Labeling
# 
# Since regression is difficult, we switch to **Classification** (Up/Down/Sideways).
# To create high-quality labels, we use a **Centered Moving Average (Gaussian Smoothing)** on the *entire* dataset.
# 
# **Why?**
# 
#  This creates a smooth trend line that uses future data (zero lag). This is valid for *target generation* because we want to teach the model what *actually happened*. We must ensure we **never** use this smoothed line as a feature.
# 
# **Classes:**
# 
#  *   **2 (UP):** Slope > Threshold
#  *   **0 (DOWN):** Slope < -Threshold
#  *   **1 (SIDEWAYS):** |Slope| <= Threshold
# 
#%%
def create_target_labels(df, sigma=1, threshold=0.0005):
    """
    Creates target labels using Gaussian smoothing (Centered MA).
    sigma: Controls smoothness (higher = smoother trend)
           Higher sigma => stronger smoothing, slower reaction to noise.
    threshold: Slope threshold for Up/Down classification
               If absolute slope is smaller than this value, we call it SIDEWAYS.
    """
    # Work on a copy to avoid modifying the original dataframe in-place
    df_target = df.copy()

    # 1. Apply Gaussian Filter (Centered Smoothing)
    # This uses the entire series (future and past) to smooth.
    # In practice this is a non-causal filter (uses future information),
    # so it is an "oracle" target, good for labeling but not for live trading.
    df_target['smoothed_close'] = gaussian_filter1d(
        df_target['close'],
        sigma=sigma
    )

    # 2. Calculate Slope (Derivative) of the smoothed line
    # We use log returns of the smoothed line to be scale-invariant:
    #    smooth_slope[t] ≈ log(S_t) - log(S_{t-1})
    # That means the slope is in relative units (percentage-ish),
    # not in raw price units.
    df_target['smooth_slope'] = np.diff(
        np.log(df_target['smoothed_close']),
        prepend=np.nan     # keep the same length as the original series
    )

    # 3. Define Classes
    # If slope is greater than +threshold  -> strong UP move (class 2).
    # If slope is smaller than -threshold  -> strong DOWN move (class 0).
    # Otherwise                             -> SIDEWAYS / neutral (class 1).
    conditions = [
        df_target['smooth_slope'] > threshold,
        df_target['smooth_slope'] < -threshold
    ]
    choices = [2, 0]  # 2=UP, 0=DOWN
    df_target['target'] = np.select(
        conditions,
        choices,
        default=1          # 1=SIDEWAYS
    )

    return df_target


# Hyperparameters for target generation.
# SIGMA controls how smooth the "oracle" line is.
# THRESHOLD controls how sensitive the labeling is to small slopes.
SIGMA = 4          # Smoothing factor (higher => smoother, fewer class flips)
THRESHOLD = 0.0004 # Slope threshold in log-return units

# Apply Target Generation
# Adjust sigma and threshold to get a balanced distribution of classes.
df_labeled = create_target_labels(df, sigma=SIGMA, threshold=THRESHOLD)

print("Class Distribution:")
print(df_labeled['target'].value_counts(normalize=True))
#%%


# Visualization of Labels
def plot_labels(df, start_idx, end_idx):
    """
    Plot a zoomed-in segment of the series with:
      - Raw close price
      - Smoothed 'oracle' price
      - Colored background according to the discrete target labels

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that already contains 'close', 'smoothed_close' and 'target'.
    start_idx : int
        Integer position (iloc) where the visible window starts.
    end_idx : int
        Integer position (iloc) where the visible window ends (exclusive).
    """
    # Select a window by integer position to zoom into a local region
    subset = df.iloc[start_idx:end_idx]

    plt.figure(figsize=(15, 7))

    # Use the subset index as X-axis values.
    # If df has a DatetimeIndex, this will be time;
    # if it is a simple RangeIndex, these will be integer positions.
    x = subset.index

    # Plot Close Price (raw data)
    plt.plot(
        x,
        subset['close'],
        label='Close Price',
        color='black',
        alpha=0.6
    )

    # Plot Smoothed Target Line (oracle, uses future information)
    plt.plot(
        x,
        subset['smoothed_close'],
        label='Smoothed Target (Oracle)',
        color='blue',
        linestyle='--',
        linewidth=2
    )

    # Color background based on Target.
    # We fill vertical bands between x[i] and x[i+1] with a color
    # that depends on the target at time i:
    #   green = UP, red = DOWN, gray = SIDEWAYS.
    #
    # Important: we use x[i] / x[i+1] here (data coordinates),
    # not just i / i+1. This keeps the background aligned with the plot.
    for i in range(len(subset) - 1):
        x0 = x[i]
        x1 = x[i + 1]
        target = subset['target'].iloc[i]

        # Default color for SIDEWAYS
        color = 'gray'
        if target == 2:
            color = 'green'
        elif target == 0:
            color = 'red'

        # axvspan draws a vertical rectangle from x0 to x1 across the full y-range
        # (ymin=0, ymax=1 are in axis coordinates, so it always spans the full height).
        plt.axvspan(
            x0,
            x1,
            color=color,
            alpha=0.2,
            lw=0
        )

    plt.title(f'Target Labeling (Sigma={SIGMA}, Thresh={THRESHOLD})')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plot a zoomed-in section.
# Note: we pass df_labeled directly (without reset_index),
# so that indices used for plotting and for axvspan stay consistent.
plot_labels(df_labeled, 1000, 1200)

#%% md
# # ## 3. HMM Integration: Modeling Market Regimes
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
# 
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
#%%

# Visualization of HMM States
def plot_hmm_states(df, start_idx, end_idx):
    """
    Plot price, smoothed oracle line, and background colored by HMM states.
    """
    subset = df.iloc[start_idx:end_idx]

    plt.figure(figsize=(15, 7))

    # X-axis = real index (time or integer positions)
    x = subset.index

    # Price and oracle line (можно так же, как в plot_labels)
    plt.plot(x, subset['close'],
             label='Close Price',
             color='black',
             alpha=0.6)
    plt.plot(x, subset['smoothed_close'],
             label='Smoothed Target (Oracle)',
             color='blue',
             linestyle='--',
             linewidth=2)

    # Background by HMM state instead of target
    for i in range(len(subset) - 1):
        x0 = x[i]
        x1 = x[i+1]
        state = subset['hmm_state'].iloc[i]

        # Map hidden states to colors (пример: 0,1,2 -> три разных цвета)
        if state == 0:
            color = 'grey'   # e.g. low-vol regime
        elif state == 1:
            color = 'green'   # e.g. medium-vol regime
        else:
            color = 'red'      # e.g. high-vol regime

        plt.axvspan(x0, x1, color=color, alpha=0.2, lw=0)

    plt.title('HMM Hidden States (Background Colors)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize a section
plot_hmm_states(df_hmm.reset_index(), 1000, 1200)
#%% md
# # ## 4. XGBoost Classifier Development
# 
# We will now build the main prediction model.
# 
# ### 4.1 Feature Engineering
# 
# We will add a rich set of features:
# *   **Volume:** Delta (Ask - Bid), Cumulative Delta, Moving Averages of Volume.
# *   **Momentum:** RSI, MACD, EMA trends.
# *   **Volatility:** Bollinger Bands (Width, %B).
# *   **HMM Features:** The state probabilities we generated in the previous step.
# 
#%%
df.ta.bbands(length=20, std=2, append=True)
df.columns
#%%
def add_features(df):
    df = df.copy()
    
    # --- Volume Features ---
    df["volume_delta"] = df["ask_vol"] - df["bid_vol"]
    for window in [4, 12, 24, 96]:
        df[f"vol_delta_rolling_{window}"] = df["volume_delta"].rolling(window).sum()
    
    # --- Momentum Features ---
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)
    df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
    
    # --- Volatility Features ---
    df.ta.bbands(length=20, std=2, append=True)
    df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    
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

#%%
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
    tree_method='hist', # Faster training
    max_depth=2,
    n_estimators=15
)

# Randomized Search (200 iterations)
print("Starting Grid Search (200 iterations)...")
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_weighted',
    cv=TimeSeriesSplit(n_splits=3),
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

print("Best Parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

#%%
# ## 5. Evaluation & Visualization
#
# We evaluate the best model on the Test set and visualize the results.

#%%
# ============================
# 1. Evaluate model on Test Set
# ============================

# Predicted class labels for the test set.
# Each element is in {0, 1, 2} corresponding to ['DOWN', 'SIDEWAYS', 'UP'].
y_pred = best_model.predict(X_test_scaled)

# Predicted class probabilities for the test set.
# Shape: (n_samples_test, 3). Each row sums to 1.0.
y_pred_proba = best_model.predict_proba(X_test_scaled)

print("\nClassification Report:")
# classification_report shows precision, recall, f1-score and support per class.
# target_names are simply human-readable names for classes 0,1,2.
print(classification_report(
    y_test,
    y_pred,
    target_names=['DOWN', 'SIDEWAYS', 'UP']
))
#%%

# ============================
# 2. Confusion Matrix
# ============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,                # show counts inside the cells
    fmt='d',                   # integer format
    cmap='Blues',
    xticklabels=['DOWN', 'SIDEWAYS', 'UP'],  # predicted labels
    yticklabels=['DOWN', 'SIDEWAYS', 'UP']   # true labels
)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()
#%% md
# 
#%%

# ============================
# 3. Feature Importance
# ============================

# XGBoost provides feature_importances_ as the gain-based importance
# for each input feature used by the model.
importance = best_model.feature_importances_

# Build a DataFrame to sort and visualize feature importances.
feat_imp = (
    pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    .sort_values('Importance', ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feat_imp,
    orient='h'
)
plt.title('Feature Importance (XGBoost Gain)')
plt.tight_layout()
plt.show()
#%%

# ============================
# 4. Attach predictions back to the full DataFrame
#    so that plotting is simple and index-safe.
# ============================

# Create empty columns for true and predicted labels in the full df.
# We will fill only the test segment.
df_full['y_true'] = np.nan
df_full['y_pred'] = np.nan

# y_test already has the correct index for the test segment (last part of df_full),
# because X_train / X_val / X_test were created from df_full WITHOUT shuffling.
test_index = y_test.index

# Write true and predicted labels into df_full aligned by index.
df_full.loc[test_index, 'y_true'] = y_test
df_full.loc[test_index, 'y_pred'] = y_pred
#%%

# ============================
# 5. Final Prediction Visualization
# ============================
def plot_predictions(df, y_test, y_pred, X_train, X_val, window_size=200):
    """
    Plot model predictions on the test segment of a time series as
    colored background behind the price.

    This function:
      1. Does NOT modify the original dataframe `df` (works on a copy).
      2. Automatically finds the test segment boundaries based on
         the lengths of X_train, X_val, X_test.
      3. Uses the 'time' column (if present) as the X-axis (dates),
         otherwise uses the existing index.
      4. Colors each bar interval according to the predicted class:
           0 = DOWN  (red)
           1 = SIDEWAYS (gray)
           2 = UP (green)

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataframe that contains at least:
          - 'close' : price series
          - 'time'  : datetime column (optional but recommended).
        This function will NOT modify `df` in-place.

    y_test : pandas.Series or np.ndarray
        True labels for the test set. Length must match X_test.

    y_pred : np.ndarray or list-like
        Predicted class labels for the test set, aligned with y_test.

    X_train : pandas.DataFrame or np.ndarray
        Training features. Only the length is used to locate the test segment.

    X_val : pandas.DataFrame or np.ndarray
        Validation features. Only the length is used to locate the test segment.

    window_size : int, optional (default=200)
        Number of test bars to display in the plot starting from the
        beginning of the test segment.
    """
    # ------------------------------------------------------------------
    # 0. Work on a copy so that the original df is never changed.
    # ------------------------------------------------------------------
    df_plot = df.copy()

    # ------------------------------------------------------------------
    # 1. Prepare time index for plotting
    # ------------------------------------------------------------------
    # If there is a 'time' column, use it as a DatetimeIndex for the copy.
    # This allows the X-axis to show actual timestamps.
    if 'time' in df_plot.columns:
        # Ensure 'time' is datetime type
        if not isinstance(df_plot['time'].dtype, pd.DatetimeTZDtype) \
           and not np.issubdtype(df_plot['time'].dtype, np.datetime64):
            df_plot['time'] = pd.to_datetime(df_plot['time'])

        df_plot = df_plot.set_index('time')
        df_plot = df_plot.sort_index()   # make sure rows are in chronological order

    # At this point, df_plot.index is either:
    #   - DatetimeIndex (if 'time' existed), or
    #   - whatever index df had originally (RangeIndex, etc.).

    # ------------------------------------------------------------------
    # 2. Locate the test segment in df_plot by position
    # ------------------------------------------------------------------
    n_total = len(df_plot)
    n_train = len(X_train)
    n_val   = len(X_val)
    n_test  = len(y_test)

    # Sanity check: n_train + n_val + n_test should not exceed n_total.
    if n_train + n_val + n_test > n_total:
        raise ValueError(
            f"Lengths inconsistent: "
            f"n_train({n_train}) + n_val({n_val}) + n_test({n_test}) > n_total({n_total})."
        )

    # Assume that train, val, test are consecutive chunks at the END of df_plot
    # after all preprocessing (rolling, shift, dropna, etc.).
    # Then the test segment corresponds to the last n_test rows of df_plot.
    test_start_pos = n_total - n_test        # integer position where test starts
    test_end_pos   = n_total                 # integer position where test ends (exclusive)

    # ------------------------------------------------------------------
    # 3. Build a small helper DataFrame only for the test segment
    # ------------------------------------------------------------------
    # We create a view of df_plot restricted to the test range and
    # attach y_test and y_pred to it, without touching the original df.
    test_slice = df_plot.iloc[test_start_pos:test_end_pos].copy()

    # Ensure y_test is an array-like aligned by order.
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    if len(test_slice) != len(y_test_arr):
        raise ValueError(
            f"Test slice length ({len(test_slice)}) != len(y_test) ({len(y_test_arr)}). "
            f"Check how you build X_train/X_val/X_test."
        )

    test_slice['y_true'] = y_test_arr
    test_slice['y_pred'] = y_pred_arr

    # ------------------------------------------------------------------
    # 4. Choose the window inside the test slice to visualize
    # ------------------------------------------------------------------
    # We start from the beginning of the test slice and take `window_size` bars.
    # plot_start = 0
    # plot_end   = min(window_size, len(test_slice))
    n_test_rows = len(test_slice)

    plot_start = n_test_rows - window_size
    plot_end   = n_test_rows

    subset = test_slice.iloc[plot_start:plot_end]

    if subset['y_pred'].isna().all():
        print("No predictions available in the selected window.")
        return

    # ------------------------------------------------------------------
    # 5. Plot: price line + background colored by predictions
    # ------------------------------------------------------------------
    plt.figure(figsize=(15, 7))

    # X-axis values: index of subset (dates if DatetimeIndex is used).
    x = subset.index

    # Plot the close price.
    plt.plot(
        x,
        subset['close'],
        color='black',
        alpha=0.6,
        label='Close Price'
    )

    # Class -> color mapping
    colors = {0: 'red', 1: 'gray', 2: 'green'}

    # Fill each interval [x[i], x[i+1]) with the color of predicted class at i.
    # Using x[i]/x[i+1] (data coordinates) ensures perfect alignment with the price.
    for i in range(len(subset) - 1):
        x0 = x[i]
        x1 = x[i + 1]
        pred = subset['y_pred'].iloc[i]

        if np.isnan(pred):
            continue

        plt.axvspan(
            x0,
            x1,
            color=colors.get(int(pred), 'gray'),
            alpha=0.2,
            lw=0
        )

    plt.title('Model Predictions on Test Segment (Background Colors)')
    plt.legend()
    plt.xticks(rotation=45)   # rotate datetime labels for readability
    plt.tight_layout()
    plt.show()
plot_predictions(df_full, y_test, y_pred, X_train, X_val, window_size=200)
#%%
def evaluate_model_performance(model, X_test, y_test):
  """
  Оценивает производительность модели и выводит отчет о классификации.

  Аргументы:
      model: Обученная модель для оценки.
      X_test: Массив признаков для тестового набора данных.
      y_test: Истинные метки для тестового набора данных.
  """
  # Получение прогнозов для тестового набора
  y_pred = model.predict(X_test)

  # Вывод отчета о классификации
  print("\nОтчет о классификации:")
  print(classification_report(
      y_test,
      y_pred,
      target_names=['DOWN', 'SIDEWAYS', 'UP']
  ))

# Пример использования:
evaluate_model_performance(best_model, X_test_scaled, y_test)
#%% md
# ## Brootforce search for optimal indicators combination ##
#%%
import pandas as pd
import pandas_ta as ta

from itertools import combinations

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES_TO_REMOVE = ['target', 'time', 'smooth_slope', 'smoothed_close']






# ==============================
#   ИГРА С КОЛИЧЕСТВОМ ТОП‑ФИЧ
# ==============================

def rerun_with_top_k(df_labeled, base_groups, fi_df, k_list,
                     test_size=0.2, n_estimators=25, random_state=42):

    '''Эта функция rerun_with_top_k выполняет отбор признаков (Feature Selection).

Её задача — проверить, можно ли улучшить точность модели, если выкинуть "мусорные" признаки и оставить только самые важные (Топ-10, Топ-20 и т.д.).

Как она работает (пошагово):
Принимает на вход:

base_groups: набор групп индикаторов, который показал лучший результат на предыдущем этапе (например, ['momentum', 'trend']).
fi_df: таблица важности признаков (Feature Importance), полученная из лучшей модели.
k_list: список чисел (например, [10, 20, 30]), сколько лучших признаков оставлять.
Цикл по k (количеству признаков):

Берет названия топ-k самых важных признаков из fi_df.
Заново формирует датафрейм, применяя индикаторы из base_groups.
УДАЛЯЕТ все колонки, кроме этих топ-k признаков и служебных колонок (target, time).
Делает сдвиг shift(1) (чтобы предсказывать будущее по прошлому).
Обучает модель заново на этом урезанном наборе данных.
Записывает точность (accuracy).
Результат: Выводит таблицу, где видно, при каком количестве признаков точность максимальна. Часто бывает, что 20 лучших признаков работают лучше, чем все 100, так как убирается шум.'''

    out = []
    print(f"[rerun_with_top_k] base_groups={base_groups}, k_list={k_list}")

    # 1. Сначала готовим ПОЛНЫЙ датасет с нужными группами (ОДИН РАЗ)
    # Очищаем от старых индикаторов, оставляем только базу
    base_cols = ['time', 'open', 'high', 'low', 'close',  'total_vol', 'target']
    cols_to_keep = [c for c in base_cols if c in df_labeled.columns]
    df_full = df_labeled[cols_to_keep].copy()

    # Применяем группы индикаторов
    for g in base_groups:
        if g in GROUP_FUNCS:
            print(f"[rerun_with_top_k] apply group: {g}")
            df_full = GROUP_FUNCS[g](df_full)

    # 2. Теперь просто фильтруем колонки в цикле
    for k in k_list:
        # Берем имена топ-K признаков
        top_feats = fi_df.head(k)["Feature"].tolist()
        print(f"[rerun_with_top_k] start k={k}, top_feats_count={len(top_feats)}")

        # Копируем уже готовый датасет с индикаторами
        df = df_full.copy()

        # Оставляем только топ-K + служебные
        # Важно: FEATURES_TO_REMOVE должны быть защищены от удаления
        cols_to_retain = set(top_feats + FEATURES_TO_REMOVE)

        # Удаляем всё лишнее
        drop_cols = [c for c in df.columns if c not in cols_to_retain]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Формируем список фичей для обучения (все что осталось, кроме служебных)
        features = [c for c in df.columns if c not in FEATURES_TO_REMOVE]

        # Сдвиг лага
        df[features] = df[features].shift(1)
        df = df.dropna()

        X = df[features]
        y = df["target"]

        # Обучение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        model = XGBClassifier(
            device='cuda',
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric='mlogloss',
            max_depth=4
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        out.append({"k": k, "accuracy": acc})
        print(f"[rerun_with_top_k] done k={k}, acc={acc:.4f}")

    summary = pd.DataFrame(out).sort_values("accuracy", ascending=False)
    print("[rerun_with_top_k] finished, best:")
    print(summary.head(10))
    return summary
#%%
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#%%

# ==================================================================
#   ОБЁРТКИ НАД КАТЕГОРИЯМИ pandas-ta
# ==================================================================

def add_momentum_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Основные осцилляторы
    df.ta.rsi(length=14, append=True)
    df.ta.roc(length=12, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.cci(length=14, append=True)       # Commodity Channel Index
    df.ta.willr(length=14, append=True)     # Williams %R
    df.ta.ao(append=True)                   # Awesome Oscillator
    df.ta.mom(length=10, append=True)       # Momentum
    df.ta.tsi(length_fast=13, length_slow=25, append=True) # True Strength Index
    df.ta.uo(append=True)                   # Ultimate Oscillator
    return df

def add_overlap_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Скользящие средние разных типов
    df.ta.ema(length=10, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.hma(length=9, append=True)        # Hull Moving Average
    df.ta.tema(length=9, append=True)       # Triple EMA

    # Трендовые наложения
    df.ta.psar(append=True)                 # Parabolic SAR
    df.ta.supertrend(length=7, multiplier=3, append=True) # Supertrend

    # VWAP (требует объем)
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    if vol_col in df.columns:
        df.ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True)
    return df

def add_trend_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.aroon(length=14, append=True)
    df.ta.vortex(length=14, append=True)    # Vortex Indicator
    df.ta.dpo(length=20, append=True)       # Detrended Price Oscillator
    df.ta.trix(length=30, append=True)      # TRIX
    df.ta.cksp(append=True)                 # Chande Kroll Stop
    return df

def add_volatility_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.atr(length=14, append=True)
    df.ta.natr(length=14, append=True)      # Normalized ATR
    df.ta.ui(length=14, append=True)        # Ulcer Index

    # Bollinger Bands (берем %B и ширину)
    bbands = df.ta.bbands(length=20, append=False)
    if bbands is not None and not bbands.empty:
        # Ищем колонки, начинающиеся с BBP (Percentage) и BBB (Bandwidth)
        bbp_cols = [c for c in bbands.columns if c.startswith('BBP')]
        bbb_cols = [c for c in bbands.columns if c.startswith('BBB')]

        if bbp_cols:
            df[bbp_cols[0]] = bbands[bbp_cols[0]]
        if bbb_cols:
            df[bbb_cols[0]] = bbands[bbb_cols[0]]

    # Keltner Channels (безопасное добавление)
    kc = df.ta.kc(append=False)
    if kc is not None and not kc.empty:
        # Пытаемся найти колонку KCP (Percentage)
        kcp_cols = [c for c in kc.columns if c.startswith('KCP')]
        if kcp_cols:
            df[kcp_cols[0]] = kc[kcp_cols[0]]

    # Donchian Channels (безопасное добавление)
    dc = df.ta.donchian(append=False)
    if dc is not None and not dc.empty:
        # Пытаемся найти колонку DCP (Percentage)
        dcp_cols = [c for c in dc.columns if c.startswith('DCP')]
        if dcp_cols:
            df[dcp_cols[0]] = dc[dcp_cols[0]]

    return df


def add_volume_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    if vol_col in df.columns:
        df.ta.obv(close=df['close'], volume=df[vol_col], append=True)
        df.ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], length=14, append=True)

        # ИСПРАВЛЕНО: метод называется .ad(), а не .adl()
        df.ta.ad(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True)

        df.ta.cmf(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True) # Chaikin Money Flow
        df.ta.eom(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True) # Ease of Movement
        df.ta.nvi(close=df['close'], volume=df[vol_col], append=True) # Negative Volume Index
        df.ta.pvi(close=df['close'], volume=df[vol_col], append=True) # Positive Volume Index
    return df


def add_statistics_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.zscore(length=30, append=True)
    df.ta.entropy(length=30, append=True)
    df.ta.kurtosis(length=30, append=True)
    df.ta.skew(length=30, append=True)
    df.ta.variance(length=30, append=True)
    df.ta.mad(length=30, append=True) # Mean Absolute Deviation
    return df

def add_candle_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасно добавляет расширенный набор свечных паттернов.
    """
    df = df.copy()

    # Расширенный список паттернов
    patterns_to_add = [
        'cdl_doji', 'cdl_hammer', 'cdl_engulfing',
        'cdl_morningstar', 'cdl_eveningstar',
        'cdl_shootingstar', 'cdl_hangingman',
        'cdl_marubozu', 'cdl_3whitesoldiers', 'cdl_3blackcrows',
        'cdl_inside', 'cdl_spinningtop'
    ]

    print("  -> Applying extended candle patterns...")
    for pattern_name in patterns_to_add:
        if hasattr(df.ta, pattern_name):
            getattr(df.ta, pattern_name)(append=True)
        else:
            # Тихо пропускаем или можно раскомментировать print для отладки
            # print(f"  -> WARNING: Candle pattern '{pattern_name}' not found.")
            pass

    return df


# ==============================
#   КОНФИГ: КАРТА ГРУПП
# ==============================

GROUP_FUNCS = {
    "momentum": add_momentum_category,
    "overlap": add_overlap_category,
    "trend": add_trend_category,
    "volatility": add_volatility_category,
    "volume": add_volume_category,
    "statistics": add_statistics_category,
    "candle": add_candle_category,
}

#%%
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


#%%
# ==================================================================
#   ОПТИМИЗИРОВАННЫЙ ПАЙПЛАЙН (СЧИТАЕМ 1 РАЗ -> ПЕРЕБИРАЕМ БЫСТРО)
# ==================================================================

def prepare_all_features(df_labeled: pd.DataFrame):
    """
    1. Приводит индекс к DatetimeIndex.
    2. Считает ВСЕ группы индикаторов сразу.
    3. Удаляет колонки, состоящие полностью из NaN.
    4. Возвращает полный датафрейм и карту {группа: [список_колонок]}.
    """
    print("[prepare_all_features] Pre-calculating ALL indicators...")
    df_all = df_labeled.copy()

    # 1. Исправляем индекс сразу (критично для VWAP)
    if 'time' in df_all.columns and not isinstance(df_all.index, pd.DatetimeIndex):
        df_all['time'] = pd.to_datetime(df_all['time'])
        df_all.set_index('time', inplace=True)

    # Запоминаем базовые колонки (чтобы случайно не удалить)
    base_cols = set(df_all.columns)
    group_features_map = {}

    # 2. Считаем все группы
    for g_name, g_func in GROUP_FUNCS.items():
        print(f"  -> Processing group: {g_name}")
        cols_before = set(df_all.columns)

        try:
            df_all = g_func(df_all)
        except Exception as e:
            print(f"    !!! CRITICAL ERROR in group '{g_name}': {e}")
            continue

        cols_after = set(df_all.columns)
        new_cols = list(cols_after - cols_before)
        group_features_map[g_name] = new_cols
        print(f"    -> Added {len(new_cols)} features.")

    # 3. Удаляем "битые" индикаторы (полностью NaN)
    nan_cols = df_all.columns[df_all.isna().all()].tolist()
    if nan_cols:
        print(f"[WARNING] Dropping {len(nan_cols)} columns that are 100% NaN: {nan_cols}")
        df_all.drop(columns=nan_cols, inplace=True)
        # Чистим карту групп
        for g in group_features_map:
            group_features_map[g] = [c for c in group_features_map[g] if c not in nan_cols]

    print(f"[prepare_all_features] Done. Total columns: {df_all.shape[1]}")
    return df_all, group_features_map


def run_fast_experiment(df_all, group_features_map, active_groups,
                        test_size=0.2, n_estimators=25, random_state=42):
    """
    Быстрый эксперимент: просто берет готовые колонки из df_all.
    """
    # 1. Собираем список нужных колонок индикаторов
    selected_indicator_cols = []
    for g in active_groups:
        selected_indicator_cols.extend(group_features_map.get(g, []))

    # 2. Добавляем базовые фичи (которые не в списке на удаление)
    #    Обычно это open, high, low, close, volume...
    all_cols = list(df_all.columns)
    # Оставляем те, которые (выбраны нами) ИЛИ (не являются индикаторами других групп И не в черном списке)
    # Проще: берем (базовые - remove) + (выбранные индикаторы)

    # Определяем базовые колонки (те, что были до индикаторов, примерно)
    # Но проще сделать так: берем всё, выкидываем индикаторы "чужих" групп

    # Все известные индикаторные колонки всех групп
    all_indicator_cols = set()
    for cols in group_features_map.values():
        all_indicator_cols.update(cols)

    # Колонки, которые не являются индикаторами (базовые цены)
    base_features = [c for c in all_cols if c not in all_indicator_cols and c not in FEATURES_TO_REMOVE]

    # Итоговый список фичей для этого эксперимента
    final_features = base_features + selected_indicator_cols

    # 3. Формируем датасет (копия среза - это быстро)
    # Добавляем target, если его нет в списке
    cols_to_take = list(set(final_features + ['target']))
    df_exp = df_all[cols_to_take].copy()

    # 4. Сдвиг фичей (Shift)
    # Сдвигаем всё, кроме таргета
    feat_cols = [c for c in df_exp.columns if c != 'target']
    df_exp[feat_cols] = df_exp[feat_cols].shift(1)

    # 5. Dropna
    rows_before = len(df_exp)
    df_exp.dropna(inplace=True)
    rows_after = len(df_exp)

    if df_exp.empty:
        return {'groups': active_groups, 'accuracy': 0, 'report': f'Empty after dropna (was {rows_before})', 'n_features': 0}

    # 6. Обучение
    X = df_exp[feat_cols]
    y = df_exp['target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    if len(X_train) == 0 or len(X_test) == 0:
         return {'groups': active_groups, 'accuracy': 0, 'report': 'Train/Test empty', 'n_features': 0}

    model = XGBClassifier(
        device='cuda',
        n_estimators=n_estimators,
        random_state=random_state,
        eval_metric='mlogloss',
        max_depth=4
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Важность признаков (опционально, можно отключить для скорости)
    fi_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return {
        'groups': active_groups,
        'accuracy': acc,
        'n_features': len(feat_cols),
        'feature_importance': fi_df
    }

def iterate_group_combos_fast(df_labeled, max_group_size=4):
    # 1. Считаем всё один раз
    df_all, group_features_map = prepare_all_features(df_labeled)

    group_names = list(group_features_map.keys())
    combo_list = []
    for r in range(1, max_group_size + 1):
        for combo in combinations(group_names, r):
            combo_list.append(list(combo))

    total_runs = len(combo_list)
    print(f"\n[iterate_group_combos_fast] Starting {total_runs} experiments...")

    results = []
    for idx, groups in enumerate(combo_list, start=1):
        res = run_fast_experiment(df_all, group_features_map, groups)

        if res['accuracy'] > 0:
            print(f"Run {idx}/{total_runs} | {groups} | Acc: {res['accuracy']:.4f} | Feats: {res['n_features']}")
        else:
            print(f"Run {idx}/{total_runs} | {groups} | SKIPPED: {res.get('report')}")

        results.append(res)

    summary = pd.DataFrame([
        {
            "groups": "+".join(r["groups"]),
            "accuracy": r["accuracy"],
            "n_features": r.get("n_features", 0)
        }
        for r in results
    ]).sort_values("accuracy", ascending=False)

    return summary, results

# ЗАПУСК
summary, results = iterate_group_combos_fast(df_labeled, max_group_size=6)

#%%

# 2) Лучший результат
best_res = max(results, key=lambda r: r["accuracy"])
fi = best_res["feature_importance"]
base_groups = best_res["groups"]
#%%

# 3) Играть количеством топ‑фич
topk_summary = rerun_with_top_k(
    df_labeled,
    base_groups=base_groups,
    fi_df=fi,
    k_list=[10, 20, 30, 40, 60, 80])


#%%
import json
import joblib

# 1. Извлекаем имена лучших 40 фичей из вашего dataframe важности (fi)
# fi - это переменная из предыдущего шага (best_res["feature_importance"])
best_k = 40
top_features = fi.head(best_k)["Feature"].tolist()

print(f"Selected {len(top_features)} features.")
print(top_features[:10], "...") # Показать первые 10

# 2. Подготовка финального датасета
# Берем df_all (который мы посчитали в iterate_group_combos_fast)
# Если переменная df_all потерялась, раскомментируйте строку ниже:
df_all, _ = prepare_all_features(df_labeled)

final_df = df_all.copy()

# Оставляем только топ-40 + target
cols_to_keep = top_features + ['target']
final_df = final_df[cols_to_keep]

# ВАЖНО: Повторяем сдвиг (Shift), так как мы обучаем заново
features_only = [c for c in final_df.columns if c != 'target']
final_df[features_only] = final_df[features_only].shift(1)
final_df.dropna(inplace=True)

# 3. Обучение финальной модели
X = final_df[top_features]
y = final_df['target'].astype(int)

# Разбиваем (или можно обучить на всем датасете для продакшена, но лучше проверить еще раз)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

final_model = XGBClassifier(
    device='cuda',
    n_estimators=100,  # Увеличили для финального качества
    random_state=42,
    eval_metric='mlogloss',
    max_depth=4
)
final_model.fit(X_train, y_train)

# Проверка
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Сохранение (Artifacts)

# Сохраняем саму модель
final_model.save_model("best_xgb_model.json")

# Сохраняем СПИСОК ФИЧЕЙ (это критически важно!)
with open("best_features_list.json", "w") as f:
    json.dump(top_features, f)

print("\n✅ Модель сохранена в 'best_xgb_model.json'")
print("✅ Список фичей сохранен в 'best_features_list.json'")
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# 1. Настройка имен классов
# 0 = Down, 1 = Sideways, 2 = Up
class_names_map = {0: 'Down', 1: 'Sideways', 2: 'Up'}

# Получаем уникальные классы из данных
unique_classes = sorted(list(set(y_test) | set(y_pred)))
target_names = [class_names_map.get(c, f"Class {c}") for c in unique_classes]

print(f"Classes detected: {unique_classes} -> {target_names}")

# 2. Текстовый отчет
print("="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=target_names))

# 3. Графики
fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Оставим 2 графика: Matrix и ROC
plt.subplots_adjust(wspace=0.3)

# --- A. Confusion Matrix (Матрица ошибок) ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 14})
axes[0].set_title('Confusion Matrix', fontsize=18)
axes[0].set_xlabel('Predicted Label', fontsize=14)
axes[0].set_ylabel('True Label', fontsize=14)
axes[0].set_xticklabels(target_names, fontsize=12)
axes[0].set_yticklabels(target_names, fontsize=12, rotation=0)

# --- B. ROC Curve (Multi-class) ---
# Для мультикласса строим One-vs-Rest ROC
y_prob = final_model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=unique_classes)

colors = ['red', 'gray', 'green'] # Red=Down, Gray=Sideways, Green=Up
for i, cls in enumerate(unique_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    label_name = class_names_map.get(cls, f"Class {cls}")
    
    # Выбираем цвет (если классов больше 3, будет циклично)
    color = colors[i] if i < len(colors) else None
    
    axes[1].plot(fpr, tpr, color=color, lw=3, label=f'{label_name} (AUC = {roc_auc:.2f})')

axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=14)
axes[1].set_ylabel('True Positive Rate', fontsize=14)
axes[1].set_title('ROC Curve (One-vs-Rest)', fontsize=18)
axes[1].legend(loc="lower right", fontsize=12)
axes[1].grid(alpha=0.3)

plt.show()
