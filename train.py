import os
# Workaround for Windows OpenMP/scikit-learn conflict
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import gaussian_filter1d
import joblib
import warnings
import json
from itertools import combinations

# --- Configuration ---
DATA_PATH_FUNDINGS = "data/processed/fundings.parquet"
DATA_PATH_KLINES = "data/processed/klines_15min_all.parquet"
DATA_PATH_VOLUMES = "data/processed/aggtrades_15min_all.parquet"
MODEL_DIR = "models"
MODEL_FILENAME = "best_xgb_model.joblib" # Changed name to reflect it's the best model
METRICS_FILENAME = "metrics.json"
BEST_FEATURES_FILENAME = "best_features_list.json" # Save list of best features

# Hyperparameters for Target Generation
SIGMA = 4
THRESHOLD = 0.0004

# Features to exclude from training
FEATURES_TO_REMOVE = ['target', 'time', 'smooth_slope', 'smoothed_close']

warnings.filterwarnings('ignore')

# ==================================================================
#   DATA LOADING & PREPROCESSING
# ==================================================================

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
    """Creates target labels using Gaussian smoothing (Oracle)."""
    df_target = df.copy()
    df_target['smoothed_close'] = gaussian_filter1d(df_target['close'], sigma=sigma)
    df_target['smooth_slope'] = np.diff(np.log(df_target['smoothed_close']), prepend=np.nan)
    
    conditions = [
        df_target['smooth_slope'] > threshold,
        df_target['smooth_slope'] < -threshold
    ]
    choices = [2, 0] # 2=UP, 0=DOWN
    df_target['target'] = np.select(conditions, choices, default=1) # 1=SIDEWAYS
    return df_target

# ==================================================================
#   FEATURE ENGINEERING CATEGORIES
# ==================================================================

def add_momentum_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.roc(length=12, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.cci(length=14, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.ao(append=True)
    df.ta.mom(length=10, append=True)
    df.ta.tsi(length_fast=13, length_slow=25, append=True)
    df.ta.uo(append=True)
    return df

def add_overlap_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.ema(length=10, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.hma(length=9, append=True)
    df.ta.tema(length=9, append=True)
    df.ta.psar(append=True)
    df.ta.supertrend(length=7, multiplier=3, append=True)
    
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    if vol_col in df.columns:
        df.ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True)
    return df

def add_trend_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.aroon(length=14, append=True)
    df.ta.vortex(length=14, append=True)
    df.ta.dpo(length=20, centered=False, append=True)
    df.ta.trix(length=30, append=True)
    df.ta.cksp(append=True)
    return df

def add_volatility_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.atr(length=14, append=True)
    df.ta.natr(length=14, append=True)
    df.ta.ui(length=14, append=True)

    bbands = df.ta.bbands(length=20, append=False)
    if bbands is not None and not bbands.empty:
        bbp_cols = [c for c in bbands.columns if c.startswith('BBP')]
        bbb_cols = [c for c in bbands.columns if c.startswith('BBB')]
        if bbp_cols: df[bbp_cols[0]] = bbands[bbp_cols[0]]
        if bbb_cols: df[bbb_cols[0]] = bbands[bbb_cols[0]]

    kc = df.ta.kc(append=False)
    if kc is not None and not kc.empty:
        kcp_cols = [c for c in kc.columns if c.startswith('KCP')]
        if kcp_cols: df[kcp_cols[0]] = kc[kcp_cols[0]]

    dc = df.ta.donchian(append=False)
    if dc is not None and not dc.empty:
        dcp_cols = [c for c in dc.columns if c.startswith('DCP')]
        if dcp_cols: df[dcp_cols[0]] = dc[dcp_cols[0]]
    return df

def add_volume_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vol_col = 'volume' if 'volume' in df.columns else 'total_vol'
    if vol_col in df.columns:
        df.ta.obv(close=df['close'], volume=df[vol_col], append=True)
        df.ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], length=14, append=True)
        df.ta.ad(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True)
        df.ta.cmf(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True)
        df.ta.eom(high=df['high'], low=df['low'], close=df['close'], volume=df[vol_col], append=True)
        df.ta.nvi(close=df['close'], volume=df[vol_col], append=True)
        df.ta.pvi(close=df['close'], volume=df[vol_col], append=True)
    return df

def add_statistics_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.ta.zscore(length=30, append=True)
    df.ta.entropy(length=30, append=True)
    df.ta.kurtosis(length=30, append=True)
    df.ta.skew(length=30, append=True)
    df.ta.variance(length=30, append=True)
    df.ta.mad(length=30, append=True)
    return df

def add_candle_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    patterns_to_add = [
        'cdl_doji', 'cdl_hammer', 'cdl_engulfing', 'cdl_morningstar', 'cdl_eveningstar',
        'cdl_shootingstar', 'cdl_hangingman', 'cdl_marubozu', 'cdl_3whitesoldiers',
        'cdl_3blackcrows', 'cdl_inside', 'cdl_spinningtop'
    ]
    for pattern_name in patterns_to_add:
        if hasattr(df.ta, pattern_name):
            getattr(df.ta, pattern_name)(append=True)
    return df

GROUP_FUNCS = {
    "momentum": add_momentum_category,
    "overlap": add_overlap_category,
    "trend": add_trend_category,
    "volatility": add_volatility_category,
    "volume": add_volume_category,
    "statistics": add_statistics_category,
    "candle": add_candle_category,
}

# ==================================================================
#   OPTIMIZED PIPELINE & FEATURE SELECTION
# ==================================================================

def prepare_all_features(df_labeled: pd.DataFrame):
    """Calculates ALL indicator groups at once."""
    print("[prepare_all_features] Pre-calculating ALL indicators...")
    df_all = df_labeled.copy()

    if 'time' in df_all.columns and not isinstance(df_all.index, pd.DatetimeIndex):
        df_all['time'] = pd.to_datetime(df_all['time'])
        df_all.set_index('time', inplace=True)

    group_features_map = {}

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

    nan_cols = df_all.columns[df_all.isna().all()].tolist()
    if nan_cols:
        print(f"[WARNING] Dropping {len(nan_cols)} columns that are 100% NaN")
        df_all.drop(columns=nan_cols, inplace=True)
        for g in group_features_map:
            group_features_map[g] = [c for c in group_features_map[g] if c not in nan_cols]

    print(f"[prepare_all_features] Done. Total columns: {df_all.shape[1]}")
    return df_all, group_features_map

def run_fast_experiment(df_all, group_features_map, active_groups, test_size=0.2, n_estimators=25, random_state=42):
    """Fast experiment: simply takes ready-made columns from df_all."""
    selected_indicator_cols = []
    for g in active_groups:
        selected_indicator_cols.extend(group_features_map.get(g, []))

    all_cols = list(df_all.columns)
    all_indicator_cols = set()
    for cols in group_features_map.values():
        all_indicator_cols.update(cols)

    base_features = [c for c in all_cols if c not in all_indicator_cols and c not in FEATURES_TO_REMOVE]
    final_features = base_features + selected_indicator_cols

    cols_to_take = list(set(final_features + ['target']))
    df_exp = df_all[cols_to_take].copy()

    feat_cols = [c for c in df_exp.columns if c != 'target']
    df_exp[feat_cols] = df_exp[feat_cols].shift(1)
    df_exp.dropna(inplace=True)

    if df_exp.empty:
        return {'groups': active_groups, 'accuracy': 0, 'report': 'Empty after dropna', 'n_features': 0}

    X = df_exp[feat_cols]
    y = df_exp['target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    if len(X_train) == 0 or len(X_test) == 0:
        return {'groups': active_groups, 'accuracy': 0, 'report': 'Train/Test empty', 'n_features': 0}

    model = XGBClassifier(
        device='cuda', n_estimators=n_estimators, random_state=random_state,
        eval_metric='mlogloss', max_depth=2
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    fi_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return {
        'groups': active_groups, 'accuracy': acc,
        'n_features': len(feat_cols), 'feature_importance': fi_df
    }

def iterate_group_combos_fast(df_labeled, max_group_size=4):
    df_all, group_features_map = prepare_all_features(df_labeled)
    group_names = list(group_features_map.keys())
    combo_list = []
    for r in range(1, max_group_size + 1):
        for combo in combinations(group_names, r):
            combo_list.append(list(combo))

    print(f"\n[iterate_group_combos_fast] Starting {len(combo_list)} experiments...")
    results = []
    for idx, groups in enumerate(combo_list, start=1):
        res = run_fast_experiment(df_all, group_features_map, groups)
        print(f"Run {idx}/{len(combo_list)} | {groups} | Acc: {res['accuracy']:.4f}")
        results.append(res)

    return results

def rerun_with_top_k(df_labeled, base_groups, fi_df, k_list, test_size=0.2, n_estimators=25, random_state=42):
    out = []
    print(f"[rerun_with_top_k] base_groups={base_groups}, k_list={k_list}")

    base_cols = ['time', 'open', 'high', 'low', 'close', 'total_vol', 'target']
    cols_to_keep = [c for c in base_cols if c in df_labeled.columns]
    df_full = df_labeled[cols_to_keep].copy()

    for g in base_groups:
        if g in GROUP_FUNCS:
            df_full = GROUP_FUNCS[g](df_full)

    for k in k_list:
        top_feats = fi_df.head(k)["Feature"].tolist()
        df = df_full.copy()
        cols_to_retain = set(top_feats + FEATURES_TO_REMOVE)
        drop_cols = [c for c in df.columns if c not in cols_to_retain]
        df = df.drop(columns=drop_cols, errors="ignore")

        features = [c for c in df.columns if c not in FEATURES_TO_REMOVE]
        df[features] = df[features].shift(1)
        df = df.dropna()

        X = df[features]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        model = XGBClassifier(
            device='cuda', n_estimators=n_estimators, random_state=random_state,
            eval_metric='mlogloss', max_depth=4
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        out.append({"k": k, "accuracy": acc})
        print(f"[rerun_with_top_k] k={k}, acc={acc:.4f}")

    summary = pd.DataFrame(out).sort_values("accuracy", ascending=False)
    return summary

# ==================================================================
#   MAIN EXECUTION
# ==================================================================

def main():
    # 1. Load Data
    df = load_and_merge_data()
    if df is None: return

    # 2. Target Generation
    df_labeled = create_target_labels(df, sigma=SIGMA, threshold=THRESHOLD)
    print("Class Distribution:")
    print(df_labeled['target'].value_counts(normalize=True))

    # 3. Find Best Group Combination
    # We limit max_group_size to 6 to cover most combinations without being too slow
    results = iterate_group_combos_fast(df_labeled, max_group_size=6)
    best_res = max(results, key=lambda r: r["accuracy"])
    base_groups = best_res["groups"]
    fi = best_res["feature_importance"]
    print(f"\nBest Group Combination: {base_groups} (Acc: {best_res['accuracy']:.4f})")

    # 4. Find Best Number of Features (Top K)
    topk_summary = rerun_with_top_k(
        df_labeled,
        base_groups=base_groups,
        fi_df=fi,
        k_list=[10, 20, 30, 40, 60, 80]
    )
    best_k_row = topk_summary.iloc[0]
    best_k = int(best_k_row['k'])
    print(f"\nBest k features: {best_k} (Acc: {best_k_row['accuracy']:.4f})")

    # 5. Prepare Final Dataset with Best K Features
    print("\nPreparing final dataset for Hyperparameter Tuning...")
    df_full = df_labeled.copy()
    for g in base_groups:
        if g in GROUP_FUNCS:
            df_full = GROUP_FUNCS[g](df_full)

    top_feats = fi.head(best_k)["Feature"].tolist()
    cols_to_retain = set(top_feats + FEATURES_TO_REMOVE)
    drop_cols = [c for c in df_full.columns if c not in cols_to_retain]
    df_final = df_full.drop(columns=drop_cols, errors="ignore")

    features = [c for c in df_final.columns if c not in FEATURES_TO_REMOVE]
    df_final[features] = df_final[features].shift(1)
    df_final = df_final.dropna()

    X = df_final[features]
    y = df_final["target"]

    # Split for RandomizedSearch
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Final Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # 6. Randomized Search for Hyperparameters
    print("\nStarting RandomizedSearchCV with Walk-Forward Validation...")
    param_dist = {
        'n_estimators': [20,100, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'max_depth': [2,3,5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3, 0.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0, 0.01, 0.1, 1, 10]
    }

    xgb = XGBClassifier(device='cuda', eval_metric='mlogloss', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=25,
        scoring='f1_weighted',
        cv=TimeSeriesSplit(n_splits=5),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_

    # 7. Final Evaluation
    print("\n--- Evaluation on Full Dataset ---")
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred, target_names=['DOWN', 'SIDEWAYS', 'UP']))

    # 8. Save Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save Model
    joblib.dump(best_model, os.path.join(MODEL_DIR, MODEL_FILENAME))
    
    # Save Metrics
    metrics = {
        "best_cv_score": random_search.best_score_,
        "best_params": random_search.best_params_,
        "best_groups": base_groups,
        "best_k": best_k,
        "final_accuracy_full": accuracy_score(y, y_pred)
    }
    with open(os.path.join(MODEL_DIR, METRICS_FILENAME), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Best Features List (Crucial for Predict.py)
    with open(os.path.join(MODEL_DIR, BEST_FEATURES_FILENAME), "w") as f:
        json.dump(features, f, indent=4)

    print(f"\nTraining Complete. Model saved to {os.path.join(MODEL_DIR, MODEL_FILENAME)}")
    print(f"Best features list saved to {os.path.join(MODEL_DIR, BEST_FEATURES_FILENAME)}")

if __name__ == "__main__":
    main()
