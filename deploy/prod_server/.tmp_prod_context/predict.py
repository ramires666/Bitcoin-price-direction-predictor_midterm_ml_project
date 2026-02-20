import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "best_xgb_model.joblib"
BEST_FEATURES_FILENAME = "best_features_list.json"

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

class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.best_features = None
        self.load_models()

    def load_models(self):
        """Loads trained model and feature list."""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, MODEL_FILENAME))
            
            with open(os.path.join(self.model_dir, BEST_FEATURES_FILENAME), "r") as f:
                self.best_features = json.load(f)
                
            print(f"Model loaded. Expecting {len(self.best_features)} features.")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            raise

    def preprocess_data(self, df):
        """Applies ALL feature engineering to ensure we have the required columns."""
        df_processed = df.copy()
        
        # Ensure time index if needed (though pandas-ta usually handles it)
        if 'time' in df_processed.columns and not isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed['time'] = pd.to_datetime(df_processed['time'])
            df_processed.set_index('time', inplace=True)

        # Apply all groups
        for g_name, g_func in GROUP_FUNCS.items():
            try:
                df_processed = g_func(df_processed)
            except Exception as e:
                print(f"Warning: Error in group '{g_name}': {e}")
                
        return df_processed

    def _prepare_features(self, df):
        """Shared preprocessing logic for single and batch prediction."""
        # 1. Generate ALL features
        df_processed = self.preprocess_data(df)
        
        # 2. Select only the features the model was trained on
        # Check if all required features exist
        missing_cols = [c for c in self.best_features if c not in df_processed.columns]
        if missing_cols:
            print(f"Warning: {len(missing_cols)} features missing from data: {missing_cols[:5]}...")
            # Fill missing with 0 to prevent crash, but this is not ideal
            for c in missing_cols:
                df_processed[c] = 0
        
        return df_processed, self.best_features

    def predict(self, df):
        """
        Generates prediction for the latest available data point.
        Returns: (prediction_class, probabilities, timestamp)
        """
        df_processed, feature_cols = self._prepare_features(df)
        
        # Get the last row (latest completed bar)
        last_row = df_processed.iloc[[-1]][feature_cols]
        
        # Check for NaNs in the last row (e.g. not enough history)
        if last_row.isna().any().any():
            print("Warning: NaNs in features. Need more history.")
            # Try to fill with previous values if possible, or return None
            # For now, return None to indicate insufficient data
            return None, None, None


        pred_class = self.model.predict(last_row)[0]
        pred_proba = self.model.predict_proba(last_row)[0]
        
        timestamp = df_processed.iloc[-1].name if isinstance(df_processed.index, pd.DatetimeIndex) else df_processed.index[-1]
        
        return pred_class, pred_proba, timestamp

    def predict_batch(self, df):
        """
        Generates predictions for the entire dataframe.
        Returns: DataFrame with 'prediction' and 'probabilities' columns
        """
        df_processed, feature_cols = self._prepare_features(df)
        
        # Drop NaNs for prediction
        df_valid = df_processed.dropna(subset=feature_cols).copy()
        
        if df_valid.empty:
            return pd.DataFrame()

        # Predict
        df_valid['prediction'] = self.model.predict(df_valid[feature_cols])
        probs = self.model.predict_proba(df_valid[feature_cols])
        df_valid['prob_down'] = probs[:, 0]
        df_valid['prob_sideways'] = probs[:, 1]
        df_valid['prob_up'] = probs[:, 2]
        
        return df_valid

if __name__ == "__main__":
    # Test with dummy data
    print("Testing Predictor...")
    dates = pd.date_range(start='2024-01-01', periods=200, freq='15T')
    df_dummy = pd.DataFrame({
        'time': dates,
        'open': np.random.rand(200) * 100 + 40000,
        'high': np.random.rand(200) * 100 + 40100,
        'low': np.random.rand(200) * 100 + 39900,
        'close': np.random.rand(200) * 100 + 40000,
        'volume': np.random.rand(200) * 10,
        'taker_buy_vol': np.random.rand(200) * 5
    })
    
    try:
        predictor = Predictor()
        cls, prob, time = predictor.predict(df_dummy)
        print(f"Time: {time}")
        print(f"Prediction: {cls} (0=DOWN, 1=SIDEWAYS, 2=UP)")
        print(f"Probabilities: {prob}")
    except Exception as e:
        print(f"Prediction failed (expected if models not trained): {e}")