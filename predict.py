import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "xgb_hmm_model.joblib"
SCALER_FILENAME = "scaler.joblib"
HMM_FILENAME = "hmm_model.joblib"
HMM_SCALER_FILENAME = "hmm_scaler.joblib"

class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.hmm_model = None
        self.hmm_scaler = None
        self.load_models()

    def load_models(self):
        """Loads trained models and scalers."""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, MODEL_FILENAME))
            self.scaler = joblib.load(os.path.join(self.model_dir, SCALER_FILENAME))
            self.hmm_model = joblib.load(os.path.join(self.model_dir, HMM_FILENAME))
            self.hmm_scaler = joblib.load(os.path.join(self.model_dir, HMM_SCALER_FILENAME))
            print("Models loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            raise

    def preprocess_data(self, df):
        """Applies the same feature engineering as training."""
        df = df.copy()
        
        # Basic Features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Volume Features
        # Note: 'taker_buy_vol' and 'maker_sell_vol' might be available from API
        # If not, we assume 'ask_vol' and 'bid_vol' are present or derived
        # For Binance klines, we usually get 'taker_buy_base_asset_volume'
        # We need to map API columns to what the model expects
        
        if 'volume_delta' not in df.columns:
             # Approximation if exact bid/ask not available:
             # volume_delta ~ 2 * taker_buy_vol - total_vol
             # (Assuming taker buy is aggressive buy, rest is aggressive sell)
             if 'taker_buy_vol' in df.columns:
                 df['volume_delta'] = 2 * df['taker_buy_vol'] - df['volume']
             else:
                 df['volume_delta'] = 0 # Fallback
        
        for window in [4, 12, 24]:
            df[f"vol_delta_rolling_{window}"] = df["volume_delta"].rolling(window).sum()
            
        # Volatility Features
        df.ta.bbands(length=20, std=2, append=True)
        df['bb_width'] = (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']) / df['BBM_20_2.0_2.0']
        
        # Trend Features
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        df['trend_ema'] = np.where(df['ema_12'] > df['ema_26'], 1, -1)
        
        return df

    def add_hmm_features(self, df):
        """Adds HMM state probabilities."""
        hmm_data = df[['log_return', 'bb_width', 'volume_delta']].copy()
        # Handle potential zeros in volume for log
        hmm_data['log_volume'] = np.log(df['volume'] + 1) 
        
        # We need to handle NaNs created by rolling windows
        # For prediction, we usually care about the LAST row, 
        # so we just need enough history to fill the windows.
        
        # Fill NaNs for HMM input to avoid errors, but keep track of valid indices
        hmm_data = hmm_data.fillna(0) 
        
        X_hmm = self.hmm_scaler.transform(hmm_data)
        state_probs = self.hmm_model.predict_proba(X_hmm)
        
        for i in range(3):
            df[f'hmm_prob_{i}'] = state_probs[:, i]
            
        return df

    def predict(self, df):
        """
        Generates prediction for the latest available data point.
        Returns: (prediction_class, probabilities, timestamp)
        """
        # 1. Preprocess
        df_processed = self.preprocess_data(df)
        
        # 2. Add HMM
        df_processed = self.add_hmm_features(df_processed)
        
        # 3. Select Features
        feature_cols = [
            'volume_delta', 'vol_delta_rolling_4', 'vol_delta_rolling_12', 'vol_delta_rolling_24',
            'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'trend_ema',
            'bb_width', 'BBP_20_2.0_2.0',
            'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'
        ]
        
        # Get the last row (latest completed bar)
        # Note: In training we shifted features by 1. 
        # For live prediction, we use the features of the *just closed* bar 
        # to predict the *next* bar's direction.
        # So we do NOT shift here, we just take the last row.
        
        last_row = df_processed.iloc[[-1]][feature_cols]
        
        # Check for NaNs in the last row (e.g. not enough history)
        if last_row.isna().any().any():
            print("Warning: NaNs in features. Need more history.")
            return None, None, None

        # 4. Scale
        X_scaled = self.scaler.transform(last_row)
        
        # 5. Predict
        pred_class = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        
        timestamp = df_processed.iloc[-1]['time'] if 'time' in df_processed.columns else df_processed.index[-1]
        
        return pred_class, pred_proba, timestamp

if __name__ == "__main__":
    # Test with dummy data
    print("Testing Predictor...")
    # Create dummy dataframe matching structure
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15T')
    df_dummy = pd.DataFrame({
        'time': dates,
        'open': np.random.rand(100) * 100 + 40000,
        'high': np.random.rand(100) * 100 + 40100,
        'low': np.random.rand(100) * 100 + 39900,
        'close': np.random.rand(100) * 100 + 40000,
        'volume': np.random.rand(100) * 10,
        'taker_buy_vol': np.random.rand(100) * 5
    })
    
    try:
        predictor = Predictor()
        cls, prob, time = predictor.predict(df_dummy)
        print(f"Time: {time}")
        print(f"Prediction: {cls} (0=DOWN, 1=SIDEWAYS, 2=UP)")
        print(f"Probabilities: {prob}")
    except Exception as e:
        print(f"Prediction failed (expected if models not trained): {e}")