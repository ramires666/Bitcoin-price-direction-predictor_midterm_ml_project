#%%
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, log_loss
# def main():

#%%



#%%
fundings = pd.read_parquet('data/processed/fundings.parquet')
klines = pd.read_parquet('data/processed/klines_15min_all.parquet')
volumes = pd.read_parquet('data/processed/aggtrades_15min_all.parquet')

# Приводим названия и типы колонок времени к одному виду
# trades: datetime -> time
volumes = volumes.rename(columns={"datetime": "time"})
# fundings: calc_time -> time
fundings = fundings.rename(columns={"calc_time": "time"})
# klines: open_time -> time
klines = klines.rename(columns={"open_time": "time"})

# Убедимся, что все три колонки в datetime64[ns, UTC] (или хотя бы datetime64[ns])
volumes["time"] = pd.to_datetime(volumes["time"], utc=True)
fundings["time"] = pd.to_datetime(fundings["time"], utc=True)
klines["time"] = pd.to_datetime(klines["time"], utc=True)


# Объединяем:
# 1) trades + klines по общему времени
df = pd.merge(volumes, klines, on="time", how="inner")

# 2) к ним присоединяем фандинг
# Если фандинг раз в 8 часов и время совпадает не всегда, можно
# либо оставить inner, либо сделать left join и потом forward-fill.
df = pd.merge(df, fundings, on="time", how="left")

# При желании можно сделать forward-fill funding_rate по времени:
df = df.sort_values("time").reset_index(drop=True)
df["funding_rate"] = df["funding_rate"].ffill()

# Сохраняем итоговый датафрейм
output_path = "data/processed/all_merged.parquet"
df.to_parquet(output_path, index=False)

# volume delta
df['volume_delta'] = df['ask_vol'] - df['bid_vol']
df['volume_delta_max'] = df['max_ask_vol'] - df['max_bid_vol']
df['volume_delta_avg'] = df['avg_ask_vol'] - df['avg_bid_vol']
df['cumulative_volume_delta_96'] = df['volume_delta'].rolling(window=96, min_periods=1).sum()
df['cumulative_volume_delta_4'] = df['volume_delta'].rolling(window=4, min_periods=1).sum()
df['cumulative_volume_delta_8'] = df['volume_delta'].rolling(window=8, min_periods=1).sum()
df['cumulative_volume_delta_16'] = df['volume_delta'].rolling(window=16, min_periods=1).sum()
df['cumulative_volume_delta_32'] = df['volume_delta'].rolling(window=32, min_periods=1).sum()

df = df.set_index("time")



#%% plot
# df = df.set_index("time")

fig, ax_price = plt.subplots(figsize=(28, 14))
df_plot = df.tail(500)

# 1) Цена на основной оси (левая)
ax_price.plot(df_plot.index, df_plot["close"], color="black", label="Close price")
ax_price.set_xlabel("Time")
ax_price.set_ylabel("Price", color="black")
ax_price.tick_params(axis="y", labelcolor="black")

# 2) Вторая ось для объёмов (правая)
ax_vol = ax_price.twinx()


ax_vol.plot(df_plot.index, df_plot["cumulative_volume_delta_96"], color="red", alpha=0.7, label="Cumsum 96 bid vol")
# ax_vol.plot(df_plot["time"], df_plot["cumulative_volume_delta_4"], color="red", alpha=0.7, label="Cumsum 4 bid vol")?
# ax_vol.plot(df_plot["time"], df_plot["cumulative_volume_delta_32"], color="red", alpha=0.7, label="Cumsum 16 bid vol")
# ax_vol.plot(df_plot["time"], df_plot["cumulative_volume_delta_32"], color="red", alpha=0.7, label="Cumsum 32 bid vol")
# ax_vol.plot(df_plot["time"], df_plot["cumulative_volume_delta_8"], color="red", alpha=0.7, label="Cumsum 8 bid vol")
# ax_vol.plot(df_plot["time"], df_plot["avg_bid_vol"], color="red", alpha=0.7, label="Avg bid vol")
# ax_vol.plot(df_plot["time"], df_plot["avg_ask_vol"], color="green", alpha=0.7, label="Avg ask vol")
# ax_vol.plot(df_plot["time"], df_plot["max_bid_vol"], color="red", alpha=0.7, label="Max bid vol")
# ax_vol.plot(df_plot["time"], df_plot["max_ask_vol"], color="green", alpha=0.7, label="Max ask vol")
ax_vol.set_ylabel("Volume", color="blue")
ax_vol.tick_params(axis="y", labelcolor="blue")

# 3) Легенда: объединяем хэндлы обеих осей
lines_price, labels_price = ax_price.get_legend_handles_labels()
lines_vol, labels_vol = ax_vol.get_legend_handles_labels()
ax_price.legend(lines_price + lines_vol, labels_price + labels_vol, loc="upper left")

ax_price.set_title("Price vs Avg / Max Bid Volume")

fig.tight_layout()
plt.show()


#%%
## technical indicators
# Trend Identification
df.ta.ema(length=9, append=True)       # Fast trend
df.ta.ema(length=21, append=True)      # Medium trend
df.ta.ema(length=50, append=True)      # Main direction filter

# Momentum - MACD (Best for 15-min regime changes)
df.ta.macd(fast=12, slow=26, signal=9, append=True)    # Standard
df.ta.macd(fast=5, slow=13, signal=1, append=True)     # Fast response

# Momentum - RSI (Mean reversion detection)
df.ta.rsi(length=7, append=True)       # Ultra-fast reversal catch
df.ta.rsi(length=14, append=True)      # Standard overbought/oversold
df.ta.rsi(length=21, append=True)      # Confirmation signal

# Volatility - Bollinger Bands
df.ta.bbands(length=20, std=2, append=True)    # Standard bands
df.ta.bbands(length=10, std=1.5, append=True)  # Tighter bands

# Volatility - ATR (Dynamic risk sizing)
df.ta.atr(length=14, append=True)      # True volatility


# Momentum - Stochastic (Short-term oversold/overbought)
df.ta.stoch(length=14, k=3, d=3, append=True)  # Standard stochastic
df.ta.stoch(length=7, k=3, d=3, append=True)   # Fast for 15m

# Momentum - CCI (Cyclical extremes, mean reversion)
df.ta.cci(length=14, append=True)      # Standard
df.ta.cci(length=20, append=True)      # Confirmation

# Trend Strength - ADX (Filter for strong trends only)
df.ta.adx(length=14, append=True)      # Trend strength filter

# Momentum - Williams %R (Alternative overbought/oversold)
df.ta.willr(length=14, append=True)    # Overbought/oversold

# Momentum - Rate of Change (Momentum intensity)
df.ta.roc(length=5, append=True)       # Fast ROC
df.ta.roc(length=10, append=True)      # Standard
df.ta.roc(length=14, append=True)      # Confirmation

# Volume - On-Balance Volume (Divergence confirmation)
df.ta.obv(append=True)                 # MACD + OBV = Sharpe 2.7357

# Momentum Divergence Detection
df.ta.mom(length=10, append=True)      # Fast momentum
df.ta.mom(length=14, append=True)      # Standard momentum

# Hilbert Transform (Adaptive cycle detection)
# df.ta.ht_sine(append=True)             # Sine wave extraction
# df.ta.ht_dcperiod(append=True)         # Dominant cycle period
df.ta.ht_trendline(append=True)        # Hilbert trendline

# Lagged price features (Essential for ML models)
df['close_lag1'] = df['close'].shift(1)
df['close_lag5'] = df['close'].shift(5)
df['close_lag10'] = df['close'].shift(10)

# Returns
df['log_return'] = np.log1p(df['close'].pct_change())

# Volatility components
df['hl_range'] = (df['high'] - df['low']) / df['close']
df['close_open_ratio'] = (df['close'] - df['open']) / df['open']

#%%
########################
# target
# Create volatility-adjusted return target
# df = df_orig.copy()
k = 1.5  # Risk parameter
# df['y'] = (df['close'].shift(-10) - df['close']) / (df['ATRr_14'] * k)
# лог-доходность между t и t+3
df['y'] = np.log(df['close'].shift(-1) / df['close'])


#%%
pd.set_option('display.max_rows', None)       # Показывать все строки
pd.set_option('display.max_columns', None)    # Показывать все колонки
pd.set_option('display.width', None)          # Отключить ограничение ширины
pd.set_option('display.max_colwidth', None)   # Полное отображение содержимого ячеек

df_orig = df.copy()
# df = df_orig.copy()
df=df.dropna()

df_prepared=df.copy()

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


#%%
# Шаг 1: Исключить
exclude_columns = ['close', 'ATRr_14', 'y']

# Шаг 2: Получить признаки
feature_columns = [col for col in df.columns if col not in exclude_columns]

# Шаг 3: Подготовить
X = df[feature_columns]
y = df['y']


############# continious split for timeseries
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]
################### continious split for timeseries

#%%# Обучение
# 1) Преобразуем в DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test,  label=y_test)

# 2) Параметры модели (аналогичные XGBRegressor)
params = {
    'objective': 'reg:squarederror',  # регрессия
    'tree_method': 'hist',
    'device': 'cuda',                 # GPU
    'max_depth': 3,
    'eta': 0.08,                      # learning_rate
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',            # метрика для мониторинга
    'seed': 42,
    'reg_lambda': 1.0,

}

# 3) Обучение с early stopping
evals = [(dtrain, 'train'), (dvalid, 'validation')]
# params = best['params']
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=50000,         # максимум деревьев
    evals=evals,
    early_stopping_rounds=50,    # остановка, если 50 раундов нет улучшения
    verbose_eval=True
)

print("Best iteration:", bst.best_iteration)
print("Best score:", bst.best_score)

# 4) Предсказания
y_pred = bst.predict(dvalid)

rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

#%%

# y_pred = model.predict(X_test)

# Теперь X_test и y_pred имеют длину test_size * длина исходного набора
test_indices = X_test.index

#%%
# Восстановление цены (БЕЗ shift!)
# predicted_close = df.loc[test_indices, 'close'] + y_pred * df.loc[test_indices, 'ATRr_14'] * k
predicted_close = df.loc[test_indices, 'close'] * np.exp(y_pred)

# График
day_start = '2025-10-30'
day_end = '2025-10-30 23:59:59'
test_indices_day = test_indices[(test_indices >= day_start) & (test_indices <= day_end)]

plt.figure(figsize=(14,7))
plt.plot(df.loc[test_indices_day].index, df.loc[test_indices_day, 'close'], label='Реальная цена')
plt.plot(predicted_close.loc[test_indices_day].index, predicted_close.loc[test_indices_day], label='Прогноз на +1 час')
plt.legend()
plt.show()





#%%

bst.get_score(importance_type='gain')
score = bst.get_score(importance_type='gain')
from operator import itemgetter
sorted_dict = dict(sorted(score.items(), key=itemgetter(1), reverse=True))
sorted_items = sorted(score.items(), key=lambda x: x[1], reverse=True)
features = pd.DataFrame(sorted_items, columns=['feature', 'score'])  # это работает!
print(features)


#%%
# истинная цена на +2 бара
true_future_close = df['close'].shift(-1)

plt.figure(figsize=(14,7))
plt.plot(
    df.loc[test_indices_day].index,
    true_future_close.loc[test_indices_day],
    label='Реальная цена на +2 бара'
)
plt.plot(
    predicted_close.loc[test_indices_day].index,
    predicted_close.loc[test_indices_day],
    label='Прогноз на +2 бара'
)
plt.legend()
plt.show()
