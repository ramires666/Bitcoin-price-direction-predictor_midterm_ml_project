import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# price_log_returns — ваш массив логретернов цены биткоина

# Random Walk прогноз (следующее значение равно текущему)
price_log_returns = list(df['log_return'])[1:]
random_walk_pred = price_log_returns[:-1]

# XGBoost подготовка и обучение (предсказание следующего шага логретерна)
X = np.array([price_log_returns[i] for i in range(len(price_log_returns)-1)]).reshape(-1, 1)
y = price_log_returns[1:]

model = xgb.XGBRegressor()
model.fit(X, y)
xgb_pred = model.predict(X)

# Метрики для random walk
rmse_rw = mean_squared_error(y, random_walk_pred)
mae_rw = mean_absolute_error(y, random_walk_pred)
r2_rw = r2_score(y, random_walk_pred)

# Метрики для XGBoost
rmse_xgb = mean_squared_error(y, xgb_pred)
mae_xgb = mean_absolute_error(y, xgb_pred)
r2_xgb = r2_score(y, xgb_pred)

print(f'Random Walk RMSE: {rmse_rw}, MAE: {mae_rw}, R²: {r2_rw}')
print(f'XGBoost RMSE: {rmse_xgb}, MAE: {mae_xgb}, R²: {r2_xgb}')


#%%
# mutual information

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

from sklearn.feature_selection import mutual_info_regression

# df — DataFrame с признаками, y — целевая колонка (Series или массив)
X = df.dropna()  # все колонки с признаками
# y — вектор с целевой переменной, например, df['y_column']

mi = mutual_info_regression(X, X.y)

# Создаем Series для удобного отображения взаимной информации
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

print(mi_series)
