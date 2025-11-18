

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


# Шаг 5: Обучить
model = xgb.XGBRegressor(tree_method='hist', device='cuda',verbosity=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Теперь X_test и y_pred имеют длину test_size * длина исходного набора
# Чтобы получить соответствующие индексы исходного DataFrame:
test_indices = X_test.index

# Восстановим цену только для тестового набора
predicted_close = df.loc[test_indices, 'close'] + y_pred * df.loc[test_indices, 'ATRr_14'] * k

# Сдвиг для визуализации цены через 4 шага
predicted_close_shifted = predicted_close.shift(4)
# predicted_close_shifted = predicted_close

#################################
day_start = '2025-10-10'
day_end = '2025-10-10 23:59:59'

# Фильтрация индексов тестовой выборки по дате
test_indices_day = test_indices[(test_indices >= day_start) & (test_indices <= day_end)]


plt.figure(figsize=(14,7))
plt.plot(df.loc[test_indices_day].index, df.loc[test_indices_day, 'close'], label='Реальная цена (тест)')
# Для предсказанной цены (тоже с такими же индексами)
plt.plot(predicted_close_shifted.loc[test_indices_day].index, predicted_close_shifted.loc[test_indices_day], label='Предсказанная цена (4 шага, тест)')
# plt.plot(df.loc[test_indices].index, df.loc[test_indices, 'close'], label='Реальная цена (тест)')
# plt.plot(df.loc[test_indices].index, predicted_close_shifted, label='Предсказанная цена (через 4 шага, тест)')
plt.legend()
plt.show()


#%%

rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

import matplotlib.pyplot as plt

predicted_close = df['close'] + y_pred * df['ATRr_14'] * k
predicted_close_shifted = predicted_close.shift(4)

# Нарисуем графики
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='Реальная цена')
plt.plot(df.index, predicted_close_shifted, label='Предсказанная цена (через 4 шага)')
plt.legend()
plt.title('Реальная цена и предсказанная цена через 4 шага')
plt.show()

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted")
plt.show()



#%%
# Train XGBoost as regressor
model = xgb.XGBRegressor(n_estimators=100, max_depth=5,verbosity=3)
model.fit(X_train, df.loc[X_train.index, 'y'])


#%%
# Signals + Position sizing
df['signal'] = np.where(df['y_pred'] > 0, 1, -1)
df['position_size'] = np.abs(df['y_pred'])  # In ATR units
df['stop_loss'] = df['close'] - (df['ATR_14'] * 1.5)
df['take_profit'] = df['close'] + (df['ATR_14'] * 2.0)


#%%
import itertools
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==== 1. Time-series split (как у тебя) ====
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test  = X.iloc[split_index:]
y_test  = y.iloc[split_index:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test,  label=y_test)

# ==== 2. Сетка гиперпараметров ====
param_grid = {
    'max_depth':        [3, 4,5,6, 7,8],
    'eta':              [0.03,0.05, 0.07, 0.09],   # learning_rate
    'subsample':        [0.5, 0.8],
    'colsample_bytree': [0.5, 0.8],
    'reg_lambda':       [1.0,3.0, 5.0],
}

# Базовые параметры, общие для всех прогонов
base_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'seed': 42,
}

num_boost_round = 5000
early_stopping_rounds = 50

results = []  # сюда будем складывать результаты по всем комбинациям

# ==== 3. Перебор комбинаций ====
keys, values = zip(*param_grid.items())

for combo in itertools.product(*values):
    # Собираем параметры для текущего прогона
    params = base_params.copy()
    params.update(dict(zip(keys, combo)))

    print("\n=== New run ===")
    print(params)

    evals = [(dtrain, 'train'), (dvalid, 'validation')]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False  # можешь поставить True для детального лога
    )

    # Лучшая итерация и score по валидации (rmse)
    best_iter = bst.best_iteration
    best_score = bst.best_score  # это rmse

    # Предсказания на тесте
    y_pred = bst.predict(dvalid, iteration_range=(0, best_iter + 1))

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"best_iter={best_iter}, val_rmse={best_score:.6f}, "
          f"test_rmse={rmse:.6f}, test_mae={mae:.6f}, test_r2={r2:.6f}")

    results.append({
        'params': params,
        'best_iteration': best_iter,
        'val_rmse': best_score,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
    })

# ==== 4. Поиск лучшей комбинации по валидации (val_rmse) ====
results_sorted = sorted(results, key=lambda x: x['val_rmse'])
best = results_sorted[0]

print("\n=== Best by validation RMSE ===")
print("Params:", best['params'])
print(f"best_iteration: {best['best_iteration']}")
print(f"val_rmse:       {best['val_rmse']:.6f}")
print(f"test_rmse:      {best['test_rmse']:.6f}")
print(f"test_mae:       {best['test_mae']:.6f}")
print(f"test_r2:        {best['test_r2']:.6f}")

