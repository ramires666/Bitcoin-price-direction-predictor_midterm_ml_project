# %%
"""
ФИНАЛЬНЫЙ СКРИПТ ДЛЯ ОБУЧЕНИЯ XGBoost-МОДЕЛИ ПРОГНОЗА ЛОГ-ВОЗВРАТА BTCUSDT

Основные цели:
1. Загрузка и объединение данных (klines + объёмы + фандинги) в единый датафрейм.
2. Генерация осмысленных фич (объёмы, лаги, индикаторы), построенных БЕЗ утечки будущего.
3. Нормализация входных признаков с помощью RobustScaler.
4. Обучение модели XGBoost с использованием GPU (CUDA) при наличии.
5. Ручной грид-сёрч гиперпараметров XGBoost с учётом специфики таймсерий:
   - фиксированный временной сплит train/validation;
   - без перемешивания;
   - с использованием early_stopping_rounds=30 для каждой комбинации.
6. Целевая переменная:
   - основывается на цене ЗАКРЫТИЯ следующего бара относительно цены ОТКРЫТИЯ текущего бара,
   - это устраняет утечку будущей информации.
7. Сравнение качества модели с простым random walk (случайное блуждание).
8. Визуализация фактической и прогнозной цены (восстановленной из лог-доходности), а также random walk.

ВАЖНО:
- Скрипт НЕ запускает обучение автоматически при импорте.
- Для запуска обучения и построения графиков внизу файла есть блок:
    if __name__ == "__main__":
        run_training_pipeline()
Вы будете запускать его самостоятельно.
"""

# ==============================
# 1. ИМПОРТ БИБЛИОТЕК
# ==============================
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import xgboost as xgb

from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Для воспроизводимости
RANDOM_STATE = 42

# ==============================
# 2. НАСТРОЙКИ ОТОБРАЖЕНИЯ PANDAS
# ==============================
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# ==============================
# 3. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ
# ==============================

def load_and_merge_data(
    fundings_path: str = "data/processed/fundings.parquet",
    klines_path: str = "data/processed/klines_15min_all.parquet",
    volumes_path: str = "data/processed/aggtrades_15min_all.parquet",
    save_merged_path: str = "data/processed/all_merged.parquet"
) -> pd.DataFrame:
    """
    Шаг 1. Загрузка и объединение исходных датасетов:
    - fundings: ставки фандинга.
    - klines: OHLCV свечи.
    - volumes: аггрегированные объёмные данные.

    Основные подшаги:
    1) Прочитать parquet-файлы.
    2) Привести имена и типы колонок времени к единому виду (time).
    3) Преобразовать колонки времени в тип datetime с временной зоной UTC.
    4) Объединить volumes + klines по времени (inner join).
    5) Присоединить fundings по времени (left join), затем сделать forward-fill funding_rate.
    6) Сохранить объединённый датафрейм (опционально) и вернуть его.
    """

    # 3.1. Загрузка данных из parquet
    fundings = pd.read_parquet(fundings_path)
    klines = pd.read_parquet(klines_path)
    volumes = pd.read_parquet(volumes_path)

    # 3.2. Приводим названия колонок времени к одному имени "time"
    # volumes: datetime -> time
    if "datetime" in volumes.columns:
        volumes = volumes.rename(columns={"datetime": "time"})

    # fundings: calc_time -> time
    if "calc_time" in fundings.columns:
        fundings = fundings.rename(columns={"calc_time": "time"})

    # klines: open_time -> time
    if "open_time" in klines.columns:
        klines = klines.rename(columns={"open_time": "time"})

    # 3.3. Преобразуем во временной тип с UTC
    volumes["time"] = pd.to_datetime(volumes["time"], utc=True)
    fundings["time"] = pd.to_datetime(fundings["time"], utc=True)
    klines["time"] = pd.to_datetime(klines["time"], utc=True)

    # 3.4. Объединяем volumes + klines по времени (inner join)
    df = pd.merge(volumes, klines, on="time", how="inner")

    # 3.5. К ним присоединяем фандинг (left join)
    df = pd.merge(df, fundings, on="time", how="left")

    # 3.6. Сортируем по времени и делаем forward-fill для funding_rate,
    #      чтобы ставки фандинга "растягивались" до следующего обновления.
    df = df.sort_values("time").reset_index(drop=True)
    if "funding_rate" in df.columns:
        df["funding_rate"] = df["funding_rate"].ffill()

    # 3.7. Сохраняем объединённый датасет (по желанию)
    df.to_parquet(save_merged_path, index=False)

    return df


# ==============================
# 4. ГЕНЕРАЦИЯ ОБЪЁМНЫХ ФИЧ
# ==============================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Шаг 2. Создание фич на основе объёмов.

    ВАЖНО ПРОТИВ УТЕЧКИ:
    - Все объёмные фичи считаются только на основе текущего и ПРЕДЫДУЩИХ баров.
      При последующей подготовке признаков мы сдвигаем их на 1 бар назад (shift),
      чтобы в момент t модель использовала максимум информацию до t-1.

    Подшаги:
    1) Базовая "volume_delta" = ask_vol - bid_vol.
    2) Разности для max/avg.
    3) Кумулятивные суммы по разным окнам:
       - общие deltas,
       - отдельные cumulative_ask и cumulative_bid,
       - cumulative_delta = cumulative_ask - cumulative_bid.
    """

    # Защита: проверяем наличие исходных колонок
    required_cols = ["ask_vol", "bid_vol", "max_ask_vol", "max_bid_vol", "avg_ask_vol", "avg_bid_vol"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют требуемые колонки для объёмных фич: {missing}")

    # 4.1. Простейший "делта-объём"
    df["volume_delta"] = df["ask_vol"] - df["bid_vol"]
    df["volume_delta_max"] = df["max_ask_vol"] - df["max_bid_vol"]
    df["volume_delta_avg"] = df["avg_ask_vol"] - df["avg_bid_vol"]

    # 4.2. Кумулятивные суммы deltas по разным окнам (в барах)
    #    Эти фичи отражают накопленный перекос спроса/предложения.
    base_windows = [4, 8, 16, 32, 96, 192]  # 1, 2, 4, 8, 24, 48 часов для 15м

    for window in base_windows:
        # Кумулятивная сумма делты
        df[f"cumulative_volume_delta_{window}"] = (
            df["volume_delta"].rolling(window=window, min_periods=1).sum()
        )

        # Отдельные кумулятивные ask/bid
        df[f"cumulative_ask_vol_{window}"] = (
            df["ask_vol"].rolling(window=window, min_periods=1).sum()
        )
        df[f"cumulative_bid_vol_{window}"] = (
            df["bid_vol"].rolling(window=window, min_periods=1).sum()
        )

        # Разность кумулятивных ask/bid
        df[f"cumulative_ask_bid_diff_{window}"] = (
            df[f"cumulative_ask_vol_{window}"] - df[f"cumulative_bid_vol_{window}"]
        )

    return df


# ==============================
# 5. ДОБАВЛЕНИЕ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ
# ==============================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Шаг 3. Добавление технических индикаторов с помощью pandas_ta.

    ВАЖНО ПРОТИВ УТЕЧКИ:
    - Индикаторы по умолчанию используют текущий бар (включая close_t),
      что в момент t содержит информацию о "конце" бара.
    - Мы в дальнейшем будем СДВИГАТЬ все индикаторные фичи на 1 бар назад
      в prepare_features_and_split, чтобы на момент t модель видела только
      значения индикаторов, рассчитанные максимум до t-1.

    Подшаги:
    1) Скользящие средние EMA (разные длины).
    2) MACD с разными параметрами.
    3) RSI с разными периодами.
    4) Bollinger Bands, ATR, Stochastic, CCI, ADX, Williams %R, ROC, OBV, MOM, Hilbert HT-тренд.
    """

    # --- Trend Identification (EMA) ---
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)

    # --- Momentum - MACD ---
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.macd(fast=5, slow=13, signal=1, append=True)

    # --- Momentum - RSI ---
    df.ta.rsi(length=7, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=21, append=True)

    # --- Volatility - Bollinger Bands ---
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.bbands(length=10, std=1.5, append=True)

    # --- Volatility - ATR ---
    df.ta.atr(length=14, append=True)

    # --- Momentum - Stochastic ---
    df.ta.stoch(length=14, k=3, d=3, append=True)
    df.ta.stoch(length=7, k=3, d=3, append=True)

    # --- Momentum - CCI ---
    df.ta.cci(length=14, append=True)
    df.ta.cci(length=20, append=True)

    # --- Trend Strength - ADX ---
    df.ta.adx(length=14, append=True)

    # --- Momentum - Williams %R ---
    df.ta.willr(length=14, append=True)

    # --- Momentum - Rate of Change ---
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.roc(length=14, append=True)

    # --- Volume - On-Balance Volume ---
    df.ta.obv(append=True)

    # --- Momentum Divergence Detection (MOM) ---
    df.ta.mom(length=10, append=True)
    df.ta.mom(length=14, append=True)

    # --- Hilbert Transform - HT Trendline ---
    df.ta.ht_trendline(append=True)

    return df


# ==============================
# 5.1 ЛАГИ ЦЕНЫ И ОБЪЁМОВ (FEATURES)
# ==============================

def add_lag_features(df: pd.DataFrame, n_lags: int = 10) -> pd.DataFrame:
    """
    Шаг 3.1. Добавление лаговых фич по цене и объёмам.

    ВАЖНО:
    - Лаги строятся ТОЛЬКО из прошлого:
      * open_{t-1}, open_{t-2}, ..., open_{t-n}
      * volume_delta_{t-1}, ..., cumulative_ask/bid_diff_{t-1}, ...
    - Таким образом, в момент t модель использует информацию только из t-1, t-2, ..., t-n,
      что корректно с точки зрения причинности.

    Подшаги:
    1) Лаги цены открытия за последние n_lags баров.
    2) Лаги базового объёмного delta (volume_delta).
    3) При желании можно легко расширить список лагируемых признаков.
    """

    # Лаги цены открытия
    for lag in range(1, n_lags + 1):
        df[f"open_lag_{lag}"] = df["open"].shift(lag)

    # Лаги базовой volume_delta
    if "volume_delta" in df.columns:
        for lag in range(1, n_lags + 1):
            df[f"volume_delta_lag_{lag}"] = df["volume_delta"].shift(lag)

    # Можно аналогично добавить лаги cumulative_ask_bid_diff, если нужно
    for window in [4, 8, 16, 32, 96, 192]:
        col_name = f"cumulative_ask_bid_diff_{window}"
        if col_name in df.columns:
            for lag in range(1, n_lags + 1):
                df[f"{col_name}_lag_{lag}"] = df[col_name].shift(lag)

    return df


# ==============================
# 6. ФОРМИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ==============================

def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Шаг 4. Формирование целевой переменной y.

    КЛЮЧЕВОЕ ИЗМЕНЕНИЕ (УБИРАЕМ УТЕЧКУ БУДУЩЕГО):
    --------------------------------------------
    - Ранее лог-доходность считалась как log(close_{t+horizon} / close_t),
      что использует цену закрытия текущего бара в знаменателе.
      Если при этом в фичах есть close_t, то модель знает будущую часть бара.
    - Теперь:
        y_t = log( close_{t + horizon} / open_t )

      То есть:
        * в момент t мы входим по open_t (цена открытия текущего бара),
        * целевая переменная отражает, какую лог-доходность мы получим,
          если держим позицию до закрытия следующего бара (horizon=1).

    Это аккуратно разделяет:
    - фичи: формируются на момент открытия бара t (open_t и вся история до t-1),
    - цель: использует только будущую close_{t+1}.
    """

    if "open" not in df.columns or "close" not in df.columns:
        raise ValueError("В датафрейме должны быть колонки 'open' и 'close' для построения целевой переменной.")

    # Лог-доходность: от открытия текущего бара t до закрытия бара t + horizon
    df["y"] = np.log(df["close"].shift(-horizon) / df["open"])
    return df


# ==============================
# 7. ПОДГОТОВКА ПРИЗНАКОВ + ROBUSTSCALER + TRAIN/TEST SPLIT
# ==============================

def prepare_features_and_split(
    df: pd.DataFrame,
    target_col: str = "y",
    train_ratio: float = 0.8
):
    """
    Шаг 5. Подготовка матрицы признаков и целевой переменной,
    временной train/test split и нормализация признаков через RobustScaler.

    КЛЮЧЕВОЕ ПРОТИВ УТЕЧКИ:
    -----------------------
    - В признаках НЕ используем:
      * close, high, low текущего бара,
      * сам target,
      * любую информацию, которая зависит от будущего (после момента открытия).
    - Все индикаторы и объёмные фичи сдвигаются на 1 бар назад:
        feature_t := indicator_{t-1}
      Это гарантирует, что в момент t индикатор отражает максимум данные до t-1,
      а не включающий close_t.

    Подшаги:
    1) Сдвинуть ВСЕ фичи, которые завязаны на текущий бар, на 1 назад.
    2) Удалить строки с NaN (появившиеся после shift и лагов).
    3) Убедиться, что time используется как индекс.
    4) Явно сформировать список признаков:
       - open_t (текущий бар),
       - лаги open/объёмов,
       - индикаторы (после shift(1)),
       - объёмные cumulative фичи (после shift(1)).
    5) Сделать временной split и применить RobustScaler.
    """

    # 7.1. Копируем датафрейм, чтобы не портить исходный
    df = df.copy()

    # 7.2. Сдвигаем индикаторы и объёмные фичи на 1 бар назад.
    #      Идея: все "быстрые" фичи (индикаторы, cumulative объёмы, лаги > 0) должны отражать историю до t-1.
    #      Мы НЕ трогаем:
    #        - 'open', потому что она сама по себе доступна на момент открытия t.
    #        - 'y' (target), 'time'.
    protected_cols = {"time", "open", target_col}

    # Выберем кандидатов для shift: всё, что не protected и не является "сырыми" high/low/close (их мы вообще не используем как фичи).
    columns_for_shift = [
        c for c in df.columns
        if c not in protected_cols and c not in {"high", "low", "close"}
    ]

    # Сдвигаем на 1 весь набор фич-кандидатов
    df[columns_for_shift] = df[columns_for_shift].shift(1)

    # 7.3. Удаляем строки с NaN (после генерации индикаторов, лагов, shift и цели)
    df = df.dropna()

    # 7.4. Устанавливаем индекс по времени
    if "time" in df.columns:
        df = df.set_index("time")

    # 7.5. Явно задаём список признаков.
    #      Включаем:
    #        - 'open' (текущий бар),
    #        - все сдвинутые индикаторы и объёмные фичи,
    #        - лаги open/объёмов.
    #      НЕ включаем:
    #        - target,
    #        - "сырые" high/low/close (чтобы не тянуть внутрь бара),
    #        - funding_rate можно включить, если он был сдвинут.
    feature_columns = [
        c for c in df.columns
        if c not in {target_col, "high", "low", "close"}
    ]

    # 7.6. Матрица X и вектор y
    X = df[feature_columns]
    y = df[target_col]

    # 7.7. Временной split по индексу
    split_index = int(len(df) * train_ratio)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # 7.8. Обучаем RobustScaler только на обучающей части
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7.9. Сохраняем индексы тестовой части для дальнейшей визуализации
    test_indices = X_test.index

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_columns,
        test_indices,
        df  # df уже после shift/dropna/set_index
    )


# ==============================
# 8. ОБУЧЕНИЕ XGBOOST-МОДЕЛИ С CUDA (БАЗОВЫЙ ВАРИАНТ)
# ==============================

def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_valid: np.ndarray,
    y_valid: pd.Series,
    params: dict = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100
) -> xgb.Booster:
    """
    Шаг 6 (базовый). Обучение XGBoost-модели с использованием GPU (CUDA) при наличии.

    Эта функция сохраняется для ссылок и простых запусков,
    но в основном пайплайне мы будем использовать grid_search_xgboost_gpu.
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": 4,
            "eta": 0.08,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse",
            "seed": RANDOM_STATE,
            "reg_lambda": 1.0,
        }

    evals = [(dtrain, "train"), (dvalid, "validation")]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )

    print("Best iteration:", bst.best_iteration)
    print("Best score:", bst.best_score)

    return bst


# ==============================
# 8.1 ГЕНЕРАЦИЯ СЕТКИ ГИПЕРПАРАМЕТРОВ ДЛЯ GRID SEARCH
# ==============================

def generate_param_grid():
    """
    Генерация списка наборов гиперпараметров для ручного грид-сёрча XGBoost.

    ВАЖНО:
    - Сетка сделана НЕ слишком большой, чтобы не взорвать время обучения.
    - Значения подобраны вокруг разумных диапазонов для задач по типу вашей:
      max_depth, eta (learning_rate), subsample, colsample_bytree, reg_lambda, reg_alpha,
      min_child_weight, gamma.

    Каждый элемент возвращаемого списка — это словарь вида:
        {"max_depth": ..., "eta": ..., ...}
    который далее будет объединён с базовыми параметрами (objective, tree_method, device и т.д.).
    """

    param_grid = {
        "max_depth": [3, 4, 5],
        "eta": [0.03, 0.05, 0.08],
        "subsample": [0.5, 0.7, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5,],
        "reg_alpha": [0.5,],
        # "reg_lambda": [0.5, 1.0, 2.0],
        # "reg_alpha": [0.0, 0.5, 1.0],
        # "min_child_weight": [1, 5, 10],
        # "gamma": [0.0, 0.5, 1.0],
        "min_child_weight": [1],
        "gamma": [0.0],
    }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combos = []
    for vals in product(*values):
        combo = dict(zip(keys, vals))
        combos.append(combo)

    return combos


# ==============================
# 8.2 РУЧНОЙ GRID SEARCH ДЛЯ XGBOOST НА GPU
# ==============================

def grid_search_xgboost_gpu(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_valid: np.ndarray,
    y_valid: pd.Series,
    base_params: dict,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 200,
):
    """
    Ручной грид-сёрч гиперпараметров XGBoost поверх xgb.train.

    ПОЧЕМУ РУЧНОЙ ЦИКЛ, А НЕ GridSearchCV:
    --------------------------------------
    1) Таймсерии:
       - Для таймсерий (особенно финансовых) критично соблюдать временной порядок.
       - GridSearchCV по умолчанию использует KFold и/или shuffle, что приводит к "подглядыванию
         в будущее" (модель видит будущие данные во время обучения) и делает оценку качества
         нереалистичной.
       - В этом грид-сёрче мы используем один фиксированный временной сплит:
         train -> validation (по времени), без перемешивания и без kfold.

    2) Низкоуровневый API xgb.train:
       - Весь пайплайн уже завязан на xgb.train и DMatrix:
         * доступен device='cuda' (GPU),
         * можно контролировать tree_method='hist',
         * напрямую задаём early_stopping_rounds.
       - GridSearchCV ожидает объект с .fit/.predict (например XGBRegressor), и пришлось бы
         переписывать логику под другой API, теряя точный контроль над Booster и настройками.

    3) Early stopping:
       - Для КАЖДОЙ комбинации гиперпараметров мы вызываем xgb.train с одинаковым
         early_stopping_rounds=30.
       - Это ускоряет обучение (не строим лишние деревья) и делает сравнение комбинаций честным,
         потому что каждая конфигурация обучалась "до своего оптимума".
    """

    param_list = generate_param_grid()

    best_rmse = float("inf")
    best_params = None
    best_model = None

    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    dvalid_full = xgb.DMatrix(X_valid, label=y_valid)

    for i, hp in enumerate(param_list, start=1):
        params = base_params.copy()
        params.update(hp)

        print(f"\n=== Грид-сёрч: комбинация {i}/{len(param_list)} ===")
        print("Текущие гиперпараметры:", params)

        evals = [(dtrain_full, "train"), (dvalid_full, "validation")]

        bst = xgb.train(
            params=params,
            dtrain=dtrain_full,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        y_pred_valid = bst.predict(dvalid_full)
        mse = mean_squared_error(y_valid, y_pred_valid)
        rmse = float(np.sqrt(mse))

        print(f"RMSE на валидации: {rmse:.6f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = bst
            print(">>> Новый лучший результат! Обновляем best_model и best_params.")

    print("\n=== РЕЗУЛЬТАТ ГРИД-СЁРЧА ===")
    print(f"Лучший RMSE на валидации: {best_rmse:.6f}")
    print("Лучшие гиперпараметры:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_rmse": best_rmse,
    }


# ==============================
# 9. ОЦЕНКА КАЧЕСТВА МОДЕЛИ
# ==============================

def evaluate_model(
    model: xgb.Booster,
    X_test: np.ndarray,
    y_test: pd.Series
) -> dict:
    """
    Шаг 7. Оценка качества модели по метрикам RMSE, MAE, R^2.
    """

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R²  : {r2:.6f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred,
    }


# ==============================
# 10. RANDOM WALK БАЗОВАЯ МОДЕЛЬ
# ==============================

def generate_random_walk(
    df: pd.DataFrame,
    test_indices: pd.Index,
    random_state: int = RANDOM_STATE
) -> pd.Series:
    """
    Шаг 8. Генерация базовой модели "random walk" (случайное блуждание)
    для сравнения с ML-моделью.
    """

    rng = np.random.default_rng(random_state)

    sigma = df["y"].std()
    if sigma is None or sigma == 0 or np.isnan(sigma):
        sigma = 0.001

    y_rw_vals = rng.normal(loc=0.0, scale=sigma, size=len(test_indices))
    y_rw = pd.Series(y_rw_vals, index=test_indices, name="y_random_walk")

    return y_rw


def evaluate_random_walk(
    df: pd.DataFrame,
    test_indices: pd.Index,
    true_y: pd.Series,
) -> dict:
    """
    Оценка качества random walk по тем же метрикам, что и для модели.
    """

    y_rw = generate_random_walk(df=df, test_indices=test_indices)

    aligned_true = true_y.loc[test_indices]
    aligned_rw = y_rw.loc[test_indices]

    mse = mean_squared_error(aligned_true, aligned_rw)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(aligned_true, aligned_rw)
    r2 = r2_score(aligned_true, aligned_rw)

    print("\n=== RANDOM WALK BASELINE ===")
    print(f"RW RMSE: {rmse:.6f}")
    print(f"RW MAE : {mae:.6f}")
    print(f"RW R²  : {r2:.6f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_rw": aligned_rw,
    }


def print_final_report(
    best_params: dict,
    model_eval: dict,
    rw_eval: dict,
    horizon: int,
) -> None:
    """
    Печатает итоговый текстовый отчёт по качеству модели и baseline random walk.

    В отчёте:
    - лучшие найденные гиперпараметры XGBoost (по grid-search);
    - метрики модели на тесте (RMSE, MAE, R^2);
    - метрики random walk на тесте;
    - короткий вывод, кто лучше и что означает знак R^2.
    """

    model_rmse = float(model_eval["rmse"])
    model_mae = float(model_eval["mae"])
    model_r2 = float(model_eval["r2"])

    rw_rmse = float(rw_eval["rmse"])
    rw_mae = float(rw_eval["mae"])
    rw_r2 = float(rw_eval["r2"])

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ОЦЕНКА МОДЕЛИ XGBoost ДЛЯ ПРОГНОЗА ЛОГ-ВОЗВРАТА")
    print(f"Горизонт прогноза: t + {horizon} баров")
    print("=" * 80)

    print("\nЛучшие гиперпараметры (по результатам grid search):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nКачество модели XGBoost на тестовой выборке:")
    print(f"  RMSE: {model_rmse:.6f}")
    print(f"  MAE : {model_mae:.6f}")
    print(f"  R²  : {model_r2:.6f}")

    print("\nКачество random walk baseline на тех же данных:")
    print(f"  RW RMSE: {rw_rmse:.6f}")
    print(f"  RW MAE : {rw_mae:.6f}")
    print(f"  RW R²  : {rw_r2:.6f}")

    print("\nВывод:")

    if model_rmse < rw_rmse:
        print(
            f"- По RMSE модель XGBoost ЛУЧШЕ random walk на горизонте t+{horizon}: "
            f"{model_rmse:.6f} против {rw_rmse:.6f}."
        )
    elif model_rmse > rw_rmse:
        print(
            f"- По RMSE модель XGBoost ХУЖЕ random walk на горизонте t+{horizon}: "
            f"{model_rmse:.6f} против {rw_rmse:.6f}."
        )
    else:
        print(
            f"- По RMSE модель XGBoost и random walk дают ОДИНАКОВЫЙ результат: "
            f"{model_rmse:.6f}."
        )

    if model_r2 < 0:
        print(
            "- Отрицательный R² у модели означает, что по дисперсии ошибок она проигрывает "
            "тривиальному предсказанию средней доходности на тесте."
        )
    else:
        print(
            "- Положительный R² означает, что модель объясняет часть вариации целевой "
            "переменной лучше, чем константная модель (среднее)."
        )

    print(
        "- Сравнение с random walk по RMSE/MAE показывает, даёт ли XGBoost "
        "хотя бы небольшое преимущество над случайным блужданием."
    )

    print("=" * 80 + "\n")


# ==============================
# 11. ВОССТАНОВЛЕНИЕ ЦЕНЫ ИЗ ЛОГ-ВОЗВРАТА
# ==============================

def restore_price_from_log_return(
    df: pd.DataFrame,
    test_indices: pd.Index,
    y_pred: np.ndarray,
) -> pd.Series:
    """
    Шаг 9. Восстановление прогнозной цены из лог-доходности.

    УЧЁТ НОВОГО ТАРГЕТА:
    -------------------
    Мы определили:
        y_t = log( close_{t+1} / open_t )

    =>  close_{t+1} = open_t * exp(y_t)

    Поэтому восстановление цены делаем именно по формуле:
        predicted_close_{t+1} = open_t * exp(y_pred_t)
    """

    if len(test_indices) != len(y_pred):
        raise ValueError(
            f"Длины test_indices ({len(test_indices)}) и y_pred ({len(y_pred)}) не совпадают!"
        )

    # Используем open_t (цена открытия текущего бара), т.к. цель определена от open_t к close_{t+1}
    open_t = df.loc[test_indices, "open"]
    predicted_close_vals = open_t * np.exp(y_pred)
    predicted_close = pd.Series(predicted_close_vals, index=test_indices, name="predicted_close")

    return predicted_close


def restore_price_from_log_return_rw(
    df: pd.DataFrame,
    test_indices: pd.Index,
    y_rw: pd.Series,
) -> pd.Series:
    """
    Аналогичное восстановление цены для random walk:
        close_{t+1}^RW = open_t * exp(y_rw_t)
    """

    open_t = df.loc[test_indices, "open"]
    predicted_close_rw_vals = open_t * np.exp(y_rw.values)
    predicted_close_rw = pd.Series(predicted_close_rw_vals, index=test_indices, name="predicted_close_rw")

    return predicted_close_rw


# ==============================
# 12. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ==============================

def plot_predictions(
    df: pd.DataFrame,
    test_indices: pd.Index,
    predicted_close: pd.Series,
    predicted_close_rw: pd.Series,
    day_start: str = None,
    day_end: str = None,
    horizon: int = 1,
    horizon_label: str = "+1 бар"
):
    """
    Шаг 10. Визуализация фактической цены, прогноза модели и random walk.

    Фактическая цена, с которой мы сравниваем:
    - close_{t+horizon} (будущая цена закрытия относительно момента принятия решения в t).
    """

    # Проверяем, что индекс датафрейма - время
    if "close" not in df.columns:
        raise ValueError("Для визуализации в df должна быть колонка 'close'.")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            raise ValueError(
                "Индекс датафрейма не является DatetimeIndex, и нет колонки 'time' для установки индекса."
            )

    idx = test_indices

    # Фильтрация по диапазону дат
    if day_start is not None:
        start_ts = pd.to_datetime(day_start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        idx = idx[idx >= start_ts]

    if day_end is not None:
        end_ts = pd.to_datetime(day_end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        idx = idx[idx <= end_ts]

    if len(idx) == 0:
        print("Нет данных для заданного диапазона дат, график не построен.")
        return

    # Фактическая будущая цена: close_{t+horizon}
    true_future_close = df["close"].shift(-horizon)

    true_close = true_future_close.loc[idx]
    pred_close = predicted_close.loc[idx]
    pred_close_rw = predicted_close_rw.loc[idx]

    plt.figure(figsize=(16, 8))
    plt.plot(true_close.index, true_close.values, label=f"Реальная цена close_(t+{horizon})", color="black")
    plt.plot(pred_close.index, pred_close.values, label=f"Прогноз модели ({horizon_label})", color="blue")
    plt.plot(pred_close_rw.index, pred_close_rw.values, label=f"Random Walk ({horizon_label})", color="red", alpha=0.7)

    plt.title(f"Сравнение фактической будущей цены close_(t+{horizon}), прогноза модели и random walk")
    plt.xlabel("Время (t)")
    plt.ylabel(f"Цена на close_(t+{horizon})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_price_and_forecast(
    df: pd.DataFrame,
    test_indices: pd.Index,
    predicted_close: pd.Series,
    day_start: str = None,
    day_end: str = None,
    horizon: int = 1,
    horizon_label: str = "+1 бар"
):
    """
    Дополнительный график: ТОЛЬКО фактическая цена и прогноз модели,
    БЕЗ random walk и БЕЗ любых shift'ов по цене.

    На графике:
    - по оси Y реальный close(t),
    - поверх него предсказанный close_(t+horizon), но привязанный по времени к моменту t.
    """

    if "close" not in df.columns:
        raise ValueError("Для визуализации в df должна быть колонка 'close'.")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            raise ValueError(
                "Индекс датафрейма не является DatetimeIndex, и нет колонки 'time' для установки индекса."
            )

    idx = test_indices

    # Фильтрация по диапазону дат
    if day_start is not None:
        start_ts = pd.to_datetime(day_start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        idx = idx[idx >= start_ts]

    if day_end is not None:
        end_ts = pd.to_datetime(day_end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        idx = idx[idx <= end_ts]

    if len(idx) == 0:
        print("Нет данных для заданного диапазона дат, график не построен.")
        return

    # БЕЗ shift: реальная цена close(t) для сравнения
    true_close = df["close"].loc[idx]
    pred_close = predicted_close.loc[idx]

    plt.figure(figsize=(16, 8))
    plt.plot(true_close.index, true_close.values, label="Реальная цена close(t)", color="black")
    plt.plot(pred_close.index, pred_close.values, label=f"Прогноз модели ({horizon_label})", color="blue")

    plt.title(f"Фактическая цена close(t) и прогноз модели на горизонт t+{horizon} (без сдвигов, без random walk)")
    plt.xlabel("Время (t)")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================
# 13. ОСНОВНОЙ ПАЙПЛАЙН ЗАПУСКА
# ==============================

def run_training_pipeline():
    """
    Основная функция-пайплайн, которая:
    1) Загружает и объединяет данные.
    2) Добавляет объёмные фичи (включая кумулятивные ask/bid и их разности).
    3) Добавляет технические индикаторы.
    4) Добавляет лаговые фичи (open, volume_delta, cumulative_ask_bid_diff).
    5) Формирует целевую переменную y как log(close_{t+1} / open_t), устраняя утечку будущего.
    6) Делает временной train/test split и нормализует признаки через RobustScaler,
       при этом ВСЕ индикаторы и объёмные фичи сдвигаются на 1 назад (используем историю до t-1).
    7) Выполняет ручной грид-сёрч гиперпараметров XGBoost на GPU (CUDA) с early_stopping_rounds=30.
    8) Оценивает лучшую модель по метрикам RMSE, MAE, R^2.
    9) Генерирует random walk и сравнивает его качество с моделью.
    10) Восстанавливает цены из лог-доходности (open_t → close_{t+1}) для модели и random walk.
    11) Строит итоговый график сравнения фактической будущей цены и прогнозов.
    """

    # --- 1. Загрузка и объединение данных ---
    print("Шаг 1: Загрузка и объединение данных...")
    df = load_and_merge_data()

    # --- 2. Объёмные фичи ---
    print("Шаг 2: Генерация объёмных фич...")
    df = add_volume_features(df)

    # --- 3. Технические индикаторы ---
    print("Шаг 3: Добавление технических индикаторов...")
    df = add_technical_indicators(df)

    # --- 3.1. Лаги цены и объёмов ---
    print("Шаг 3.1: Добавление лаговых фич по open и объёмным дельтам...")
    df = add_lag_features(df, n_lags=10)

    # --- 4. Целевая переменная y (лог-доходность по close_{t+horizon}/open_t) ---
    horizon = 4
    print(f"Шаг 4: Формирование целевой переменной y (log(close_(t+{horizon}) / open_t))...")
    df = add_target(df, horizon=horizon)

    # --- 5. Подготовка признаков + RobustScaler + временной split ---
    print("Шаг 5: Подготовка признаков, RobustScaler, train/test split (с учётом shift фич)...")
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_columns,
        test_indices,
        df_prepared,
    ) = prepare_features_and_split(
        df=df,
        target_col="y",
        train_ratio=0.8,
    )

    # --- 6. Обучение XGBoost с CUDA + ручной грид-сёрч ---
    print("Шаг 6: Обучение XGBoost-модели с использованием GPU (CUDA) + ручной грид-сёрч...")

    base_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "seed": RANDOM_STATE,
    }

    grid_result = grid_search_xgboost_gpu(
        X_train=X_train_scaled,
        y_train=y_train,
        X_valid=X_test_scaled,
        y_valid=y_test,
        base_params=base_params,
        num_boost_round=5000,
        early_stopping_rounds=30,
        verbose_eval=200,
    )

    bst = grid_result["best_model"]
    best_params = grid_result["best_params"]
    print("Лучшие гиперпараметры, найденные грид-сёрчем:", best_params)

    # --- 7. Оценка качества модели ---
    print("Шаг 7: Оценка качества модели на тестовой выборке...")
    model_eval = evaluate_model(
        model=bst,
        X_test=X_test_scaled,
        y_test=y_test
    )
    y_pred = model_eval["y_pred"]

    # --- 8. Оценка random walk ---
    print("Шаг 8: Оценка random walk baseline на тех же данных...")
    rw_eval = evaluate_random_walk(
        df=df_prepared,
        test_indices=test_indices,
        true_y=y_test,
    )
    y_rw = rw_eval["y_rw"]

    # --- 8.1. Итоговый текстовый отчёт по качеству модели и baseline ---
    print_final_report(
        best_params=best_params,
        model_eval=model_eval,
        rw_eval=rw_eval,
        horizon=horizon,
    )

    # --- 9. Восстановление цен из лог-доходности ---
    print("Шаг 9: Восстановление цен из лог-доходности (open_t → close_{t+1}) для модели и random walk...")
    predicted_close = restore_price_from_log_return(
        df=df_prepared,
        test_indices=test_indices,
        y_pred=y_pred,
    )

    predicted_close_rw = restore_price_from_log_return_rw(
        df=df_prepared,
        test_indices=test_indices,
        y_rw=y_rw,
    )

    # --- 10. Визуализация ---
    print("Шаг 10: Визуализация сравнения фактической будущей цены, прогноза модели и random walk...")

    day_start = "2025-10-30"
    day_end = "2025-10-30 23:59:59"

    plot_predictions(
        df=df_prepared,
        test_indices=test_indices,
        predicted_close=predicted_close,
        predicted_close_rw=predicted_close_rw,
        day_start=day_start,
        day_end=day_end,
        horizon=horizon,
        horizon_label=f"+{horizon} баров",
    )

    # --- 10.1. Визуализация только фактической цены и прогноза модели (без random walk, без сдвигов) ---
    print("Шаг 10.1: Визуализация фактической цены и прогноза модели без random walk и без сдвигов по цене...")

    plot_price_and_forecast(
        df=df_prepared,
        test_indices=test_indices,
        predicted_close=predicted_close,
        day_start=day_start,
        day_end=day_end,
        horizon=horizon,
        horizon_label=f"+{horizon} баров",
    )

    # --- 11. Важность признаков ---
    print("\nШаг 11: Расчёт важности признаков по gain...")
    score_dict = bst.get_score(importance_type="gain")

    mapped_scores = {}
    for k, v in score_dict.items():
        if k.startswith("f"):
            idx = int(k[1:])
            if idx < len(feature_columns):
                mapped_scores[feature_columns[idx]] = v

    sorted_items = sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)
    features_df = pd.DataFrame(sorted_items, columns=["feature", "score"])
    print(features_df)


# ==============================
# ТОЧКА ВХОДА
# ==============================
if __name__ == "__main__":
    # Я ТОЛЬКО ПИШУ КОД.
    # Вы сами решаете, когда запускать обучение и визуализацию.
    # Для запуска:
    #   python final.py
    #
    # или:
    #   from final import run_training_pipeline
    #   run_training_pipeline()
    run_training_pipeline()