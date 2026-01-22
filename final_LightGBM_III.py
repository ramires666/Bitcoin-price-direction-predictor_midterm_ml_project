# %%
"""
ФИНАЛЬНЫЙ СКРИПТ ДЛЯ ОБУЧЕНИЯ LightGBM-МОДЕЛИ ПРОГНОЗА ЛОГ-ВОЗВРАТА BTCUSDT

Основные цели:
1. Загрузка и объединение данных (klines + объёмы + фандинги) в единый датафрейм.
2. Генерация осмысленных фич (объёмы, лаги, индикаторы), построенных БЕЗ утечки будущего.
3. Нормализация входных признаков с помощью RobustScaler.
4. Обучение модели градиентного бустинга LightGBM с использованием GPU (CUDA) при наличии.
5. Ручной грид-сёрч гиперпараметров LightGBM с учётом специфики таймсерий:
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

Данный файл является вариантом исходного final.py, адаптированным под LightGBM вместо XGBoost.
"""

# ==============================
# 1. ИМПОРТ БИБЛИОТЕК
# ==============================
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, early_stopping

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
# Флаг использования GPU в LightGBM (по умолчанию отключён, т.е. используется CPU)
USE_GPU = False

# Конфигурация фич для варианта III (компактный набор индикаторов + лаги)
MAX_LAG_III = 24

# Выбранные технические индикаторы из разных групп:
# - тренд: EMA_50
# - волатильность: ATR_14
# - моментум/осциллятор: RSI_14
# - объём/поток: OBV
# - объёмные дельты: volume_delta, volume_delta_max
SELECTED_INDICATORS = [
    "RSI_14",           # моментум / осциллятор
    "EMA_50",           # тренд
    "ATR_14",           # волатильность
    "OBV",              # объём/поток
    "volume_delta",     # дельта объёмов (ask_vol - bid_vol)
    "volume_delta_max", # дельта максимумов объёмов (max_ask_vol - max_bid_vol)
]

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

def add_lag_features(
    df: pd.DataFrame,
    horizon: int,
    max_lag: int = MAX_LAG_III,
) -> pd.DataFrame:
    """
    Шаг 3.1. Добавление лаговых фич по выбранным техническим индикаторам.

    ВАРИАНТ III (аскетичный набор признаков):
    ---------------------------------------
    - Строим лаги только для нескольких заранее выбранных индикаторов из разных групп:
      тренд / волатильность / моментум / объём.
    - Для каждого индикатора создаём лаги t-1, t-2, ..., t-max_lag.
    - Само значение индикатора в момент t-1 попадает в модель через признак *_lag_1.

    Параметр horizon оставлен для симметрии с add_target и основным пайплайном,
    но на расчёт лагов напрямую не влияет.
    """

    df = df.copy()

    for col in SELECTED_INDICATORS:
        if col not in df.columns:
            # Индикатор мог не посчитаться из-за отсутствующих данных – просто пропускаем его.
            continue
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Удаляем строки с NaN, появившимися из-за лагов в начале серии
    df = df.dropna().copy()

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

    ВАРИАНТ III (компактный набор признаков):
    ----------------------------------------
    - В этой версии в модель подаётся минимальный, «сконцентрированный» набор фич:
      * несколько технических индикаторов разных типов (тренд / волатильность / моментум / объём),
      * лаги каждого индикатора от 1 до MAX_LAG_III (по умолчанию 24 баров назад).
    - В качестве признаков используются:
      * *_lag_1, ..., *_lag_MAX_LAG_III для индикаторов из SELECTED_INDICATORS,
      * по желанию — сами индикаторы в момент t-1 (без суффикса _lag_), если они присутствуют в df.

    Против утечки будущего:
    - Базовые индикаторы и другие «мгновенные» фичи (объёмы и т.п.) сдвигаются на 1 бар назад:
        indicator_t_feature := indicator_{t-1}
      Это делается для всех колонок-кандидатов, КРОМЕ лаговых признаков (*_lag_*) и служебных колонок.
    - Лаговые признаки *_lag_k уже отражают значения индикаторов в t-k и НЕ сдвигаются дополнительно.

    Подшаги:
    1) Сдвинуть на 1 бар все быстрые фичи (индикаторы, объёмы и т.п.), кроме *_lag_* и служебных колонок.
    2) Удалить строки с NaN (после генерации индикаторов, лагов, shift и цели).
    3) Убедиться, что time используется как индекс.
    4) Сформировать список признаков как:
         - лаги выбранных индикаторов (*_lag_*),
         - опционально сами индикаторы t-1 (SELECTED_INDICATORS).
    5) Сделать временной split и применить RobustScaler.
    """

    # 7.1. Копируем датафрейм, чтобы не портить исходный
    df = df.copy()

    # 7.2. Сдвигаем индикаторы и объёмные фичи на 1 бар назад,
    #      НО НЕ трогаем:
    #        - 'open' (известна на момент открытия t),
    #        - целевую переменную (target_col),
    #        - 'time',
    #        - "сырые" high/low/close (как и раньше),
    #        - лаговые признаки *_lag_* по выбранным индикаторам.
    protected_cols = {"time", "open", target_col}

    lag_prefixes = [f"{ind}_lag_" for ind in SELECTED_INDICATORS]

    columns_for_shift: list[str] = []
    for c in df.columns:
        if c in protected_cols or c in {"high", "low", "close"}:
            continue
        # Не сдвигаем лаговые признаки выбранных индикаторов
        if any(c.startswith(prefix) for prefix in lag_prefixes):
            continue
        columns_for_shift.append(c)

    if columns_for_shift:
        df[columns_for_shift] = df[columns_for_shift].shift(1)

    # 7.3. Удаляем строки с NaN (после генерации индикаторов, лагов, shift и цели)
    df = df.dropna()

    # 7.4. Устанавливаем индекс по времени
    if "time" in df.columns:
        df = df.set_index("time")

    # 7.5. Формирование компактного набора признаков для варианта III
    all_cols = df.columns

    # Лаги выбранных индикаторов
    lag_cols = [
        c
        for c in all_cols
        if any(c.startswith(f"{ind}_lag_") for ind in SELECTED_INDICATORS)
    ]

    # Опционально добавляем сами индикаторы (они уже сдвинуты на 1 бар назад выше)
    base_indicator_cols = [c for c in SELECTED_INDICATORS if c in all_cols]

    feature_columns = base_indicator_cols + lag_cols

    if not feature_columns:
        raise ValueError(
            "Не удалось сформировать список признаков для варианта III: "
            "найден пустой feature_columns. Проверьте, что индикаторы и их лаги были рассчитаны."
        )

    # 7.6. Матрица X и вектор y
    X = df[feature_columns].copy()
    y = df[target_col].copy()

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
# 8. ОБУЧЕНИЕ LIGHTGBM-МОДЕЛИ С GPU (БАЗОВЫЙ ВАРИАНТ)
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
) -> LGBMRegressor:
    """
    Шаг 6 (базовый). Обучение LightGBM-модели (по умолчанию на CPU; GPU при наличии и включённом флаге USE_GPU).

    ВНИМАНИЕ:
    - Имя функции сохранено train_xgboost_gpu для обратной совместимости с исходным скриптом,
      но внутри теперь используется LGBMRegressor (LightGBM).
    - В основном пайплайне ниже всё равно используется grid_search_xgboost_gpu,
      который также адаптирован под LightGBM.
    """

    # Базовые параметры по умолчанию, если не переданы явно
    default_params = {
        "objective": "regression",
        "learning_rate": 0.08,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        # По умолчанию используем CPU; при USE_GPU=True и наличии GPU-сборки LightGBM можно переключиться на GPU
        "device_type": "gpu" if USE_GPU else "cpu",
        "boosting_type": "gbdt",
        "n_estimators": num_boost_round,
        # Подавляем лишний логгинг LightGBM в stdout
        "verbosity": -1,
        # Опционально форсируем col-wise, чтобы избежать информационного сообщения о выборе стратегии
        "force_col_wise": True,
    }

    if params is not None:
        default_params.update(params)

    model = LGBMRegressor(**default_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
    )

    if hasattr(model, "best_iteration_"):
        print("Best iteration:", model.best_iteration_)
    else:
        print("Best iteration: not available (no early stopping information).")

    # Для совместимости с исходным кодом возвращаем обученную модель
    return model


# ==============================
# 8.1 ГЕНЕРАЦИЯ СЕТКИ ГИПЕРПАРАМЕТРОВ ДЛЯ GRID SEARCH
# ==============================

def generate_param_grid():
    """
    Генерация списка наборов гиперпараметров для ручного грид-сёрча LightGBM.

    ВАЖНО:
    - Сетка сделана НЕ слишком большой, чтобы не взорвать время обучения.
    - Значения подобраны вокруг разумных диапазонов для задач по типу вашей:
      max_depth, learning_rate (eta), subsample, colsample_bytree, reg_lambda, reg_alpha,
      min_child_weight (в LightGBM ближе всего min_child_samples), gamma (в LightGBM ближе всего min_split_gain).

    Каждый элемент возвращаемого списка — это словарь вида:
        {"max_depth": ..., "learning_rate": ..., ...}
    который далее будет объединён с базовыми параметрами LightGBM.
    """

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.5, 0.7, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5,],
        "reg_alpha": [0.5,],
        # "reg_lambda": [0.5, 1.0, 2.0],
        # "reg_alpha": [0.0, 0.5, 1.0],
        # "min_child_weight": [1, 5, 10],
        # "gamma": [0.0, 0.5, 1.0],
        "min_child_samples": [1],
        "min_split_gain": [0.0],
    }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combos = []
    for vals in product(*values):
        combo = dict(zip(keys, vals))
        combos.append(combo)

    return combos


# ==============================
# 8.2 РУЧНОЙ GRID SEARCH ДЛЯ LIGHTGBM НА GPU
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
    Ручной грид-сёрч гиперпараметров LightGBM поверх LGBMRegressor.

    ПОЧЕМУ РУЧНОЙ ЦИКЛ, А НЕ GridSearchCV:
    --------------------------------------
    1) Таймсерии:
       - Для таймсерий (особенно финансовых) критично соблюдать временной порядок.
       - GridSearchCV по умолчанию использует KFold и/или shuffle, что приводит к "подглядыванию
         в будущее" (модель видит будущие данные во время обучения) и делает оценку качества
         нереалистичной.
       - В этом грид-сёрче мы используем один фиксированный временной сплит:
         train -> validation (по времени), без перемешивания и без kfold.

    2) Контроль над пайплайном:
       - Весь пайплайн уже завязан на ручной цикл и явную передачу train/valid.
       - Мы сохраняем тот же подход, что и в исходном скрипте с XGBoost, но через LGBMRegressor
         (LightGBM), включая ручной вызов early_stopping_rounds.

    3) Early stopping:
       - Для КАЖДОЙ комбинации гиперпараметров мы вызываем .fit с одинаковым
         early_stopping_rounds=early_stopping_rounds.
       - Это ускоряет обучение (не строим лишние деревья) и делает сравнение комбинаций честным,
         потому что каждая конфигурация обучалась "до своего оптимума".
    """

    param_list = generate_param_grid()

    best_rmse = float("inf")
    best_params = None
    best_model = None

    for i, hp in enumerate(param_list, start=1):
        params = base_params.copy()
        params.update(hp)

        print(f"\n=== Грид-сёрч: комбинация {i}/{len(param_list)} ===")
        print("Текущие гиперпараметры:", params)

        model = LGBMRegressor(
            n_estimators=num_boost_round,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )

        y_pred_valid = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred_valid)
        rmse = float(np.sqrt(mse))

        print(f"RMSE на валидации: {rmse:.6f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model
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
    model: LGBMRegressor,
    X_test: np.ndarray,
    y_test: pd.Series
) -> dict:
    """
    Шаг 7. Оценка качества модели по метрикам RMSE, MAE, R^2.
    """

    y_pred = model.predict(X_test)

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
    - лучшие найденные гиперпараметры LightGBM (по grid-search);
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
    print("ИТОГОВАЯ ОЦЕНКА МОДЕЛИ LightGBM ДЛЯ ПРОГНОЗА ЛОГ-ВОЗВРАТА")
    print(f"Горизонт прогноза: t + {horizon} баров")
    print("=" * 80)

    print("\nЛучшие гиперпараметры (по результатам grid search):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nКачество модели LightGBM на тестовой выборке:")
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
            f"- По RMSE модель LightGBM ЛУЧШЕ random walk на горизонте t+{horizon}: "
            f"{model_rmse:.6f} против {rw_rmse:.6f}."
        )
    elif model_rmse > rw_rmse:
        print(
            f"- По RMSE модель LightGBM ХУЖЕ random walk на горизонте t+{horizon}: "
            f"{model_rmse:.6f} против {rw_rmse:.6f}."
        )
    else:
        print(
            f"- По RMSE модель LightGBM и random walk дают ОДИНАКОВЫЙ результат: "
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
        "- Сравнение с random walk по RMSE/MAE показывает, даёт ли LightGBM "
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
    Основная функция-пайплайн (LightGBM, вариант III с компактным набором признаков), которая:
    1) Загружает и объединяет данные.
    2) Добавляет объёмные фичи (включая кумулятивные ask/bid и их разности).
    3) Добавляет технические индикаторы.
    4) Добавляет лаговые фичи ТОЛЬКО по нескольким выбранным техиндикаторам
       (лаги от 1 до MAX_LAG_III для каждого индикатора).
    5) Формирует целевую переменную y как log(close_{t+horizon} / open_t), устраняя утечку будущего.
    6) Делает временной train/test split и нормализует признаки через RobustScaler,
       при этом базовые индикаторы и объёмные фичи сдвигаются на 1 назад (используем историю до t-1),
       а лаговые признаки *_lag_* не сдвигаются повторно.
    7) Выполняет ручной грид-сёрч гиперпараметров LightGBM (CPU по умолчанию; GPU при наличии и включённом флаге USE_GPU) с early_stopping_rounds=30.
    8) Оценивает лучшую модель по метрикам RMSE, MAE, R^2.
    9) Генерирует random walk и сравнивает его качество с моделью.
    10) Восстанавливает цены из лог-доходности (open_t → close_{t+1}) для модели и random walk.
    11) Строит итоговый график сравнения фактической будущей цены и прогнозов.

    ОСОБЕННОСТЬ ВЕРСИИ III:
    - В отличие от варианта II, здесь в модель подаются ТОЛЬКО лаговые признаки выбранных
      техиндикаторов (и, опционально, сами индикаторы в момент t-1), без сырых цен/объёмов
      и прочих дополнительных фич. Это делает набор признаков более «аскетичным» и интерпретируемым.
    """

    # --- 1. Загрузка и объединение данных ---
    print("Шаг 1: Загрузка и объединение данных...")
    df = load_and_merge_data()

    # Горизонт прогноза (используется и в add_target, и в текстах отчётов/графиков)
    horizon = 4

    # --- 2. Объёмные фичи ---
    print("Шаг 2: Генерация объёмных фич...")
    df = add_volume_features(df)

    # --- 3. Технические индикаторы ---
    print("Шаг 3: Добавление технических индикаторов...")
    df = add_technical_indicators(df)

    # --- 3.1. Лаги выбранных техиндикаторов ---
    print(f"Шаг 3.1: Добавление лаговых фич по выбранным индикаторам (1–{MAX_LAG_III} баров)...")
    df = add_lag_features(df, horizon=horizon, max_lag=MAX_LAG_III)

    # --- 4. Целевая переменная y (лог-доходность по close_{t+horizon}/open_t) ---
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

    # --- 6. Обучение LightGBM с GPU + ручной грид-сёрч ---
    print("Шаг 6: Обучение LightGBM-модели с использованием GPU (CUDA) + ручной грид-сёрч...")

    base_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        # По умолчанию используем CPU; при USE_GPU=True и наличии GPU-сборки LightGBM можно переключиться на GPU
        "device_type": "gpu" if USE_GPU else "cpu",
        "random_state": RANDOM_STATE,
        # Подавляем лишний логгинг LightGBM в stdout
        "verbosity": -1,
        # Опционально форсируем col-wise, чтобы избежать информационного сообщения о выборе стратегии
        "force_col_wise": True,
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

    # --- Дополнительная оценка практической полезности прогноза ---
    try:
        # Лог-доходности в процентах
        y_test_array = np.asarray(y_test)
        y_pred_array = np.asarray(y_pred)
        abs_true = np.abs(y_test_array)
        abs_err = np.abs(y_pred_array - y_test_array)

        avg_abs_true_pct = float(abs_true.mean() * 100.0)
        avg_abs_err_pct = float(abs_err.mean() * 100.0)
        ratio = float(avg_abs_err_pct / (avg_abs_true_pct + 1e-8))

        # Оценка в абсолютных ценах
        true_future_close = df_prepared["close"].shift(-horizon)
        test_close = true_future_close.loc[test_indices]
        aligned_test_close = test_close.loc[predicted_close.index]

        # Строим датафрейм для аккуратного выравнивания и отбрасываем NaN
        df_price_eval = pd.DataFrame(
            {
                "pred": predicted_close,
                "true": aligned_test_close,
            }
        ).dropna()

        if df_price_eval.empty:
            print(
                "\nНевозможно оценить ошибку по цене: после выравнивания не осталось валидных значений (все NaN)."
            )
            avg_abs_price_err = float("nan")
            avg_price_level = float("nan")
            avg_abs_price_err_pct_price = float("nan")
        else:
            avg_abs_price_err = float(
                np.mean(
                    np.abs(
                        df_price_eval["pred"].values - df_price_eval["true"].values
                    )
                )
            )
            avg_price_level = float(np.mean(df_price_eval["true"].values))
            avg_abs_price_err_pct_price = float(
                avg_abs_price_err / (avg_price_level + 1e-8) * 100.0
            )

        print("\n=== Оценка практической полезности прогноза ===")
        print(
            f"Средняя абсолютная величина реального движения (лог-доходность): {avg_abs_true_pct:.3f}%"
        )
        print(
            f"Средняя абсолютная ошибка прогноза (лог-доходность): {avg_abs_err_pct:.3f}%"
        )
        print(
            f"Относительная ошибка к амплитуде движения: {ratio:.3f} "
            "(чем ближе к 1, тем менее полезен прогноз)"
        )
        if np.isnan(avg_abs_price_err) or np.isnan(avg_abs_price_err_pct_price):
            print(
                "Средняя абсолютная ошибка по цене: оценку посчитать не удалось (недостаточно валидных данных)."
            )
        else:
            print(
                f"Средняя абсолютная ошибка по цене: {avg_abs_price_err:.2f} USDT "
                f"({avg_abs_price_err_pct_price:.3f}% от средней цены)"
            )

        # Краткая текстовая интерпретация метрик
        if not np.isnan(ratio):
            if ratio >= 0.9:
                print(
                    "Вывод: ошибка прогноза по лог-доходности практически равна типичному движению. "
                    "Модель даёт мало практической пользы."
                )
            elif ratio >= 0.6:
                print(
                    "Вывод: модель даёт умеренный сигнал, но значительная часть движения остаётся непредсказанной."
                )
            else:
                print(
                    "Вывод: модель объясняет заметную часть движения и может быть полезной как часть торговой системы."
                )

        if not np.isnan(avg_abs_price_err_pct_price):
            if avg_abs_price_err_pct_price <= 0.5:
                print(
                    "Ошибка по цене менее ~0.5% от средней цены — практическая ценность прогноза может быть низкой."
                )
            elif avg_abs_price_err_pct_price <= 1.5:
                print(
                    "Ошибка по цене около 0.5–1.5% от средней цены — сигнал может быть погранично полезным."
                )
            else:
                print(
                    "Ошибка по цене больше 1.5% от средней цены — при достаточной частоте таких движений "
                    "сигнал может быть практически полезным."
                )
    except Exception as e:
        print("\nНе удалось вычислить блок практической полезности прогноза:", repr(e))

    # --- 11. Важность признаков ---
    print("\nШаг 11: Расчёт важности признаков по gain (LightGBM)...")
    booster = bst.booster_ if hasattr(bst, "booster_") else bst
    importance_gain = booster.feature_importance(importance_type="gain")

    # Используем реальные имена признаков из prepare_features_and_split
    if len(importance_gain) != len(feature_columns):
        print(
            f"ВНИМАНИЕ: длина importance_gain ({len(importance_gain)}) "
            f"не совпадает с количеством признаков ({len(feature_columns)})."
        )
        print("Будут использованы имена признаков из booster.feature_name().")
        feature_names = booster.feature_name()
        features_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "score": importance_gain,
                }
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
    else:
        features_df = (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "score": importance_gain,
                }
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

    print(features_df.head(50))


# ==============================
# ТОЧКА ВХОДА
# ==============================
if __name__ == "__main__":
    # Я ТОЛЬКО ПИШУ КОД.
    # Вы сами решаете, когда запускать обучение и визуализацию.
    # Для запуска третьей версии LightGBM-скрипта:
    #   python final_LightGBM_III.py
    #
    # или:
    #   from final_LightGBM_III import run_training_pipeline
    #   run_training_pipeline()
    run_training_pipeline()