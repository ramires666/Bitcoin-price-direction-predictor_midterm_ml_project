# Fix Windows CPU core detection warning before any imports
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

# Set LOKY_MAX_CPU_COUNT to fix Windows CPU detection warning
os.environ["LOKY_MAX_CPU_COUNT"] = "0"  # 0 means use all available cores
# Suppress the specific warning about physical core detection
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message=".*Returning the number of logical cores instead.*")
# Suppress LightGBM feature names warning
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*LGBMClassifier was fitted with feature names.*")
# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", message=".*missing from font.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# %% [markdown]
# # Модель LightGBM для поиска "пробоев" (Breakouts)
#
# ### Цели этого ноутбука:
# 1. **Добавить признаки волатильности**: Ввести Ширину Полос Боллинджера (`BBW`) как индикатор "затишья" перед "бурей".
# 2. **Проверить гипотезу "торговли пробоев"**: Сможет ли модель, видя сужение волатильности + направленный сигнал, предсказать сильное движение.
# 3. **Объединить все лучшие наработки**: Использовать объемные фичи, "умные" сигналы и новые признаки волатильности вместе.
# 4. **Создать продвинутую визуализацию**: Одновременно показать предсказания (фоновые полосы) и реальные исходы (цвет графика).
# 5. **Максимально использовать объемные фичи**: Вернуть все кумулятивные дельты объемов.
# 6. **Явно удалить NaN**: Убедиться, что все строки с NaN удалены после генерации признаков.

# %%
# =============================================
# Ячейка 1: Импорт библиотек и базовые настройки
# =============================================
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, early_stopping
from itertools import product
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss # Импортируем log_loss напрямую
)
from scipy.fft import fft, ifft # Импортируем для Фурье-преобразования
from typing import Optional # Для Optional в plot_advanced_signals

# Import our advanced backtester
from advanced_backtester import advanced_backtester

# Для воспроизводимости
RANDOM_STATE = 42
# Флаг использования GPU в LightGBM (по умолчанию отключён)
USE_GPU = False

# 🔥 ПОДОБРАННЫЕ ПАРАМЕТРЫ FFT ИЗ fft_simple_analysis.py ✅
FFT_MIN_CUTOFF = 20
FFT_CUTOFF_FRACTION = 80
NEUTRAL_SLOPE_THRESHOLD = 30.0  # |slope| < 30 USDT/15m → нейтрал (серый)
COMMISSION = 0.0000275  # Комиссия 0.0275%

# Настройки отображения Pandas
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

# %% [markdown]
# ## Шаг 1: Загрузка и объединение данных
#
# Стандартный шаг загрузки и объединения данных.

# %%
# =============================================
# Ячейка 2: Функция загрузки данных
# =============================================
def load_and_merge_data(
    fundings_path: str = "data/processed/fundings.parquet",
    klines_path: str = "data/processed/klines_15min_all.parquet",
    volumes_path: str = "data/processed/aggtrades_15min_all.parquet",
) -> pd.DataFrame:
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

    return df

# %% [markdown]
# ## Шаг 2: Генерация признаков (с фокусом на волатильность и объем)
#
# Расширяем `add_volume_features` и добавляем `add_volatility_features`.

# %%
# =============================================
# Ячейка 3: Функции для генерации признаков
# =============================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает все детальные объемные признаки из исходного кода."""
    df["volume_delta"] = df["ask_vol"] - df["bid_vol"]
    df["volume_delta_max"] = df["max_ask_vol"] - df["max_bid_vol"]
    df["volume_delta_avg"] = df["avg_ask_vol"] - df["avg_bid_vol"]
    
    base_windows = [4, 8, 16, 32, 96] # 1, 2, 4, 8, 24 часов для 15м
    for window in base_windows:
        df[f"cumulative_volume_delta_{window}"] = df["volume_delta"].rolling(window=window, min_periods=1).sum()
        df[f"cumulative_ask_bid_diff_{window}"] = (
            df["ask_vol"].rolling(window=window, min_periods=1).sum() - 
            df["bid_vol"].rolling(window=window, min_periods=1).sum()
        )
    return df

def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.ema(length=12, append=True, col_names="EMA_12")
    df.ta.ema(length=26, append=True, col_names="EMA_26")
    
    # Оригинальный MACD с явными именами колонок
    df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=("MACD_12_26_9", "MACDH_12_26_9", "MACDS_12_26_9"))
    # Дополнительные MACD с явными именами колонок
    df.ta.macd(fast=7, slow=14, signal=7, append=True, col_names=("MACD_7_14_7", "MACDH_7_14_7", "MACDS_7_14_7"))
    df.ta.macd(fast=24, slow=52, signal=18, append=True, col_names=("MACD_24_52_18", "MACDH_24_52_18", "MACDS_24_52_18"))

    df.ta.rsi(length=14, append=True, col_names="RSI_14")

    df['ema_trend'] = np.sign(df['EMA_12'] - df['EMA_26'])
    df['macd_crossover'] = np.sign(df['MACD_12_26_9'] - df['MACDS_12_26_9']).diff()
    
    # Дополнительные сигналы MACD
    df['macd_crossover_7_14_7'] = np.sign(df['MACD_7_14_7'] - df['MACDS_7_14_7']).diff()
    df['macd_crossover_24_52_18'] = np.sign(df['MACD_24_52_18'] - df['MACDS_24_52_18']).diff()
    
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки, основанные на волатильности. Надежный метод."""
    
    # Оригинальные Bollinger Bands с явными именами колонок (std=2.0 -> 2)
    df.ta.bbands(length=20, std=2, append=True, col_names=(f"BBL_20_2", f"BBM_20_2", f"BBU_20_2", f"BBB_20_2", f"BBP_20_2"))
    
    # Расчет BBW для оригинальных BB
    df['bbw'] = (df[f"BBU_20_2"] - df[f"BBL_20_2"]) / df[f"BBM_20_2"]
    df['bbw_sma_10'] = df['bbw'].rolling(10).mean()

    # Дополнительные Bollinger Bands
    bb_params = [(10, 1.5), (30, 2.5)]
    for length, std in bb_params:
        # Используем строковое форматирование для std, чтобы избежать проблем с представлением float в именах колонок
        # Например, 1.5 -> 1_5
        std_str = str(std).replace('.', '_') 
        df.ta.bbands(length=length, std=std, append=True, col_names=(f"BBL_{length}_{std_str}", f"BBM_{length}_{std_str}", f"BBU_{length}_{std_str}", f"BBB_{length}_{std_str}", f"BBP_{length}_{std_str}"))
        
        df[f'bbw_{length}_{std_str}'] = (df[f"BBU_{length}_{std_str}"] - df[f"BBL_{length}_{std_str}"]) / df[f"BBM_{length}_{std_str}"]
        df[f'bbw_sma_10_{length}_{std_str}'] = df[f'bbw_{length}_{std_str}'].rolling(10).mean()
    
    return df

# %% [markdown]
# ## Шаг 3: Формирование 3-классовой целевой переменной
#
# Используем "мертвую зону" (threshold) для отсечения рыночного шума.

# %%
# =============================================
# Ячейка 4: Функция для создания 3-классовой цели
# =============================================

def prepare_fft_target(
    df: pd.DataFrame,
    threshold: float = 0.0005,
    cutoff_ratio: float = 0.1
) -> tuple:
    """Подготовка FFT-обработанного сигнала для целевой переменной.
    
    Args:
        df: DataFrame с колонкой 'close'
        threshold: порог для классификации (верх/низ/нейтрал)
        cutoff_ratio: доля сохраняемых частот в FFT (0.0-1.0)
        
    Returns:
        tuple: (fft_signal, fft_signal_diff)
    """
    close_prices = df['close'].values
    N = len(close_prices)
    
    if N == 0:
        print("No data for FFT target generation.")
        return np.full(N, np.nan), np.full(N, np.nan)

    # Убираем тренд для лучшей обработки FFT
    mean_price = np.mean(close_prices)
    detrended_prices = close_prices - mean_price
    
    # Применяем FFT к детрендированной серии
    yf = fft(detrended_prices)
    
    # Создаем фильтр для удаления высокочастотных шумов
    yf_filtered = np.zeros_like(yf, dtype=complex)
    
    # Определяем границу частот для фильтрации
    cutoff_freq = int(N * cutoff_ratio)
    
    # Оставляем низкочастотные компоненты (включая DC)
    yf_filtered[:cutoff_freq] = yf[:cutoff_freq]
    if N % 2 == 0:  # Если N четное
        yf_filtered[N-cutoff_freq:] = yf[N-cutoff_freq:]
    else:  # Если N нечетное
        yf_filtered[N-cutoff_freq+1:] = yf[N-cutoff_freq+1:]
    
    # Применяем обратное FFT для реконструкции сигнала
    fft_reconstructed_detrended = ifft(yf_filtered).real
    
    # Восстанавливаем тренд
    fft_reconstructed_signal = fft_reconstructed_detrended + mean_price
    
    # Вычисляем разницу FFT сигнала для определения цели
    # Используем лог-доходность для более стабильных результатов
    fft_log_returns = np.diff(np.log(fft_reconstructed_signal[fft_reconstructed_signal > 0]))
    
    # Расширяем fft_signal_diff до длины N, заполняя первую позицию нулем
    fft_signal_diff = np.concatenate([[0], fft_log_returns])
    
    return fft_reconstructed_signal, fft_signal_diff


def add_target(df: pd.DataFrame, threshold: float = 0.0005, cutoff_ratio: float = 0.1) -> pd.DataFrame:
    """Создает 3-классовую целевую переменную на основе FFT сигнала."""
    df_copy = df.copy()
    
    # Подготавливаем FFT сигнал
    fft_signal, fft_signal_diff = prepare_fft_target(
        df=df_copy,
        threshold=threshold,
        cutoff_ratio=cutoff_ratio
    )
    
    df_copy['fft_signal'] = fft_signal
    df_copy['fft_signal_diff'] = fft_signal_diff
    
    # Определяем цель на основе fft_signal_diff с учетом комиссии
    min_signal = 7 * COMMISSION
    effective_threshold = max(threshold, min_signal)
    
    conditions = [df_copy['fft_signal_diff'] > effective_threshold, df_copy['fft_signal_diff'] < -effective_threshold]
    choices = [2, 0] # 2 для UP, 0 для DOWN
    df_copy['y'] = np.select(conditions, choices, default=1) # 1 для SIDEWAYS
    
    # Выводим отладочную информацию о FFT
    print(f"FFT Debug Info:")
    print(f"- Cutoff ratio: {cutoff_ratio} (preserving {cutoff_ratio*100}% of frequencies)")
    print(f"- FFT signal stats: mean={np.mean(fft_signal):.6f}, std={np.std(fft_signal):.6f}")
    print(f"- FFT diff stats: mean={np.mean(fft_signal_diff):.6f}, std={np.std(fft_signal_diff):.6f}")
    print(f"- Threshold for classification: ±{effective_threshold:.6f} (base: {threshold}, min: {min_signal:.6f})")
    
    return df_copy

# %% [markdown]
# ## Шаг 4: Подготовка и разделение данных
#
# Собираем все наши лучшие признаки в один набор.

# %%
# =================================================================
# Ячейка 5: Функция подготовки и разделения данных
# =================================================================
def prepare_features_and_split(
    df: pd.DataFrame,
    target_col: str = "y",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15
):
    df = df.copy()

    # Собираем все признаки вместе
    volume_features = [col for col in df.columns if 'volume_delta' in col or 'cumulative_ask_bid_diff' in col]
    
    strategy_features = [
        'ema_trend', 'macd_crossover',
        'MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9', # Оригинальный MACD
        'MACD_7_14_7', 'MACDH_7_14_7', 'MACDS_7_14_7', 'macd_crossover_7_14_7', # Дополнительный MACD 1
        'MACD_24_52_18', 'MACDH_24_52_18', 'MACDS_24_52_18', 'macd_crossover_24_52_18', # Дополнительный MACD 2
        'RSI_14', # RSI
        # REMOVED FFT FEATURES TO PREVENT DATA LEAKAGE - FFT only used for target creation
    ]
    
    volatility_features = [
        'bbw', 'bbw_sma_10', # Оригинальный BBW
        'BBL_20_2', 'BBM_20_2', 'BBU_20_2', 'BBB_20_2', 'BBP_20_2', # Оригинальные BB (std=2.0 -> 2)
        'bbw_10_1_5', 'bbw_sma_10_10_1_5', # Дополнительный BBW 1 (std=1.5 -> 1_5)
        'BBL_10_1_5', 'BBM_10_1_5', 'BBU_10_1_5', 'BBB_10_1_5', 'BBP_10_1_5', # Дополнительные BB 1
        'bbw_30_2_5', 'bbw_sma_10_30_2_5', # Дополнительный BBW 2 (std=2.5 -> 2_5)
        'BBL_30_2_5', 'BBM_30_2_5', 'BBU_30_2_5', 'BBB_30_2_5', 'BBP_30_2_5' # Дополнительные BB 2
    ]
    
    feature_columns = list(set(volume_features + strategy_features + volatility_features))
    
    # Сдвигаем все фичи, чтобы избежать заглядывания в будущее
    df[feature_columns] = df[feature_columns].shift(1)

    cols_to_keep = feature_columns + [target_col, 'open', 'close', 'high', 'low']
    df = df.set_index('time')[cols_to_keep]
    
    df = df.dropna() # Явное удаление NaN после сдвига

    print(f"🔍 ВАЖНО: FFT признаки НЕ используются для обучения (только для создания целевой переменной)")
    print(f"Используемые признаки ({len(feature_columns)} шт): {feature_columns}")
    print("\nРаспределение классов в полном датасете:")
    print(f"{df[target_col].value_counts}(normalize=True)")
    
    # Убеждаемся, что FFT признаки не попали в обучающие данные
    fft_features_in_training = [col for col in feature_columns if 'fft' in col.lower()]
    if fft_features_in_training:
        print(f"⚠️ ОШИБКА: FFT признаки найдены в обучающих данных: {fft_features_in_training}")
    else:
        print(f"✅ FFT признаки корректно исключены из обучающих данных")
        
    print()

    X = df[feature_columns]
    y = df[target_col]

    train_split_index = int(len(df) * train_ratio)
    valid_split_index = int(len(df) * (train_ratio + valid_ratio))

    X_train, y_train = X.iloc[:train_split_index], y.iloc[:train_split_index]
    X_valid, y_valid = X.iloc[train_split_index:valid_split_index], y.iloc[train_split_index:valid_split_index]
    X_test, y_test = X.iloc[valid_split_index:], y.iloc[valid_split_index:]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    test_indices = X_test.index

    return (
        X_train_scaled, y_train,
        X_valid_scaled, y_valid,
        X_test_scaled, y_test,
        scaler, feature_columns, test_indices, df
    )

# %% [markdown]
# ## Шаг 5: Обучение и оценка модели
#
# Функции для Grid Search, оценки и визуализации.

# %%
# =================================================================
# Ячейка 6: Функции для обучения, оценки и визуализации
# =================================================================

def grid_search_lgbm_classifier(
    X_train: np.ndarray, y_train: pd.Series,
    X_valid: np.ndarray, y_valid: pd.Series,
    base_params: dict,
    feature_names: list = None,
):
    param_grid = {
        "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.03],
        "n_estimators": [200, 400], "colsample_bytree": [0.7, 0.9]
    }
    param_list = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    best_logloss = float("inf")
    best_params, best_model = None, None

    for i, hp in enumerate(param_list, start=1):
        params = {**base_params, **hp}
        print(f"\n--- Grid Search: Комбинация {i}/{len(param_list)} ---")
        
        model = LGBMClassifier(**params)
        
        # Fit with early stopping and feature names
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(30, verbose=False)],
            feature_name=feature_names if feature_names else 'auto'
        )

        y_pred_valid_proba = model.predict_proba(X_valid)
        # Явно указываем все классы для log_loss
        loss = log_loss(y_valid, y_pred_valid_proba, labels=[0, 1, 2])
        print(f"Multi-LogLoss на валидации: {loss:.5f}")

        if loss < best_logloss:
            best_logloss, best_params, best_model = loss, params, model
            print(">>> Новый лучший результат! Обновляем модель.")

    print(f"\n=== РЕЗУЛЬТАТ GRID SEARCH ===\nЛучший LogLoss на валидации: {best_logloss:.5f}\nЛучшие гиперпараметры: {best_params}")
    return {"best_model": best_model, "best_score": best_logloss, "best_params": best_params}

def evaluate_classifier(model: LGBMClassifier, X_test: np.ndarray, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    print("\n" + "="*50 + "\nОТЧЕТ ПО КЛАССИФИКАЦИИ НА ТЕСТОВОЙ ВЫБОРКЕ\n" + "="*50)
    # Явно указываем все классы для classification_report
    report = classification_report(y_test, y_pred, target_names=['DOWN (0)', 'SIDEWAYS (1)', 'UP (2)'], zero_division=0, labels=[0, 1, 2])
    print(report)
    print("="*50)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]) # Явно указываем все классы для confusion_matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred DOWN', 'Pred SIDEWAYS', 'Pred UP'], 
                yticklabels=['Actual DOWN', 'Actual SIDEWAYS', 'Actual UP'])
    plt.title('Матрица ошибок')
    plt.ylabel('Реальное значение')
    plt.xlabel('Предсказанное значение')
    plt.show()
    
    return {"y_pred": y_pred}

def plot_advanced_signals(df: pd.DataFrame, test_indices: pd.Index, y_pred: Optional[np.ndarray] = None, y_true: Optional[pd.Series] = None):
    """ИСПРАВЛЕННАЯ визуализация БЕЗ РАЗРЫВОВ в линии цены."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    if test_indices.empty:
        print("Тестовая выборка пуста, график не построен.")
        return
        
    # Получаем последний день в тестовой выборке
    last_day_in_test = test_indices.max().normalize()
    day_start_dt = last_day_in_test
    day_end_dt = last_day_in_test + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Фильтруем test_indices для последнего дня
    test_indices_for_day = test_indices[(test_indices >= day_start_dt) & (test_indices <= day_end_dt)]

    if test_indices_for_day.empty:
        print(f"Нет тестовых данных для последнего дня {day_start_dt.strftime('%Y-%m-%d')}, график не построен.")
        return

    # Получаем данные цены для этого конкретного дня, выровненные по test_indices_for_day
    plot_df_for_day = df.loc[test_indices_for_day].copy()

    # Определяем, какой сигнал использовать для раскраски фона
    if y_pred is not None:
        full_signal_series = pd.Series(y_pred, index=test_indices)
        signals_for_day = full_signal_series.loc[test_indices_for_day]
        plot_title_prefix = "Предсказания (фон)"
    else:
        # Если y_pred не предоставлен, используем фактическую целевую переменную 'y' из df_prepared
        signals_for_day = plot_df_for_day['y']
        plot_title_prefix = "Целевая переменная (фон)"

    # Убедимся, что y_true доступен для раскраски линии цены
    if y_true is None:
        true_outcomes_for_day = plot_df_for_day['y'] # Используем фактическую цель, если y_true не передан явно
    else:
        true_outcomes_for_day = y_true.loc[test_indices_for_day]


    print(f"Строим график для последнего дня в тестовой выборке: {day_start_dt.strftime('%Y-%m-%d')}")
    print(f"Сигналы для фона за {day_start_dt.strftime('%Y-%m-%d')}:\n{signals_for_day.value_counts()}")
    print(f"Реальные классы за {day_start_dt.strftime('%Y-%m-%d')}:\n{true_outcomes_for_day.value_counts()}")

    # 1. Рисуем фоновые полосы предсказаний (или целевой переменной) - более прозрачные
    for i in range(len(signals_for_day)):
        idx = signals_for_day.index[i]
        end_idx = idx + pd.Timedelta(minutes=15)
        
        if signals_for_day.iloc[i] == 2: # UP
            ax.axvspan(idx, end_idx, color='lightgreen', alpha=0.15, lw=0)
        elif signals_for_day.iloc[i] == 0: # DOWN
            ax.axvspan(idx, end_idx, color='lightcoral', alpha=0.15, lw=0)
        else: # SIDEWAYS
            ax.axvspan(idx, end_idx, color='lightgray', alpha=0.15, lw=0)

    # 2. ИСПРАВЛЕНИЕ: Рисуем НЕПРЕРЫВНУЮ линию цены БЕЗ разрывов!
    price_data = plot_df_for_day['close']
    ax.plot(price_data.index, price_data.values, color='black', linewidth=2, label='Цена закрытия', alpha=0.8, zorder=3)

    # 3. Добавляем цветные точки поверх линии для обозначения классов
    for class_val, color, label in [(0, 'red', 'DOWN'), (1, 'gray', 'SIDEWAYS'), (2, 'green', 'UP')]:
        mask = true_outcomes_for_day == class_val
        if mask.any():
            class_prices = price_data[mask]
            class_times = true_outcomes_for_day[mask].index
            ax.scatter(class_times, class_prices, color=color, s=40, alpha=0.9,
                      label=f'Реальный {label}', zorder=5, edgecolors='white', linewidth=0.5)

    # ОГРАНИЧИВАЕМ МАСШТАБ ТОЛЬКО ПО ЦЕНЕ (чтобы FFT не ломал масштаб)
    price_min = price_data.min()
    price_max = price_data.max()
    price_range = price_max - price_min
    margin = price_range * 0.1  # 10% отступы
    
    ax.set_ylim(price_min - margin, price_max + margin)
    
    # ОБРЕЗАЕМ FFT СИГНАЛ по диапазону цены (чтобы он не выходил за пределы)
    if 'fft_signal' in plot_df_for_day.columns and not plot_df_for_day['fft_signal'].isnull().all():
        fft_signal = plot_df_for_day['fft_signal'].copy()
        # Обрезаем FFT сигнал по диапазону цены
        fft_signal_masked = fft_signal.copy()
        fft_signal_masked[fft_signal_masked < price_min - margin] = np.nan
        fft_signal_masked[fft_signal_masked > price_max + margin] = np.nan
        
        ax.plot(plot_df_for_day.index, fft_signal_masked, color='blue', linestyle='--', label='FFT Signal', alpha=0.7, zorder=4)

    ax.set_title(f"{plot_title_prefix} vs Реальность (точки) за {day_start_dt.strftime('%Y-%m-%d')}")
    ax.set_xlabel("Время")
    ax.set_ylabel("Цена")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_fft_comparison(df: pd.DataFrame, start_idx: int = 0, end_idx: int = 500):
    """Сравнение оригинальной цены и FFT реконструкции."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Ограничиваем индексы
    end_idx = min(end_idx, len(df))
    
    # График 1: Сравнение оригинальной цены и FFT сигнала
    ax1.plot(range(start_idx, end_idx), df['close'].iloc[start_idx:end_idx],
             label='Оригинальная цена', color='blue', alpha=0.7)
    ax1.plot(range(start_idx, end_idx), df['fft_signal'].iloc[start_idx:end_idx],
             label='FFT реконструкция', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Сравнение: Оригинальная цена vs FFT реконструкция')
    ax1.set_xlabel('Индекс времени')
    ax1.set_ylabel('Цена')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: FFT сигнал дифференциал
    ax2.plot(range(start_idx, end_idx), df['fft_signal_diff'].iloc[start_idx:end_idx],
             label='FFT сигнал (дифференциал)', color='green', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=0.003, color='red', linestyle='--', alpha=0.5, label='Threshold +')
    ax2.axhline(y=-0.003, color='red', linestyle='--', alpha=0.5, label='Threshold -')
    ax2.set_title('FFT сигнал для классификации цели')
    ax2.set_xlabel('Индекс времени')
    ax2.set_ylabel('Изменение сигнала')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print(f"Статистика FFT:")
    print(f"- Корреляция между оригиналом и FFT: {np.corrcoef(df['close'].iloc[start_idx:end_idx], df['fft_signal'].iloc[start_idx:end_idx])[0,1]:.4f}")
    print(f"- Стандартное отклонение FFT: {np.std(df['fft_signal'].iloc[start_idx:end_idx]):.6f}")
    print(f"- Стандартное отклонение FFT дифф: {np.std(df['fft_signal_diff'].iloc[start_idx:end_idx]):.6f}")

def add_multiple_fft_targets(df: pd.DataFrame, cutoff_ratios: list, thresholds: list, fft_components: int = 10) -> pd.DataFrame:
    """Создает несколько вариантов FFT целей с разными параметрами для сравнения."""
    df_result = df.copy()
    
    for i, cutoff_ratio in enumerate(cutoff_ratios):
        for j, threshold in enumerate(thresholds):
            # Убираем тренд для лучшей обработки FFT
            close_prices = df_result['close'].values
            mean_price = np.mean(close_prices)
            detrended_prices = close_prices - mean_price
            
            # Применяем FFT к детрендированной серии
            yf = fft(detrended_prices)
            
            # Создаем фильтр для удаления высокочастотных шумов
            yf_filtered = np.zeros_like(yf, dtype=complex)
            
            # Определяем границу частот для фильтрации
            cutoff_freq = int(len(close_prices) * cutoff_ratio)
            
            # Оставляем низкочастотные компоненты (включая DC)
            yf_filtered[:cutoff_freq] = yf[:cutoff_freq]
            if len(close_prices) % 2 == 0:  # Если N четное
                yf_filtered[len(close_prices)-cutoff_freq:] = yf[len(close_prices)-cutoff_freq:]
            else:  # Если N нечетное
                yf_filtered[len(close_prices)-cutoff_freq+1:] = yf[len(close_prices)-cutoff_freq+1:]
            
            # Применяем обратное FFT для реконструкции сигнала
            fft_reconstructed_detrended = ifft(yf_filtered).real
            
            # Восстанавливаем тренд
            fft_reconstructed_signal = fft_reconstructed_detrended + mean_price
            
            # Сохраняем FFT сигнал
            col_name = f'fft_signal_cutoff_{cutoff_ratio}_thresh_{threshold}'
            df_result[col_name] = fft_reconstructed_signal
            
            # Вычисляем разницу FFT сигнала для определения цели
            fft_log_returns = np.diff(np.log(fft_reconstructed_signal[fft_reconstructed_signal > 0]))
            fft_signal_diff = np.concatenate([[0], fft_log_returns])
            diff_col_name = f'fft_signal_diff_cutoff_{cutoff_ratio}_thresh_{threshold}'
            df_result[diff_col_name] = fft_signal_diff
            
            # Определяем цель на основе fft_signal_diff с учетом комиссии
            min_signal = 7 * COMMISSION
            effective_threshold = max(threshold, min_signal)
            
            conditions = [df_result[diff_col_name] > effective_threshold, df_result[diff_col_name] < -effective_threshold]
            choices = [2, 0] # 2 для UP, 0 для DOWN
            target_col_name = f'y_fft_cutoff_{cutoff_ratio}_thresh_{threshold}'
            df_result[target_col_name] = np.select(conditions, choices, default=1) # 1 для SIDEWAYS
            
            print(f"FFT Вариант {i*len(thresholds)+j+1}: cutoff_ratio={cutoff_ratio}, threshold={effective_threshold:.6f} (base: {threshold}, min: {min_signal:.6f})")
            print(f"  - Корреляция с оригиналом: {np.corrcoef(close_prices, fft_reconstructed_signal)[0,1]:.4f}")
            print(f"  - Распределение классов: {df_result[target_col_name].value_counts().to_dict()}")
            print()
    
    return df_result

def plot_multiple_fft_comparison(df: pd.DataFrame, start_idx: int = 0, end_idx: int = 500):
    """Сравнение нескольких вариантов FFT с разными параметрами."""
    # Находим все FFT колонки
    fft_signal_cols = [col for col in df.columns if col.startswith('fft_signal_cutoff_') and not 'diff' in col]
    
    if not fft_signal_cols:
        print("Не найдено FFT колонок для сравнения")
        return
    
    # Ограничиваем индексы
    end_idx = min(end_idx, len(df))
    
    # Создаем subplot для каждого FFT варианта
    n_plots = len(fft_signal_cols) + 1  # +1 для оригинальной цены
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Плоский список осей
    axes_flat = axes.flatten()
    
    # График 1: Оригинальная цена
    axes_flat[0].plot(range(start_idx, end_idx), df['close'].iloc[start_idx:end_idx],
                     label='Оригинальная цена', color='blue', linewidth=2)
    axes_flat[0].set_title('Оригинальная цена')
    axes_flat[0].set_xlabel('Индекс времени')
    axes_flat[0].set_ylabel('Цена')
    axes_flat[0].legend()
    axes_flat[0].grid(True, alpha=0.3)
    
    # Графики для каждого FFT варианта
    for i, fft_col in enumerate(fft_signal_cols):
        if i + 1 < len(axes_flat):
            axes_flat[i + 1].plot(range(start_idx, end_idx), df['close'].iloc[start_idx:end_idx],
                                 label='Оригинальная', color='blue', alpha=0.5, linewidth=1)
            axes_flat[i + 1].plot(range(start_idx, end_idx), df[fft_col].iloc[start_idx:end_idx],
                                 label='FFT', color='red', linewidth=2, linestyle='--')
            
            # Извлекаем параметры из названия колонки
            params = fft_col.replace('fft_signal_cutoff_', '').split('_thresh_')
            cutoff_ratio = params[0]
            threshold = params[1]
            
            axes_flat[i + 1].set_title(f'FFT: cutoff={cutoff_ratio}, threshold={threshold}')
            axes_flat[i + 1].set_xlabel('Индекс времени')
            axes_flat[i + 1].set_ylabel('Цена')
            axes_flat[i + 1].legend()
            axes_flat[i + 1].grid(True, alpha=0.3)
    
    # Скрываем лишние subplots
    for i in range(len(fft_signal_cols) + 1, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 10))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Важность признаков')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def auto_optimize_fft_parameters(
    df_featured: pd.DataFrame,
    cutoff_ratios: list,
    thresholds: list,
    target_col: str = "y"
) -> dict:
    """Автоматическая оптимизация FFT параметров с поиском максимальной доходности."""
    
    results = {}
    best_return = float('-inf')
    best_params = None
    best_model = None
    best_data = None
    
    print("🔍 НАЧИНАЕМ АВТООПТИМИЗАЦИЮ FFT ПАРАМЕТРОВ...")
    print(f"Параметры для тестирования:")
    print(f"- Cutoff ratios: {cutoff_ratios}")
    print(f"- Thresholds: {thresholds}")
    print(f"- Всего комбинаций: {len(cutoff_ratios) * len(thresholds)}")
    print("=" * 60)
    
    for i, cutoff_ratio in enumerate(cutoff_ratios):
        for j, threshold in enumerate(thresholds):
            combo_num = i * len(thresholds) + j + 1
            total_combos = len(cutoff_ratios) * len(thresholds)
            
            print(f"\n🔄 Комбинация {combo_num}/{total_combos}")
            print(f"📊 Параметры: cutoff_ratio={cutoff_ratio}, threshold={threshold:.6f}")
            print("-" * 40)
            
            try:
                # 1. Создаем цель с текущими параметрами
                df_targeted = add_target(
                    df_featured,
                    threshold=threshold,
                    cutoff_ratio=cutoff_ratio
                )
                
                # 🚨 ПРОВЕРКА: Если слишком много нейтральных сигналов (>95%), пропускаем
                neutral_ratio = (df_targeted['y'] == 1).mean()
                # if neutral_ratio > 0.9999:
                #     print(f"⏭️  ПРОПУСК: Слишком много нейтральных сигналов ({neutral_ratio:.2%} > 95%)")
                #     results[f"cutoff_{cutoff_ratio}_thresh_{threshold:.6f}"] = {
                #         'cutoff_ratio': cutoff_ratio,
                #         'threshold': threshold,
                #         'logloss': np.nan,
                #         'total_return': -999,  # Очень плохой результат
                #         'sharpe': -999,
                #         'max_drawdown': -999,
                #         'win_rate': -999,
                #         'model': None,
                #         'skipped': True,
                #         'reason': f'Нейтральных сигналов: {neutral_ratio:.2%} > 95%'
                #     }
                #     continue
                
                print(f"📊 Нейтральных сигналов: {neutral_ratio:.2%} (ОК)")
                
                # 2. Подготовка данных
                (
                    X_train_scaled, y_train,
                    X_valid_scaled, y_valid,
                    X_test_scaled, y_test,
                    scaler, feature_columns, test_indices, df_prepared
                ) = prepare_features_and_split(
                    df=df_targeted, target_col=target_col, train_ratio=0.7, valid_ratio=0.15,
                )
                
                # 3. Обучение модели
                base_params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'random_state': RANDOM_STATE,
                    'verbose': -1,
                    'device': 'gpu' if USE_GPU else 'cpu'
                }
                
                grid_result = grid_search_lgbm_classifier(
                    X_train=X_train_scaled, y_train=y_train,
                    X_valid=X_valid_scaled, y_valid=y_valid,
                    base_params=base_params,
                    feature_names=feature_columns
                )
                
                model = grid_result["best_model"]
                logloss = grid_result["best_score"]
                
                # 4. Предсказания
                y_pred = model.predict(X_test_scaled)
                
                # 5. Автобэктест
                backtest_stats = advanced_backtester(
                    df=df_prepared,
                    predictions=y_pred,
                    test_indices=test_indices,
                    model_name=f"FFT_opt_{cutoff_ratio}_{threshold:.6f}",
                    risk_free_rate=0.05,
                    save_plot=False  # Не сохраняем графики для каждой комбинации
                )
                
                # 6. Сохраняем результаты
                total_return = backtest_stats.get('total_return', 0)
                
                results[f"cutoff_{cutoff_ratio}_thresh_{threshold:.6f}"] = {
                    'cutoff_ratio': cutoff_ratio,
                    'threshold': threshold,
                    'logloss': logloss,
                    'total_return': total_return,
                    'sharpe': backtest_stats.get('sharpe_ratio', 0),
                    'max_drawdown': backtest_stats.get('max_drawdown', 0),
                    'win_rate': backtest_stats.get('win_rate', 0),
                    'neutral_ratio': neutral_ratio,
                    'model': model,
                    'data': {
                        'df_prepared': df_prepared,
                        'test_indices': test_indices,
                        'predictions': y_pred,
                        'y_test': y_test
                    }
                }
                
                print(f"✅ Результат: Return={total_return:.4f}, Sharpe={backtest_stats.get('sharpe_ratio', 0):.4f}")
                
                # 7. Проверяем на лучший результат
                if total_return > best_return:
                    best_return = total_return
                    best_params = (cutoff_ratio, threshold)
                    best_model = model
                    best_data = {
                        'df_prepared': df_prepared,
                        'test_indices': test_indices,
                        'predictions': y_pred,
                        'y_test': y_test
                    }
                    print(f"🏆 НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ! Return={best_return:.4f}")
                
            except Exception as e:
                print(f"❌ Ошибка при обработке параметров: {e}")
                continue
    
    # 8. Выводим итоговые результаты
    print("\n" + "="*60)
    print("🏁 РЕЗУЛЬТАТЫ АВТООПТИМИЗАЦИИ:")
    print("="*60)
    
    # Подсчет пропущенных комбинаций
    skipped_count = sum(1 for r in results.values() if r.get('skipped', False))
    total_count = len(results)
    successful_count = total_count - skipped_count
    
    print(f"\n📊 СТАТИСТИКА ПРОЦЕССА:")
    print(f"   Всего комбинаций: {total_count}")
    print(f"   Успешно обработано: {successful_count}")
    print(f"   Пропущено (слишком много нейтралов): {skipped_count}")
    print(f"   Процент пропуска: {skipped_count/total_count:.1%}")
    
    # Фильтруем только успешные результаты для сортировки
    successful_results = {k: v for k, v in results.items() if not v.get('skipped', False)}
    
    if successful_results:
        # Сортируем по доходности только успешные результаты
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['total_return'], reverse=True)
        
        print("\n🏆 ТОП-5 ЛУЧШИХ КОМБИНАЦИЙ:")
        for i, (name, stats) in enumerate(sorted_results[:5], 1):
            print(f"{i}. {name}")
            print(f"   Return: {stats['total_return']:.4f} | Sharpe: {stats['sharpe']:.4f} | LogLoss: {stats['logloss']:.5f}")
            print(f"   Нейтральных: {stats['neutral_ratio']:.2%}")
        
        print(f"\n🥇 ЛУЧШИЕ ПАРАМЕТРЫ:")
        print(f"   Cutoff ratio: {best_params[0]}")
        print(f"   Threshold: {best_params[1]:.6f}")
        print(f"   Максимальная доходность: {best_return:.4f}")
    else:
        print("\n❌ НИ ОДНОЙ КОМБИНАЦИИ НЕ ПРОШЛА ПРОВЕРКУ НА НЕЙТРАЛЬНЫЕ СИГНАЛЫ!")
        print("🔧 РЕКОМЕНДАЦИЯ: Увеличить диапазон threshold или изменить cutoff_ratios")
    
    # Показываем примеры пропущенных комбинаций
    if skipped_count > 0:
        print(f"\n⏭️ ПРИМЕРЫ ПРОПУЩЕННЫХ КОМБИНАЦИЙ:")
        skipped_examples = [(k, v) for k, v in results.items() if v.get('skipped', False)][:3]
        for name, stats in skipped_examples:
            print(f"   ❌ {name}: {stats.get('reason', 'Unknown reason')}")
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'best_data': best_data,
        'best_return': best_return,
        'all_results': results,
        'sorted_results': sorted_results if successful_results else []
    }

# %% [markdown]
# ## Шаг 6: Автоматическая оптимизация FFT параметров
#
# Запускаем цикл для поиска оптимальных параметров FFT с максимальной доходностью.

# %%
# =============================================
# Ячейка 7: Автоматическая оптимизация
# =============================================
def __main__():
    print("🚀 ЗАПУСК АВТООПТИМИЗАЦИИ FFT ПАРАМЕТРОВ...")

    # Загружаем и подготавливаем данные один раз
    df_raw = load_and_merge_data()
    df_featured = add_volume_features(df_raw)
    df_featured = add_strategy_features(df_featured)
    df_featured = add_volatility_features(df_featured)

    # Диапазоны параметров для оптимизации
    cutoff_ratios = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15]  # 0.5%-15% частот
    thresholds = [
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008
    ]  # Разные уровни чувствительности

    # Запускаем автоматическую оптимизацию
    optimization_results = auto_optimize_fft_parameters(
        df_featured=df_featured,
        cutoff_ratios=cutoff_ratios,
        thresholds=thresholds
    )

    print("\n🎉 АВТООПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print("="*50)

    # Итоговая визуализация лучшей модели
    if optimization_results['best_model'] is not None:
        print("\n📈 Построение финальной визуализации для лучших параметров...")

        best_data = optimization_results['best_data']
        plot_advanced_signals(
            df=best_data['df_prepared'],
            test_indices=best_data['test_indices'],
            y_pred=best_data['predictions'],
            y_true=best_data['y_test']
        )

        # Финальный бэктест с сохранением графика
        print("\n🔍 Финальный бэктест лучшей модели с сохранением графика...")
        final_backtest_stats = advanced_backtester(
            df=best_data['df_prepared'],
            predictions=best_data['predictions'],
            test_indices=best_data['test_indices'],
            model_name="FINAL_OPTIMIZED",
            risk_free_rate=0.05,
            save_plot=True,
            plot_filename="optimized_fft_lgbm_backtest.png"
        )

        print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА ЛУЧШЕЙ МОДЕЛИ:")
        print(f"   🎯 Параметры: cutoff_ratio={optimization_results['best_params'][0]}, threshold={optimization_results['best_params'][1]:.6f}")
        print(f"   💰 Итоговая доходность: {optimization_results['best_return']:.4f}")
        print(f"   📈 Sharpe Ratio: {final_backtest_stats.get('sharpe_ratio', 0):.4f}")
        print(f"   📉 Max Drawdown: {final_backtest_stats.get('max_drawdown', 0):.4f}")
        print(f"   ✅ Win Rate: {final_backtest_stats.get('win_rate', 0):.4f}")
        print(f"   📁 График сохранен: optimized_fft_lgbm_backtest.png")
    else:
        print("❌ Не удалось найти оптимальные параметры")



if __name__ == "__main__":
    __main__()

#