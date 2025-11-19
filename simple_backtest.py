import pandas as pd

def run_simple_backtest(df, start_date=None, end_date=None, debug=False):
    """
    Runs a simple backtest on a DataFrame that already contains a 'prediction' column.
    The DataFrame should contain all data including the warm-up period.
    The backtest will be performed only on the range between start_date and end_date.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Filter the combined data to the actual backtest range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if df.empty:
        if debug: print("DEBUG: DataFrame is empty AFTER filtering for date range.")
        return 0.0, 0, [], pd.DataFrame()

    trades = []
    current_position = 0
    entry_price = 0.0
    entry_time = None
    total_return_pct = 0.0

    if debug:
        print("\nDEBUG: Starting backtest loop...")
        print("Time, Price, Prediction, Desired Pos, Current Pos, Action")

    for time, row in df.iterrows():
        prediction = row['prediction']
        price = row['close']
        action = "HOLD"
        
        desired_position = 1 if prediction == 2 else -1 if prediction == 0 else 0

        if current_position != 0:
            if (current_position == 1 and desired_position == -1) or \
               (current_position == -1 and desired_position == 1):
                
                exit_price = price
                if entry_price == 0: continue
                
                trade_return_pct = ((exit_price - entry_price) / entry_price) * 100 if current_position == 1 else ((entry_price - exit_price) / entry_price) * 100
                total_return_pct += trade_return_pct
                direction = "LONG" if current_position == 1 else "SHORT"
                action = f"CLOSE {direction}"

                trades.append({
                    "entry_time": entry_time, "exit_time": time, "entry_price": float(entry_price),
                    "exit_price": float(exit_price), "direction": direction, "return_pct": float(trade_return_pct)
                })
                
                current_position = desired_position
                entry_price = price
                entry_time = time
                action += f" & OPEN {'LONG' if desired_position==1 else 'SHORT'}"
        
        elif desired_position != 0:
            current_position = desired_position
            entry_price = price
            entry_time = time
            action = f"OPEN {'LONG' if desired_position==1 else 'SHORT'}"

        if debug:
            print(f"{time}, {price:.2f}, {int(prediction)}, {desired_position}, {current_position}, {action}")

    if current_position != 0 and not df.empty:
        last_price = df['close'].iloc[-1]
        last_time = df.index[-1]
        
        if entry_price > 0:
            exit_price = last_price
            trade_return_pct = ((exit_price - entry_price) / entry_price) * 100 if current_position == 1 else ((entry_price - exit_price) / entry_price) * 100
            total_return_pct += trade_return_pct
            direction = "LONG" if current_position == 1 else "SHORT"
            action = f"FINAL CLOSE {direction}"

            trades.append({
                "entry_time": entry_time, "exit_time": last_time, "entry_price": float(entry_price),
                "exit_price": float(exit_price), "direction": direction, "return_pct": float(trade_return_pct)
            })
            if debug:
                print(f"{last_time}, {last_price:.2f}, N/A, 0, {current_position}, {action} -> Return: {trade_return_pct:.2f}%")

    trades_count = len(trades)
    if debug:
        print(f"\nDEBUG: Backtest finished. Total trades: {trades_count}, Total return: {total_return_pct:.2f}%")

    return total_return_pct, trades_count, trades, df
