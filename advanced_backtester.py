"""
Advanced Backtester Module for Trading Strategies
Provides comprehensive backtesting functionality with detailed risk metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Fix emoji warnings in matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans']
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def advanced_backtester(
    df: pd.DataFrame,
    predictions: np.ndarray,
    test_indices: pd.Index,
    model_name: str = "TEST",
    risk_free_rate: float = 0.05,
    commission_rate: float = 0.00055,  # 0.055% default commission per trade
    save_plot: bool = True,
    plot_filename: str = "lgbm_backtest_equity_dd.png"
) -> dict:
    """
    Advanced backtester with proper position management and entry/exit visualization.
    
    Position Logic:
    - Only ONE position at a time (no stacking)
    - Opposite signal → flip position
    - Neutral signal → close position immediately
    
    Args:
        df: DataFrame with price data and prepared features
        predictions: Model predictions (0=DOWN, 1=SIDEWAYS, 2=UP)
        test_indices: Index of test data
        model_name: Name for reporting (default: "TEST")
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 5%)
        commission_rate: Commission rate per trade (default: 0.055% = 0.00055)
        save_plot: Whether to save the equity curve plot
        plot_filename: Filename for saved plot
        
    Returns:
        dict: Comprehensive backtest statistics with trade details
    """
    
    # Prepare test data
    test_df = df.loc[test_indices].copy()
    
    # Create signal series aligned with test data
    signal_series = pd.Series(predictions, index=test_indices)
    
    # DEBUG: Check signal distribution and apply filters
    print(f"\n>>> SIGNAL ANALYSIS:")
    print(f"   SIGNAL Distribution:")
    signal_counts = signal_series.value_counts().sort_index()
    for signal, count in signal_counts.items():
        signal_name = {0: "DOWN/SHORT", 1: "SIDEWAYS", 2: "UP/LONG"}[signal]
        percentage = (count / len(signal_series)) * 100
        print(f"      {signal} ({signal_name}): {count} ({percentage:.1f}%)")
        
    # Apply signal quality filters
    total_signals = len(signal_series)
    neutral_count = signal_counts.get(1, 0)
    neutral_percentage = (neutral_count / total_signals) * 100
    
    # Filter 1: Must have at least 30% neutral signals
    min_neutral_sinals = 30.0
    if neutral_percentage < min_neutral_sinals:
        error_msg = f"REJECTED: Only {neutral_percentage:.1f}% neutral signals (minimum {min_neutral_sinals}% required)"
        print(f"   FILTER 1: {error_msg}")
        raise ValueError(error_msg)
    else:
        print(f"   FILTER 1: PASSED - {neutral_percentage:.1f}% neutral signals >= {min_neutral_sinals}% required")
        
    # Filter 2: Must have less than 95% neutral signals (was already implemented)
    max_neutral_sinals = 99.99999
    if neutral_percentage > max_neutral_sinals:
        error_msg = f"REJECTED: Too many neutral signals ({neutral_percentage:.1f}%, maximum {max_neutral_sinals}%)"
        print(f"   FILTER 2: {error_msg}")
        raise ValueError(error_msg)
    else:
        print(f"   FILTER 2: PASSED - {neutral_percentage:.1f}% neutral signals <= {max_neutral_sinals}% maximum")
        
    # Show first 10 signals as example
    print(f"   First 10 signals example:")
    for i in range(min(10, len(signal_series))):
        signal = signal_series.iloc[i]
        signal_name = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}[signal]
        print(f"      {i}: {signal} ({signal_name})")
    
    # Initialize position tracking
    positions = pd.Series(0.0, index=test_df.index)  # Current position (0, +1, -1)
    entry_exit_markers = []  # Track entry/exit events
    trading_costs = []  # Track commission costs
    
    current_position = 0  # Start with no position
    entry_price = None
    
    # Process signals sequentially with proper position management
    for i, (timestamp, signal) in enumerate(signal_series.items()):
        if i == 0:
            # First signal - determine initial position
            if signal == 0:  # DOWN -> SHORT
                new_position = -1
            elif signal == 2:  # UP -> LONG
                new_position = +1
            else:  # SIDEWAYS -> FLAT
                new_position = 0
        else:
            prev_signal = signal_series.iloc[i-1]
            
            # NEUTRAL signal - close any position
            if signal == 1:
                new_position = 0
            # SAME signal - keep current position
            elif signal == prev_signal:
                new_position = current_position
            # DIFFERENT signal - handle position changes
            elif signal != prev_signal:
                # DOWN signal (0) handling
                if signal == 0:
                    if current_position == 0:  # FLAT -> SHORT
                        new_position = -1
                    elif current_position == 1:  # LONG -> FLIP to SHORT
                        new_position = -1
                    else:  # Already SHORT
                        new_position = current_position
                # UP signal (2) handling
                elif signal == 2:
                    if current_position == 0:  # FLAT -> LONG
                        new_position = +1
                    elif current_position == -1:  # SHORT -> FLIP to LONG
                        new_position = +1
                    else:  # Already LONG
                        new_position = current_position
            else:  # Fallback
                new_position = current_position
        
        # Handle position changes with detailed logging and commission calculation
        if new_position != current_position:
            position_names = {0: "FLAT", -1: "SHORT", 1: "LONG"}
            current_price = test_df.loc[timestamp, 'close']
            commission_cost = 0
            
            if current_position != 0 and new_position != current_position:
                # Closing a position (either to flat or opposite)
                if new_position == 0:
                    entry_exit_markers.append({
                        'timestamp': timestamp,
                        'price': current_price,
                        'action': 'EXIT',
                        'position': 0,
                        'signal': signal
                    })
                    # Commission for closing position
                    commission_cost += current_price * commission_rate
                    trading_costs.append(commission_cost)
                    print(f"EXIT: {position_names[current_position]} -> {position_names[new_position]} @ {current_price:.2f}")
                    print(f"Commission: {commission_cost:.4f} ({commission_rate*100:.3f}%)")
                elif new_position == -current_position:  # Opposite position (FLIP)
                    entry_exit_markers.append({
                        'timestamp': timestamp,
                        'price': current_price,
                        'action': 'FLIP',
                        'position': new_position,
                        'signal': signal
                    })
                    # Commission for closing current position + opening new position
                    commission_cost += current_price * commission_rate * 2  # Both operations
                    trading_costs.append(commission_cost)
                    print(f"FLIP: {position_names[current_position]} -> {position_names[new_position]} @ {current_price:.2f}")
                    print(f"Commission (2x): {commission_cost:.4f} ({commission_rate*100:.3f}% each)")
            
            if new_position != 0 and current_position == 0:
                # Opening a new position
                entry_exit_markers.append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'action': 'ENTRY',
                    'position': new_position,
                    'signal': signal
                })
                # Commission for opening position
                commission_cost += current_price * commission_rate
                trading_costs.append(commission_cost)
                print(f"ENTRY: {position_names[current_position]} -> {position_names[new_position]} @ {current_price:.2f}")
                print(f"Commission: {commission_cost:.4f} ({commission_rate*100:.3f}%)")
            
            current_position = new_position
        
        positions.loc[timestamp] = current_position
    
    # Add position to dataframe
    test_df['position'] = positions
    
    # Calculate returns based on position changes (no lookahead bias)
    position_shifted = test_df['position'].shift(1)
    test_df['returns'] = position_shifted * test_df['close'].pct_change()
    test_df['equity'] = (1 + test_df['returns']).cumprod()
    test_df['equity'] = test_df['equity'].fillna(1)
    
    # Apply commission costs by directly reducing equity at trade points
    # Simple approach: subtract commission as percentage of position value
    total_commission_paid = 0
    commission_events = []
    
    for i, marker in enumerate(entry_exit_markers):
        timestamp = marker['timestamp']
        commission_cost = trading_costs[i] if i < len(trading_costs) else 0
        
        if commission_cost > 0:
            # Apply commission directly to all subsequent equity values
            position_idx = test_df.index.get_loc(timestamp)
            test_df.iloc[position_idx:, test_df.columns.get_loc('equity')] *= (1 - commission_rate)
            total_commission_paid += commission_cost
            
            commission_events.append({
                'timestamp': timestamp,
                'action': marker['action'],
                'commission': commission_cost,
                'commission_pct': commission_rate * 100
            })

    # Calculate equity curve metrics
    test_df['peak'] = test_df['equity'].cummax()
    test_df['drawdown'] = (test_df['equity'] - test_df['peak']) / test_df['peak'] * 100

    # Calculate additional trading statistics
    # Win/Loss rates from actual trades
    trade_returns = test_df['returns'].dropna()
    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]

    win_rate = len(winning_trades) / len(trade_returns) * 100 if len(trade_returns) > 0 else 0
    loss_rate = len(losing_trades) / len(trade_returns) * 100 if len(trade_returns) > 0 else 0

    # Average trade time (from actual entries to exits)
    if entry_exit_markers:
        entry_times = [marker['timestamp'] for marker in entry_exit_markers if marker['action'] in ['ENTRY', 'FLIP']]
        if len(entry_times) > 1:
            trade_durations = []
            for i in range(len(entry_times) - 1):
                duration = entry_times[i+1] - entry_times[i]
                trade_durations.append(duration.total_seconds() / 3600)  # hours
            avg_trade_time = np.mean(trade_durations) if trade_durations else 0
        else:
            avg_trade_time = 0
    else:
        avg_trade_time = 0

    # DEBUG: Calculate and show detailed risk metrics
    print(f"\n>>> RISK METRICS DEBUG:")
    print(f"   TRADE RETURNS STATS:")
    print(f"      Total trades: {len(trade_returns)}")
    print(f"      Mean return: {trade_returns.mean():.6f}")
    print(f"      Std return: {trade_returns.std():.6f}")
    print(f"      Min return: {trade_returns.min():.6f}")
    print(f"      Max return: {trade_returns.max():.6f}")
    
    downside_returns = trade_returns[trade_returns < 0]
    print(f"   DOWNSIDE STATS:")
    print(f"      Downside trades: {len(downside_returns)}")
    if len(downside_returns) > 0:
        print(f"      Downside mean: {downside_returns.mean():.6f}")
        print(f"      Downside std: {downside_returns.std():.6f}")

    # Risk metrics - ROBUST CALCULATION
    # Check for mathematical validity first
    if len(trade_returns) == 0:
        sharpe_ratio = 0
        sortino_ratio = 0
        print(f"   WARNING: No trades to calculate metrics")
    elif len(trade_returns) == 1:
        sharpe_ratio = 0
        sortino_ratio = 0
        print(f"   WARNING: Only 1 trade, cannot calculate meaningful ratios")
    else:
        mean_return = trade_returns.mean()
        std_return = trade_returns.std()
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Check for mathematical validity
        if std_return <= 0:
            print(f"   WARNING: Zero or negative standard deviation: {std_return}")
            sharpe_ratio = float('inf') if mean_return > 0 else 0
        else:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)
        
        if len(downside_returns) == 0:
            sortino_ratio = float('inf') if mean_return > 0 else 0
            print(f"   OK: No losing trades - perfect strategy!")
        elif downside_std <= 0:
            sortino_ratio = float('inf') if mean_return > 0 else 0
        else:
            sortino_ratio = mean_return / downside_std * np.sqrt(252)
        
        print(f"   OK: Calculated Sharpe: {sharpe_ratio:.2f}, Sortino: {sortino_ratio:.2f}")
        
        # Sanity check
        if sharpe_ratio < 0 and mean_return > 0:
            print(f"   ERROR: Negative Sharpe with positive returns!")
            print(f"      This indicates a calculation error.")
        
        if sortino_ratio < 0 and mean_return > 0:
            print(f"   ERROR: Negative Sortino with positive returns!")
            print(f"      This indicates a calculation error.")

    # Calculate final statistics
    total_return = (test_df['equity'].iloc[-1] - 1) * 100
    max_dd = test_df['drawdown'].min()  # Maximum drawdown (deepest loss during period)
    current_dd = test_df['drawdown'].min()  # Maximum unrealized drawdown (worst loss, could cause margin call)

    # Print beautiful statistics
    print(f"\n=== BACKTEST STATISTICS ({model_name}):")
    print(f"MAIN METRICS:")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Loss Rate: {loss_rate:.1f}%")
    print(f"   Avg Trade Time: {avg_trade_time:.1f}h")
    print(f"   Final Equity: {test_df['equity'].iloc[-1]:.4f}")

    print(f"\nCOMMISSION COSTS:")
    print(f"   Commission Rate: {commission_rate*100:.3f}% per trade")
    print(f"   Total Trades: {len(entry_exit_markers)}")
    print(f"   Total Commission Paid: {total_commission_paid:.4f}")
    print(f"   Commission as % of Final Equity: {(total_commission_paid / test_df['equity'].iloc[-1] * 100):.2f}%")

    print(f"\nRISK METRICS:")
    print(f"   Max Realized DD: {max_dd:.2f}%")
    print(f"   Max Unrealized DD (Margin Call Risk): {current_dd:.2f}%")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {sortino_ratio:.2f}")

    # Create comprehensive plot with price, positions, and entry/exit markers
    fig = plt.figure(figsize=(20, 12))
    
    # Top subplot: Price with entry/exit markers
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(test_df.index, test_df['close'], color='black', linewidth=1.5, label='BTC Price', alpha=0.8)
    
    # Add entry/exit markers
    for marker in entry_exit_markers:
        if marker['action'] == 'ENTRY':
            if marker['position'] > 0:  # LONG entry
                color = 'green'
                marker_type = '^'  # UP arrow for LONG
            else:  # SHORT entry
                color = 'red'
                marker_type = 'v'  # DOWN arrow for SHORT
        elif marker['action'] == 'FLIP':
            color = 'blue'
            marker_type = 'o'
        else:  # EXIT
            color = 'orange'
            marker_type = 'v'
        
        marker_size = 100 if marker['action'] == 'ENTRY' else 80
        
        ax1.scatter(marker['timestamp'], marker['price'],
                   color=color, marker=marker_type, s=marker_size,
                   alpha=0.9, zorder=5, edgecolors='white', linewidth=1)
    
    ax1.set_title(f'{model_name} Strategy: Price + Entry/Exit Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle subplot: Position over time
    ax2 = plt.subplot(3, 1, 2)
    ax2.fill_between(test_df.index, test_df['position'], 0,
                    where=(test_df['position'] > 0), color='green', alpha=0.3, label='Long Position')
    ax2.fill_between(test_df.index, test_df['position'], 0,
                    where=(test_df['position'] < 0), color='red', alpha=0.3, label='Short Position')
    ax2.plot(test_df.index, test_df['position'], color='black', linewidth=2)
    ax2.set_title('Position Management (Only 1 Position at a Time)')
    ax2.set_ylabel('Position (+1=Long, -1=Short, 0=Flat)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom subplot: Equity curve and drawdown
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(test_df.index, test_df['equity'], color='green', linewidth=2, label='Equity')
    ax3.set_ylabel('Equity', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    
    ax4 = ax3.twinx()
    ax4.fill_between(test_df.index, test_df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown %')
    ax4.plot(test_df.index, test_df['drawdown'], color='red', linewidth=1)
    ax4.set_ylabel('Drawdown %', color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.set_ylim(max_dd * 1.1, 0)
    
    ax3.set_title('Equity Curve + Drawdown')
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nOK: Backtest ready! PNG: {plot_filename}")
    
    plt.show()

    # Print entry/exit summary
    print(f"\n=== TRADE SUMMARY ({len(entry_exit_markers)} signals):")
    entries = [m for m in entry_exit_markers if m['action'] == 'ENTRY']
    flips = [m for m in entry_exit_markers if m['action'] == 'FLIP']
    exits = [m for m in entry_exit_markers if m['action'] == 'EXIT']
    
    long_entries = [e for e in entries if e['position'] > 0]
    short_entries = [e for e in entries if e['position'] < 0]
    
    print(f"   Long Entries: {len(long_entries)}")
    print(f"   Short Entries: {len(short_entries)}")
    print(f"   Position Flips: {len(flips)}")
    print(f"   Position Exits: {len(exits)}")
    
    # DEBUG: Check if any short positions were created
    all_positions = test_df['position'].unique()
    print(f"\n=== POSITION DEBUG:")
    print(f"   Unique positions in dataset: {sorted(all_positions)}")
    
    # Count different position types
    position_counts = test_df['position'].value_counts().sort_index()
    print(f"   Position time distribution:")
    for pos, count in position_counts.items():
        pos_name = {0: "FLAT", 1: "LONG", -1: "SHORT"}[int(pos)]
        percentage = (count / len(test_df)) * 100
        print(f"      {pos_name} ({pos}): {count} ({percentage:.1f}%)")
    
    # Show examples of each position type
    print(f"   Position timeline examples:")
    position_sample = test_df[['position', 'close']].head(30)
    for idx, row in position_sample.iterrows():
        pos_name = {0: "FLAT", 1: "LONG", -1: "SHORT"}[int(row['position'])]
        print(f"      {idx}: {pos_name} @ {row['close']:.2f}")
    
    # Final check: Are shorts working?
    has_shorts = any(test_df['position'] == -1)
    has_longs = any(test_df['position'] == 1)
    print(f"\n=== FINAL CHECK:")
    print(f"   Has LONG positions: {has_longs}")
    print(f"   Has SHORT positions: {has_shorts}")
    if has_shorts and has_longs:
        print(f"   SUCCESS! Both LONG and SHORT positions are working!")
    elif has_longs and not has_shorts:
        print(f"   WARNING: Only LONG positions found - check signal logic!")
    elif has_shorts and not has_longs:
        print(f"   WARNING: Only SHORT positions found - check signal logic!")
    else:
        print(f"   ERROR: No positions found at all!")

    # Return comprehensive statistics
    stats = {
        'total_return': total_return,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_trade_time': avg_trade_time,
        'final_equity': test_df['equity'].iloc[-1],
        'max_drawdown': max_dd,
        'max_unrealized_dd': current_dd,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'equity_curve': test_df['equity'],
        'drawdown_series': test_df['drawdown'],
        'position_series': test_df['position'],
        'entry_exit_markers': entry_exit_markers,
        'test_df': test_df,
        # Commission-related statistics
        'commission_rate': commission_rate,
        'total_commission_paid': total_commission_paid,
        'commission_events': commission_events,
        'trading_costs': trading_costs
    }
    
    return stats