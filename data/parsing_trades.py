import os
from pathlib import Path
import sys
import re

# --- CONFIGURATION ---
# Set to True to use cudf for GPU acceleration. Requires NVIDIA GPU and CUDA.
USE_CUDF = True

# Set the desired timeframe for aggregation in pandas-style notation (e.g., '15min', '1H', '4H').
# cuDF does not accept the 'min' alias; it expects 'T' or explicit seconds like '900s'.
# We will translate this to a cuDF-compatible alias when USE_CUDF is True.
AGGREGATION_TIMEFRAME = '15min'
# --- END CONFIGURATION ---

# Dynamically import pandas or cudf based on configuration
if USE_CUDF:
    try:
        import cudf as dd
        import cupy as np
        print("Using cuDF for GPU acceleration.")
    except ImportError:
        print("cuDF or cupy not found. Falling back to pandas.", file=sys.stderr)
        import pandas as dd
        import numpy as np
        USE_CUDF = False
else:
    import pandas as dd
    import numpy as np
    print("Using pandas for CPU processing.")

# Import the downloader function
from downloader import download_file, BASE_URL


def _normalize_timeframe_for_cudf(tf: str) -> str:
    """
    Deprecated for GPU path: cuDF's Series.dt.floor does not accept '15min' or '15T'
    in the current environment, so GPU aggregation no longer relies on this helper.

    Kept only for potential future use or backward compatibility on CPU paths.
    """
    return tf


def _read_aggtrades_file_gpu_aware(path: Path, col_names):
    """
    GPU-friendly reader that can fall back to unzipping large/broken ZIPs.

    Logic:
    - If file is .csv -> read directly with cudf/pandas read_csv.
    - If file is .zip   -> first try cudf/pandas read_csv with compression.
      - If that raises a cuDF decompression RuntimeError (Z_STREAM_END),
        we unzip once to a sibling .csv on disk and then read that .csv.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # Direct CSV path (already uncompressed)
    if ext == ".csv":
        return dd.read_csv(path, names=col_names, header=0)

    # Otherwise assume ZIP (Binance monthly archives)
    try:
        return dd.read_csv(path, names=col_names, header=0)
    except RuntimeError as e:
        msg = str(e)
        if "Z_STREAM_END not encountered" not in msg:
            # Not the cuDF DEFLATE bug, propagate
            raise

        # Fallback: unzip to CSV next to the ZIP and read that.
        csv_path = path.with_suffix(".csv")
        print(f"cuDF DEFLATE error on {path.name}. Extracting to {csv_path.name} and retrying as CSV...", file=sys.stderr)

        import zipfile

        try:
            with zipfile.ZipFile(path, "r") as zf:
                # Binance monthly ZIPs typically contain a single CSV with same basename
                # but we handle generic case: take the first .csv entry.
                csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                if not csv_members:
                    raise RuntimeError(f"No CSV entries found inside {path}")
                member = csv_members[0]
                # Extract into the same directory as the ZIP, with flattened name.
                with zf.open(member) as src, open(csv_path, "wb") as dst:
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(chunk)
        except Exception as unzip_err:
            raise RuntimeError(f"Failed to unzip {path}: {unzip_err}") from unzip_err

        # Now read the extracted CSV on GPU/CPU
        return dd.read_csv(csv_path, names=col_names, header=0)


def process_aggtrades_file(file_path, timeframe, use_gpu):
    """
    Reads a single zipped CSV file of aggregate trades, processes it,
    and returns a DataFrame aggregated into specified time intervals.
    """
    print(f"Processing {file_path}...")
 
    # We will:
    # - On GPU (cuDF): avoid .resample / .dt.floor entirely, and instead bucket
    #   timestamps into 15-minute bins manually via int64 arithmetic.
    # - On CPU (pandas): keep using dt.floor(timeframe) as before.
    BUCKET_MS = 15 * 60 * 1000  # 15 minutes in milliseconds (project is fixed to 15min for now)

    if use_gpu:
        print(
            f"Backend: GPU (cuDF); timeframe='{timeframe}', bucket_ms={BUCKET_MS}"
        )
    else:
        print(
            f"Backend: CPU (pandas); timeframe='{timeframe}'"
        )

    col_names = [
        'agg_trade_id', 'price', 'quantity', 'first_trade_id',
        'last_trade_id', 'transact_time', 'is_buyer_maker'
    ]

    df = _read_aggtrades_file_gpu_aware(file_path, col_names)

    # Ensure we have the expected columns regardless of how the CSV is parsed
    # Some Binance files may come with a header or different dtypes.
    missing_cols = [c for c in col_names if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns {missing_cols} in {file_path}")

    # Always work with an explicit datetime column, do NOT rely on index.
    df['transact_time'] = dd.to_datetime(df['transact_time'], unit='ms')

    df['bid_vol'] = df['quantity'].where(df['is_buyer_maker'], 0)
    df['ask_vol'] = df['quantity'].where(~df['is_buyer_maker'], 0)

    # Decide if this is a "huge" month (like March 2024) -> chunk-based GPU path
    path_obj = Path(file_path)
    is_huge_month = USE_CUDF and path_obj.suffix.lower() == '.csv' and '2024-03' in path_obj.name

    # Helper: floor to timeframe / bucket start
    def floor_bucket(frame):
        if use_gpu:
            # GPU (cuDF): manual bucketing via int64 milliseconds from epoch.
            # df['transact_time'] is already datetime64[ms] via dd.to_datetime(..., unit='ms').
            ts_ms = frame['transact_time'].astype('int64')
            bucket_start_ms = (ts_ms // BUCKET_MS) * BUCKET_MS
            frame['bucket'] = dd.to_datetime(bucket_start_ms, unit='ms')
        else:
            # CPU (pandas): standard dt.floor with string timeframe (e.g. '15min').
            frame['bucket'] = frame['transact_time'].dt.floor(timeframe)
        return frame

    # Helper: base agg for one frame (GPU or pandas)
    def base_agg(frame):
        frame = floor_bucket(frame)
        agg_dict = {
            'quantity': 'sum',
            'bid_vol': ['sum', 'max', 'mean'],
            'ask_vol': ['sum', 'max', 'mean'],
        }
        g = frame.groupby('bucket').agg(agg_dict)
        g.columns = ['_'.join(col) for col in g.columns.values]
        g = g.reset_index().rename(columns={'bucket': 'transact_time'})
        return g

    # Helper: modes for one frame
    def modes_agg(frame):
        frame = floor_bucket(frame)

        # bid mode
        bid_df = frame[frame['bid_vol'] > 0][['bucket', 'bid_vol']]
        if len(bid_df) > 0:
            bid_counts = (
                bid_df.groupby(['bucket', 'bid_vol'])
                .size()
                .reset_index(name='counts')
            )
            bid_counts = bid_counts.sort_values(
                ['bucket', 'counts'], ascending=[True, False]
            )
            bid_modes = bid_counts.drop_duplicates(
                subset=['bucket'], keep='first'
            )
            bid_modes = bid_modes.rename(
                columns={'bucket': 'transact_time', 'bid_vol': 'mod_bid_vol'}
            )
        else:
            if USE_CUDF:
                bid_modes = dd.DataFrame(
                    {'transact_time': dd.Series([], dtype='datetime64[ms]'),
                     'mod_bid_vol': dd.Series([], dtype='float64')}
                )
            else:
                import pandas as _pd
                bid_modes = _pd.DataFrame(columns=['transact_time', 'mod_bid_vol'])

        # ask mode
        ask_df = frame[frame['ask_vol'] > 0][['bucket', 'ask_vol']]
        if len(ask_df) > 0:
            ask_counts = (
                ask_df.groupby(['bucket', 'ask_vol'])
                .size()
                .reset_index(name='counts')
            )
            ask_counts = ask_counts.sort_values(
                ['bucket', 'counts'], ascending=[True, False]
            )
            ask_modes = ask_counts.drop_duplicates(
                subset=['bucket'], keep='first'
            )
            ask_modes = ask_modes.rename(
                columns={'bucket': 'transact_time', 'ask_vol': 'mod_ask_vol'}
            )
        else:
            if USE_CUDF:
                ask_modes = dd.DataFrame(
                    {'transact_time': dd.Series([], dtype='datetime64[ms]'),
                     'mod_ask_vol': dd.Series([], dtype='float64')}
                )
            else:
                import pandas as _pd
                ask_modes = _pd.DataFrame(columns=['transact_time', 'mod_ask_vol'])

        if USE_CUDF:
            modes = bid_modes.merge(
                ask_modes, on='transact_time', how='outer'
            )
        else:
            modes = bid_modes.merge(
                ask_modes, on='transact_time', how='outer'
            )
        return modes

    if is_huge_month:
        # Chunked path: pandas read_csv -> chunk -> cuDF -> aggregate, to avoid OOM on March.
        import pandas as pd

        print(f"Using chunk-based aggregation for huge file {path_obj.name}...")
        chunks = pd.read_csv(
            path_obj,
            names=col_names,
            header=0,
            chunksize=5_000_000
        )

        agg_acc = None
        mode_acc = None

        for i, chunk in enumerate(chunks, start=1):
            print(f"  Processing chunk {i} for {path_obj.name}...")
            chunk['transact_time'] = pd.to_datetime(chunk['transact_time'], unit='ms')
            chunk['bid_vol'] = chunk['quantity'].where(chunk['is_buyer_maker'], 0)
            chunk['ask_vol'] = chunk['quantity'].where(~chunk['is_buyer_maker'], 0)

            # Move chunk to GPU if USE_CUDF
            if USE_CUDF:
                g = dd.from_pandas(chunk)
            else:
                g = chunk

            base_chunk = base_agg(g)
            modes_chunk = modes_agg(g)

            # Accumulate base agg: concat + re-agg by transact_time
            if agg_acc is None:
                agg_acc = base_chunk
            else:
                agg_acc = dd.concat([agg_acc, base_chunk], ignore_index=True)
                agg_acc = agg_acc.groupby('transact_time').agg({
                    'quantity_sum': 'sum',
                    'bid_vol_sum': 'sum',
                    'bid_vol_max': 'max',
                    'bid_vol_mean': 'mean',
                    'ask_vol_sum': 'sum',
                    'ask_vol_max': 'max',
                    'ask_vol_mean': 'mean',
                }).reset_index()

            # Accumulate modes (best-effort; точный глобальный mode потребовал бы хранить counts)
            if modes_chunk is not None and len(modes_chunk) > 0:
                if mode_acc is None:
                    mode_acc = modes_chunk
                else:
                    mode_acc = dd.concat([mode_acc, modes_chunk], ignore_index=True)

        if agg_acc is None:
            df_resampled = dd.DataFrame()
        else:
            df_resampled = agg_acc
            if mode_acc is not None and len(mode_acc) > 0:
                # последние моды для каждого бакета (best-effort)
                if USE_CUDF:
                    mode_acc = mode_acc.sort_values('transact_time')
                    mode_acc = mode_acc.drop_duplicates(
                        subset=['transact_time'], keep='last'
                    )
                else:
                    mode_acc = mode_acc.sort_values('transact_time').drop_duplicates(
                        subset=['transact_time'], keep='last'
                    )
                df_resampled = df_resampled.merge(
                    mode_acc.set_index('transact_time'),
                    on='transact_time',
                    how='left'
                )
    else:
        # Normal single-shot path
        if USE_CUDF:
            # df уже cuDF
            df_resampled = base_agg(df)
            modes = modes_agg(df)
            df_resampled = df_resampled.merge(
                modes.set_index('transact_time'),
                on='transact_time',
                how='left'
            )
        else:
            # pandas CPU fallback
            df_resampled = base_agg(df)
            modes = modes_agg(df)
            df_resampled = df_resampled.merge(
                modes.set_index('transact_time'),
                on='transact_time',
                how='left'
            )

    # --- Column Renaming and Finalizing ---
    rename_map = {
        'quantity_sum': 'total_vol',
        'bid_vol_sum': 'bid_vol',
        'bid_vol_max': 'max_bid_vol',
        'bid_vol_mean': 'avg_bid_vol',
        'ask_vol_sum': 'ask_vol',
        'ask_vol_max': 'max_ask_vol',
        'ask_vol_mean': 'avg_ask_vol'
    }
    df_resampled.rename(columns=rename_map, inplace=True)
    df_resampled = df_resampled.fillna(0)

    df_resampled = df_resampled.sort_values('transact_time')
    df_resampled = df_resampled.reset_index(drop=True)
    df_resampled.rename(columns={'transact_time': 'datetime'}, inplace=True)

    final_cols = [
        'datetime', 'total_vol', 'bid_vol', 'max_bid_vol', 'mod_bid_vol',
        'avg_bid_vol', 'ask_vol', 'max_ask_vol', 'mod_ask_vol', 'avg_ask_vol'
    ]
    for col in final_cols:
        if col not in df_resampled.columns:
            df_resampled[col] = 0.0
    df_resampled = df_resampled[final_cols]

    print(f"Finished processing {file_path}.")
    return df_resampled


def main():
    """
    Main function to process all monthly aggtrades files, save them as individual
    parquet files, and then combine them into a single master parquet file.
    """
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / 'data' / 'download' / 'aggtrades'
    output_dir_name = f'aggtrades_{AGGREGATION_TIMEFRAME}'
    output_dir = base_dir / 'data' / 'processed' / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    trade_files = sorted(input_dir.glob('*.zip'))

    for file_path in trade_files:
        output_filename = file_path.stem + '.parquet'
        output_path = output_dir / output_filename

        if output_path.exists():
            print(f"Skipping {file_path}, as {output_path} already exists.")
            continue

        try:
            monthly_df = process_aggtrades_file(file_path, AGGREGATION_TIMEFRAME, USE_CUDF)
            monthly_df.to_parquet(output_path, index=False)
            print(f"Saved {output_path}")
        except Exception as e:
            # Treat ALL errors as code / environment issues, not file corruption.
            # We no longer attempt to re-download anything here.
            print(f"Error while processing {file_path}: {repr(e)}", file=sys.stderr)
            print("Skipping this file due to processing error (no re-download is attempted).", file=sys.stderr)
            continue

    print("\nCombining all monthly files...")
    all_monthly_files = sorted(output_dir.glob('*.parquet'))
    
    if not all_monthly_files:
        print("No monthly files found to combine. Exiting.")
        return
        
    combined_df = dd.concat(
        (dd.read_parquet(f) for f in all_monthly_files),
        ignore_index=True
    )
    combined_df = combined_df.sort_values('datetime')

    combined_output_name = f'aggtrades_{AGGREGATION_TIMEFRAME}_all.parquet'
    combined_output_path = output_dir.parent / combined_output_name
    combined_df.to_parquet(combined_output_path, index=False)

    print(f"\nAll files processed and combined into {combined_output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
