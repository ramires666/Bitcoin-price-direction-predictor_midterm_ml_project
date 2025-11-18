import sys
from pathlib import Path
import zipfile

import pandas as pd


def read_klines_zip(path: Path) -> pd.DataFrame:
    """
    Read a single Binance klines archive like:
    BTCUSDT-15m-YYYY-MM.zip

    Inside it expects a CSV with columns:
    open_time,open,high,low,close,volume,close_time,quote_volume,count,
    taker_buy_volume,taker_buy_quote_volume,ignore
    """
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV entries found inside {path}")
        member = csv_members[0]
        with zf.open(member) as f:
            df = pd.read_csv(f)

    # Ensure required columns are present
    expected_cols = {"open_time", "open", "high", "low", "close", "count"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")

    # Keep only needed columns
    df = df[["open_time", "open", "high", "low", "close", "count"]].copy()

    # Convert open_time from ms to timezone-aware datetime (UTC)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    return df


def main():
    """
    Aggregate all monthly BTCUSDT 15m klines into a single Parquet:
    data/processed/klines_15min_all.parquet

    Output columns:
      - open_time (datetime64[ns, UTC])
      - open
      - high
      - low
      - close
      - count
    """
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "download" / "klines"
    output_path = base_dir / "data" / "processed" / "klines_15min_all.parquet"

    input_files = sorted(input_dir.glob("BTCUSDT-15m-*.zip"))
    if not input_files:
        print(f"No kline zip files found in {input_dir}", file=sys.stderr)
        return

    dfs = []
    for fp in input_files:
        print(f"Processing {fp}...")
        try:
            df_month = read_klines_zip(fp)
            dfs.append(df_month)
        except Exception as e:
            print(f"Error while processing {fp}: {repr(e)}", file=sys.stderr)
            # Skip bad month but keep going
            continue

    if not dfs:
        print("No dataframes were created; nothing to save.", file=sys.stderr)
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values("open_time").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(output_path, index=False)
    print(f"Saved combined klines data to {output_path}")


if __name__ == "__main__":
    main()