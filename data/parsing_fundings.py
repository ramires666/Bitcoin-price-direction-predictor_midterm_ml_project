import sys
from pathlib import Path
import zipfile

import pandas as pd


def read_funding_zip(path: Path) -> pd.DataFrame:
    """
    Read a single Binance funding archive like:
    BTCUSDT-fundingRate-YYYY-MM.zip

    Inside it expects a CSV with columns:
    calc_time,funding_interval_hours,last_funding_rate
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
    expected_cols = {"calc_time", "last_funding_rate"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")

    # Keep only needed columns and rename
    df = df[["calc_time", "last_funding_rate"]].copy()
    df.rename(columns={"last_funding_rate": "funding_rate"}, inplace=True)

    # Convert calc_time from ms to timezone-aware datetime (UTC)
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)

    return df


def main():
    """
    Aggregate all monthly BTCUSDT funding rate files into a single Parquet:
    data/processed/fundings.parquet

    Output columns:
      - calc_time (datetime64[ns, UTC])
      - funding_rate (float)
    """
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "download" / "fundings"
    output_path = base_dir / "data" / "processed" / "fundings.parquet"

    input_files = sorted(input_dir.glob("BTCUSDT-fundingRate-*.zip"))
    if not input_files:
        print(f"No funding zip files found in {input_dir}", file=sys.stderr)
        return

    dfs = []
    for fp in input_files:
        print(f"Processing {fp}...")
        try:
            df_month = read_funding_zip(fp)
            dfs.append(df_month)
        except Exception as e:
            print(f"Error while processing {fp}: {repr(e)}", file=sys.stderr)
            # Skip bad month but keep going
            continue

    if not dfs:
        print("No dataframes were created; nothing to save.", file=sys.stderr)
        return

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values("calc_time").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(output_path, index=False)
    print(f"Saved combined funding data to {output_path}")


if __name__ == "__main__":
    main()