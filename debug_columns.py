import pandas as pd
import os

def check_columns():
    fundings_path = "data/processed/fundings.parquet"
    klines_path = "data/processed/klines_15min_all.parquet"
    volumes_path = "data/processed/aggtrades_15min_all.parquet"

    print(f"Checking columns for files in {os.getcwd()}")

    if os.path.exists(klines_path):
        try:
            klines = pd.read_parquet(klines_path)
            print(f"\n--- {klines_path} columns ---")
            print(klines.columns.tolist())
        except Exception as e:
            print(f"Error reading {klines_path}: {e}")
    else:
        print(f"{klines_path} not found")

    if os.path.exists(volumes_path):
        try:
            volumes = pd.read_parquet(volumes_path)
            print(f"\n--- {volumes_path} columns ---")
            print(volumes.columns.tolist())
        except Exception as e:
            print(f"Error reading {volumes_path}: {e}")
    else:
        print(f"{volumes_path} not found")

    # Simulate the merge
    if os.path.exists(klines_path) and os.path.exists(volumes_path):
        try:
            klines = pd.read_parquet(klines_path)
            volumes = pd.read_parquet(volumes_path)
            
            if "datetime" in volumes.columns: volumes = volumes.rename(columns={"datetime": "time"})
            if "open_time" in klines.columns: klines = klines.rename(columns={"open_time": "time"})
            
            # Ensure time is datetime
            volumes["time"] = pd.to_datetime(volumes["time"], utc=True)
            klines["time"] = pd.to_datetime(klines["time"], utc=True)

            print("\n--- Merging volumes and klines ---")
            df = pd.merge(volumes, klines, on="time", how="inner")
            print("Merged DataFrame columns:")
            print(df.columns.tolist())
        except Exception as e:
            print(f"Error during merge simulation: {e}")

if __name__ == "__main__":
    check_columns()