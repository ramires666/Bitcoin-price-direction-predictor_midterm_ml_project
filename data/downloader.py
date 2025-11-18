import os
import requests
from datetime import datetime, timedelta

# --- Configuration ---
BASE_URL = "https://data.binance.vision/data/futures/um/monthly"
DESTINATION_FOLDER = r"C:\_PYTH\projects\XGB_fints_project\data\download"
COIN_SYMBOL = "BTCUSDT"      # Symbol to download
KLINE_TIMEFRAME = "15m"      # Timeframe for klines (e.g., "1m", "5m", "15m", "1h")

# Define the data types and their corresponding URL path templates
# {symbol}, {timeframe}, {year}, {month} will be replaced
DATA_TYPES = {
    "fundingRate": "fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month}.zip",
    "aggTrades": "aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month}.zip",
    "klines": "klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month}.zip"
}

START_YEAR = 2024
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 10
# ---------------------

def generate_dates():
    """Generates year and month pairs from start to end date."""
    current_date = datetime(START_YEAR, START_MONTH, 1)
    end_date = datetime(END_YEAR, END_MONTH, 1)
    while current_date <= end_date:
        yield current_date.year, f"{current_date.month:02d}"
        # Move to the next month
        current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)


def download_file(url, destination_path):
    """Downloads a file from a URL to a destination path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {os.path.basename(destination_path)}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")
        return False

def main():
    """Main function to download all the required files."""
    # This mapping connects the data type from DATA_TYPES to the desired subfolder
    folder_mapping = {
        "fundingRate": "fundings",
        "aggTrades": "aggtrades",
        "klines": "klines"
    }

    print(f"Base destination folder: {DESTINATION_FOLDER}")
    print(f"Downloading data for symbol: {COIN_SYMBOL}")
    print(f"Klines timeframe: {KLINE_TIMEFRAME}")

    for year, month in generate_dates():
        for data_type, path_template in DATA_TYPES.items():
            # Determine the correct subfolder
            subfolder = folder_mapping.get(data_type, data_type)
            target_folder = os.path.join(DESTINATION_FOLDER, subfolder)
            
            # Create destination folder and subfolder if they don't exist
            os.makedirs(target_folder, exist_ok=True)

            # Format the path template with all required parts
            url_path = path_template.format(
                symbol=COIN_SYMBOL,
                timeframe=KLINE_TIMEFRAME,
                year=year,
                month=month
            )
            
            # The filename is the last part of the URL path
            filename = url_path.split('/')[-1]
            
            url = f"{BASE_URL}/{url_path}"
            
            destination_path = os.path.join(target_folder, filename)

            # Check if file already exists
            if os.path.exists(destination_path):
                print(f"Skipping existing file: {filename} in {subfolder}")
                continue

            print(f"Attempting to download {url} to {subfolder}...")
            download_file(url, destination_path)

if __name__ == "__main__":
    main()
