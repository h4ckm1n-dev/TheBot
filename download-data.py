import csv
from datetime import datetime, timedelta

import ccxt
import pytz
from rich.console import Console

console = Console()

# Initialize the Binance Exchange
binance = ccxt.binance(
    {
        "enableRateLimit": True,
    }
)


def fetch_ohlcv_in_chunks(symbol, start_date, end_date, timeframe) -> list:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data in chunks for a given symbol, date range, and timeframe.

    Args:
        symbol (Any): The symbol for which OHLCV data is to be fetched.
        start_date (Any): The start date of the data range.
        end_date (Any): The end date of the data range.
        timeframe (Any): The timeframe for the OHLCV data (e.g., '1h', '1d').

    Returns:
        list: A list of OHLCV data fetched in chunks.
    """
    all_data = []
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=365), end_date)
        since = binance.parse8601(current_date.strftime("%Y-%m-%d") + "T00:00:00Z")
        data = binance.fetch_ohlcv(symbol, timeframe, since)
        all_data.extend(data)
        current_date = next_date
    return all_data


def download_crypto_data(symbols, start_date, end_date, timeframe) -> None:
    """
    Downloads OHLCV (Open, High, Low, Close, Volume) data for multiple symbols within a given date range and timeframe.

    Args:
        symbols (Any): List of symbols for which OHLCV data is to be fetched.
        start_date (Any): The start date of the data range.
        end_date (Any): The end date of the data range.
        timeframe (Any): The timeframe for the OHLCV data (e.g., '1h', '1d').

    Returns:
        None
    """
    for symbol in symbols:
        console.print(f"Fetching OHLCV data for {symbol}...", style="bold blue")
        ohlcv_data = fetch_ohlcv_in_chunks(symbol, start_date, end_date, timeframe)
        console.print(
            f"OHLCV data for {symbol} fetched successfully.", style="bold green"
        )

        # Specify your CSV file path dynamically
        csv_file_path = f'./data/{symbol.replace("/", "_")}_price_data.csv'

        # Writing to CSV
        with open(csv_file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Open", "High", "Low", "Close", "Volume"])
            for row in ohlcv_data:
                # Formatting the timestamp to a more readable, timezone-aware format
                utc_time = datetime.fromtimestamp(row[0] / 1000, pytz.utc)
                row[0] = utc_time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow(row)

        console.print(
            f"Data for {symbol} successfully saved to {csv_file_path}",
            style="bold green",
        )


# Your desired date range and other parameters
start_date = datetime(2018, 1, 1)
end_date = datetime.now()
symbols = [
    "BTC/USDT",  # Bitcoin
    "ETH/USDT",  # Ethereum
    "BNB/USDT",  # Binance Coin
    # Add more symbols if needed
]
TIMEFRAME = "1h"

# Download the data
download_crypto_data(symbols, start_date, end_date, TIMEFRAME)
