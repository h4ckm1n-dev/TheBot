import csv
from datetime import datetime, timedelta

import ccxt
import pytz  # Import the pytz module for timezone handling
from rich.console import Console

console = Console()

# Initialize the Binance Exchange
binance = ccxt.binance(
    {
        "enableRateLimit": True,  # important for Binance to avoid IP bans
    }
)


def fetch_ohlcv_in_chunks(symbol, start_date, end_date, timeframe):
    all_data = []
    current_date = start_date
    while current_date < end_date:
        next_date = min(
            current_date + timedelta(days=365), end_date
        )  # Adjust the chunk size as needed
        since = binance.parse8601(current_date.strftime("%Y-%m-%d") + "T00:00:00Z")
        data = binance.fetch_ohlcv(symbol, timeframe, since)
        all_data.extend(data)
        current_date = next_date
    return all_data


def download_crypto_data(symbols, start_date, end_date, timeframe):
    for symbol in symbols:
        console.print(f"Fetching OHLCV data for {symbol}...", style="bold blue")
        ohlcv_data = fetch_ohlcv_in_chunks(symbol, start_date, end_date, timeframe)
        console.print(
            f"OHLCV data for {symbol} fetched successfully.", style="bold green"
        )

        # Specify your CSV file path dynamically
        csv_file_path = f'./data/{symbol.replace("/", "_")}_price_data.csv'

        # Writing to CSV
        with open(csv_file_path, "w", newline="") as file:
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
start_date = datetime(2018, 1, 1)  # Example: starting from January 1, 2018
end_date = datetime.now()  # Up to the current date
symbols = [
    "BTC/USDT",  # Bitcoin
    "ETH/USDT",  # Ethereum
    "BNB/USDT",  # Binance Coin
    "ADA/USDT",  # Cardano
    "SOL/USDT",  # Solana
    "XRP/USDT",  # Ripple
    "DOT/USDT",  # Polkadot
    "LUNA/USDT",  # Terra (consider recent events for Terra)
    "DOGE/USDT",  # Dogecoin
    "AVAX/USDT",  # Avalanche
    "SHIB/USDT",  # Shiba Inu
    "MATIC/USDT",  # Polygon
    "LTC/USDT",  # Litecoin
    "UNI/USDT",  # Uniswap
    "LINK/USDT",  # Chainlink
    "ALGO/USDT",  # Algorand
    "XLM/USDT",  # Stellar
    "VET/USDT",  # VeChain
    "AXS/USDT",  # Axie Infinity
    "ATOM/USDT",  # Cosmos
    "FTT/USDT",  # FTX Token
    "TRX/USDT",  # TRON
    "ETC/USDT",  # Ethereum Classic
    "FIL/USDT",  # Filecoin
    "THETA/USDT",  # Theta Network
    "XTZ/USDT",  # Tezos
    "EOS/USDT",  # EOS
    "AAVE/USDT",  # Aave
    "KSM/USDT",  # Kusama
    "NEO/USDT",  # NEO
    "MKR/USDT",  # Maker
    "COMP/USDT",  # Compound
    "ZEC/USDT",  # Zcash
    "WAVES/USDT",  # Waves
    "DASH/USDT",  # Dash
    "SNX/USDT",  # Synthetix
    "DCR/USDT",  # Decred
    "XEM/USDT",  # NEM
    "QTUM/USDT",  # Qtum
    "ZIL/USDT",  # Zilliqa
    "BAT/USDT",  # Basic Attention Token
    "ENJ/USDT",  # Enjin Coin
    "MANA/USDT",  # Decentraland
    "SUSHI/USDT",  # SushiSwap
    "YFI/USDT",  # Yearn.finance
    "UMA/USDT",  # UMA
    "ICX/USDT",  # ICON
    "ONT/USDT",  # Ontology
    "ZRX/USDT",  # 0x
]  # Simplified list for illustration
timeframe = "1d"

# Download the data
download_crypto_data(symbols, start_date, end_date, timeframe)
