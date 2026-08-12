import os
import yfinance as yf
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "raw_stock_data.csv")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"
]
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

print(f"Fetching multi-asset historical data through {END_DATE} ({len(TICKERS)} assets)...")

all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    df['Ticker'] = ticker
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nFetch complete! Saved {len(combined_df)} multi-asset rows up to {END_DATE} in '{OUTPUT_FILE}'.")