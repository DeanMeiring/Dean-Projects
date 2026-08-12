import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "raw_stock_data.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "final_stock_training.csv")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Raw data file not found at '{INPUT_FILE}'. Run 'fetch_stock_data.py' first.")

df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

FEATURE_COLUMNS = [
    'sma_14', 'sma_50', 'macd', 'macd_signal', 'rsi_14', 
    'daily_return', 'volatility_14d', 'volatility_30d', 'volume_ratio'
]

processed_groups = []

for ticker, group in df.groupby('Ticker'):
    group = group.copy()
    
    # 1. Moving Averages & Trend
    group['sma_14'] = group['Close'].rolling(window=14).mean()
    group['sma_50'] = group['Close'].rolling(window=50).mean()
    group['ema_12'] = group['Close'].ewm(span=12, adjust=False).mean()
    group['ema_26'] = group['Close'].ewm(span=26, adjust=False).mean()
    
    group['macd'] = group['ema_12'] - group['ema_26']
    group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
    
    # 2. RSI 14
    delta = group['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    group['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 3. Returns & Volatility
    group['daily_return'] = group['Close'].pct_change()
    group['volatility_14d'] = group['daily_return'].rolling(window=14).std()
    group['volatility_30d'] = group['daily_return'].rolling(window=30).std()
    
    # 4. Volume Ratio
    group['volume_sma_14'] = group['Volume'].rolling(window=14).mean()
    group['volume_ratio'] = group['Volume'] / (group['volume_sma_14'] + 1e-9)
    
    # 5. Target Creation (+1% for stocks, +3% for crypto over 2 days)
    target_return = 0.03 if "-USD" in ticker else 0.01
    group['future_2d_return'] = (group['Close'].shift(-2) - group['Close']) / group['Close']
    group['target'] = (group['future_2d_return'] > target_return).astype(float)
    
    # Keep latest data points by only dropping missing feature values, not target NaNs
    processed_groups.append(group.dropna(subset=FEATURE_COLUMNS))

final_df = pd.concat(processed_groups, ignore_index=True)

feature_cols = [
    'Date', 'Ticker', 'Close', 'sma_14', 'sma_50', 'macd', 'macd_signal', 'rsi_14', 
    'daily_return', 'volatility_14d', 'volatility_30d', 'volume_ratio', 'target'
]

final_df = final_df[feature_cols]
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"Feature engineering complete! Total samples preserved: {len(final_df)}")
print(f"Saved dataset to: '{OUTPUT_FILE}'")