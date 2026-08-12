import os
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORM_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "final_stock_training.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "stock_xgboost_model.json")
CSV_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "multi_asset_2d_forecast.csv")
PLOT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "multi_asset_2d_trendline.png")

MODERATE_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
HIGH_VOL_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]

FEATURE_COLUMNS = [
    'sma_14', 'sma_50', 'macd', 'macd_signal', 'rsi_14', 
    'daily_return', 'volatility_14d', 'volatility_30d', 'volume_ratio'
]

def get_current_live_price(ticker, fallback_price):
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.get('lastPrice')
        if price is not None and not np.isnan(price):
            return round(float(price), 2)
    except Exception:
        pass
    return round(float(fallback_price), 2)

def categorize_signal(prob, category):
    threshold = 0.40 if category == "Moderate" else 0.45
    if prob >= threshold:
        return f"BULLISH (P >= {threshold})"
    elif prob >= (threshold - 0.08):
        return "NEUTRAL / WATCHLIST"
    else:
        return "BEARISH / WEAK"

if not os.path.exists(TRANSFORM_DATA_PATH):
    raise FileNotFoundError(f"Transformed dataset not found at '{TRANSFORM_DATA_PATH}'. Run feature engineering first.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Run 'train_stock_xgboost.py' first.")

df_transformed = pd.read_csv(TRANSFORM_DATA_PATH)
df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

forecast_results = []
latest_prices = {}

print("================ GENERATING 2-DAY ADVANCE PREDICTIONS ================\n")

# Process Moderate Volatility Stocks
for ticker in MODERATE_TICKERS:
    t_df = df_transformed[df_transformed['Ticker'] == ticker].sort_values('Date').tail(10)
    
    if t_df.empty:
        continue
        
    latest_close = float(t_df['Close'].iloc[-1])
    live_price = get_current_live_price(ticker, latest_close)
    latest_prices[ticker] = live_price
    
    for idx, row in t_df.iterrows():
        input_data = pd.DataFrame([row[FEATURE_COLUMNS]])
        prob = float(model.predict_proba(input_data)[:, 1][0])
        
        obs_date = row['Date']
        target_forecast_date = (obs_date + pd.offsets.BusinessDay(2)).strftime('%Y-%m-%d')
        
        forecast_results.append({
            'Category': 'Moderate (Stocks)',
            'Ticker': ticker,
            'Current_Price': live_price,
            'Observation_Date': obs_date.strftime('%Y-%m-%d'),
            'Target_Forecast_Date': target_forecast_date,
            '2D_Bullish_Probability': round(prob, 4),
            'Signal': categorize_signal(prob, "Moderate")
        })

# Process High Volatility Crypto
for ticker in HIGH_VOL_TICKERS:
    t_df = df_transformed[df_transformed['Ticker'] == ticker].sort_values('Date').tail(10)
    
    if t_df.empty:
        continue
        
    latest_close = float(t_df['Close'].iloc[-1])
    live_price = get_current_live_price(ticker, latest_close)
    latest_prices[ticker] = live_price
    
    for idx, row in t_df.iterrows():
        input_data = pd.DataFrame([row[FEATURE_COLUMNS]])
        prob = float(model.predict_proba(input_data)[:, 1][0])
        
        obs_date = row['Date']
        target_forecast_date = (obs_date + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        
        forecast_results.append({
            'Category': 'High Volatility (Crypto)',
            'Ticker': ticker,
            'Current_Price': live_price,
            'Observation_Date': obs_date.strftime('%Y-%m-%d'),
            'Target_Forecast_Date': target_forecast_date,
            '2D_Bullish_Probability': round(prob, 4),
            'Signal': categorize_signal(prob, "High")
        })

df_forecast = pd.DataFrame(forecast_results)
df_forecast.to_csv(CSV_OUTPUT_PATH, index=False)

# Render Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

df_mod = df_forecast[df_forecast['Category'] == 'Moderate (Stocks)']
for ticker in MODERATE_TICKERS:
    t_data = df_mod[df_mod['Ticker'] == ticker]
    if not t_data.empty:
        lbl = f"{ticker} (${latest_prices.get(ticker, 0):,.2f})"
        ax1.plot(t_data['Target_Forecast_Date'], t_data['2D_Bullish_Probability'], marker='o', linewidth=2, label=lbl)

ax1.axhline(y=0.40, color='g', linestyle='--', label='Bullish Threshold (0.40)')
ax1.set_title('Moderate Volatility Stocks (Target Forecast Date)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Target Forecast Date (+2 Business Days)', fontsize=10)
ax1.set_ylabel('2-Day Bullish Probability', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=9)

df_high = df_forecast[df_forecast['Category'] == 'High Volatility (Crypto)']
for ticker in HIGH_VOL_TICKERS:
    t_data = df_high[df_high['Ticker'] == ticker]
    if not t_data.empty:
        lbl = f"{ticker} (${latest_prices.get(ticker, 0):,.2f})"
        ax2.plot(t_data['Target_Forecast_Date'], t_data['2D_Bullish_Probability'], marker='s', linewidth=2, label=lbl)

ax2.axhline(y=0.45, color='g', linestyle='--', label='Bullish Threshold (0.45)')
ax2.set_title('High Volatility Crypto (Target Forecast Date)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Target Forecast Date (+2 Calendar Days)', fontsize=10)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=9)

plt.ylim([0, 1.05])
fig.suptitle('2-Day Advance Forecast (Target Dates)', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
plt.close()

print(f"Saved forecast CSV to: '{CSV_OUTPUT_PATH}'")
print(f"Saved updated trendline image to: '{PLOT_OUTPUT_PATH}'")