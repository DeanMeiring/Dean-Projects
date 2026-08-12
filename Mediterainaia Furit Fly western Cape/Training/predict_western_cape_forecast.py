import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Dynamic Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "western_cape_medfly_xgboost.json")
CSV_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "wc_outbreak_forecast_14day.csv")
PLOT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "wc_outbreak_risk_trendlines.png")

# 2. Key Western Cape Agricultural Hubs
REGIONS = {
    "Grabouw": {"lat": -34.1525, "lon": 19.0142},
    "Ceres": {"lat": -33.3712, "lon": 19.3101},
    "Citrusdal": {"lat": -32.5858, "lon": 19.0131},
    "Stellenbosch": {"lat": -33.9322, "lon": 18.8644},
    "Hex River Valley": {"lat": -33.4731, "lon": 19.6781}
}

FEATURE_COLUMNS = [
    'temp_14d_mean', 'temp_30d_mean', 'humidity_14d_mean', 
    'vpd_14d_max', 'soil_temp_14d_mean', 'soil_moisture_14d_mean', 
    'degree_days_30d_sum', 'rain_14d_sum'
]

def fetch_weather_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 30,
        "forecast_days": 14,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "vapor_pressure_deficit_max",
            "soil_temperature_0_to_7cm_mean",
            "soil_moisture_0_to_7cm_mean",
            "rain_sum"
        ],
        "timezone": "Africa/Johannesburg"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "daily" not in data:
        raise ValueError(f"API Request Failed: {data}")
        
    df = pd.DataFrame(data["daily"])
    df['time'] = pd.to_datetime(df['time'])
    return df

def categorize_risk(prob):
    if prob >= 0.65:
        return "HIGH (Trigger Bait/SIT)"
    elif prob >= 0.35:
        return "MODERATE (Inspect Traps)"
    else:
        return "LOW (Routine Monitoring)"

# 3. Load Trained XGBoost Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at '{MODEL_PATH}'. Run 'train_xgboost.py' first.")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

forecast_results = []

print("================ WESTERN CAPE MEDFLY OUTBREAK FORECAST ================\n")

# 4. Processing Regions
for region_name, coords in REGIONS.items():
    print(f"Fetching 14-day weather forecast for {region_name}...")
    df_weather = fetch_weather_forecast(coords["lat"], coords["lon"])
    
    # Identify the 14 forecast dates (excluding the past 30 days buffer)
    today = pd.Timestamp.now().normalize()
    forecast_dates = df_weather[df_weather['time'] >= today]['time'].unique()
    
    for eval_date in forecast_dates:
        eval_date = pd.to_datetime(eval_date)
        
        # 14-day and 30-day preceding rolling windows relative to forecast date
        w14 = df_weather[(df_weather['time'] <= eval_date) & (df_weather['time'] > eval_date - pd.Timedelta(days=14))]
        w30 = df_weather[(df_weather['time'] <= eval_date) & (df_weather['time'] > eval_date - pd.Timedelta(days=30))]
        
        if len(w14) >= 7 and len(w30) >= 15:
            dd_30 = np.maximum(0, w30['temperature_2m_mean'] - 10.0).sum()
            
            features = pd.DataFrame([{
                'temp_14d_mean': w14['temperature_2m_mean'].mean(),
                'temp_30d_mean': w30['temperature_2m_mean'].mean(),
                'humidity_14d_mean': w14['relative_humidity_2m_mean'].mean(),
                'vpd_14d_max': w14['vapor_pressure_deficit_max'].max(),
                'soil_temp_14d_mean': w14['soil_temperature_0_to_7cm_mean'].mean(),
                'soil_moisture_14d_mean': w14['soil_moisture_0_to_7cm_mean'].mean(),
                'degree_days_30d_sum': dd_30,
                'rain_14d_sum': w14['rain_sum'].sum()
            }])[FEATURE_COLUMNS]
            
            risk_prob = float(model.predict_proba(features)[:, 1][0])
            
            forecast_results.append({
                'Region': region_name,
                'Date': eval_date.strftime('%Y-%m-%d'),
                'Risk_Probability': round(risk_prob, 4),
                'Risk_Level': categorize_risk(risk_prob),
                'Temp_14d_Mean': round(features['temp_14d_mean'].values[0], 1),
                'Humidity_14d_Mean': round(features['humidity_14d_mean'].values[0], 1),
                'Degree_Days_30d': round(features['degree_days_30d_sum'].values[0], 1)
            })

df_forecast = pd.DataFrame(forecast_results)

# 5. Output Summary Table
print("\n--- 14-DAY OUTBREAK RISK SUMMARY ---")
summary_table = df_forecast.pivot(index='Date', columns='Region', values='Risk_Probability')
print(summary_table.to_string())

# Save CSV
df_forecast.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"\nSaved raw predictions to: '{CSV_OUTPUT_PATH}'")

# 6. Generate Trendline Chart
plt.figure(figsize=(12, 6))
for region in df_forecast['Region'].unique():
    region_data = df_forecast[df_forecast['Region'] == region]
    plt.plot(region_data['Date'], region_data['Risk_Probability'], marker='o', linewidth=2, label=region)

plt.axhline(y=0.65, color='r', linestyle='--', label='High Risk Threshold (0.65)')
plt.axhline(y=0.35, color='orange', linestyle=':', label='Moderate Risk Threshold (0.35)')

plt.title('Western Cape 14-Day Medfly Outbreak Risk Forecast Trendline', fontsize=14, fontweight='bold')
plt.xlabel('Forecast Date', fontsize=11)
plt.ylabel('Outbreak Probability P(Outbreak)', fontsize=11)
plt.xticks(rotation=45)
plt.ylim([0, 1.05])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
plt.close()

print(f"Saved trendline chart to: '{PLOT_OUTPUT_PATH}'")