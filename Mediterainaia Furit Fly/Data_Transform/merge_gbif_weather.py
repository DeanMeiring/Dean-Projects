import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

occurrences_path = os.path.join(SCRIPT_DIR, "gbif_medfly_occurrences_wc.csv")
weather_path = os.path.join(SCRIPT_DIR, "western_cape_weather_historical_expanded.csv")
output_path = os.path.join(SCRIPT_DIR, "final_gbif_weather_training.csv")

# 1. Load Data
df_occ = pd.read_csv(occurrences_path)
df_weather = pd.read_csv(weather_path)

# Ensure weather timestamps are datetime objects
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather = df_weather.sort_values('time').reset_index(drop=True)

weather_min_date = df_weather['time'].min()
weather_max_date = df_weather['time'].max()

# Convert dates cleanly, coercing unparseable range strings (e.g. '1915-01/1915-04') to NaT
df_occ['event_date'] = pd.to_datetime(df_occ['event_date'], errors='coerce', utc=True)

# Drop unparseable rows and convert timezone to naive
df_occ = df_occ.dropna(subset=['event_date']).copy()
df_occ['event_date'] = df_occ['event_date'].dt.tz_convert(None)

# Filter occurrences to match the time frame of your weather dataset
df_occ = df_occ[
    (df_occ['event_date'] >= weather_min_date) & 
    (df_occ['event_date'] <= weather_max_date)
].copy()

print(f"Found {len(df_occ)} GBIF occurrences matching weather timeline ({weather_min_date.date()} to {weather_max_date.date()}).")

if len(df_occ) == 0:
    raise ValueError(
        "No GBIF occurrence dates overlap with your weather date range. "
        "Ensure your weather fetching script covers the years present in GBIF occurrences."
    )

# 2. Extract Preceding Weather Windows for Outbreaks (Class 1)
presence_records = []

for idx, row in df_occ.iterrows():
    event_date = row['event_date']
    
    window_14 = df_weather[(df_weather['time'] <= event_date) & 
                           (df_weather['time'] > event_date - pd.Timedelta(days=14))]
    window_30 = df_weather[(df_weather['time'] <= event_date) & 
                           (df_weather['time'] > event_date - pd.Timedelta(days=30))]
    
    if len(window_14) >= 7:
        degree_days_30 = np.maximum(0, window_30['temperature_2m_mean'] - 10.0).sum()
        
        presence_records.append({
            'event_date': event_date,
            'temp_14d_mean': window_14['temperature_2m_mean'].mean(),
            'temp_30d_mean': window_30['temperature_2m_mean'].mean(),
            'humidity_14d_mean': window_14['relative_humidity_2m_mean'].mean(),
            'vpd_14d_max': window_14['vapor_pressure_deficit_max'].max(),
            'soil_temp_14d_mean': window_14['soil_temperature_0_to_7cm_mean'].mean(),
            'soil_moisture_14d_mean': window_14['soil_moisture_0_to_7cm_mean'].mean(),
            'degree_days_30d_sum': degree_days_30,
            'rain_14d_sum': window_14['rain_sum'].sum(),
            'outbreak': 1
        })

df_pres = pd.DataFrame(presence_records)

# 3. Generate Absence Records (Class 0)
np.random.seed(42)
absence_dates = df_weather[(df_weather['time'].dt.month.isin([6, 7, 8]))]['time'].sample(
    n=len(df_pres), replace=True
)

absence_records = []
for sample_date in absence_dates:
    window_14 = df_weather[(df_weather['time'] <= sample_date) & 
                           (df_weather['time'] > sample_date - pd.Timedelta(days=14))]
    window_30 = df_weather[(df_weather['time'] <= sample_date) & 
                           (df_weather['time'] > sample_date - pd.Timedelta(days=30))]
    
    if len(window_14) >= 7:
        degree_days_30 = np.maximum(0, window_30['temperature_2m_mean'] - 10.0).sum()
        
        absence_records.append({
            'event_date': sample_date,
            'temp_14d_mean': window_14['temperature_2m_mean'].mean(),
            'temp_30d_mean': window_30['temperature_2m_mean'].mean(),
            'humidity_14d_mean': window_14['relative_humidity_2m_mean'].mean(),
            'vpd_14d_max': window_14['vapor_pressure_deficit_max'].max(),
            'soil_temp_14d_mean': window_14['soil_temperature_0_to_7cm_mean'].mean(),
            'soil_moisture_14d_mean': window_14['soil_moisture_0_to_7cm_mean'].mean(),
            'degree_days_30d_sum': degree_days_30,
            'rain_14d_sum': window_14['rain_sum'].sum(),
            'outbreak': 0
        })

df_abs = pd.DataFrame(absence_records)

# 4. Combine and Save Final Training Set
df_final = pd.concat([df_pres, df_abs], ignore_index=True)
df_final.to_csv(output_path, index=False)

print(f"Dataset successfully built!")
print(f"Outbreaks (Class 1): {len(df_pres)} | Non-outbreaks (Class 0): {len(df_abs)}")
print(f"Saved to: '{output_path}'")