import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

occurrences_path = os.path.join(SCRIPT_DIR, "ca_gbif_occurrences.csv")
weather_path = os.path.join(SCRIPT_DIR, "california_weather_historical.csv")
output_path = os.path.join(SCRIPT_DIR, "final_ca_gbif_weather_training.csv")

df_occ = pd.read_csv(occurrences_path)
df_weather = pd.read_csv(weather_path)

df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather = df_weather.sort_values('time').reset_index(drop=True)

weather_min_date = df_weather['time'].min()
weather_max_date = df_weather['time'].max()

df_occ['event_date'] = pd.to_datetime(df_occ['event_date'], errors='coerce', utc=True)
df_occ = df_occ.dropna(subset=['event_date']).copy()
df_occ['event_date'] = df_occ['event_date'].dt.tz_convert(None)

df_occ = df_occ[
    (df_occ['event_date'] >= weather_min_date) & 
    (df_occ['event_date'] <= weather_max_date)
].copy()

# Class 1: Outbreak Events
presence_records = []
for idx, row in df_occ.iterrows():
    event_date = row['event_date']
    
    w14 = df_weather[(df_weather['time'] <= event_date) & (df_weather['time'] > event_date - pd.Timedelta(days=14))]
    w30 = df_weather[(df_weather['time'] <= event_date) & (df_weather['time'] > event_date - pd.Timedelta(days=30))]
    
    if len(w14) >= 7:
        dd_30 = np.maximum(0, w30['temperature_2m_mean'] - 10.0).sum()
        
        presence_records.append({
            'event_date': event_date,
            'temp_14d_mean': w14['temperature_2m_mean'].mean(),
            'temp_30d_mean': w30['temperature_2m_mean'].mean(),
            'humidity_14d_mean': w14['relative_humidity_2m_mean'].mean(),
            'vpd_14d_max': w14['vapor_pressure_deficit_max'].max(),
            'soil_temp_14d_mean': w14['soil_temperature_0_to_7cm_mean'].mean(),
            'soil_moisture_14d_mean': w14['soil_moisture_0_to_7cm_mean'].mean(),
            'degree_days_30d_sum': dd_30,
            'rain_14d_sum': w14['rain_sum'].sum(),
            'outbreak': 1
        })

df_pres = pd.DataFrame(presence_records)

# Class 0: Baseline Non-Outbreak Events
np.random.seed(42)
absence_dates = df_weather[(df_weather['time'].dt.month.isin([12, 1, 2]))]['time'].sample(
    n=len(df_pres), replace=True
)

absence_records = []
for sample_date in absence_dates:
    w14 = df_weather[(df_weather['time'] <= sample_date) & (df_weather['time'] > sample_date - pd.Timedelta(days=14))]
    w30 = df_weather[(df_weather['time'] <= sample_date) & (df_weather['time'] > sample_date - pd.Timedelta(days=30))]
    
    if len(w14) >= 7:
        dd_30 = np.maximum(0, w30['temperature_2m_mean'] - 10.0).sum()
        
        absence_records.append({
            'event_date': sample_date,
            'temp_14d_mean': w14['temperature_2m_mean'].mean(),
            'temp_30d_mean': w30['temperature_2m_mean'].mean(),
            'humidity_14d_mean': w14['relative_humidity_2m_mean'].mean(),
            'vpd_14d_max': w14['vapor_pressure_deficit_max'].max(),
            'soil_temp_14d_mean': w14['soil_temperature_0_to_7cm_mean'].mean(),
            'soil_moisture_14d_mean': w14['soil_moisture_0_to_7cm_mean'].mean(),
            'degree_days_30d_sum': dd_30,
            'rain_14d_sum': w14['rain_sum'].sum(),
            'outbreak': 0
        })

df_abs = pd.DataFrame(absence_records)

df_final = pd.concat([df_pres, df_abs], ignore_index=True)
df_final.to_csv(output_path, index=False)

print(f"Dataset built! Outbreaks (Class 1): {len(df_pres)} | Non-outbreaks (Class 0): {len(df_abs)}")
print(f"Saved to: '{output_path}'")