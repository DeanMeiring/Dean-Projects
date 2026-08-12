import os
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "california_weather_historical.csv")

URL = "https://archive-api.open-meteo.com/v1/archive"

# California Central Valley Centroid (Fresno)
params = {
    "latitude": 36.7468,
    "longitude": -119.7726,
    "start_date": "1970-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "vapor_pressure_deficit_max",
        "soil_temperature_0_to_7cm_mean",
        "soil_moisture_0_to_7cm_mean",
        "rain_sum"
    ],
    "timezone": "America/Los_Angeles"
}

print("Fetching historical weather dataset for California from Open-Meteo...")
response = requests.get(URL, params=params)
data = response.json()

if "daily" in data:
    df_weather = pd.DataFrame(data["daily"])
    df_weather.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully downloaded weather data ({len(df_weather)} days). Saved to '{OUTPUT_FILE}'.")
else:
    print("Failed to fetch weather data:", data)