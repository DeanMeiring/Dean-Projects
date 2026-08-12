import os
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "western_cape_weather_historical_expanded.csv")

# Open-Meteo Historical Weather API for Western Cape (Cape Town / Winelands centroid)
URL = "https://archive-api.open-meteo.com/v1/archive"

# Set start date back to cover your GBIF occurrence dates
params = {
    "latitude": -33.9249,
    "longitude": 18.4241,
    "start_date": "1940-01-01",  # Expand back to 2010 (or earliest year from Step 1)
    "end_date": "2025-12-31",
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

print("Fetching extended historical weather dataset from Open-Meteo...")
response = requests.get(URL, params=params)
data = response.json()

if "daily" in data:
    df_weather = pd.DataFrame(data["daily"])
    df_weather.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully downloaded weather data ({len(df_weather)} days). Saved to '{OUTPUT_FILE}'.")
else:
    print("Failed to fetch weather data:", data)