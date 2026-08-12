import requests
import pandas as pd

def fetch_western_cape_weather(lat=-33.9322, lon=18.8644, start_date="2023-01-01", end_date="2023-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "relative_humidity_2m_mean", "rain_sum"],
        "timezone": "Africa/Johannesburg"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    daily_data = data["daily"]
    df = pd.DataFrame(daily_data)
    
    # Calculate Degree-Days (Base threshold = 10°C)
    df['degree_days'] = df['temperature_2m_mean'].apply(lambda t: max(0, t - 10.0))
    
    df.to_csv("western_cape_weather_2023.csv", index=False)
    print(f"Downloaded {len(df)} daily weather records for lat: {lat}, lon: {lon}")
    return df

if __name__ == "__main__":
    fetch_western_cape_weather()