"""
Fetch the extra weather needed for the v3 features, per region:
  - hourly air temperature (for chill portions, Dynamic Model)
  - daily max wind speed, max gusts and dominant wind direction (Cape Doctor)

Uses the Open-Meteo historical archive (same source as fetch_weather_history_v2.py).
Each region is cached to its own CSV so a rate-limit (HTTP 429) only costs a retry.

Output: wc_v3_hourly_temp.csv, wc_v3_daily_wind.csv
"""
import json
import os
import time

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COORDS_PATH = os.path.join(SCRIPT_DIR, "wc_v2_region_coords.json")
FTD_PATH = os.path.join(SCRIPT_DIR, "wc_trap_ftd_long.csv")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache_v3")
HOURLY_OUT = os.path.join(SCRIPT_DIR, "wc_v3_hourly_temp.csv")
WIND_OUT = os.path.join(SCRIPT_DIR, "wc_v3_daily_wind.csv")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get(params, label):
    for attempt in range(8):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()
            reason = "rate-limited"
        except (requests.RequestException, ValueError) as e:  # timeouts, bad/empty responses
            reason = f"error ({type(e).__name__})"
        wait = 30 * (attempt + 1)
        print(f"  {label}: {reason}, retrying in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{label}: failed after 8 attempts")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    coords = json.load(open(COORDS_PATH))
    ftd = pd.read_csv(FTD_PATH)
    end = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    hourly, wind = [], []
    for region, c in coords.items():
        first_year = int(ftd.loc[ftd.region == region, "year"].min())
        start = f"{first_year - 1}-03-01"  # covers the previous winter's chill season
        slug = region.replace(" & ", "_").replace(" ", "_")

        h_path = os.path.join(CACHE_DIR, f"hourly_{slug}.csv")
        if not os.path.exists(h_path):
            data = get({"latitude": c["lat"], "longitude": c["lon"], "start_date": start, "end_date": end,
                        "hourly": "temperature_2m", "timezone": "Africa/Johannesburg"}, f"{region} hourly")
            pd.DataFrame(data["hourly"]).assign(region=region).to_csv(h_path, index=False)
            print(f"{region}: hourly saved", flush=True)
            time.sleep(10)
        hourly.append(pd.read_csv(h_path))

        w_path = os.path.join(CACHE_DIR, f"wind_{slug}.csv")
        if not os.path.exists(w_path):
            data = get({"latitude": c["lat"], "longitude": c["lon"], "start_date": start, "end_date": end,
                        "daily": "wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
                        "timezone": "Africa/Johannesburg"}, f"{region} wind")
            pd.DataFrame(data["daily"]).assign(region=region).to_csv(w_path, index=False)
            print(f"{region}: wind saved", flush=True)
            time.sleep(10)
        wind.append(pd.read_csv(w_path))

    pd.concat(hourly).to_csv(HOURLY_OUT, index=False)
    pd.concat(wind).to_csv(WIND_OUT, index=False)
    print(f"Saved {HOURLY_OUT} and {WIND_OUT}")


if __name__ == "__main__":
    main()
