import argparse
import json
import os
import time
from datetime import date

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COORDS_PATH = os.path.join(SCRIPT_DIR, "wc_v2_region_coords.json")
FTD_PATH = os.path.join(SCRIPT_DIR, "wc_trap_ftd_long.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "wc_v2_weather_history.csv")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "vapor_pressure_deficit_max",
    "soil_temperature_0_to_7cm_mean",
    "soil_moisture_0_to_7cm_mean",
    "rain_sum",
]


def fetch_region_weather(region, lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARS,
        "timezone": "Africa/Johannesburg",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "daily" not in data:
        raise ValueError(f"Archive API request failed for {region}: {data}")

    df = pd.DataFrame(data["daily"])
    df["region"] = region
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch per-region historical daily weather covering the FTD trap-data span."
    )
    parser.add_argument("--coords", default=COORDS_PATH)
    parser.add_argument("--ftd", default=FTD_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.coords):
        raise FileNotFoundError(f"'{args.coords}' not found. Run geocode_regions.py first.")
    if not os.path.exists(args.ftd):
        raise FileNotFoundError(f"'{args.ftd}' not found. Run parse_ftd_excel.py first.")

    with open(args.coords) as f:
        coords = json.load(f)

    df_ftd = pd.read_csv(args.ftd)

    frames = []
    for region, region_coords in coords.items():
        region_years = df_ftd.loc[df_ftd["region"] == region, "year"]
        if region_years.empty:
            print(f"WARNING: no FTD rows for '{region}', skipping weather fetch.")
            continue

        # 35-day buffer before the earliest week so the first data point still
        # gets a full 30-day rolling window.
        start_date = f"{int(region_years.min()) - 1}-12-01"
        end_year = min(int(region_years.max()), date.today().year)
        end_date = f"{end_year}-12-31" if end_year < date.today().year else date.today().isoformat()

        print(f"Fetching weather for {region} ({region_coords['lat']:.4f}, "
              f"{region_coords['lon']:.4f}) from {start_date} to {end_date}...")

        df_region = fetch_region_weather(
            region, region_coords["lat"], region_coords["lon"], start_date, end_date
        )
        frames.append(df_region)
        time.sleep(0.5)  # be polite to the free API

    df_weather = pd.concat(frames, ignore_index=True)
    df_weather.to_csv(args.output, index=False)
    print(f"Saved {len(df_weather)} region-days of weather history to '{args.output}'")


if __name__ == "__main__":
    main()
