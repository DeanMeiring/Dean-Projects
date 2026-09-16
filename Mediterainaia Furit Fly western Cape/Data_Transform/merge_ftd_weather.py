import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FTD_PATH = os.path.join(SCRIPT_DIR, "wc_trap_ftd_long.csv")
WEATHER_PATH = os.path.join(SCRIPT_DIR, "wc_v2_weather_history.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "wc_v2_training_regression.csv")

FEATURE_COLUMNS = [
    "temp_14d_mean", "temp_30d_mean", "humidity_14d_mean",
    "vpd_14d_max", "soil_temp_14d_mean", "soil_moisture_14d_mean",
    "degree_days_30d_sum", "rain_14d_sum",
]


def week_to_date(year, week):
    # Monday of the given ISO week. FTD weeks are simple sequential trap-read
    # weeks (1..52/53), not strict ISO weeks, but this gives a consistent,
    # reproducible date anchor to build weather windows around.
    return pd.Timestamp.fromisocalendar(int(year), int(min(week, 53)), 1)


def build_features(window_14, window_30):
    if len(window_14) < 7 or len(window_30) < 15:
        return None

    degree_days_30 = np.maximum(0, window_30["temperature_2m_mean"] - 10.0).sum()

    return {
        "temp_14d_mean": window_14["temperature_2m_mean"].mean(),
        "temp_30d_mean": window_30["temperature_2m_mean"].mean(),
        "humidity_14d_mean": window_14["relative_humidity_2m_mean"].mean(),
        "vpd_14d_max": window_14["vapor_pressure_deficit_max"].max(),
        "soil_temp_14d_mean": window_14["soil_temperature_0_to_7cm_mean"].mean(),
        "soil_moisture_14d_mean": window_14["soil_moisture_0_to_7cm_mean"].mean(),
        "degree_days_30d_sum": degree_days_30,
        "rain_14d_sum": window_14["rain_sum"].sum(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge weekly FTD trap counts with preceding weather windows into a regression training set."
    )
    parser.add_argument("--ftd", default=FTD_PATH)
    parser.add_argument("--weather", default=WEATHER_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.ftd):
        raise FileNotFoundError(f"'{args.ftd}' not found. Run parse_ftd_excel.py first.")
    if not os.path.exists(args.weather):
        raise FileNotFoundError(f"'{args.weather}' not found. Run fetch_weather_history_v2.py first.")

    df_ftd = pd.read_csv(args.ftd)
    df_weather = pd.read_csv(args.weather)
    df_weather["time"] = pd.to_datetime(df_weather["time"])

    records = []
    skipped = 0
    for region, group in df_ftd.groupby("region"):
        region_weather = df_weather[df_weather["region"] == region].sort_values("time")
        if region_weather.empty:
            print(f"WARNING: no weather rows for '{region}', skipping {len(group)} FTD rows.")
            skipped += len(group)
            continue

        for _, row in group.iterrows():
            event_date = week_to_date(row["year"], row["week"])

            window_14 = region_weather[
                (region_weather["time"] <= event_date)
                & (region_weather["time"] > event_date - pd.Timedelta(days=14))
            ]
            window_30 = region_weather[
                (region_weather["time"] <= event_date)
                & (region_weather["time"] > event_date - pd.Timedelta(days=30))
            ]

            features = build_features(window_14, window_30)
            if features is None:
                skipped += 1
                continue

            features.update({
                "region": region,
                "year": int(row["year"]),
                "week": int(row["week"]),
                "event_date": event_date.strftime("%Y-%m-%d"),
                "ftd": float(row["ftd"]),
            })
            records.append(features)

    df_final = pd.DataFrame.from_records(records)
    df_final = df_final[["region", "year", "week", "event_date"] + FEATURE_COLUMNS + ["ftd"]]
    df_final.to_csv(args.output, index=False)

    print(f"Built {len(df_final)} training rows ({skipped} rows skipped: missing weather or short window).")
    print(f"Saved to: '{args.output}'")


if __name__ == "__main__":
    main()
