import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xgboost as xgb

# 1. Dynamic Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "western_cape_medfly_regressor.json")
THRESHOLDS_PATH = os.path.join(SCRIPT_DIR, "wc_v2_risk_thresholds.json")
COORDS_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "wc_v2_region_coords.json")
CSV_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "wc_v2_ftd_forecast_14day.csv")
PLOT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "wc_v2_ftd_forecast_trendlines.png")

FEATURE_COLUMNS = [
    "temp_14d_mean", "temp_30d_mean", "humidity_14d_mean",
    "vpd_14d_max", "soil_temp_14d_mean", "soil_moisture_14d_mean",
    "degree_days_30d_sum", "rain_14d_sum",
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
            "rain_sum",
        ],
        "timezone": "Africa/Johannesburg",
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    if "daily" not in data:
        raise ValueError(f"API Request Failed: {data}")

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def load_thresholds():
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(
            f"'{THRESHOLDS_PATH}' not found. Run train_xgboost_regressor.py first."
        )
    with open(THRESHOLDS_PATH) as f:
        return json.load(f)


def categorize_risk(ftd, thresholds):
    if ftd >= thresholds["moderate_high_boundary"]:
        return "HIGH (Trigger Bait/SIT)"
    elif ftd >= thresholds["low_moderate_boundary"]:
        return "MODERATE (Inspect Traps)"
    else:
        return "LOW (Routine Monitoring)"


def run_forecast():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at '{MODEL_PATH}'. Run train_xgboost_regressor.py first.")
    if not os.path.exists(COORDS_PATH):
        raise FileNotFoundError(f"'{COORDS_PATH}' not found. Run geocode_regions.py first.")

    with open(COORDS_PATH) as f:
        regions = json.load(f)

    thresholds = load_thresholds()

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    forecast_results = []

    print("================ WESTERN CAPE MEDFLY FTD FORECAST (v2, real trap-data regressor) ================\n")

    for region_name, coords in regions.items():
        print(f"Fetching 14-day weather forecast for {region_name}...")
        df_weather = fetch_weather_forecast(coords["lat"], coords["lon"])

        today = pd.Timestamp.now().normalize()
        forecast_dates = df_weather[df_weather["time"] >= today]["time"].unique()

        for eval_date in forecast_dates:
            eval_date = pd.to_datetime(eval_date)

            w14 = df_weather[(df_weather["time"] <= eval_date) & (df_weather["time"] > eval_date - pd.Timedelta(days=14))]
            w30 = df_weather[(df_weather["time"] <= eval_date) & (df_weather["time"] > eval_date - pd.Timedelta(days=30))]

            if len(w14) >= 7 and len(w30) >= 15:
                dd_30 = np.maximum(0, w30["temperature_2m_mean"] - 10.0).sum()

                features = pd.DataFrame([{
                    "temp_14d_mean": w14["temperature_2m_mean"].mean(),
                    "temp_30d_mean": w30["temperature_2m_mean"].mean(),
                    "humidity_14d_mean": w14["relative_humidity_2m_mean"].mean(),
                    "vpd_14d_max": w14["vapor_pressure_deficit_max"].max(),
                    "soil_temp_14d_mean": w14["soil_temperature_0_to_7cm_mean"].mean(),
                    "soil_moisture_14d_mean": w14["soil_moisture_0_to_7cm_mean"].mean(),
                    "degree_days_30d_sum": dd_30,
                    "rain_14d_sum": w14["rain_sum"].sum(),
                }])[FEATURE_COLUMNS]

                predicted_ftd = float(max(0.0, model.predict(features)[0]))

                forecast_results.append({
                    "Region": region_name,
                    "Date": eval_date.strftime("%Y-%m-%d"),
                    "Predicted_FTD": round(predicted_ftd, 4),
                    "Risk_Level": categorize_risk(predicted_ftd, thresholds),
                    "Temp_14d_Mean": round(features["temp_14d_mean"].values[0], 1),
                    "Humidity_14d_Mean": round(features["humidity_14d_mean"].values[0], 1),
                    "Degree_Days_30d": round(features["degree_days_30d_sum"].values[0], 1),
                })

    df_forecast = pd.DataFrame(forecast_results)
    return df_forecast, thresholds


def save_outputs(df_forecast, thresholds):
    print("\n--- 14-DAY FTD FORECAST SUMMARY ---")
    summary_table = df_forecast.pivot(index="Date", columns="Region", values="Predicted_FTD")
    print(summary_table.to_string())

    df_forecast.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved raw predictions to: '{CSV_OUTPUT_PATH}'")

    plt.figure(figsize=(12, 6))
    for region in df_forecast["Region"].unique():
        region_data = df_forecast[df_forecast["Region"] == region]
        plt.plot(region_data["Date"], region_data["Predicted_FTD"], marker="o", linewidth=2, label=region)

    plt.axhline(y=thresholds["moderate_high_boundary"], color="r", linestyle="--",
                label=f"High Risk Threshold ({thresholds['moderate_high_boundary']:.3f})")
    plt.axhline(y=thresholds["low_moderate_boundary"], color="orange", linestyle=":",
                label=f"Moderate Risk Threshold ({thresholds['low_moderate_boundary']:.3f})")

    plt.title("Western Cape 14-Day Medfly FTD Forecast Trendline (real trap-data regressor)", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Date", fontsize=11)
    plt.ylabel("Predicted FTD (flies/trap/day)", fontsize=11)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"Saved trendline chart to: '{PLOT_OUTPUT_PATH}'")


if __name__ == "__main__":
    df_forecast, thresholds = run_forecast()
    save_outputs(df_forecast, thresholds)
