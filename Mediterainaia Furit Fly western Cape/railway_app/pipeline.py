"""
Self-contained geocode -> fetch weather -> merge -> train -> forecast pipeline
for the Railway-deployed Western Cape Medfly regressor.

This is a consolidated, deploy-local copy of the logic in
`../Data_Transform/{geocode_regions,fetch_weather_history_v2,merge_ftd_weather}.py`
and `../Training/{train_xgboost_regressor,predict_western_cape_forecast_v2}.py`.

Why a copy instead of importing those directly: Railway's "root directory"
service setting scopes the entire build to `railway_app/` — sibling folders
like `Data_Transform/` and `Training/` are never uploaded to the build
context, so `sys.path` tricks pointing at them resolve to paths that don't
exist in the deployed container (this is exactly what broke the first
deploy: `ModuleNotFoundError: No module named 'geocode_regions'`). The
`Data_Transform`/`Training` scripts remain the canonical CLI-runnable
versions per the repo README for local/manual runs; this module is the
Railway runtime's independent copy. If you change the feature engineering,
model config, or risk-threshold logic, change it in both places.
"""
import json
import logging
import os
import time
from datetime import date, datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# This runs unattended on a background thread with no other way to observe
# it (this pipeline's own dev sandbox has no network route to the deployed
# service's HTTP endpoints, so /pipeline/status can't be polled from there
# either) — logging to stdout is the only way progress and failures are
# ever visible, via `railway logs` / the Railway dashboard.
logger = logging.getLogger("medfly_pipeline")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FTD_LONG_PATH = os.path.join(DATA_DIR, "wc_trap_ftd_long.csv")
COORDS_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_region_coords.json")
WEATHER_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_weather_history.csv")
TRAINING_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_training_regression.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "western_cape_medfly_regressor.json")
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_risk_thresholds.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_model_metrics.json")
REGRESSOR_DASHBOARD_PATH = os.path.join(ARTIFACTS_DIR, "medfly_regressor_dashboard.png")
FORECAST_CSV_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_ftd_forecast_14day.csv")
FORECAST_CHART_PATH = os.path.join(ARTIFACTS_DIR, "wc_v2_ftd_forecast_trendlines.png")

FEATURE_COLUMNS = [
    "temp_14d_mean", "temp_30d_mean", "humidity_14d_mean",
    "vpd_14d_max", "soil_temp_14d_mean", "soil_moisture_14d_mean",
    "degree_days_30d_sum", "rain_14d_sum",
]

# --- Step 1: geocode regions --------------------------------------------

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

REGION_QUERIES = {
    "Warm Bokkeveld": ["Warm Bokkeveld", "Op-die-Berg", "Ceres"],
    "Wolseley": ["Wolseley"],
    "Tulbagh": ["Tulbagh"],
    "Elgin & Grabouw": ["Grabouw", "Elgin, Western Cape"],
    "Vyeboom": ["Vyeboom", "Villiersdorp"],
}


def _geocode_query(query):
    resp = requests.get(
        GEOCODE_URL,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    for r in results:
        if r.get("country_code") == "ZA" and r.get("admin1", "").lower().startswith("western cape"):
            return r
    return results[0] if results else None


def geocode_regions():
    coords = {}
    for region, queries in REGION_QUERIES.items():
        found, used_query = None, None
        for query in queries:
            found = _geocode_query(query)
            used_query = query
            if found:
                break
            time.sleep(0.2)
        if not found:
            raise RuntimeError(f"Could not geocode '{region}' with any of {queries}.")
        coords[region] = {
            "lat": found["latitude"],
            "lon": found["longitude"],
            "matched_name": found.get("name"),
            "admin1": found.get("admin1"),
            "query_used": used_query,
        }
        logger.info(
            "geocoded %r -> %r (%s), %.4f, %.4f [query=%r]",
            region, found.get("name"), found.get("admin1"),
            found["latitude"], found["longitude"], used_query,
        )

    with open(COORDS_PATH, "w") as f:
        json.dump(coords, f, indent=2)
    return coords


# --- Step 2: fetch historical weather per region ------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_mean", "relative_humidity_2m_mean", "vapor_pressure_deficit_max",
    "soil_temperature_0_to_7cm_mean", "soil_moisture_0_to_7cm_mean", "rain_sum",
]


def _fetch_region_weather(region, lat, lon, start_date, end_date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": DAILY_VARS, "timezone": "Africa/Johannesburg",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "daily" not in data:
        raise ValueError(f"Archive API request failed for {region}: {data}")
    df = pd.DataFrame(data["daily"])
    df["region"] = region
    return df


def fetch_weather_history(coords, df_ftd):
    frames = []
    for region, region_coords in coords.items():
        region_years = df_ftd.loc[df_ftd["region"] == region, "year"]
        if region_years.empty:
            continue
        start_date = f"{int(region_years.min()) - 1}-12-01"
        end_year = min(int(region_years.max()), date.today().year)
        end_date = f"{end_year}-12-31" if end_year < date.today().year else date.today().isoformat()
        logger.info("fetching weather for %r: %s to %s", region, start_date, end_date)
        df_region = _fetch_region_weather(region, region_coords["lat"], region_coords["lon"], start_date, end_date)
        logger.info("got %d days of weather for %r", len(df_region), region)
        frames.append(df_region)
        time.sleep(0.5)

    df_weather = pd.concat(frames, ignore_index=True)
    df_weather.to_csv(WEATHER_HISTORY_PATH, index=False)
    return df_weather


# --- Step 3: merge FTD + weather into training rows ----------------------

def _week_to_date(year, week):
    return pd.Timestamp.fromisocalendar(int(year), int(min(week, 53)), 1)


def _build_features(window_14, window_30):
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


def merge_ftd_weather(df_ftd, df_weather):
    df_weather = df_weather.copy()
    df_weather["time"] = pd.to_datetime(df_weather["time"])

    records = []
    for region, group in df_ftd.groupby("region"):
        region_weather = df_weather[df_weather["region"] == region].sort_values("time")
        if region_weather.empty:
            continue
        for _, row in group.iterrows():
            event_date = _week_to_date(row["year"], row["week"])
            window_14 = region_weather[(region_weather["time"] <= event_date) & (region_weather["time"] > event_date - pd.Timedelta(days=14))]
            window_30 = region_weather[(region_weather["time"] <= event_date) & (region_weather["time"] > event_date - pd.Timedelta(days=30))]
            features = _build_features(window_14, window_30)
            if features is None:
                continue
            features.update({
                "region": region, "year": int(row["year"]), "week": int(row["week"]),
                "event_date": event_date.strftime("%Y-%m-%d"), "ftd": float(row["ftd"]),
            })
            records.append(features)

    df_final = pd.DataFrame.from_records(records)
    df_final = df_final[["region", "year", "week", "event_date"] + FEATURE_COLUMNS + ["ftd"]]
    df_final.to_csv(TRAINING_PATH, index=False)
    logger.info("merged %d training rows across %d regions", len(df_final), df_final["region"].nunique())
    return df_final


# --- Step 4: train the regressor -----------------------------------------

def train_regressor():
    if not os.path.exists(TRAINING_PATH):
        raise FileNotFoundError(f"'{TRAINING_PATH}' not found — run merge_ftd_weather first.")

    df = pd.read_csv(TRAINING_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["ftd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, eval_metric="rmse", random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), a_min=0, a_max=None)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    model.save_model(MODEL_PATH)

    moderate_cutoff, high_cutoff = y.quantile([0.35, 0.65]).values
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump({
            "low_moderate_boundary": float(moderate_cutoff),
            "moderate_high_boundary": float(high_cutoff),
            "method": "35th/65th percentile of observed FTD in training data",
        }, f, indent=2)

    importance = model.get_booster().get_score(importance_type="gain")
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(df)), "n_regions": int(df["region"].nunique()),
            "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
            "rmse": rmse, "mae": mae, "r2": r2,
            "feature_importance": {k: float(v) for k, v in importance.items()},
        }, f, indent=2)

    _save_regressor_dashboard(df, y_test, y_pred, importance, rmse, mae, r2, moderate_cutoff, high_cutoff)
    logger.info("trained regressor: rmse=%.4f mae=%.4f r2=%.4f n_samples=%d", rmse, mae, r2, len(df))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n_samples": len(df)}


def _save_regressor_dashboard(df, y_test, y_pred, importance, rmse, mae, r2, moderate_cutoff, high_cutoff):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Western Cape Medfly XGBoost Regressor Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)

    axes[0, 0].scatter(y_test, y_pred, alpha=0.4, s=18, color="#2b5c8f")
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="Perfect fit")
    axes[0, 0].set_xlabel("Actual FTD"); axes[0, 0].set_ylabel("Predicted FTD")
    axes[0, 0].set_title("Predicted vs Actual FTD", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="upper left")

    residuals = y_test.values - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.4, s=18, color="#8f2b5c")
    axes[0, 1].axhline(y=0, color="black", linewidth=1)
    axes[0, 1].set_xlabel("Predicted FTD"); axes[0, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[0, 1].set_title("Residual Plot", fontsize=12, fontweight="bold")

    df_imp = pd.DataFrame({"Feature": list(importance.keys()), "Gain": list(importance.values())}).sort_values("Gain")
    axes[1, 0].barh(df_imp["Feature"], df_imp["Gain"], color="#2b5c8f")
    axes[1, 0].set_title("Top Feature Importance (Gain)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Gain Score")

    axes[1, 1].axis("off")
    summary_text = (
        f"--- DATASET OVERVIEW ---\nTotal Training Samples: {len(df)}\nRegions: {df['region'].nunique()}\n"
        f"Years Covered: {df['year'].min()}-{df['year'].max()}\nTarget: FTD (flies/trap/day, continuous)\n\n"
        f"--- EVALUATION METRICS ---\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR^2 Score: {r2:.4f}\n\n"
        f"--- RISK THRESHOLDS (data-driven, not entomological standard) ---\n"
        f"LOW < {moderate_cutoff:.3f} <= MODERATE < {high_cutoff:.3f} <= HIGH\n\n"
        f"--- MODEL CONFIGURATION ---\nAlgorithm: XGBoost Regressor\nTrees (n_estimators): 150\n"
        f"Max Depth: 4 | Learning Rate: 0.03\nSaved Model: western_cape_medfly_regressor.json"
    )
    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=11, family="monospace", verticalalignment="center",
                     bbox=dict(boxstyle="round,pad=1", facecolor="#f4f6f9", alpha=0.8, edgecolor="#ccc"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(REGRESSOR_DASHBOARD_PATH, dpi=300)
    plt.close()


# --- Step 5: live 14-day forecast ----------------------------------------

def _fetch_weather_forecast(lat, lon):
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "past_days": 30, "forecast_days": 14,
            "daily": [
                "temperature_2m_mean", "relative_humidity_2m_mean", "vapor_pressure_deficit_max",
                "soil_temperature_0_to_7cm_mean", "soil_moisture_0_to_7cm_mean", "rain_sum",
            ],
            "timezone": "Africa/Johannesburg",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "daily" not in data:
        raise ValueError(f"API Request Failed: {data}")
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def _categorize_risk(ftd, thresholds):
    if ftd >= thresholds["moderate_high_boundary"]:
        return "HIGH (Trigger Bait/SIT)"
    elif ftd >= thresholds["low_moderate_boundary"]:
        return "MODERATE (Inspect Traps)"
    return "LOW (Routine Monitoring)"


def run_forecast():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet.")
    if not os.path.exists(COORDS_PATH):
        raise FileNotFoundError("Regions not geocoded yet.")

    with open(COORDS_PATH) as f:
        regions = json.load(f)
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    forecast_results = []
    for region_name, coords in regions.items():
        df_weather = _fetch_weather_forecast(coords["lat"], coords["lon"])
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
                    "Region": region_name, "Date": eval_date.strftime("%Y-%m-%d"),
                    "Predicted_FTD": round(predicted_ftd, 4),
                    "Risk_Level": _categorize_risk(predicted_ftd, thresholds),
                    "Temp_14d_Mean": round(features["temp_14d_mean"].values[0], 1),
                    "Humidity_14d_Mean": round(features["humidity_14d_mean"].values[0], 1),
                    "Degree_Days_30d": round(features["degree_days_30d_sum"].values[0], 1),
                })

    df_forecast = pd.DataFrame(forecast_results)
    return df_forecast, thresholds


def save_forecast_outputs(df_forecast, thresholds):
    df_forecast.to_csv(FORECAST_CSV_PATH, index=False)

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
    plt.savefig(FORECAST_CHART_PATH, dpi=300)
    plt.close()


# --- Orchestration ---------------------------------------------------------

def run_full_pipeline(on_step=None):
    """Runs geocode -> fetch -> merge -> train. `on_step(name)` is called
    before each step, if given, so a caller can report progress."""
    def step(name):
        logger.info("=== step: %s ===", name)
        if on_step:
            on_step(name)

    step("geocoding regions")
    coords = geocode_regions()

    step("loading trap-count data")
    if not os.path.exists(FTD_LONG_PATH):
        raise FileNotFoundError(f"'{FTD_LONG_PATH}' not found — should be committed to the repo.")
    df_ftd = pd.read_csv(FTD_LONG_PATH)

    step("fetching historical weather")
    df_weather = fetch_weather_history(coords, df_ftd)

    step("merging FTD + weather")
    merge_ftd_weather(df_ftd, df_weather)

    step("training regressor")
    metrics = train_regressor()

    return metrics
