"""
Railway-deployed service for the Western Cape Medfly regressor (v2, trained
on Ghian du Toit's real FTD trap-count data instead of GBIF occurrence proxies).

This process has normal outbound internet access (unlike the sandbox this
pipeline was originally written in), so it's the one place the geocoding /
historical weather / live forecast calls to Open-Meteo actually run.

Endpoints:
  GET  /health           - liveness check
  POST /pipeline/run     - (re)runs geocode -> fetch weather history -> merge -> train.
                            Idempotent; safe to re-run to refresh the model on new FTD data.
  GET  /forecast         - live 14-day FTD forecast per region (JSON). 404s with a clear
                            message if /pipeline/run hasn't been called yet.
  GET  /forecast/chart   - the trendline PNG from the most recent forecast.
"""
import os
import runpy
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WC_DIR = os.path.dirname(APP_DIR)  # "...Mediterainaia Furit Fly western Cape"
DATA_TRANSFORM_DIR = os.path.join(WC_DIR, "Data_Transform")
TRAINING_DIR = os.path.join(WC_DIR, "Training")

# These directories have spaces in their path components, which is fine for
# sys.path (it's just a string) — only the `import <module>` names below need
# to be valid identifiers, and they are.
sys.path.insert(0, DATA_TRANSFORM_DIR)
sys.path.insert(0, TRAINING_DIR)

app = FastAPI(title="Western Cape Medfly Forecast API", version="2.0.0")

MODEL_PATH = os.path.join(TRAINING_DIR, "western_cape_medfly_regressor.json")
CHART_PATH = os.path.join(TRAINING_DIR, "wc_v2_ftd_forecast_trendlines.png")


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": os.path.exists(MODEL_PATH)}


@app.post("/pipeline/run")
def run_pipeline():
    """Geocode regions, fetch historical weather, merge with FTD data, and train.

    Requires `Data_Transform/wc_trap_ftd_long.csv` to already be committed in
    the repo (produced by parse_ftd_excel.py, which needs the source .xlsx
    that isn't checked in — that step runs once, locally, not here).
    """
    import geocode_regions
    import fetch_weather_history_v2
    import merge_ftd_weather

    train_script_path = os.path.join(TRAINING_DIR, "train_xgboost_regressor.py")

    try:
        geocode_regions.main()
        fetch_weather_history_v2.main()
        merge_ftd_weather.main()
        # train_xgboost_regressor.py is a flat script (matches the existing
        # repo's classifier-training style), not a function — run it fresh
        # with runpy each call so re-training actually re-executes rather
        # than hitting Python's module-import cache on a second POST.
        runpy.run_path(train_script_path, run_name="__main__")
    except Exception as exc:  # surface pipeline failures to the caller
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return {"status": "trained", "model_path": MODEL_PATH}


@app.get("/forecast")
def get_forecast():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=409,
            detail="Model not trained yet. POST /pipeline/run first.",
        )

    import predict_western_cape_forecast_v2 as predict_mod

    try:
        df_forecast, thresholds = predict_mod.run_forecast()
        predict_mod.save_outputs(df_forecast, thresholds)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Forecast failed: {exc}") from exc

    return {
        "thresholds": thresholds,
        "forecast": df_forecast.to_dict(orient="records"),
    }


@app.get("/forecast/chart")
def get_forecast_chart():
    if not os.path.exists(CHART_PATH):
        raise HTTPException(
            status_code=404,
            detail="No chart yet. Call GET /forecast first to generate one.",
        )
    return FileResponse(CHART_PATH, media_type="image/png")
