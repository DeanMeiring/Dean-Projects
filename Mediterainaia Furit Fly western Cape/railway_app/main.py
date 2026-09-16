"""
Railway-deployed service for the Western Cape Medfly regressor (v2, trained
on Ghian du Toit's real FTD trap-count data instead of GBIF occurrence proxies).

This process has normal outbound internet access (unlike the sandbox this
pipeline was originally written in), so it's the one place the geocoding /
historical weather / live forecast calls to Open-Meteo actually run.

Endpoints:
  GET  /                 - HTML dashboard (dataset stats, model status, live forecast, charts)
  GET  /health           - liveness check
  GET  /stats            - dataset + model status as JSON, including which real-world place
                            each trap-data region name geocoded to (verify before trusting
                            the forecast — a wrong match silently poisons the weather features)
  POST /pipeline/run     - starts geocode -> fetch weather history -> merge -> train in the
                            background and returns immediately (the full run can take minutes
                            against live Open-Meteo; a synchronous request risked a proxy
                            timeout looking like a failure even when training kept going
                            server-side). Poll GET /pipeline/status for progress.
  GET  /pipeline/status  - current/last pipeline run status, step log, and any error.
  GET  /forecast         - live 14-day FTD forecast per region (JSON). 404s with a clear
                            message if /pipeline/run hasn't completed yet.
  GET  /forecast/chart   - the trendline PNG from the most recent forecast.
  GET  /dashboard/regressor.png  - v2 regressor performance dashboard (predicted-vs-actual,
                                     residuals, feature importance, KPIs)
  GET  /dashboard/classifier.png - v1 GBIF-proxy classifier performance dashboard, for comparison

Set AUTO_TRAIN_ON_START=true to kick off one pipeline run automatically on
boot if no model exists yet (background thread; doesn't block the health
check the Railway deploy waits on).
"""
import json
import os
import runpy
import sys
import threading
import traceback
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WC_DIR = os.path.dirname(APP_DIR)  # "...Mediterainaia Furit Fly western Cape"
DATA_TRANSFORM_DIR = os.path.join(WC_DIR, "Data_Transform")
TRAINING_DIR = os.path.join(WC_DIR, "Training")
STATIC_DIR = os.path.join(APP_DIR, "static")

# These directories have spaces in their path components, which is fine for
# sys.path (it's just a string) — only the `import <module>` names below need
# to be valid identifiers, and they are.
sys.path.insert(0, DATA_TRANSFORM_DIR)
sys.path.insert(0, TRAINING_DIR)

app = FastAPI(title="Western Cape Medfly Forecast API", version="2.0.0")

MODEL_PATH = os.path.join(TRAINING_DIR, "western_cape_medfly_regressor.json")
CHART_PATH = os.path.join(TRAINING_DIR, "wc_v2_ftd_forecast_trendlines.png")
THRESHOLDS_PATH = os.path.join(TRAINING_DIR, "wc_v2_risk_thresholds.json")
METRICS_PATH = os.path.join(TRAINING_DIR, "wc_v2_model_metrics.json")
FTD_LONG_PATH = os.path.join(DATA_TRANSFORM_DIR, "wc_trap_ftd_long.csv")
V1_MODEL_PATH = os.path.join(TRAINING_DIR, "western_cape_medfly_xgboost.json")
V2_REGRESSOR_DASHBOARD_PATH = os.path.join(TRAINING_DIR, "medfly_regressor_dashboard.png")
V1_CLASSIFIER_DASHBOARD_PATH = os.path.join(TRAINING_DIR, "medfly_model_dashboard.png")
REGION_COORDS_PATH = os.path.join(DATA_TRANSFORM_DIR, "wc_v2_region_coords.json")

_pipeline_lock = threading.Lock()
_pipeline_state = {
    "status": "idle",  # idle | running | success | failed
    "step": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _run_pipeline_locked():
    """The actual geocode -> fetch -> merge -> train sequence. Runs on a
    background thread; updates _pipeline_state as it goes so /pipeline/status
    has something meaningful to report while a run is in flight."""
    import geocode_regions
    import fetch_weather_history_v2
    import merge_ftd_weather

    train_script_path = os.path.join(TRAINING_DIR, "train_xgboost_regressor.py")

    # status/started_at are already set by _start_pipeline_if_idle before
    # this thread was launched — only this thread touches step/error/finished_at.
    try:
        _pipeline_state["step"] = "geocoding regions"
        geocode_regions.main()

        _pipeline_state["step"] = "fetching historical weather"
        fetch_weather_history_v2.main()

        _pipeline_state["step"] = "merging FTD + weather"
        merge_ftd_weather.main()

        _pipeline_state["step"] = "training regressor"
        # train_xgboost_regressor.py is a flat script (matches the existing
        # repo's classifier-training style), not a function — run it fresh
        # with runpy each call so re-training actually re-executes rather
        # than hitting Python's module-import cache on a second run.
        runpy.run_path(train_script_path, run_name="__main__")

        _pipeline_state.update(status="success", step="done")
    except Exception:
        _pipeline_state.update(status="failed", error=traceback.format_exc())
    finally:
        _pipeline_state["finished_at"] = datetime.now(timezone.utc).isoformat()


def _start_pipeline_if_idle():
    """Returns True if this call started a run, False if one was already in flight."""
    if not _pipeline_lock.acquire(blocking=False):
        return False
    try:
        if _pipeline_state["status"] == "running":
            return False
        # Set before starting the thread, not inside it — otherwise a status
        # check immediately after this call can race the thread and still
        # see the stale previous state.
        _pipeline_state.update(status="running", step="starting", error=None,
                                started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
        thread = threading.Thread(target=_run_pipeline_locked, daemon=True)
        thread.start()
        return True
    finally:
        _pipeline_lock.release()


@app.on_event("startup")
def _maybe_auto_train():
    if os.environ.get("AUTO_TRAIN_ON_START", "").lower() in ("1", "true", "yes"):
        if not os.path.exists(MODEL_PATH):
            _start_pipeline_if_idle()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path) as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": os.path.exists(MODEL_PATH)}


@app.get("/stats")
def get_stats():
    stats = {
        "v1_classifier": {
            "trained": os.path.exists(V1_MODEL_PATH),
            "label_source": "GBIF occurrence records (45 points) + synthetic winter-only negatives",
            "target": "binary outbreak/no-outbreak classification",
        },
        "v2_regressor": {
            "trained": os.path.exists(MODEL_PATH),
            "label_source": "real trap-count data (FTD), shared by industry contact",
            "target": "continuous FTD regression",
        },
        "trap_dataset": None,
        "v2_metrics": None,
        "v2_thresholds": None,
        "region_geocoding": None,
    }

    if os.path.exists(FTD_LONG_PATH):
        df = pd.read_csv(FTD_LONG_PATH)
        stats["trap_dataset"] = {
            "total_records": int(len(df)),
            "regions": sorted(df["region"].unique().tolist()),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "records_per_region": df.groupby("region").size().to_dict(),
        }

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            stats["v2_metrics"] = json.load(f)

    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            stats["v2_thresholds"] = json.load(f)

    if os.path.exists(REGION_COORDS_PATH):
        with open(REGION_COORDS_PATH) as f:
            stats["region_geocoding"] = json.load(f)

    return stats


@app.post("/pipeline/run")
def run_pipeline():
    """Kick off geocode -> fetch weather history -> merge -> train in the
    background and return immediately. Requires
    `Data_Transform/wc_trap_ftd_long.csv` to already be committed in the repo
    (produced by parse_ftd_excel.py, which needs the source .xlsx that isn't
    checked in — that step runs once, locally, not here). Poll
    GET /pipeline/status for progress; check /stats.region_geocoding once it
    finishes to verify each region matched the right real-world place.
    """
    started = _start_pipeline_if_idle()
    return {"request": "started" if started else "already_running", **_pipeline_state}


@app.get("/pipeline/status")
def pipeline_status():
    return _pipeline_state


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


@app.get("/dashboard/regressor.png")
def get_regressor_dashboard():
    if not os.path.exists(V2_REGRESSOR_DASHBOARD_PATH):
        raise HTTPException(status_code=404, detail="v2 not trained yet. POST /pipeline/run first.")
    return FileResponse(V2_REGRESSOR_DASHBOARD_PATH, media_type="image/png")


@app.get("/dashboard/classifier.png")
def get_classifier_dashboard():
    if not os.path.exists(V1_CLASSIFIER_DASHBOARD_PATH):
        raise HTTPException(status_code=404, detail="v1 classifier dashboard not found in repo.")
    return FileResponse(V1_CLASSIFIER_DASHBOARD_PATH, media_type="image/png")
