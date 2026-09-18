"""
Railway-deployed service for the Western Cape Medfly regressor (v2, trained
on Ghian du Toit's real FTD trap-count data instead of GBIF occurrence proxies).

This process has normal outbound internet access (unlike the sandbox this
pipeline was originally written in), so it's the one place the geocoding /
historical weather / live forecast calls to Open-Meteo actually run.

All pipeline logic lives in `pipeline.py`, self-contained within this
directory — see that file's docstring for why (Railway's "root directory"
service setting scopes the build to `railway_app/` only, so this module
cannot reach the sibling `Data_Transform/`/`Training/` folders the CLI
scripts live in).

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
import threading
import traceback
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

import pipeline

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
DATA_DIR = os.path.join(APP_DIR, "data")

app = FastAPI(title="Western Cape Medfly Forecast API", version="2.0.0")

V1_MODEL_PATH = os.path.join(DATA_DIR, "western_cape_medfly_xgboost.json")
V1_CLASSIFIER_DASHBOARD_PATH = os.path.join(DATA_DIR, "medfly_model_dashboard.png")

_pipeline_lock = threading.Lock()
_pipeline_state = {
    "status": "idle",  # idle | running | success | failed
    "step": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _run_pipeline_locked():
    """Runs on a background thread; updates _pipeline_state as it goes so
    /pipeline/status has something meaningful to report while in flight."""
    try:
        pipeline.run_full_pipeline(on_step=lambda name: _pipeline_state.__setitem__("step", name))
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
        if not os.path.exists(pipeline.MODEL_PATH):
            _start_pipeline_if_idle()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path) as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": os.path.exists(pipeline.MODEL_PATH)}


@app.get("/stats")
def get_stats():
    stats = {
        "v1_classifier": {
            "trained": os.path.exists(V1_MODEL_PATH),
            "label_source": "GBIF occurrence records (45 points) + synthetic winter-only negatives",
            "target": "binary outbreak/no-outbreak classification",
        },
        "v2_regressor": {
            "trained": os.path.exists(pipeline.MODEL_PATH),
            "label_source": "real trap-count data (FTD), shared by industry contact",
            "target": "continuous FTD regression",
        },
        "trap_dataset": None,
        "v2_metrics": None,
        "v2_thresholds": None,
        "region_geocoding": None,
    }

    if os.path.exists(pipeline.FTD_LONG_PATH):
        df = pd.read_csv(pipeline.FTD_LONG_PATH)
        stats["trap_dataset"] = {
            "total_records": int(len(df)),
            "regions": sorted(df["region"].unique().tolist()),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "records_per_region": df.groupby("region").size().to_dict(),
        }

    if os.path.exists(pipeline.METRICS_PATH):
        with open(pipeline.METRICS_PATH) as f:
            stats["v2_metrics"] = json.load(f)

    if os.path.exists(pipeline.THRESHOLDS_PATH):
        with open(pipeline.THRESHOLDS_PATH) as f:
            stats["v2_thresholds"] = json.load(f)

    if os.path.exists(pipeline.COORDS_PATH):
        with open(pipeline.COORDS_PATH) as f:
            stats["region_geocoding"] = json.load(f)

    return stats


@app.post("/pipeline/run")
def run_pipeline():
    """Kick off geocode -> fetch weather history -> merge -> train in the
    background and return immediately. Poll GET /pipeline/status for
    progress; check /stats.region_geocoding once it finishes to verify each
    region matched the right real-world place.
    """
    started = _start_pipeline_if_idle()
    return {"request": "started" if started else "already_running", **_pipeline_state}


@app.get("/pipeline/status")
def pipeline_status():
    return _pipeline_state


@app.get("/forecast")
def get_forecast():
    if not os.path.exists(pipeline.MODEL_PATH):
        raise HTTPException(
            status_code=409,
            detail="Model not trained yet. POST /pipeline/run first.",
        )

    try:
        df_forecast, thresholds = pipeline.run_forecast()
        pipeline.save_forecast_outputs(df_forecast, thresholds)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Forecast failed: {exc}") from exc

    return {
        "thresholds": thresholds,
        "forecast": df_forecast.to_dict(orient="records"),
    }


@app.get("/forecast/chart")
def get_forecast_chart():
    if not os.path.exists(pipeline.FORECAST_CHART_PATH):
        raise HTTPException(
            status_code=404,
            detail="No chart yet. Call GET /forecast first to generate one.",
        )
    return FileResponse(pipeline.FORECAST_CHART_PATH, media_type="image/png")


@app.get("/dashboard/regressor.png")
def get_regressor_dashboard():
    if not os.path.exists(pipeline.REGRESSOR_DASHBOARD_PATH):
        raise HTTPException(status_code=404, detail="v2 not trained yet. POST /pipeline/run first.")
    return FileResponse(pipeline.REGRESSOR_DASHBOARD_PATH, media_type="image/png")


@app.get("/dashboard/classifier.png")
def get_classifier_dashboard():
    if not os.path.exists(V1_CLASSIFIER_DASHBOARD_PATH):
        raise HTTPException(status_code=404, detail="v1 classifier dashboard not found in repo.")
    return FileResponse(V1_CLASSIFIER_DASHBOARD_PATH, media_type="image/png")
