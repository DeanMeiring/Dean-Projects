"""
Experiment (e): what drives the predictions? SHAP explanations for two XGBoost models.

  Model A "weather + season"  : weather windows + week of year + region (no trap history)
  Model B "1-week forecast"   : Model A's inputs as of last week + last trap readings

Both are trained on 2017-2026 (the period after the 2014-2017 drop in FTD) on log1p(FTD).
SHAP values are in log-FTD units: positive = pushes the prediction up.

Outputs (Training/experiments/):
  shap_<model>_beeswarm.png, shap_<model>_importance.csv, shap_weather_dependence.png

Run: python shap_analysis.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from horizon_experiments import BASE, DATA_DIR, OUT_DIR, TRAIN_PATH, WEATHER_PATH, WX, build, daily_windows

NICE = {
    "temp_14d_mean": "Temp, 14d mean", "temp_30d_mean": "Temp, 30d mean", "temp_90d_mean": "Temp, 90d mean",
    "humidity_14d_mean": "Humidity, 14d mean", "vpd_14d_max": "VPD, 14d max",
    "soil_temp_14d_mean": "Soil temp, 14d mean", "soil_moisture_14d_mean": "Soil moisture, 14d mean",
    "degree_days_30d_sum": "Degree-days, 30d", "degree_days_90d_sum": "Degree-days, 90d",
    "rain_14d_sum": "Rain, 14d", "rain_60d_sum": "Rain, 60d",
    "week_sin": "Season (week sin)", "week_cos": "Season (week cos)",
    "last_log": "Last trap reading", "prev_log": "Reading before that", "roll4_log": "Avg of last 4 readings",
    "weeks_since_reading": "Weeks since reading",
}


def nice(c):
    if c.startswith("asof_"):
        return NICE[c[5:]] + " (last wk)"
    if c.startswith("region_"):
        return "Region: " + c[7:]
    return NICE.get(c, c)


def explain(df, feats, tag):
    X = pd.get_dummies(df[feats], columns=["region"], dtype=float)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, random_state=42)
    model.fit(X, df["log_ftd"])
    sv = shap.TreeExplainer(model)(X)
    sv.feature_names = [nice(c) for c in X.columns]

    imp = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(os.path.join(OUT_DIR, f"shap_{tag}_importance.csv"))

    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False)
    plt.title({"weather": "Model A — weather + season (no trap history)",
               "forecast": "Model B — 1-week-ahead forecast with trap history"}[tag], fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_{tag}_beeswarm.png"), dpi=200, bbox_inches="tight")
    plt.close()
    return sv, imp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_csv(TRAIN_PATH)
    df = build(raw, daily_windows(pd.read_csv(WEATHER_PATH)), h=1)
    df = df[df.year >= 2017].reset_index(drop=True)

    sv_a, imp_a = explain(df, WX + ["week_sin", "week_cos", "region"], "weather")
    sv_b, imp_b = explain(df, BASE + [f"asof_{c}" for c in WX], "forecast")

    # Dependence plots for the 4 most influential weather features in Model A.
    top_wx = [f for f in imp_a.index if f in {nice(c) for c in WX}][:4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    names = list(sv_a.feature_names)
    for ax, f in zip(axes, top_wx):
        i = names.index(f)
        ax.scatter(sv_a.data[:, i], sv_a.values[:, i], s=8, alpha=0.35, color="#2a78d6", edgecolors="none")
        ax.axhline(0, color="#6b6a65", lw=1)
        ax.set_xlabel(f)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SHAP value (effect on log FTD)")
    fig.suptitle("How the top weather features push predicted FTD up or down (Model A)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_weather_dependence.png"), dpi=200)
    plt.close()

    print("Model A (weather + season) — mean |SHAP|:\n", imp_a.round(4).to_string())
    print("\nModel B (1-week forecast) — mean |SHAP|:\n", imp_b.round(4).head(12).to_string())
    share = imp_b[[i for i in imp_b.index if i in ("Last trap reading", "Reading before that", "Avg of last 4 readings")]].sum() / imp_b.sum()
    print(f"\nModel B: trap history accounts for {share:.0%} of total |SHAP|")


if __name__ == "__main__":
    main()
