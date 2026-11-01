"""
Experiment (a): how far ahead can we forecast medfly FTD?

For a horizon of h weeks, every feature is computed "as of" h weeks before the
target week — the last trap reading and the weather up to that day. Only the
calendar (week of year) and region are known for the target week itself.

Feature variants compared at each horizon:
  - traps only          : recent trap readings + season + region (no weather)
  - traps + weather     : the above + weather windows ending on the forecast day
  - traps + future wx   : weather windows ending on the TARGET day instead —
                          an upper bound, as if the weather forecast were perfect

Baselines: persistence (repeat the last trap reading) and the seasonal average.

Run: python horizon_experiments.py
"""
import os
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from model_experiments import preprocessor, score, seasonal_baseline

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data_Transform")
TRAIN_PATH = os.path.join(DATA_DIR, "wc_v2_training_regression.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "wc_v2_weather_history.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "experiments")

HORIZONS = [1, 2, 4, 8]
WX = ["temp_14d_mean", "temp_30d_mean", "temp_90d_mean", "humidity_14d_mean",
      "vpd_14d_max", "soil_temp_14d_mean", "soil_moisture_14d_mean",
      "degree_days_30d_sum", "degree_days_90d_sum", "rain_14d_sum", "rain_60d_sum"]
TRAPS = ["last_log", "prev_log", "roll4_log", "weeks_since_reading"]
BASE = TRAPS + ["week_sin", "week_cos", "region"]

VARIANTS = {
    "traps only": BASE,
    "traps + weather": BASE + [f"asof_{c}" for c in WX],
    "traps + future wx": BASE + WX,
}

MODELS = {
    "Ridge regression": (lambda: Ridge(alpha=1.0), True),
    "Random Forest": (lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42), False),
    "XGBoost (log FTD)": (lambda: xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8,
                                                   colsample_bytree=0.8, random_state=42), False),
    "LightGBM": (lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1), False),
}


def daily_windows(weather):
    """Rolling weather windows for every day, per region (window ends on that day)."""
    w = weather.copy()
    w["time"] = pd.to_datetime(w["time"])
    w["dd"] = np.maximum(0, w["temperature_2m_mean"] - 10.0)
    out = []
    for region, g in w.groupby("region"):
        g = g.set_index("time").sort_index()
        t, h, r = g["temperature_2m_mean"], g["relative_humidity_2m_mean"], g["rain_sum"]
        out.append(pd.DataFrame({
            "region": region,
            "temp_14d_mean": t.rolling("14D").mean(), "temp_30d_mean": t.rolling("30D").mean(),
            "temp_90d_mean": t.rolling("90D").mean(), "humidity_14d_mean": h.rolling("14D").mean(),
            "vpd_14d_max": g["vapor_pressure_deficit_max"].rolling("14D").max(),
            "soil_temp_14d_mean": g["soil_temperature_0_to_7cm_mean"].rolling("14D").mean(),
            "soil_moisture_14d_mean": g["soil_moisture_0_to_7cm_mean"].rolling("14D").mean(),
            "degree_days_30d_sum": g["dd"].rolling("30D").sum(), "degree_days_90d_sum": g["dd"].rolling("90D").sum(),
            "rain_14d_sum": r.rolling("14D").sum(), "rain_60d_sum": r.rolling("60D").sum(),
        }).reset_index().rename(columns={"time": "date"}))
    return pd.concat(out)


def build(df, wx_daily, h):
    """Features for forecasting each row's FTD h weeks in advance."""
    df = df[["region", "year", "week", "event_date", "ftd"]].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["log_ftd"] = np.log1p(df["ftd"])
    df["forecast_date"] = df["event_date"] - pd.Timedelta(weeks=h)
    df = df.sort_values(["region", "event_date"]).reset_index(drop=True)

    # Most recent trap reading on or before the forecast date (merge_asof, per region).
    hist = df[["region", "event_date", "log_ftd"]].rename(columns={"event_date": "read_date", "log_ftd": "last_log"})
    g = hist.groupby("region")["last_log"]
    hist["prev_log"] = g.shift(1)
    hist["roll4_log"] = g.transform(lambda s: s.rolling(4, min_periods=2).mean())
    df = pd.merge_asof(df.sort_values("forecast_date"), hist.sort_values("read_date"),
                       left_on="forecast_date", right_on="read_date", by="region", direction="backward")
    df["weeks_since_reading"] = (df["event_date"] - df["read_date"]).dt.days / 7
    stale = df["weeks_since_reading"] > h + 3  # gap in the trap record, no recent reading
    df.loc[stale, ["last_log", "prev_log", "roll4_log"]] = np.nan

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    target_wx = wx_daily.rename(columns={"date": "event_date"})
    asof_wx = wx_daily.rename(columns={"date": "forecast_date", **{c: f"asof_{c}" for c in WX}})
    df = df.merge(target_wx, on=["region", "event_date"], how="left")
    df = df.merge(asof_wx, on=["region", "forecast_date"], how="left")
    return df.dropna(subset=["last_log"]).reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_csv(TRAIN_PATH)
    wx_daily = daily_windows(pd.read_csv(WEATHER_PATH))

    results = []
    for h in HORIZONS:
        df = build(raw, wx_daily, h)
        for split, train in (("train 2017-21", df[df.year.between(2017, 2021)]),
                             ("train 2010-21", df[df.year <= 2021])):
            test = df[df.year >= 2022]
            cutoffs = train["ftd"].quantile([0.35, 0.65]).values
            y = test["ftd"].values
            base = {"horizon_weeks": h, "split": split, "n_test": len(test)}
            results.append({**base, "variant": "-", "model": "Persistence (repeat last reading)",
                            **score(y, np.expm1(test["last_log"].values), cutoffs)})
            results.append({**base, "variant": "-", "model": "Seasonal average (no ML)",
                            **score(y, seasonal_baseline(train, test), cutoffs)})
            for vname, feats in VARIANTS.items():
                for mname, (make, scale) in MODELS.items():
                    pipe = make_pipeline(preprocessor(feats, scale), make())
                    pipe.fit(train[feats], train["log_ftd"])
                    pred = np.expm1(pipe.predict(test[feats]))
                    results.append({**base, "variant": vname, "model": mname, **score(y, pred, cutoffs)})
            print(f"h={h} {split} done", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, "horizon_comparison.csv"), index=False)
    plot(res[res.split == "train 2017-21"])

    pd.set_option("display.width", 220)
    r = res[res.split == "train 2017-21"]
    print("\nR2 (log scale), train 2017-21 -> test 2022-26")
    r = r.assign(row=np.where(r.variant == "-", r.model, r.model + " | " + r.variant))
    print(r.pivot_table(index="row", columns="horizon_weeks", values="R2_log").round(3).sort_values(1, ascending=False))
    print("\nRisk-band accuracy")
    print(r.pivot_table(index="row", columns="horizon_weeks", values="RiskBand_Acc").round(3).sort_values(1, ascending=False))


def plot(r):
    """R2 (log) and risk-band accuracy vs horizon: best ML model per variant + baselines."""
    colors = {"traps only": "#2a78d6", "traps + weather": "#eb6834", "traps + future wx": "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in ((axes[0], "R2_log", "R² (log FTD)"), (axes[1], "RiskBand_Acc", "Risk-level accuracy")):
        for v, c in colors.items():
            best = r[r.variant == v].groupby("horizon_weeks")[metric].max()
            ax.plot(best.index, best.values, color=c, lw=2, marker="o", ms=8, label=f"Best ML: {v}")
        for m, ls in (("Persistence (repeat last reading)", "--"), ("Seasonal average (no ML)", ":")):
            s = r[r.model == m].set_index("horizon_weeks")[metric]
            ax.plot(s.index, s.values, color="#6b6a65", lw=2, ls=ls, marker="o", ms=6, label=m)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon (weeks ahead)")
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", color="#e5e4e0")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("How far ahead can medfly trap counts be forecast?  (train 2017–21, test 2022–26)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_comparison.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
