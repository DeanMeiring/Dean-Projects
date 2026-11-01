"""
Model comparison experiments for the Western Cape medfly FTD regressor (v2).

Compares a range of ML models against the current XGBoost regressor, across
several feature sets and two ways of splitting train/test:

  - random : 75/25 random split (same as train_xgboost_regressor.py)
  - time   : train on 2010-2021, test on 2022-2026 (an honest "future" test,
             since neighbouring weeks are near-duplicates under a random split)

Every model predicts log1p(FTD) (the target is heavily skewed) except the
"raw" reference models, then predictions are converted back to FTD for scoring.

Run (after merge_ftd_weather.py has produced the training CSV):
    python model_experiments.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data_Transform")
TRAIN_PATH = os.path.join(DATA_DIR, "wc_v2_training_regression.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "wc_v2_weather_history.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "experiments")

WEATHER = [
    "temp_14d_mean", "temp_30d_mean", "humidity_14d_mean",
    "vpd_14d_max", "soil_temp_14d_mean", "soil_moisture_14d_mean",
    "degree_days_30d_sum", "rain_14d_sum",
]
LONG_WEATHER = ["temp_90d_mean", "degree_days_60d_sum", "degree_days_90d_sum",
                "rain_60d_sum", "humidity_30d_mean"]
SEASON = ["week_sin", "week_cos"]
LAGS = ["lag1_log", "lag2_log", "roll4_log"]

FEATURE_SETS = {
    "weather":               WEATHER,
    "weather+season+region": WEATHER + SEASON + ["region"],
    "+long_windows":         WEATHER + LONG_WEATHER + SEASON + ["region"],
    "+last_weeks_ftd":       WEATHER + LONG_WEATHER + SEASON + ["region"] + LAGS,
}


# ---------------------------------------------------------------- features
def add_features(df, weather):
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    weather = weather.copy()
    weather["time"] = pd.to_datetime(weather["time"])
    weather["dd"] = np.maximum(0, weather["temperature_2m_mean"] - 10.0)

    # Longer windows: medfly populations build up over a season, so 60-90
    # days of accumulated heat may matter more than the last 2-4 weeks.
    rows = []
    for region, w in weather.groupby("region"):
        w = w.set_index("time").sort_index()
        roll = pd.DataFrame({
            "temp_90d_mean": w["temperature_2m_mean"].rolling("90D").mean(),
            "degree_days_60d_sum": w["dd"].rolling("60D").sum(),
            "degree_days_90d_sum": w["dd"].rolling("90D").sum(),
            "rain_60d_sum": w["rain_sum"].rolling("60D").sum(),
            "humidity_30d_mean": w["relative_humidity_2m_mean"].rolling("30D").mean(),
        })
        roll["region"] = region
        rows.append(roll.reset_index().rename(columns={"time": "event_date"}))
    df = df.merge(pd.concat(rows), on=["region", "event_date"], how="left")

    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # Previous trap readings (only valid if the reading is recent, <=3 weeks back).
    df = df.sort_values(["region", "event_date"]).reset_index(drop=True)
    df["log_ftd"] = np.log1p(df["ftd"])
    g = df.groupby("region")
    for k in (1, 2):
        lag = g["log_ftd"].shift(k)
        gap = (df["event_date"] - g["event_date"].shift(k)).dt.days
        df[f"lag{k}_log"] = lag.where(gap <= 7 * (k + 2))
    df["roll4_log"] = g["log_ftd"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    return df


# ---------------------------------------------------------------- models
def preprocessor(features, scale):
    num = [f for f in features if f != "region"]
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), *( [StandardScaler()] if scale else [] ))
    parts = [("num", num_pipe, num)]
    if "region" in features:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]))
    return ColumnTransformer(parts)


XGB_CURRENT = dict(n_estimators=150, max_depth=4, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)

MODELS = {
    # name: (estimator factory, needs scaling, target space)
    "XGBoost (current, raw FTD)":  (lambda: xgb.XGBRegressor(**XGB_CURRENT), False, "raw"),
    "XGBoost Tweedie (raw FTD)":   (lambda: xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3,
                                                             n_estimators=400, max_depth=4, learning_rate=0.03,
                                                             subsample=0.8, colsample_bytree=0.8, random_state=42), False, "raw"),
    "XGBoost (log FTD)":           (lambda: xgb.XGBRegressor(**XGB_CURRENT), False, "log"),
    "XGBoost tuned (log FTD)":     (lambda: xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.02,
                                                             subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                                                             reg_lambda=2.0, random_state=42), False, "log"),
    "LightGBM":                    (lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                                                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                              colsample_bytree=0.8, random_state=42, verbose=-1), False, "log"),
    "HistGradientBoosting":        (lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04,
                                                                          random_state=42), False, "log"),
    "Random Forest":               (lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                                  n_jobs=-1, random_state=42), False, "log"),
    "Extra Trees":                 (lambda: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3,
                                                                n_jobs=-1, random_state=42), False, "log"),
    "Ridge regression":            (lambda: Ridge(alpha=1.0), True, "log"),
    "k-Nearest Neighbours":        (lambda: KNeighborsRegressor(n_neighbors=15, weights="distance"), True, "log"),
    "SVR (RBF kernel)":            (lambda: SVR(C=1.0, epsilon=0.05), True, "log"),
    "Neural net (MLP)":            (lambda: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000,
                                                         early_stopping=True, random_state=42), True, "log"),
}


def seasonal_baseline(train, test):
    """No ML: predict the average (log) FTD seen for that region in that week of the year."""
    tab = train.groupby(["region", "week"])["log_ftd"].mean()
    reg = train.groupby("region")["log_ftd"].mean()
    keys = list(zip(test["region"], test["week"]))
    pred = np.array([tab.get(k, reg.get(k[0], train["log_ftd"].mean())) for k in keys])
    return np.expm1(pred)


# ---------------------------------------------------------------- scoring
def score(y_true, y_pred, cutoffs):
    y_pred = np.clip(y_pred, 0, None)
    band = lambda v: np.digitize(v, cutoffs)  # 0=LOW, 1=MODERATE, 2=HIGH
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "R2_log": r2_score(np.log1p(y_true), np.log1p(y_pred)),
        "Spearman": spearmanr(y_true, y_pred).correlation,
        "RiskBand_Acc": float(np.mean(band(y_true) == band(y_pred))),
    }


def splits(df):
    tr, te = train_test_split(df.index, test_size=0.25, random_state=42)
    yield "random", df.loc[tr], df.loc[te]
    yield "time", df[df["year"] <= 2021], df[df["year"] >= 2022]
    # FTD in Tulbagh/Wolseley collapsed ~2014-2017 (likely a control programme,
    # not weather). Training only on the post-drop era tests whether weather
    # predicts anything once that regime change is out of the picture.
    yield "time_recent", df[df["year"].between(2017, 2021)], df[df["year"] >= 2022]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = add_features(pd.read_csv(TRAIN_PATH), pd.read_csv(WEATHER_PATH))

    results = []
    for split_name, train, test in splits(df):
        cutoffs = train["ftd"].quantile([0.35, 0.65]).values
        y_true = test["ftd"].values
        results.append({"split": split_name, "features": "(none)", "model": "Seasonal average (no ML)",
                        **score(y_true, seasonal_baseline(train, test), cutoffs)})

        for fs_name, feats in FEATURE_SETS.items():
            for name, (make, scale, space) in MODELS.items():
                pipe = make_pipeline(preprocessor(feats, scale), make())
                y_fit = train["ftd"] if space == "raw" else train["log_ftd"]
                pipe.fit(train[feats], y_fit)
                pred = pipe.predict(test[feats])
                pred = pred if space == "raw" else np.expm1(pred)
                results.append({"split": split_name, "features": fs_name, "model": name,
                                **score(y_true, pred, cutoffs)})
                print(f"{split_name:6s} | {fs_name:22s} | {name:28s} | R2_log={results[-1]['R2_log']:.3f}", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    for split_name in ("random", "time", "time_recent"):
        print(f"\n===== {split_name.upper()} SPLIT — top 15 by R2 (log scale) =====")
        print(res[res.split == split_name].sort_values("R2_log", ascending=False)
              .head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
