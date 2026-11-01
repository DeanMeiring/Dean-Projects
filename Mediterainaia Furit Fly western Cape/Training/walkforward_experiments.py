"""
Walk-forward validation: trap-count regression + outbreak-week classification.

Folds  : train on 2017..Y, test on year Y+1, for Y = 2019..2025 (7 test years, 2020-2026).
         Training starts in 2017, after the 2014-2017 drop in FTD.
Horizon: forecast 1, 2 and 4 weeks ahead. Every input is as of the forecast day
         (latest trap readings, weather and v3 features up to that day).

Outbreak week: FTD at or above the 75th percentile of FTD in that fold's training years.
This is a data-driven cut-off, not an industry action threshold.

Models
  regression     : Random Forest, XGBoost, LightGBM, Ridge  (target log1p FTD)
  classification : Random Forest, XGBoost, Logistic regression (balanced class weights)
  baselines      : persistence (repeat last reading), seasonal average for region+week,
                   degree-day model (Ridge / threshold rule on 90-day degree-days only)

Statistics: Wilcoxon signed-rank test, Random Forest vs XGBoost, paired over
fold x region scores (and over folds alone).

Outputs (experiments/): walkforward_folds.csv, walkforward_summary.csv,
                        walkforward_wilcoxon.csv, walkforward_f1.png
Run: python walkforward_experiments.py
"""
import os
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, r2_score, recall_score
from sklearn.pipeline import make_pipeline

from horizon_experiments import BASE, OUT_DIR, TRAIN_PATH, WEATHER_PATH, WX, build, daily_windows
from model_experiments import preprocessor

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "wc_v3_daily_features.csv")

HORIZONS = [1, 2, 4]
TEST_YEARS = range(2020, 2027)
OUTBREAK_PCTL = 0.75
SEED = 42

REGRESSORS = {
    "Random Forest": (lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED), False),
    "XGBoost": (lambda: xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8,
                                         colsample_bytree=0.8, random_state=SEED), False),
    "LightGBM": (lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1), False),
    "Ridge regression": (lambda: Ridge(alpha=1.0), True),
}


def classifiers(pos_weight):
    return {
        "Random Forest": (lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                                                         n_jobs=-1, random_state=SEED), False),
        "XGBoost": (lambda: xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8,
                                              colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                              eval_metric="logloss", random_state=SEED), False),
        "Logistic regression": (lambda: LogisticRegression(class_weight="balanced", max_iter=2000), True),
    }


def load(h):
    raw = pd.read_csv(TRAIN_PATH)
    df = build(raw, daily_windows(pd.read_csv(WEATHER_PATH)), h)
    v3 = pd.read_csv(V3_PATH, parse_dates=["date"])
    v3_cols = [c for c in v3.columns if c not in ("region", "date")]
    v3 = v3.rename(columns={"date": "forecast_date", **{c: f"asof_{c}" for c in v3_cols}})
    df = df.merge(v3, on=["region", "forecast_date"], how="left")
    feats = BASE + [f"asof_{c}" for c in WX] + [f"asof_{c}" for c in v3_cols]
    return df, feats


def seasonal(train, test, col):
    tab = train.groupby(["region", "week"])[col].mean()
    reg = train.groupby("region")[col].mean()
    return np.array([tab.get((r, w), reg.get(r, train[col].mean())) for r, w in zip(test.region, test.week)])


def reg_scores(y_log, pred_log):
    return {"R2_log": r2_score(y_log, pred_log), "MAE": mean_absolute_error(np.expm1(y_log), np.expm1(pred_log))}


def clf_scores(y, pred):
    return {"Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0)}


def per_region(rows, base, test, y, pred, kind):
    """Score the whole fold, and each region separately (used for the paired tests)."""
    fn = reg_scores if kind == "regression" else clf_scores
    rows.append({**base, "region": "ALL", **fn(y, pred)})
    for region in test.region.unique():
        m = (test.region == region).values
        if kind == "classification" and y[m].sum() == 0:
            continue  # no outbreak weeks for this region-year: F1 undefined
        if kind == "regression" and m.sum() < 5:
            continue
        rows.append({**base, "region": region, **fn(y[m], pred[m])})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for h in HORIZONS:
        df, feats = load(h)
        for test_year in TEST_YEARS:
            train = df[df.year.between(2017, test_year - 1)]
            test = df[df.year == test_year]
            if test.empty:
                continue
            cut = train.ftd.quantile(OUTBREAK_PCTL)
            y_tr, y_te = (train.ftd >= cut).astype(int).values, (test.ftd >= cut).astype(int).values
            ylog_te = test.log_ftd.values
            base = {"horizon_weeks": h, "test_year": test_year, "n_test": len(test),
                    "outbreak_cutoff_ftd": round(cut, 3), "outbreak_share_test": y_te.mean()}

            # ---- regression
            R = lambda m: {**base, "task": "regression", "model": m}
            per_region(rows, R("Persistence (baseline)"), test, ylog_te, test.last_log.values, "regression")
            per_region(rows, R("Seasonal average (baseline)"), test, ylog_te, seasonal(train, test, "log_ftd"), "regression")
            dd = ["asof_gdd96_90d_sum", "week_sin", "week_cos", "region"]
            p = make_pipeline(preprocessor(dd, True), Ridge(alpha=1.0)).fit(train[dd], train.log_ftd)
            per_region(rows, R("Degree-day model (baseline)"), test, ylog_te, p.predict(test[dd]), "regression")
            for name, (make, scale) in REGRESSORS.items():
                p = make_pipeline(preprocessor(feats, scale), make()).fit(train[feats], train.log_ftd)
                per_region(rows, R(name), test, ylog_te, p.predict(test[feats]), "regression")

            # ---- classification
            C = lambda m: {**base, "task": "classification", "model": m}
            per_region(rows, C("Persistence (baseline)"), test, y_te, (np.expm1(test.last_log.values) >= cut).astype(int), "classification")
            train_flag = train.assign(flag=y_tr)
            per_region(rows, C("Seasonal average (baseline)"), test, y_te, (seasonal(train_flag, test, "flag") >= 0.5).astype(int), "classification")
            # Degree-day threshold rule: outbreak if 90-day degree-days >= a cut-off chosen to maximise F1 on training data.
            gdd_tr = train["asof_gdd96_90d_sum"].values
            cands = np.nanpercentile(gdd_tr, np.arange(5, 96, 1))
            best = max(cands, key=lambda t: f1_score(y_tr, (gdd_tr >= t).astype(int), zero_division=0))
            per_region(rows, C("Degree-day threshold (baseline)"), test, y_te,
                       (test["asof_gdd96_90d_sum"].values >= best).astype(int), "classification")
            pos_w = (y_tr == 0).sum() / max(1, y_tr.sum())
            for name, (make, scale) in classifiers(pos_w).items():
                p = make_pipeline(preprocessor(feats, scale), make()).fit(train[feats], y_tr)
                per_region(rows, C(name), test, y_te, p.predict(test[feats]), "classification")
            print(f"h={h} test {test_year} done", flush=True)

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(OUT_DIR, "walkforward_folds.csv"), index=False)

    allr = folds[folds.region == "ALL"]
    summ = []
    for (task, h, m), g in allr.groupby(["task", "horizon_weeks", "model"]):
        metrics = ["R2_log", "MAE"] if task == "regression" else ["Precision", "Recall", "F1"]
        summ.append({"task": task, "horizon_weeks": h, "model": m,
                     **{f"{k}_mean": g[k].mean() for k in metrics}, **{f"{k}_sd": g[k].std() for k in metrics}})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT_DIR, "walkforward_summary.csv"), index=False)

    # ---- Wilcoxon: Random Forest vs XGBoost
    tests = []
    for task, metric in (("regression", "R2_log"), ("classification", "F1")):
        for h in HORIZONS:
            for level, sub in (("fold x region", folds[folds.region != "ALL"]), ("fold", allr)):
                s = sub[(sub.task == task) & (sub.horizon_weeks == h)]
                piv = s.pivot_table(index=["test_year", "region"], columns="model", values=metric).dropna(
                    subset=["Random Forest", "XGBoost"])
                diff = piv["Random Forest"] - piv["XGBoost"]
                stat, pval = wilcoxon(piv["Random Forest"], piv["XGBoost"]) if (diff != 0).any() else (np.nan, 1.0)
                tests.append({"task": task, "metric": metric, "horizon_weeks": h, "paired_over": level,
                              "n_pairs": len(piv), "RF_median": piv["Random Forest"].median(),
                              "XGB_median": piv["XGBoost"].median(), "median_diff_RF_minus_XGB": diff.median(),
                              "W": stat, "p_value": pval})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT_DIR, "walkforward_wilcoxon.csv"), index=False)

    plot(summ)
    pd.set_option("display.width", 220)
    print("\n=== Regression (mean over 7 test years) ===")
    r = summ[summ.task == "regression"].pivot_table(index="model", columns="horizon_weeks", values="R2_log_mean")
    print(r.round(3).sort_values(1, ascending=False))
    print("\n=== Outbreak classification: F1 (mean over 7 test years) ===")
    c = summ[summ.task == "classification"]
    print(c.pivot_table(index="model", columns="horizon_weeks", values="F1_mean").round(3).sort_values(1, ascending=False))
    print("\nPrecision / Recall at 2 weeks:")
    print(c[c.horizon_weeks == 2][["model", "Precision_mean", "Recall_mean", "F1_mean", "F1_sd"]].round(3).to_string(index=False))
    print("\n=== Wilcoxon RF vs XGBoost ===")
    print(tests.round(4).to_string(index=False))


def plot(summ):
    c = summ[summ.task == "classification"]
    order = ["Random Forest", "XGBoost", "Logistic regression", "Persistence (baseline)",
             "Seasonal average (baseline)", "Degree-day threshold (baseline)"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6b6a65", "#a5a49f", "#cfcdc6"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.13
    for i, (m, col) in enumerate(zip(order, colors)):
        g = c[c.model == m].set_index("horizon_weeks").reindex(HORIZONS)
        xs = np.arange(len(HORIZONS)) + (i - 2.5) * width
        ax.bar(xs, g.F1_mean, width * 0.9, yerr=g.F1_sd, color=col, label=m, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": "#6b6a65"})
    ax.set_xticks(np.arange(len(HORIZONS)), [f"{h} week{'s' if h > 1 else ''} ahead" for h in HORIZONS])
    ax.set_ylabel("F1-score (mean ± SD over 7 test years)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e5e4e0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title("Outbreak-week detection, walk-forward 2020–2026", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "walkforward_f1.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
