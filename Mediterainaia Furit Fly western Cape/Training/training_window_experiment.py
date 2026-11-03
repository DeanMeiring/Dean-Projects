"""
Does training on only recent years help? (Supervisor question, 3 Nov 2026.)

Same walk-forward test years (2020-2026) and horizons (1, 2, 4 weeks) as walkforward_experiments.py,
but the training window changes:
  - All years      : 2010 .. Y   (includes the high-catch years before the 2014-2017 drop)
  - From 2017      : 2017 .. Y   (the main study setting)
  - Last 5 years   : Y-4 .. Y
  - Last 3 years   : Y-2 .. Y

The outbreak cut-off is always taken from the 2017..Y training years, so every window is scored
against the same outbreak labels and the F1 scores are directly comparable.

Outputs: experiments/training_window_folds.csv, experiments/training_window_summary.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from model_experiments import preprocessor
from walkforward_experiments import (HORIZONS, OUT_DIR, OUTBREAK_PCTL, REGRESSORS, TEST_YEARS, classifiers,
                                     clf_scores, load, reg_scores)

warnings.filterwarnings("ignore")

WINDOWS = {
    "All years (2010+)": lambda y: (2010, y - 1),
    "From 2017 (main)": lambda y: (2017, y - 1),
    "Last 5 years": lambda y: (y - 5, y - 1),
    "Last 3 years": lambda y: (y - 3, y - 1),
}
MODELS = ["Random Forest", "XGBoost"]


def main():
    rows = []
    for h in HORIZONS:
        df, feats = load(h)
        for test_year in TEST_YEARS:
            test = df[df.year == test_year]
            if test.empty:
                continue
            ref = df[df.year.between(2017, test_year - 1)]
            cut = ref.ftd.quantile(OUTBREAK_PCTL)
            y_te = (test.ftd >= cut).astype(int).values
            for wname, span in WINDOWS.items():
                lo, hi = span(test_year)
                train = df[df.year.between(lo, hi)]
                y_tr = (train.ftd >= cut).astype(int).values
                pos_w = (y_tr == 0).sum() / max(1, y_tr.sum())
                base = {"horizon_weeks": h, "test_year": test_year, "window": wname,
                        "train_from": int(train.year.min()), "n_train": len(train)}
                for name in MODELS:
                    make, scale = REGRESSORS[name]
                    p = make_pipeline(preprocessor(feats, scale), make()).fit(train[feats], train.log_ftd)
                    rows.append({**base, "task": "regression", "model": name,
                                 **reg_scores(test.log_ftd.values, p.predict(test[feats]))})
                    make, scale = classifiers(pos_w)[name]
                    p = make_pipeline(preprocessor(feats, scale), make()).fit(train[feats], y_tr)
                    rows.append({**base, "task": "classification", "model": name,
                                 **clf_scores(y_te, p.predict(test[feats]))})
            print(f"h={h} test {test_year} done", flush=True)

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(OUT_DIR, "training_window_folds.csv"), index=False)
    metrics = ["R2_log", "MAE", "Precision", "Recall", "F1"]
    summ = (folds.groupby(["task", "horizon_weeks", "model", "window"])[[m for m in metrics if m in folds]]
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.dropna(axis=1, how="all").reset_index()
    summ.to_csv(os.path.join(OUT_DIR, "training_window_summary.csv"), index=False)

    order = list(WINDOWS)
    print("\n=== R2 (log), mean over 7 test years ===")
    r = summ[summ.task == "regression"].pivot_table(index=["model", "window"], columns="horizon_weeks", values="R2_log_mean")
    print(r.reindex(order, level=1).round(3))
    print("\n=== Outbreak F1, mean over 7 test years ===")
    c = summ[summ.task == "classification"].pivot_table(index=["model", "window"], columns="horizon_weeks", values="F1_mean")
    print(c.reindex(order, level=1).round(3))


if __name__ == "__main__":
    main()
