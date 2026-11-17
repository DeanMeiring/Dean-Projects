"""
Extra report items (2-week horizon, walk-forward test years 2020-2026, same settings as walkforward_experiments.py):
  Figure_4.9_forecast_vs_actual.png  - Random Forest forecast vs actual FTD for two districts
  Figure_4.10_confusion_matrix.png   - Random Forest outbreak warnings: hits, misses, false alarms
  Table_B1_model_settings.csv        - settings of every model (Appendix B)
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.pipeline import make_pipeline

from model_experiments import preprocessor
from walkforward_experiments import OUTBREAK_PCTL, REGRESSORS, TEST_YEARS, classifiers, load

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figures")
plt.rcParams["font.family"] = "Times New Roman"
INK, BLUE, ORANGE, GREY = "#1f2937", "#2b6cb0", "#dd6b20", "#9ca3af"
H = 2

df, feats = load(H)
parts = []
for ty in TEST_YEARS:
    train = df[df.year.between(2017, ty - 1)]
    test = df[df.year == ty].copy()
    if test.empty:
        continue
    cut = train.ftd.quantile(OUTBREAK_PCTL)
    y_tr = (train.ftd >= cut).astype(int).values
    make, scale = REGRESSORS["Random Forest"]
    reg = make_pipeline(preprocessor(feats, scale), make()).fit(train[feats], train.log_ftd)
    pos_w = (y_tr == 0).sum() / max(1, y_tr.sum())
    cmake, cscale = classifiers(pos_w)["Random Forest"]
    clf = make_pipeline(preprocessor(feats, cscale), cmake()).fit(train[feats], y_tr)
    test["pred_ftd"] = np.expm1(reg.predict(test[feats]))
    test["actual_flag"] = (test.ftd >= cut).astype(int)
    test["pred_flag"] = clf.predict(test[feats])
    test["cutoff"] = cut
    parts.append(test)
    print("fold", ty, "done", flush=True)
res = pd.concat(parts)
date_col = "event_date" if "event_date" in res.columns else "forecast_date"
res[date_col] = pd.to_datetime(res[date_col])

# ---------- Figure 4.9: forecast vs actual ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
for ax, region in zip(axes, ["Wolseley", "Elgin & Grabouw"]):
    g = res[res.region.str.contains(region.split(" ")[0])].sort_values(date_col).reset_index(drop=True)
    # break the lines where readings stop for more than 5 weeks (winter gaps), instead of drawing across them
    gap = g[date_col].diff().dt.days > 35
    g.loc[gap, ["ftd", "pred_ftd"]] = np.nan
    ax.plot(g[date_col], g.ftd, color=GREY, lw=1.6, label="Actual trap catch")
    ax.plot(g[date_col], g.pred_ftd, color=BLUE, lw=1.6, label="Random Forest forecast (2 weeks ahead)")
    ax.set_title(region, loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel("FTD (flies/trap/day)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure_4.9_forecast_vs_actual.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Figure 4.10: confusion matrix ----------
cm = confusion_matrix(res.actual_flag, res.pred_flag, labels=[1, 0])
fig, ax = plt.subplots(figsize=(6.4, 5.2))
names = [["Hit\n(outbreak warned)", "Miss\n(outbreak not warned)"],
         ["False alarm\n(warned, no outbreak)", "Correct quiet week"]]
colours = [["#c6f6d5", "#fed7d7"], ["#feebc8", "#edf2f7"]]
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fc=colours[i][j], ec="white", lw=3))
        ax.text(j + 0.5, 1.62 - i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
        ax.text(j + 0.5, 1.3 - i, names[i][j], ha="center", va="center", fontsize=11, color=INK)
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5], ["Warning given", "No warning"], fontsize=12)
ax.set_yticks([1.5, 0.5], ["Outbreak week", "Normal week"], fontsize=12)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure_4.10_confusion_matrix.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
rec, prec = recall_score(res.actual_flag, res.pred_flag), precision_score(res.actual_flag, res.pred_flag)
print("confusion [[TP FN][FP TN]]:", cm.tolist(), f"recall={rec:.3f} precision={prec:.3f}")

# ---------- Table B1: model settings ----------
rows = [
    ("Random Forest", "Regression and classification (rounds 1 and 2)", "400 trees; at least 3 weeks per leaf; balanced class weights for classification"),
    ("Extra Trees", "Regression (round 1)", "400 trees; at least 3 weeks per leaf"),
    ("XGBoost (raw and log FTD)", "Regression (rounds 1 and 2) and classification", "150 trees; maximum depth 4; learning rate 0.03; 80% of rows and features per tree; outbreak weeks up-weighted for classification"),
    ("XGBoost tuned", "Regression (round 1)", "600 trees; maximum depth 5; learning rate 0.02; 70% of features per tree; minimum child weight 5; L2 penalty 2"),
    ("XGBoost Tweedie", "Regression (round 1)", "Tweedie loss (power 1.3) on raw FTD; 400 trees; maximum depth 4; learning rate 0.03"),
    ("LightGBM", "Regression (rounds 1 and 2)", "500 trees; 15 leaves per tree; learning rate 0.03; 80% of rows and features per tree"),
    ("HistGradientBoosting", "Regression (round 1)", "400 iterations; learning rate 0.04"),
    ("Ridge regression", "Regression (rounds 1 and 2)", "Penalty alpha = 1; inputs standardised"),
    ("Logistic regression", "Classification (round 2)", "Balanced class weights; inputs standardised"),
    ("k-Nearest Neighbours", "Regression (round 1)", "15 neighbours, weighted by distance; inputs standardised"),
    ("Support Vector Regression", "Regression (round 1)", "RBF kernel; C = 1; epsilon = 0.05; inputs standardised"),
    ("Neural network (MLP)", "Regression (round 1)", "Two hidden layers (64 and 32 units); L2 penalty 0.001; early stopping; inputs standardised"),
    ("All models", "-", "Random seed 42; missing values filled with the training median; district one-hot encoded; target log(1 + FTD) unless stated"),
]
pd.DataFrame(rows, columns=["Model", "Used for", "Settings"]).to_csv(os.path.join(OUT, "Table_B1_model_settings.csv"), index=False)
print("saved")
