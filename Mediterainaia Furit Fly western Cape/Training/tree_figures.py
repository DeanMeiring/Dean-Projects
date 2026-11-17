"""
Report figures that show how the tree models work:
  Figure_3.2_example_tree.png   - one shallow decision tree fitted to the real data (2-week horizon, 2017-2025)
  Figure_3.3_rf_vs_xgboost.png  - how Random Forest (bagging) and XGBoost (boosting) combine many trees

The example tree is for illustration only: the study's Random Forest averages 400 deeper trees.
"""
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.tree import DecisionTreeRegressor, plot_tree

from make_report_figures import label
from walkforward_experiments import load

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figures")
plt.rcParams["font.family"] = "Times New Roman"
INK, BLUE, PALE, ORANGE = "#1f2937", "#2b6cb0", "#e8f0fa", "#f6ad55"

# ---------- Figure 3.2: example tree ----------
df, feats = load(2)
num = [c for c in feats if c != "region" and np.issubdtype(df[c].dtype, np.number)]
train = df[df.year.between(2017, 2025)].dropna(subset=["log_ftd"])
X = train[num].fillna(train[num].median())
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=42).fit(X, train.log_ftd)

fig, ax = plt.subplots(figsize=(15, 7))
texts = plot_tree(tree, feature_names=[label(c) for c in num], filled=False, impurity=False,
                  rounded=True, fontsize=11, ax=ax)
for t in texts:
    s = t.get_text()
    m = re.search(r"value = ([-\d.]+)", s)
    if m:
        ftd = np.expm1(float(m.group(1)))
        s = s.replace(m.group(0), f"predicted FTD = {ftd:.2f}")
    s = re.sub(r"samples = (\d+)", r"\1 weeks", s)
    # the trap feature is stored as log(1 + FTD); show the threshold in FTD
    s = re.sub(r"Last trap reading <= ([-\d.]+)",
               lambda m: f"Last trap reading <= {np.expm1(float(m.group(1))):.2f} FTD", s)
    t.set_text(s)
    bb = t.get_bbox_patch()
    if bb is not None:
        bb.set_facecolor(PALE); bb.set_edgecolor(BLUE)
fig.savefig(os.path.join(OUT, "Figure_3.2_example_tree.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("tree splits:", [label(num[i]) for i in tree.tree_.feature if i >= 0])


# ---------- Figure 3.3: bagging vs boosting ----------
def box(ax, x, y, w, h, text, fc=PALE, ec=BLUE, size=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.2, color=INK))


fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2))
for ax in (a1, a2):
    ax.set_xlim(0, 11); ax.set_ylim(0, 3); ax.axis("off")
a2.set_ylim(0.9, 3)

a1.text(0.1, 2.8, "Random Forest (bagging): trees grown independently, then averaged", fontsize=13, fontweight="bold", color=INK)
box(a1, 0.2, 0.9, 1.8, 1.0, "Training data")
for k, y in enumerate([2.0, 1.15, 0.3]):
    box(a1, 3.0, y, 2.4, 0.65, f"Tree {k + 1}: random sample\nof weeks and features" if k < 2 else "... 400 trees", size=10)
    arrow(a1, (2.05, 1.4), (2.95, y + 0.32)); arrow(a1, (5.45, y + 0.32), (6.6, 1.4))
box(a1, 6.65, 0.95, 1.7, 0.9, "Average", fc="#eef0f2", ec=INK, bold=True)
arrow(a1, (8.4, 1.4), (9.0, 1.4)); box(a1, 9.05, 0.95, 1.8, 0.9, "Forecast", fc=ORANGE, ec="#c05621", bold=True)

a2.text(0.1, 2.8, "XGBoost (boosting): each tree corrects the errors of the trees before it", fontsize=13, fontweight="bold", color=INK)
box(a2, 0.2, 1.5, 1.8, 1.0, "Training data")
xs = [2.6, 4.75, 6.9]
for k, x in enumerate(xs):
    box(a2, x, 1.55, 1.7, 0.9, ["Tree 1:\nfirst guess", "Tree 2: fixes\nTree 1's errors", "... 150 trees"][k], size=10)
arrow(a2, (2.05, 2.0), (2.55, 2.0)); arrow(a2, (4.35, 2.0), (4.7, 2.0)); arrow(a2, (6.5, 2.0), (6.85, 2.0))
arrow(a2, (8.65, 2.0), (9.0, 2.0)); box(a2, 9.05, 1.55, 1.8, 0.9, "Sum =\nforecast", fc=ORANGE, ec="#c05621", bold=True)
a2.text(5.6, 1.15, "errors (residuals) passed on to the next tree", ha="center", fontsize=10.5, fontstyle="italic", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure_3.3_rf_vs_xgboost.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
