"""
Final figures and tables for Chapter 4 (Results) of the DST481 report.

Run AFTER walkforward_experiments.py (it reads walkforward_summary.csv / _wilcoxon.csv).
Everything is written to Training/report_figures/, numbered in chapter order:

  Data overview
    Table 4.1  Trap records and FTD summary per district
    Figure 4.1 Mean FTD per year per district (log scale)
    Figure 4.2 Mean FTD by month per district (seasonality)
  RQ1 Features
    Table 4.2  Feature list: source, window, biological rationale
    Figure 4.3 Spearman correlation of each feature with log FTD (2017-2026)
  RQ2 Models
    Table 4.3  Walk-forward R2 (log FTD) per model and horizon
    Table 4.4  Wilcoxon signed-rank tests, Random Forest vs XGBoost
  RQ3 Horizon / outbreak detection
    Figure 4.4 Walk-forward R2 by forecast horizon
    Figure 4.5 Outbreak-week F1 by horizon
    Table 4.5  Precision, recall and F1 by model and horizon
  RQ4 Drivers (SHAP, XGBoost trained on 2017-2026)
    Figure 4.6 SHAP share by feature group: 2-week forecast vs climate-only model
    Figure 4.7 SHAP beeswarm, climate-only model
    Figure 4.8 SHAP dependence, top climate features
    Table 4.6  Mean |SHAP| per feature, both models
    Table 4.7  Seasonal-confound check for accumulated features (chill, degree-days)
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

from horizon_experiments import BASE, TRAIN_PATH, WEATHER_PATH, build, daily_windows

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "experiments")
OUT = os.path.join(SCRIPT_DIR, "report_figures")
V3_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "wc_v3_daily_features.csv")

ACCENT, SECOND, GREY, LIGHT, GRID, INK = "#2a78d6", "#eb6834", "#6b6a65", "#c3c2b7", "#e5e4e0", "#2b2a27"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})

# Climate features for the final models (base-10 degree-days replaced by the 9.6 C versions).
CLIMATE = ["temp_14d_mean", "temp_30d_mean", "temp_90d_mean", "humidity_14d_mean", "vpd_14d_max",
           "soil_temp_14d_mean", "soil_moisture_14d_mean", "rain_14d_sum", "rain_60d_sum"]
V3 = ["cp_30d_sum", "cp_season", "gdd96_30d_sum", "gdd96_60d_sum", "gdd96_90d_sum", "gdd96_season",
      "wind_max_14d_mean", "gust_max_14d_max", "cape_doctor_days_14d"]

FEATURE_INFO = [  # name, label, group, source, rationale
    ("last_log", "Last trap reading", "Trap history", "Trap data", "Current population level; populations change gradually week to week"),
    ("prev_log", "Reading before that", "Trap history", "Trap data", "Direction of recent population change"),
    ("roll4_log", "Mean of last 4 readings", "Trap history", "Trap data", "Smooths week-to-week trap noise"),
    ("weeks_since_reading", "Weeks since last reading", "Trap history", "Trap data", "How stale the latest reading is"),
    ("week_sin", "Season (sine of week)", "Season", "Calendar", "Annual cycle of host fruit and fly activity (peak March-May)"),
    ("week_cos", "Season (cosine of week)", "Season", "Calendar", "Second component of the annual cycle"),
    ("temp_14d_mean", "Temperature, 14-day mean", "Temperature", "Open-Meteo", "Development rate of immature stages rises with temperature"),
    ("temp_30d_mean", "Temperature, 30-day mean", "Temperature", "Open-Meteo", "Roughly one generation of development"),
    ("temp_90d_mean", "Temperature, 90-day mean", "Temperature", "Open-Meteo", "Season-long build-up of the population"),
    ("soil_temp_14d_mean", "Soil temperature, 14-day mean", "Temperature", "Open-Meteo", "Pupae develop in the soil"),
    ("gdd96_30d_sum", "Degree-days (9.6°C), 30 days", "Degree-days", "Derived", "Heat units above the lower developmental threshold (Grout & Stoltz, 2007)"),
    ("gdd96_60d_sum", "Degree-days (9.6°C), 60 days", "Degree-days", "Derived", "About 338 degree-days complete one immature cycle"),
    ("gdd96_90d_sum", "Degree-days (9.6°C), 90 days", "Degree-days", "Derived", "Multi-generation heat accumulation"),
    ("gdd96_season", "Degree-days (9.6°C), since 1 July", "Degree-days", "Derived", "Heat accumulated since the start of spring"),
    ("cp_30d_sum", "Chill portions, 30 days", "Chill (Dynamic Model)", "Derived (hourly)", "Recent winter chill (Erez et al., 1990)"),
    ("cp_season", "Chill portions, since 1 March", "Chill (Dynamic Model)", "Derived (hourly)", "Winter chill governing host-tree budburst and fruit timing"),
    ("humidity_14d_mean", "Relative humidity, 14-day mean", "Moisture", "Open-Meteo", "Adult survival and fecundity"),
    ("vpd_14d_max", "Vapour pressure deficit, 14-day max", "Moisture", "Open-Meteo", "Desiccation stress on adults"),
    ("soil_moisture_14d_mean", "Soil moisture, 14-day mean", "Moisture", "Open-Meteo", "Pupal survival in the soil"),
    ("rain_14d_sum", "Rainfall, 14 days", "Moisture", "Open-Meteo", "Short-term moisture"),
    ("rain_60d_sum", "Rainfall, 60 days", "Moisture", "Open-Meteo", "Seasonal moisture"),
    ("wind_max_14d_mean", "Max wind speed, 14-day mean", "Wind (Cape Doctor)", "Open-Meteo", "Wind limits flight and trap catch"),
    ("gust_max_14d_max", "Max wind gust, 14 days", "Wind (Cape Doctor)", "Open-Meteo", "Extreme wind events"),
    ("cape_doctor_days_14d", "Cape Doctor days, 14 days", "Wind (Cape Doctor)", "Derived", "Strong south-easterly days; possible dispersal effect"),
]
LABEL = {n: lab for n, lab, *_ in FEATURE_INFO}
GROUP = {n: g for n, _, g, *_ in FEATURE_INFO}
GROUP_ORDER = ["Trap history", "Season", "Region", "Temperature", "Degree-days", "Chill (Dynamic Model)", "Moisture",
               "Wind (Cape Doctor)"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


def label(col):
    c = col[5:] if col.startswith("asof_") else col
    if c.startswith("region_"):
        return "Region: " + c[7:]
    return LABEL.get(c, c)


def group(col):
    c = col[5:] if col.startswith("asof_") else col
    return "Region" if c.startswith("region_") else GROUP.get(c, "Other")


def load_frame(h):
    raw = pd.read_csv(TRAIN_PATH)
    df = build(raw, daily_windows(pd.read_csv(WEATHER_PATH)), h)
    v3 = pd.read_csv(V3_PATH, parse_dates=["date"])
    df = df.merge(v3.rename(columns={"date": "event_date"}), on=["region", "event_date"], how="left")
    df = df.merge(v3.rename(columns={"date": "forecast_date", **{c: f"asof_{c}" for c in V3}}),
                  on=["region", "forecast_date"], how="left")
    return df


# ------------------------------------------------------------------ data overview
def data_overview():
    d = pd.read_csv(TRAIN_PATH, parse_dates=["event_date"])
    t = d.groupby("region").agg(Weekly_readings=("ftd", "size"), First_year=("year", "min"), Last_year=("year", "max"),
                                Mean_FTD=("ftd", "mean"), Median_FTD=("ftd", "median"), Max_FTD=("ftd", "max"),
                                Zero_weeks_pct=("ftd", lambda s: 100 * (s == 0).mean()))
    t.loc["All districts"] = [len(d), d.year.min(), d.year.max(), d.ftd.mean(), d.ftd.median(), d.ftd.max(),
                              100 * (d.ftd == 0).mean()]
    t.round(3).to_csv(os.path.join(OUT, "Table_4.1_trap_data_summary.csv"))

    yearly = d.groupby(["region", "year"]).ftd.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(2013.5, 2017, color=GRID, zorder=0)
    ax.text(2015.25, 12, "2014–2017 drop", ha="center", color=GREY, fontsize=10)
    for region, g in yearly.groupby("region"):
        hot = region in ("Tulbagh", "Wolseley")
        ax.plot(g.year, g.ftd, marker="o", ms=3.5, lw=2 if hot else 1.3, color=ACCENT if hot else LIGHT, zorder=3 if hot else 2)
        first = g.iloc[0]
        ax.text(first.year - 0.3, first.ftd, region, ha="right", va="center", fontsize=9,
                color=INK if hot else GREY, fontweight="bold" if hot else "normal")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 20)
    ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax.set_xlim(2005.3, 2026.5)
    ax.spines["left"].set_bounds(0.01, 20)
    ax.set_xticks([2010, 2014, 2017, 2020, 2022, 2026])
    ax.set_ylabel("Mean FTD (flies/trap/day, log scale)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Mean annual FTD per district, 2010–2026")
    save(fig, "Figure_4.1_ftd_by_year.png")

    d["month"] = d.event_date.dt.month
    monthly = d[d.year >= 2017].groupby(["region", "month"]).ftd.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (region, g) in enumerate(monthly.groupby("region")):
        ax.plot(g.month, g.ftd, marker="o", ms=3.5, lw=1.5, color=LIGHT, label="Individual districts" if i == 0 else None)
    allm = d[d.year >= 2017].groupby("month").ftd.mean()
    ax.plot(allm.index, allm.values, lw=2.5, color=ACCENT, marker="o", ms=4, label="All districts")
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0.6, 12.4)
    ax.set_ylabel("Mean FTD (flies/trap/day)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Seasonal pattern of FTD by month, 2017–2026")
    save(fig, "Figure_4.2_ftd_by_month.png")


# ------------------------------------------------------------------ RQ1 features
def features_table_and_correlation(df):
    pd.DataFrame([{"Feature": lab, "Group": g, "Source": src, "Biological rationale": why}
                  for _, lab, g, src, why in FEATURE_INFO]).to_csv(
        os.path.join(OUT, "Table_4.2_features.csv"), index=False)

    recent = df[df.year >= 2017]
    cols = CLIMATE + V3 + ["week_sin", "week_cos"]
    rho = pd.Series({c: spearmanr(recent[c], recent.log_ftd, nan_policy="omit").correlation for c in cols})
    rho = rho.reindex(rho.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [ACCENT if group(c) in ("Degree-days", "Chill (Dynamic Model)", "Wind (Cape Doctor)") else LIGHT
              for c in rho.index]
    ax.barh([label(c) for c in rho.index], rho.values, color=colors, height=0.7)
    ax.axvline(0, color=GREY, lw=0.8)
    for y, v in enumerate(rho.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("Spearman correlation with log(1 + FTD), same week, 2017–2026")
    ax.set_xlim(-0.75, 0.75)
    ax.set_title("Correlation of each feature with trap catch\n(blue = new biological and wind features)")
    save(fig, "Figure_4.3_feature_correlation.png")


# ------------------------------------------------------------------ RQ2 / RQ3 from walk-forward
def walkforward_outputs():
    s = pd.read_csv(os.path.join(EXP_DIR, "walkforward_summary.csv"))
    w = pd.read_csv(os.path.join(EXP_DIR, "walkforward_wilcoxon.csv"))
    reg, clf = s[s.task == "regression"], s[s.task == "classification"]

    t3 = reg.assign(cell=reg.R2_log_mean.map("{:.2f}".format) + " (" + reg.R2_log_sd.map("{:.2f}".format) + ")") \
        .pivot(index="model", columns="horizon_weeks", values="cell")
    t3.columns = [f"R2 log, {h} wk ahead, mean (SD)" for h in t3.columns]
    t3.loc[reg.groupby("model").R2_log_mean.mean().sort_values(ascending=False).index].to_csv(
        os.path.join(OUT, "Table_4.3_walkforward_r2.csv"))

    w.round(4).to_csv(os.path.join(OUT, "Table_4.4_wilcoxon_rf_vs_xgb.csv"), index=False)

    t5 = clf[["model", "horizon_weeks", "Precision_mean", "Recall_mean", "F1_mean", "F1_sd"]].round(3) \
        .sort_values(["horizon_weeks", "F1_mean"], ascending=[True, False])
    t5.to_csv(os.path.join(OUT, "Table_4.5_outbreak_classification.csv"), index=False)

    hs = sorted(reg.horizon_weeks.unique())
    style = {"Random Forest": (ACCENT, "-", 2.5), "XGBoost": (SECOND, "-", 2), "LightGBM": (LIGHT, "-", 1.5),
             "Ridge regression": (GREY, "-", 1.5), "Persistence (baseline)": (GREY, "--", 1.5),
             "Seasonal average (baseline)": (GREY, ":", 1.5), "Degree-day model (baseline)": (LIGHT, "--", 1.5)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ends = {}
    for m, (c, ls, lw) in style.items():
        g = reg[reg.model == m].set_index("horizon_weeks").reindex(hs)
        ax.plot(hs, g.R2_log_mean, color=c, ls=ls, lw=lw, marker="o", ms=5)
        ends[m] = g.R2_log_mean.iloc[-1]
    # End-of-line labels, nudged apart so close lines don't overlap.
    placed = []
    for m, v in sorted(ends.items(), key=lambda kv: -kv[1]):
        y = min([v] + [p - 0.035 for p in placed[-1:]])
        placed.append(y)
        c = style[m][0]
        ax.text(hs[-1] + 0.12, y, f"{m} ({v:.2f})", va="center", fontsize=9, color=INK if c in (ACCENT, SECOND) else GREY)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(hs, [f"{h} week{'s' if h > 1 else ''}" for h in hs])
    ax.set_xlim(0.8, 6.2)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("R² on log(1 + FTD), mean of 7 test years")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Trap-count forecast accuracy by horizon (walk-forward, 2020–2026)")
    save(fig, "Figure_4.4_walkforward_r2_by_horizon.png")

    order = ["Random Forest", "XGBoost", "Logistic regression", "Persistence (baseline)",
             "Seasonal average (baseline)", "Degree-day threshold (baseline)"]
    colors = [ACCENT, SECOND, "#8fb4e3", GREY, "#a5a49f", LIGHT]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.13
    for i, (m, c) in enumerate(zip(order, colors)):
        g = clf[clf.model == m].set_index("horizon_weeks").reindex(hs)
        xs = np.arange(len(hs)) + (i - 2.5) * width
        ax.bar(xs, g.F1_mean, width * 0.9, yerr=g.F1_sd, color=c, label=m, capsize=2.5,
               error_kw={"elinewidth": 0.8, "ecolor": GREY})
    ax.set_xticks(np.arange(len(hs)), [f"{h} week{'s' if h > 1 else ''} ahead" for h in hs])
    ax.set_ylabel("F1-score, mean ± SD of 7 test years")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    ax.set_title("Outbreak-week detection by model and horizon (walk-forward, 2020–2026)")
    save(fig, "Figure_4.5_outbreak_f1_by_horizon.png")


# ------------------------------------------------------------------ RQ4 SHAP
def shap_model(df, feats, target="log_ftd"):
    X = pd.get_dummies(df[feats], columns=["region"], dtype=float)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, random_state=42).fit(X, df[target])
    sv = shap.TreeExplainer(model)(X)
    return X, sv


def shap_outputs():
    # Model A: 2-week-ahead forecast (the recommended model): trap history + as-of climate.
    fa = load_frame(2)
    fa = fa[fa.year >= 2017].reset_index(drop=True)
    feats_a = BASE + [f"asof_{c}" for c in CLIMATE + V3]
    Xa, sva = shap_model(fa, feats_a)
    # Model B: climate-only (no trap history), climate windows ending in the target week.
    fb = load_frame(1)
    fb = fb[fb.year >= 2017].reset_index(drop=True)
    feats_b = CLIMATE + V3 + ["week_sin", "week_cos", "region"]
    Xb, svb = shap_model(fb, feats_b)

    rows = []
    for name, X, sv in (("2-week forecast", Xa, sva), ("Climate-only", Xb, svb)):
        imp = np.abs(sv.values).mean(0)
        for col, v in zip(X.columns, imp):
            rows.append({"model": name, "feature": label(col), "group": group(col), "mean_abs_shap": v})
    imp = pd.DataFrame(rows)
    imp["share_pct"] = 100 * imp.mean_abs_shap / imp.groupby("model").mean_abs_shap.transform("sum")
    imp.sort_values(["model", "mean_abs_shap"], ascending=[True, False]).round(4).to_csv(
        os.path.join(OUT, "Table_4.6_shap_importance.csv"), index=False)

    # Figure 4.6: share of total |SHAP| by feature group, both models side by side.
    grp = imp.groupby(["model", "group"]).share_pct.sum().unstack(0).reindex(GROUP_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(grp))
    ax.barh(y - 0.2, grp["2-week forecast"], 0.38, color=ACCENT, label="2-week forecast (with trap history)")
    ax.barh(y + 0.2, grp["Climate-only"], 0.38, color=LIGHT, label="Climate-only model")
    for i, (a, b) in enumerate(zip(grp["2-week forecast"], grp["Climate-only"])):
        ax.text(a + 0.8, i - 0.2, f"{a:.0f}%", va="center", fontsize=9)
        ax.text(b + 0.8, i + 0.2, f"{b:.0f}%", va="center", fontsize=9, color=GREY)
    ax.set_yticks(y, grp.index)
    ax.invert_yaxis()
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("What drives the predictions: SHAP share by feature group")
    save(fig, "Figure_4.6_shap_groups.png")

    # Figure 4.7: beeswarm, climate-only model.
    svb.feature_names = [label(c) for c in Xb.columns]
    plt.figure()
    shap.plots.beeswarm(svb, max_display=15, show=False)
    plt.gcf().set_size_inches(8, 6.5)
    plt.title("SHAP values, climate-only model (2017–2026)", fontweight="bold")
    plt.gcf().savefig(os.path.join(OUT, "Figure_4.7_shap_beeswarm_climate.png"))
    plt.close()
    print("saved Figure_4.7_shap_beeswarm_climate.png")

    # Figure 4.8: dependence for the 4 strongest climate features (no season/region).
    climate_idx = [i for i, c in enumerate(Xb.columns) if group(c) not in ("Season", "Region")]
    top = sorted(climate_idx, key=lambda i: -np.abs(svb.values[:, i]).mean())[:4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, i in zip(axes, top):
        ax.scatter(svb.data[:, i], svb.values[:, i], s=6, alpha=0.35, color=ACCENT, edgecolors="none")
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel(svb.feature_names[i], fontsize=10)
    axes[0].set_ylabel("SHAP value (effect on log FTD)")
    fig.suptitle("How the strongest climate features change predicted FTD (climate-only model)", fontweight="bold")
    save(fig, "Figure_4.8_shap_dependence.png")

    print("\nSHAP share by group (%):\n", grp.round(1))
    print("\nTop 10 features, climate-only model:")
    print(imp[imp.model == "Climate-only"].nlargest(10, "mean_abs_shap")[["feature", "share_pct"]].round(1).to_string(index=False))


def seasonal_confound_check():
    """Table 4.7: are season-accumulated features just the calendar in disguise?

    For each accumulated feature: the share of its variation explained by region + week of
    year, and its SHAP share before vs after replacing it with this year's anomaly
    (value minus the usual value for that region and week).
    """
    df = load_frame(1)
    df = df[df.year >= 2017].reset_index(drop=True)
    seasonal = ["cp_season", "gdd96_season", "temp_90d_mean"]
    calendar_r2 = {}
    for c in seasonal:
        clim = df.groupby(["region", "week"])[c].transform("mean")
        calendar_r2[c] = 1 - ((df[c] - clim) ** 2).sum() / ((df[c] - df[c].mean()) ** 2).sum()
        df[c + "_anom"] = df[c] - clim

    def shares(feats):
        X, sv = shap_model(df, feats)
        imp = pd.Series(np.abs(sv.values).mean(0), index=X.columns)
        return 100 * imp / imp.sum()

    base = CLIMATE + V3 + ["week_sin", "week_cos", "region"]
    before = shares(base)
    after = shares([c for c in base if c not in seasonal] + [c + "_anom" for c in seasonal])
    pd.DataFrame([{"Feature": label(c), "Variation explained by region + week (%)": round(100 * calendar_r2[c], 1),
                   "SHAP share, raw feature (%)": round(before[c], 1),
                   "SHAP share, year-to-year anomaly (%)": round(after[c + "_anom"], 1)} for c in seasonal]) \
        .to_csv(os.path.join(OUT, "Table_4.7_seasonal_confound.csv"), index=False)
    print("saved Table_4.7_seasonal_confound.csv")


def main():
    os.makedirs(OUT, exist_ok=True)
    data_overview()
    features_table_and_correlation(load_frame(1))
    walkforward_outputs()
    shap_outputs()
    seasonal_confound_check()


if __name__ == "__main__":
    main()
