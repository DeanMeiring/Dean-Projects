import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Dynamic Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "wc_v2_training_regression.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "western_cape_medfly_regressor.json")
DASHBOARD_PATH = os.path.join(SCRIPT_DIR, "medfly_regressor_dashboard.png")
THRESHOLDS_PATH = os.path.join(SCRIPT_DIR, "wc_v2_risk_thresholds.json")
METRICS_PATH = os.path.join(SCRIPT_DIR, "wc_v2_model_metrics.json")

FEATURE_COLUMNS = [
    "temp_14d_mean", "temp_30d_mean", "humidity_14d_mean",
    "vpd_14d_max", "soil_temp_14d_mean", "soil_moisture_14d_mean",
    "degree_days_30d_sum", "rain_14d_sum",
]

# 2. Load Dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Training dataset not found at '{DATA_PATH}'. Run parse_ftd_excel.py, "
        "geocode_regions.py, fetch_weather_history_v2.py and merge_ftd_weather.py first."
    )

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS]
y = df["ftd"]

# 3. Train/Test Split (regression target, no stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train Model
model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="rmse",
    random_state=42,
)

model.fit(X_train, y_train)

# 5. Model Predictions & Metrics
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None)  # FTD can't be negative

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

# Save model weights
model.save_model(MODEL_PATH)

# Risk thresholds derived from the empirical FTD distribution (35th/65th
# percentile of observed trap counts) rather than an assumed absolute cutoff
# — there's no industry-standard FTD threshold available to this pipeline,
# so this is a data-driven proxy, not an entomological standard. Document
# this clearly wherever these thresholds are surfaced (dashboard, API, dissertation).
moderate_cutoff, high_cutoff = y.quantile([0.35, 0.65]).values
with open(THRESHOLDS_PATH, "w") as f:
    json.dump({
        "low_moderate_boundary": float(moderate_cutoff),
        "moderate_high_boundary": float(high_cutoff),
        "method": "35th/65th percentile of observed FTD in training data",
    }, f, indent=2)

# 6. Build High-Resolution PNG Visual Dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Western Cape Medfly XGBoost Regressor Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Predicted vs Actual scatter
axes[0, 0].scatter(y_test, y_pred, alpha=0.4, s=18, color="#2b5c8f")
max_val = max(y_test.max(), y_pred.max())
axes[0, 0].plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="Perfect fit")
axes[0, 0].set_xlabel("Actual FTD")
axes[0, 0].set_ylabel("Predicted FTD")
axes[0, 0].set_title("Predicted vs Actual FTD", fontsize=12, fontweight="bold")
axes[0, 0].legend(loc="upper left")

# Panel 2: Residuals
residuals = y_test.values - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.4, s=18, color="#8f2b5c")
axes[0, 1].axhline(y=0, color="black", linewidth=1)
axes[0, 1].set_xlabel("Predicted FTD")
axes[0, 1].set_ylabel("Residual (Actual - Predicted)")
axes[0, 1].set_title("Residual Plot", fontsize=12, fontweight="bold")

# Panel 3: Feature Importance (Gain Score)
importance = model.get_booster().get_score(importance_type="gain")
df_imp = pd.DataFrame({"Feature": list(importance.keys()), "Gain": list(importance.values())})
df_imp = df_imp.sort_values(by="Gain", ascending=True)

axes[1, 0].barh(df_imp["Feature"], df_imp["Gain"], color="#2b5c8f")
axes[1, 0].set_title("Top Feature Importance (Gain)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Gain Score")

# Panel 4: Executive Summary / KPI Text Box
axes[1, 1].axis("off")
summary_text = (
    f"--- DATASET OVERVIEW ---\n"
    f"Total Training Samples: {len(df)}\n"
    f"Regions: {df['region'].nunique()}\n"
    f"Years Covered: {df['year'].min()}-{df['year'].max()}\n"
    f"Target: FTD (flies/trap/day, continuous)\n\n"
    f"--- EVALUATION METRICS ---\n"
    f"RMSE: {rmse:.4f}\n"
    f"MAE: {mae:.4f}\n"
    f"R^2 Score: {r2:.4f}\n\n"
    f"--- RISK THRESHOLDS (data-driven, not entomological standard) ---\n"
    f"LOW < {moderate_cutoff:.3f} <= MODERATE < {high_cutoff:.3f} <= HIGH\n\n"
    f"--- MODEL CONFIGURATION ---\n"
    f"Algorithm: XGBoost Regressor\n"
    f"Trees (n_estimators): 150\n"
    f"Max Depth: 4 | Learning Rate: 0.03\n"
    f"Saved Model: western_cape_medfly_regressor.json"
)

axes[1, 1].text(
    0.05, 0.5, summary_text, fontsize=11, family="monospace",
    verticalalignment="center", bbox=dict(boxstyle="round,pad=1", facecolor="#f4f6f9", alpha=0.8, edgecolor="#ccc")
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(DASHBOARD_PATH, dpi=300)
plt.close()

# Machine-readable metrics for the dashboard UI, alongside the PNG.
with open(METRICS_PATH, "w") as f:
    json.dump({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_regions": int(df["region"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "feature_importance": {k: float(v) for k, v in importance.items()},
    }, f, indent=2)

print(f"\n================ SUCCESS ================")
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R^2: {r2:.4f}")
print(f"Visual dashboard generated and saved to: '{DASHBOARD_PATH}'")
