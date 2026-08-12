import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

# 1. Dynamic Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "final_gbif_weather_training.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "western_cape_medfly_xgboost.json")
DASHBOARD_PATH = os.path.join(SCRIPT_DIR, "medfly_model_dashboard.png")

# 2. Load Dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training dataset not found at '{DATA_PATH}'. Run 'merge_gbif_weather.py' first.")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['event_date', 'time', 'outbreak'], errors='ignore')
y = df['outbreak']

# 3. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Handle Imbalance & Train Model
neg_count, pos_count = np.bincount(y_train)
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# 5. Model Predictions & Metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Save model weights
model.save_model(MODEL_PATH)

# 6. Build High-Resolution PNG Visual Dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Mediterranean Fruit Fly (Medfly) XGBoost Model Performance Dashboard", fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Outbreak (0)', 'Outbreak (1)'],
            yticklabels=['No Outbreak (0)', 'Outbreak (1)'])
axes[0, 0].set_title("Confusion Matrix", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("True Label")

# Panel 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold')
axes[0, 1].legend(loc="lower right")

# Panel 3: Feature Importance (Gain Score)
importance = model.get_booster().get_score(importance_type='gain')
df_imp = pd.DataFrame({'Feature': list(importance.keys()), 'Gain': list(importance.values())})
df_imp = df_imp.sort_values(by='Gain', ascending=True)

axes[1, 0].barh(df_imp['Feature'], df_imp['Gain'], color='#2b5c8f')
axes[1, 0].set_title("Top Feature Importance (Gain)", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Gain Score")

# Panel 4: Executive Summary / KPI Text Box
axes[1, 1].axis('off')
summary_text = (
    f"--- DATASET OVERVIEW ---\n"
    f"Total Training Samples: {len(df)}\n"
    f"Class 1 (Outbreak Events): {sum(y == 1)}\n"
    f"Class 0 (Baseline Events): {sum(y == 0)}\n\n"
    f"--- EVALUATION METRICS ---\n"
    f"ROC-AUC Score: {auc_score:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall: {rec:.4f}\n"
    f"F1-Score: {f1:.4f}\n\n"
    f"--- MODEL CONFIGURATION ---\n"
    f"Algorithm: XGBoost Classifier\n"
    f"Trees (n_estimators): 150\n"
    f"Max Depth: 4 | Learning Rate: 0.03\n"
    f"Saved Model: western_cape_medfly_xgboost.json"
)

axes[1, 1].text(
    0.05, 0.5, summary_text, fontsize=11, family='monospace',
    verticalalignment='center', bbox=dict(boxstyle='round,pad=1', facecolor='#f4f6f9', alpha=0.8, edgecolor='#ccc')
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(DASHBOARD_PATH, dpi=300)
plt.close()

print(f"\n================ SUCCESS ================")
print(f"Visual dashboard generated and saved to: '{DASHBOARD_PATH}'")