import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the processed Western Cape dataset
dataset_path = 'western_cape_medfly_training.csv'

# Handle relative path if executing from root or subfolder
if not os.path.exists(dataset_path):
    dataset_path = 'Mediterainaia Furit Fly/western_cape_medfly_training.csv'

df = pd.read_csv(dataset_path)

# 2. Separate features (X) and target (y)
# Drop non-predictive columns like timestamp
X = df.drop(columns=['time', 'outbreak'], errors='ignore')
y = df['outbreak']

print("Features used for training:")
print(list(X.columns))

# 3. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Handle Class Imbalance ratio
neg_count, pos_count = np.bincount(y_train)
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

# 5. Initialize and Train XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=20
)

# 6. Evaluate Model Performance
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n================ MODEL EVALUATION ================")
print(classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Outbreak (1)']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Save Feature Importance Plot
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10, importance_type='gain', title='Western Cape Medfly - Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('medfly_feature_importance.png')
print("\nFeature importance plot saved as 'medfly_feature_importance.png'.")

# 8. Save Trained Model Artifact
model.save_model("western_cape_medfly_xgboost.json")
print("Model saved to 'western_cape_medfly_xgboost.json'.")