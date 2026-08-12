import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Data_Transform", "final_stock_training.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "stock_xgboost_model.json")

df = pd.read_csv(DATA_PATH)

# Drop rows without future labels for training
df_train = df.dropna(subset=['target']).copy()
df_train['target'] = df_train['target'].astype(int)

FEATURE_COLUMNS = [
    'sma_14', 'sma_50', 'macd', 'macd_signal', 'rsi_14', 
    'daily_return', 'volatility_14d', 'volatility_30d', 'volume_ratio'
]

X = df_train[FEATURE_COLUMNS]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

model.save_model(MODEL_PATH)
print(f"Model saved to '{MODEL_PATH}'")