"""
Gupio Option 2 - Bank Term Deposit Subscription Prediction

Run from the project root:
    python src/train.py

The script expects:
    data/bank-full.csv

It excludes the fields prohibited by the Gupio pre-contact prediction rule:
contact, day, month, duration, campaign
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
LEAKAGE_COLS = ["contact", "day", "month", "duration", "campaign"]

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-full.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop(columns=LEAKAGE_COLS + ["y"])
y = (df["y"] == "yes").astype(int)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

def make_preprocessor(scale_numeric=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale_numeric else "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor(True)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor(False)),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ]),
    "Logistic Regression (Balanced)": Pipeline([
        ("preprocessor", make_preprocessor(True)),
        ("model", LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ]),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "TN": cm[0,0], "FP": cm[0,1], "FN": cm[1,0], "TP": cm[1,1],
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

final_model = models["Logistic Regression (Balanced)"]
joblib.dump(final_model, MODEL_DIR / "final_model.joblib")

print(results.to_string(index=False))
print("\nFinal model: Logistic Regression (Balanced)")
print("Excluded for pre-contact prediction:", ", ".join(LEAKAGE_COLS))
