"""Train the diabetes model and save it as api/model.pkl.

Run from the repo root:  python model/train.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]

df = pd.read_csv(ROOT / "model" / "diabetes.csv", header=None, names=FEATURES + ["outcome"])

# In this dataset a 0 in these columns means "not measured", not a real value.
for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
    df[col] = df[col].replace(0, np.nan)

X, y = df[FEATURES].values, df["outcome"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42
    ),
}

best_name, best_score, best_pipe = None, -1.0, None
for name, clf in candidates.items():
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"{name:20s} CV ROC-AUC = {score:.3f}")
    if score > best_score:
        best_name, best_score, best_pipe = name, score, pipe

best_pipe.fit(X_train, y_train)
print(f"Selected {best_name}; held-out accuracy = {best_pipe.score(X_test, y_test):.3f}")

# Refit on all data for the deployed model.
best_pipe.fit(X, y)
out = ROOT / "api" / "model.pkl"
with open(out, "wb") as f:
    pickle.dump({"model": best_pipe, "features": FEATURES, "name": best_name}, f)
print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB)")
