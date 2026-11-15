"""Train the diabetes model and save it as api/model.pkl.

Run from the repo root:
    pip install -r model/requirements-train.txt
    python model/train.py

scikit-learn is used for training only. The pickle stores the learned
parameters as plain Python numbers, so the deployed API needs no
scikit-learn/numpy/scipy (they are too big for a Vercel function).
"""
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
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
# In this dataset a 0 in these columns means "not measured", not a real value.
ZERO_MEANS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

df = pd.read_csv(ROOT / "model" / "diabetes.csv", header=None, names=FEATURES + ["outcome"])
for col in ZERO_MEANS_MISSING:
    df[col] = df[col].replace(0, np.nan)

X, y = df[FEATURES].values, df["outcome"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
pipe.fit(X_train, y_train)
print(f"CV ROC-AUC = {cv_auc:.3f}; held-out accuracy = {pipe.score(X_test, y_test):.3f}")

# Refit on all data for the deployed model.
pipe.fit(X, y)
imputer, scaler, clf = pipe.named_steps.values()
params = {
    "name": "logistic_regression",
    "features": FEATURES,
    "zero_means_missing": ZERO_MEANS_MISSING,
    "medians": [float(v) for v in imputer.statistics_],
    "means": [float(v) for v in scaler.mean_],
    "scales": [float(v) for v in scaler.scale_],
    "coef": [float(v) for v in clf.coef_[0]],
    "intercept": float(clf.intercept_[0]),
}


def predict_proba(p, row):
    """Same maths as api/predict.py; used here to verify the export."""
    z = p["intercept"]
    for i, v in enumerate(row):
        if v is None or math.isnan(v):
            v = p["medians"][i]
        z += p["coef"][i] * (v - p["means"][i]) / p["scales"][i]
    return 1.0 / (1.0 + math.exp(-z))


# Check the exported parameters reproduce scikit-learn exactly.
sk = pipe.predict_proba(X)[:, 1]
ours = np.array([predict_proba(params, list(r)) for r in X])
max_diff = float(np.abs(sk - ours).max())
assert max_diff < 1e-9, f"export mismatch: {max_diff}"
print(f"Export verified against scikit-learn (max diff {max_diff:.1e})")

out = ROOT / "api" / "model.pkl"
with open(out, "wb") as f:
    pickle.dump(params, f)
print(f"Saved {out} ({out.stat().st_size} bytes)")
