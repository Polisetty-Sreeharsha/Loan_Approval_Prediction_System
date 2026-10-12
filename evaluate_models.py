
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(ROOT, "Datasets", "dataset.csv"))
df = df.drop(columns=["loan_id"])
df["loan_status"] = df["loan_status"].map({"y": 1, "n": 0})

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_split=10,
        min_samples_leaf=2, max_features="log2",
        random_state=42, n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n=== CROSS-VALIDATION ON TRAINING DATA ===")

for name, estimator in candidates.items():
    pipe = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", estimator)
    ])
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1
    )
    print(
        f"{name}: "
        f"Accuracy={scores['test_accuracy'].mean():.4f}, "
        f"F1={scores['test_f1'].mean():.4f}, "
        f"ROC-AUC={scores['test_roc_auc'].mean():.4f}"
    )

print("\n=== HELD-OUT TEST COMPARISON ===")

saved_model = joblib.load(
    os.path.join(ROOT, "models", "loan_approval_model.pkl")
)

# Evaluate the existing saved model without changing it.
models_to_test = {"Current saved model": saved_model}

# Train the alternative on training data only.
alternative = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])
alternative.fit(X_train, y_train)
models_to_test["Logistic Regression"] = alternative

for name, model in models_to_test.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, prob):.4f}")
    print(f"True rejections: {tn}")
    print(f"False approvals: {fp}")
    print(f"False rejections: {fn}")
    print(f"True approvals:  {tp}")

print("\nExisting model file was not modified.")
