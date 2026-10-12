
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

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

candidate = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

candidate.fit(X_train, y_train)

os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
save_path = os.path.join(ROOT, "models", "loan_approval_candidate.pkl")
joblib.dump(candidate, save_path)

print("Candidate model saved:", save_path)
print("Current production model was not changed.")
