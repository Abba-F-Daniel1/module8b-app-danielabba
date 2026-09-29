"""
Train the NCD referral-risk model.

Reads the NCD community-screening dataset (Module 8A training data),
cleans it, trains a scikit-learn pipeline, and saves it as model.pkl.

Run:  python train.py
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/ncd_screening.csv"
MODEL_PATH = "model.pkl"

TARGET = "referred"
NUMERIC_FEATURES = [
    "patient_age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "blood_glucose_mmol_l",
]
CATEGORICAL_FEATURES = [
    "sex",
    "smoker",
    "family_history_ncd",
    "condition",
    "diagnosis_status",
    "controlled",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # The raw export mixes short and long sex labels; normalise them.
    df["sex"] = df["sex"].replace({"F": "Female", "M": "Male"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline().fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pipe.predict(X_test))
    auc = roc_auc_score(y_test, proba)
    print(f"Holdout accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
    print(f"Referral base rate: {y.mean():.3f}")

    # Bundle the pipeline with the metadata the app needs at inference time.
    joblib.dump(
        {
            "model": pipe,
            "features": FEATURES,
            "metrics": {"accuracy": acc, "roc_auc": auc},
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
