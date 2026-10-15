"""Train a customer churn classifier on the IBM Telco dataset.

Two candidate pipelines are compared with 5-fold cross-validation on the
training split. The winner is refit on the full training split, evaluated
once on the held-out test split, and saved together with its metrics.

Preprocessing lives inside the saved pipeline, so the serving container
applies exactly the same transformations that were used in training.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # TotalCharges is stored as text and is blank for brand-new customers
    # (tenure 0), who have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Store SeniorCitizen as Yes/No so every categorical field shares one format.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

    y = (df[TARGET] == "Yes").astype(int)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--data", type=Path, default=Path("data/telco_churn.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Loaded {len(X)} rows; churn rate {y.mean():.1%}")

    # Model selection uses only the training split; the test split is touched once.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, pipeline in build_candidates().items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_results[name] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        print(f"CV ROC-AUC {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["mean"])
    best_model = build_candidates()[best_name].fit(X_train, y_train)
    test_metrics = evaluate(best_model, X_test, y_test)
    print(f"Selected {best_name}; test metrics: {test_metrics}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.out / "model.joblib")

    metrics = {
        "selected_model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "cv_roc_auc": cv_results,
        "test": test_metrics,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {args.out.resolve()}")


if __name__ == "__main__":
    main()
