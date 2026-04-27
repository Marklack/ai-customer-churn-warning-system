"""Churn prediction model: training, persistence, and evaluation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_loader import get_X_y

MODEL_PATH = Path("output/churn_model.pkl")


def train_churn_model(df: pd.DataFrame, verbose: bool = True) -> GradientBoostingClassifier:
    """Train a gradient boosting classifier on customer data and return the fitted model."""
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    if verbose:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"\n=== Churn Model Performance ===")
        print(f"ROC-AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    return model


def save_model(model: GradientBoostingClassifier, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> GradientBoostingClassifier:
    return joblib.load(path)


def get_feature_importances(model: GradientBoostingClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances from the trained model."""
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
