"""SHAP-based feature explainability: per-customer risk factor narratives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier

from src.data_loader import get_feature_columns

# Human-readable templates for each feature, keyed by feature name.
# Tuple: (positive_impact_phrase, negative_impact_phrase)
_FEATURE_NARRATIVE = {
    "monthly_revenue":             ("high monthly revenue", "low monthly revenue"),
    "contract_months_remaining":   ("contract expiring soon", "long time left on contract"),
    "support_tickets_90d":         ("high volume of support tickets", "low support ticket volume"),
    "product_usage_score":         ("low product usage score", "strong product usage score"),
    "days_since_login":            ("not logged in recently", "logs in regularly"),
    "nps_score":                   ("low NPS / detractor sentiment", "high NPS / promoter sentiment"),
    "payment_delays":              ("history of payment delays", "no payment delays"),
}


def _build_explainer(model: GradientBoostingClassifier, X: pd.DataFrame) -> shap.TreeExplainer:
    return shap.TreeExplainer(model, data=X, feature_perturbation="interventional")


def compute_shap_values(
    model: GradientBoostingClassifier, df: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return raw SHAP values array and the feature DataFrame used to compute them."""
    features = get_feature_columns()
    X = df[features]
    explainer = _build_explainer(model, X)
    shap_values = explainer.shap_values(X)
    return np.array(shap_values), X


def get_top_risk_factors(
    row: pd.Series,
    shap_vals: np.ndarray,
    feature_names: list[str],
    n_top: int = 3,
) -> list[str]:
    """
    Return up to n_top human-readable risk factor strings for a single customer.
    Only factors that increase churn risk (positive SHAP) are included.
    Falls back to top absolute contributors if none are positive.
    """
    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: x[1], reverse=True)

    # Prefer features driving churn (positive SHAP); fall back to absolute top
    drivers = [p for p in pairs if p[1] > 0] or pairs
    drivers = drivers[:n_top]

    narratives = []
    for feature, shap_val in drivers:
        pos_phrase, neg_phrase = _FEATURE_NARRATIVE.get(feature, (feature, feature))
        phrase = pos_phrase if shap_val > 0 else neg_phrase
        magnitude = abs(shap_val)
        strength = "strongly " if magnitude > 0.15 else ""
        narratives.append(f"{phrase} ({strength}{'increases' if shap_val > 0 else 'reduces'} churn risk)")

    return narratives


def add_risk_factors_to_df(
    df: pd.DataFrame,
    model: GradientBoostingClassifier,
) -> pd.DataFrame:
    """
    Compute SHAP values for the full scored DataFrame and attach a
    top_risk_factors list column to each row.
    """
    features = get_feature_columns()
    shap_values, X = compute_shap_values(model, df)

    risk_factors: list[list[str]] = []
    for i, row in enumerate(X.itertuples(index=False)):
        row_series = pd.Series(row._asdict())
        factors = get_top_risk_factors(row_series, shap_values[i], features)
        risk_factors.append(factors)

    result = df.copy()
    result["top_risk_factors"] = risk_factors
    return result


def get_global_feature_importance(
    model: GradientBoostingClassifier, df: pd.DataFrame
) -> pd.DataFrame:
    """Return mean absolute SHAP values per feature for global importance display."""
    features = get_feature_columns()
    shap_values, _ = compute_shap_values(model, df)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
