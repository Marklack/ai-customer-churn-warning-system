"""Score every customer with churn probability and assign a risk tier."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.data_loader import get_feature_columns

# Inclusive lower bound, exclusive upper bound → label
_RISK_THRESHOLDS = [
    (0.80, 1.01, "Critical"),
    (0.60, 0.80, "High"),
    (0.35, 0.60, "Medium"),
    (0.00, 0.35, "Low"),
]

RISK_TIER_ORDER = ["Critical", "High", "Medium", "Low"]
RISK_TIER_COLORS = {
    "Critical": "#FF4B4B",
    "High":     "#FF8C00",
    "Medium":   "#FFD700",
    "Low":      "#2ECC71",
}


def _assign_risk_tier(prob: float) -> str:
    for lo, hi, label in _RISK_THRESHOLDS:
        if lo <= prob < hi:
            return label
    return "Critical"


def score_customers(df: pd.DataFrame, model: GradientBoostingClassifier) -> pd.DataFrame:
    """
    Append churn_probability and risk_tier to a copy of df.
    Returns customers sorted by churn_probability descending.
    """
    features = get_feature_columns()
    X = df[features]

    probs = model.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["churn_probability"] = probs
    scored["risk_tier"] = scored["churn_probability"].apply(_assign_risk_tier)
    scored["risk_tier"] = pd.Categorical(
        scored["risk_tier"], categories=RISK_TIER_ORDER, ordered=True
    )
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    return scored


def compute_at_risk_mrr(scored_df: pd.DataFrame, min_tier: str = "High") -> float:
    """Sum monthly_revenue for customers at or above a given risk tier."""
    tiers = RISK_TIER_ORDER[: RISK_TIER_ORDER.index(min_tier) + 1]
    return scored_df.loc[scored_df["risk_tier"].isin(tiers), "monthly_revenue"].sum()
