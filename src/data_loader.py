"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "monthly_revenue",
    "contract_months_remaining",
    "support_tickets_90d",
    "product_usage_score",
    "days_since_login",
    "nps_score",
    "payment_delays",
]

REVENUE_TIER_BINS = [0, 2_000, 10_000, float("inf")]
REVENUE_TIER_LABELS = ["Low (<$2k)", "Mid ($2k–$10k)", "High (>$10k)"]


def load_customer_data(filepath: str | Path = "data/sample_customer_data.csv") -> pd.DataFrame:
    """Load raw customer CSV and return a cleaned DataFrame."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset="customer_id")
    df["revenue_tier"] = pd.cut(
        df["monthly_revenue"],
        bins=REVENUE_TIER_BINS,
        labels=REVENUE_TIER_LABELS,
    )
    return df


def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into feature matrix X and target y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["churned"].copy()
    return X, y
