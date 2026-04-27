"""Pydantic schemas for structured LLM output and data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChurnExplanation(BaseModel):
    """Structured AI-generated churn risk explanation for a single customer."""

    plain_english_explanation: str = Field(
        ...,
        description="Plain-English explanation of why this customer is at risk of churning",
    )
    retention_action: str = Field(
        ...,
        description="Specific, concrete retention action for the account team to take immediately",
    )
    talking_points: list[str] = Field(
        ...,
        description="3-5 talking points the account manager should raise in their next customer conversation",
    )
    urgency_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        ...,
        description="How urgently the account team needs to act on this customer",
    )


class CustomerRiskProfile(BaseModel):
    """Full risk profile for a single customer, combining ML scores and AI insights."""

    customer_id: str
    customer_name: str
    segment: str
    monthly_revenue: float
    account_owner: str
    churn_probability: float
    risk_tier: str
    top_risk_factors: list[str]
    explanation: ChurnExplanation | None = None


class ExecutiveSummary(BaseModel):
    """AI-generated executive summary of the churn landscape."""

    total_customers_analyzed: int
    high_risk_count: int
    critical_risk_count: int
    total_at_risk_mrr: float
    top_segments_at_risk: list[str]
    recommended_priorities: list[str]
    summary_narrative: str
