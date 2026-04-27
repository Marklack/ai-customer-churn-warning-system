"""LLM-powered churn explanations and executive summaries via Anthropic Claude."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from src.schemas import ChurnExplanation, ExecutiveSummary

# System prompt cached once per session using Anthropic prompt caching
_SYSTEM_PROMPT = """You are a senior B2B customer success strategist with deep expertise in
churn prevention. You analyze customer health signals from a SaaS/subscription business and
produce actionable retention guidance for account managers.

Your analysis is:
- Specific and data-driven (reference actual metric values)
- Commercially focused (tie risk to revenue impact)
- Practical (talking points an account manager can use tomorrow)
- Appropriately urgent without causing panic

Always provide structured output exactly matching the requested tool schema."""

_EXEC_SUMMARY_SYSTEM = """You are a Chief Customer Officer preparing a concise executive
briefing on customer churn risk for the leadership team. Your summaries are:
- Strategic and revenue-focused
- Clear about the most critical accounts and segments
- Actionable at the organizational level
- Written for a C-suite audience in plain business language"""


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


def _customer_tool_schema() -> dict[str, Any]:
    return {
        "name": "generate_churn_analysis",
        "description": "Generate structured churn risk analysis and retention guidance for a B2B customer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plain_english_explanation": {
                    "type": "string",
                    "description": (
                        "2-3 sentence plain-English explanation of why this customer is "
                        "at churn risk, referencing their specific metric values."
                    ),
                },
                "retention_action": {
                    "type": "string",
                    "description": (
                        "One specific, concrete retention action the account team "
                        "should take within the next 5 business days."
                    ),
                },
                "talking_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3 to 5 talking points for the account manager's next conversation with this customer.",
                    "minItems": 3,
                    "maxItems": 5,
                },
                "urgency_level": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High", "Critical"],
                    "description": "How urgently the account team needs to act.",
                },
            },
            "required": [
                "plain_english_explanation",
                "retention_action",
                "talking_points",
                "urgency_level",
            ],
        },
    }


def _exec_summary_tool_schema() -> dict[str, Any]:
    return {
        "name": "generate_executive_summary",
        "description": "Generate a structured executive summary of the portfolio churn risk landscape.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary_narrative": {
                    "type": "string",
                    "description": "3-4 sentence executive narrative describing the overall churn situation.",
                },
                "recommended_priorities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3-5 organizational priorities to address churn risk.",
                    "minItems": 3,
                    "maxItems": 5,
                },
                "top_segments_at_risk": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The 1-3 customer segments with the highest concentration of churn risk.",
                },
            },
            "required": ["summary_narrative", "recommended_priorities", "top_segments_at_risk"],
        },
    }


def generate_churn_explanation(customer: dict[str, Any]) -> ChurnExplanation:
    """
    Call Claude to produce a structured churn explanation for a single customer.
    Uses prompt caching on the system prompt to reduce cost across bulk runs.
    """
    client = _get_client()
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    risk_factors_text = "\n".join(f"  • {f}" for f in customer.get("top_risk_factors", []))

    user_message = f"""Analyze the churn risk for the following customer and call the generate_churn_analysis tool.

CUSTOMER PROFILE
----------------
Name:     {customer['customer_name']}
ID:       {customer['customer_id']}
Segment:  {customer['segment']}
MRR:      ${customer['monthly_revenue']:,.0f}/month
Owner:    {customer['account_owner']}

CHURN SIGNAL
------------
Churn Probability: {customer['churn_probability']:.1%}
Risk Tier:         {customer['risk_tier']}

TOP RISK DRIVERS (from SHAP explainability)
--------------------------------------------
{risk_factors_text}

RAW METRICS
-----------
Contract months remaining : {customer['contract_months_remaining']}
Support tickets (90 days) : {customer['support_tickets_90d']}
Product usage score       : {customer['product_usage_score']}/100
Days since last login     : {customer['days_since_login']}
NPS score                 : {customer['nps_score']}
Payment delays            : {customer['payment_delays']}"""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                # Cache the system prompt — reused across all customer calls
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_customer_tool_schema()],
        tool_choice={"type": "tool", "name": "generate_churn_analysis"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "generate_churn_analysis":
            return ChurnExplanation(**block.input)

    raise RuntimeError(f"Claude did not return tool use for customer {customer.get('customer_id')}")


def generate_executive_summary(
    scored_df,
    total_at_risk_mrr: float,
    critical_count: int,
    high_count: int,
) -> ExecutiveSummary:
    """
    Generate an AI executive summary of the overall churn risk picture.
    scored_df should be the full scored and explained DataFrame.
    """
    import pandas as pd

    client = _get_client()
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Build a compact portfolio snapshot for the LLM
    segment_risk = (
        scored_df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_churn_prob=("churn_probability", "mean"),
            at_risk_mrr=("monthly_revenue", lambda x: x[scored_df.loc[x.index, "churn_probability"] >= 0.6].sum()),
        )
        .sort_values("avg_churn_prob", ascending=False)
        .to_string()
    )

    owner_risk = (
        scored_df[scored_df["churn_probability"] >= 0.6]
        .groupby("account_owner")["customer_id"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .to_string()
    )

    top_critical = scored_df[scored_df["risk_tier"] == "Critical"][
        ["customer_name", "segment", "monthly_revenue", "churn_probability"]
    ].head(5).to_string(index=False)

    user_message = f"""Prepare an executive churn risk briefing by calling the generate_executive_summary tool.

PORTFOLIO SNAPSHOT
------------------
Total customers analyzed : {len(scored_df)}
Critical risk customers  : {critical_count}
High risk customers      : {high_count}
Total at-risk MRR        : ${total_at_risk_mrr:,.0f}

RISK BY SEGMENT
---------------
{segment_risk}

TOP AT-RISK ACCOUNT OWNERS
---------------------------
{owner_risk}

TOP CRITICAL ACCOUNTS
---------------------
{top_critical}"""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _EXEC_SUMMARY_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_exec_summary_tool_schema()],
        tool_choice={"type": "tool", "name": "generate_executive_summary"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "generate_executive_summary":
            return ExecutiveSummary(
                total_customers_analyzed=len(scored_df),
                high_risk_count=high_count,
                critical_risk_count=critical_count,
                total_at_risk_mrr=total_at_risk_mrr,
                **block.input,
            )

    raise RuntimeError("Claude did not return executive summary tool use.")
