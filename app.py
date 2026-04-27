"""
AI Customer Churn Early Warning System
Streamlit dashboard — run with: streamlit run app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_customer_data
from src.explainability import add_risk_factors_to_df, get_global_feature_importance
from src.llm_recommendations import generate_churn_explanation, generate_executive_summary
from src.model import train_churn_model
from src.report_generator import flatten_llm_columns, generate_executive_report, generate_retention_csv
from src.schemas import ExecutiveSummary
from src.scoring import RISK_TIER_COLORS, RISK_TIER_ORDER, compute_at_risk_mrr, score_customers

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Churn Early Warning",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/sample_customer_data.csv"


# ── Cached heavy operations ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training churn model…")
def _get_model_and_data():
    df = load_customer_data(DATA_PATH)
    model = train_churn_model(df, verbose=False)
    return model, df


@st.cache_data(show_spinner="Scoring customers…")
def _get_full_scored_df(_model, data_mtime: float) -> pd.DataFrame:
    """Rescore whenever the CSV changes (data_mtime is the cache key)."""
    df = load_customer_data(DATA_PATH)
    scored = score_customers(df, _model)
    scored = add_risk_factors_to_df(scored, _model)
    return scored


@st.cache_data(show_spinner="Computing global feature importance…")
def _get_global_importance(_model, data_mtime: float) -> pd.DataFrame:
    df = load_customer_data(DATA_PATH)
    return get_global_feature_importance(_model, df)


def _data_mtime() -> float:
    try:
        return Path(DATA_PATH).stat().st_mtime
    except FileNotFoundError:
        return 0.0


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _ensure_llm_cache() -> None:
    if "llm_cache" not in st.session_state:
        st.session_state["llm_cache"] = {}
    if "exec_summary" not in st.session_state:
        st.session_state["exec_summary"] = None


def _get_or_generate_explanation(customer: dict) -> None:
    cid = customer["customer_id"]
    if cid in st.session_state["llm_cache"]:
        return
    with st.spinner(f"Generating AI insights for {customer['customer_name']}…"):
        try:
            explanation = generate_churn_explanation(customer)
            st.session_state["llm_cache"][cid] = explanation
        except Exception as exc:
            st.error(f"LLM error for {cid}: {exc}")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _risk_badge(tier: str) -> str:
    color = RISK_TIER_COLORS.get(tier, "#888")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600">{tier}</span>'


def _fmt_currency(val: float) -> str:
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if val >= 1_000:
        return f"${val/1_000:.0f}k"
    return f"${val:.0f}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("⚠️ Churn Warning")
    st.sidebar.markdown("*AI-powered retention intelligence*")
    st.sidebar.divider()

    st.sidebar.header("Filters")

    # Risk tier filter
    all_tiers = [t for t in RISK_TIER_ORDER if t in df["risk_tier"].values]
    selected_tiers = st.sidebar.multiselect(
        "Risk Tier",
        options=all_tiers,
        default=all_tiers,
    )

    # Segment filter
    segments = sorted(df["segment"].unique())
    selected_segments = st.sidebar.multiselect(
        "Segment",
        options=segments,
        default=segments,
    )

    # Revenue tier filter
    rev_tiers = df["revenue_tier"].cat.categories.tolist()
    selected_rev_tiers = st.sidebar.multiselect(
        "Revenue Tier",
        options=rev_tiers,
        default=rev_tiers,
    )

    # Account owner filter
    owners = sorted(df["account_owner"].unique())
    selected_owners = st.sidebar.multiselect(
        "Account Owner",
        options=owners,
        default=owners,
    )

    # Minimum churn probability slider
    min_prob = st.sidebar.slider(
        "Min Churn Probability",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        format="%d%%",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Data: {DATA_PATH}")

    # Apply filters
    mask = (
        df["risk_tier"].isin(selected_tiers)
        & df["segment"].isin(selected_segments)
        & df["revenue_tier"].isin(selected_rev_tiers)
        & df["account_owner"].isin(selected_owners)
        & (df["churn_probability"] >= min_prob / 100)
    )
    return df[mask].copy()


# ── KPI cards ──────────────────────────────────────────────────────────────────

def _render_kpis(filtered_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    critical = (filtered_df["risk_tier"] == "Critical").sum()
    high = (filtered_df["risk_tier"] == "High").sum()
    medium = (filtered_df["risk_tier"] == "Medium").sum()
    at_risk_mrr = compute_at_risk_mrr(filtered_df, "High")
    avg_prob = filtered_df["churn_probability"].mean() * 100 if len(filtered_df) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", len(filtered_df), delta=f"{len(filtered_df) - len(full_df)} vs. all")
    c2.metric(
        "🔴 Critical Risk",
        critical,
        delta=None,
        help="Churn probability ≥ 80%",
    )
    c3.metric("🟠 High Risk", high, help="Churn probability 60–80%")
    c4.metric("🟡 Medium Risk", medium, help="Churn probability 35–60%")
    c5.metric("💰 At-Risk MRR", _fmt_currency(at_risk_mrr), help="High + Critical customers only")


# ── Charts ────────────────────────────────────────────────────────────────────

def _render_charts(filtered_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(
            filtered_df,
            x="churn_probability",
            color="risk_tier",
            color_discrete_map=RISK_TIER_COLORS,
            category_orders={"risk_tier": RISK_TIER_ORDER},
            nbins=25,
            labels={"churn_probability": "Churn Probability", "risk_tier": "Risk Tier"},
            opacity=0.85,
        )
        fig.update_layout(
            bargap=0.05,
            legend_title_text="Risk Tier",
            xaxis_tickformat=".0%",
            margin=dict(t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Global Feature Importance")
        fig2 = px.bar(
            importance_df.head(7),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Reds",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
        )
        fig2.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Risk by segment bubble chart
    st.subheader("Risk Landscape by Segment")
    seg_df = (
        filtered_df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_churn_prob=("churn_probability", "mean"),
            total_mrr=("monthly_revenue", "sum"),
        )
        .reset_index()
    )
    if not seg_df.empty:
        fig3 = px.scatter(
            seg_df,
            x="avg_churn_prob",
            y="total_mrr",
            size="customers",
            color="segment",
            text="segment",
            labels={
                "avg_churn_prob": "Avg Churn Probability",
                "total_mrr": "Total MRR ($)",
                "customers": "# Customers",
            },
            size_max=60,
        )
        fig3.update_traces(textposition="top center")
        fig3.update_layout(
            xaxis_tickformat=".0%",
            margin=dict(t=10, b=10),
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── Customer table & detail cards ─────────────────────────────────────────────

def _render_customer_table(filtered_df: pd.DataFrame) -> None:
    st.subheader(f"Customer Risk List ({len(filtered_df)} accounts)")

    if filtered_df.empty:
        st.info("No customers match the current filters.")
        return

    # Display-friendly columns
    display = filtered_df[[
        "customer_id", "customer_name", "segment", "account_owner",
        "monthly_revenue", "churn_probability", "risk_tier",
        "contract_months_remaining", "support_tickets_90d",
        "product_usage_score", "days_since_login", "nps_score", "payment_delays",
    ]].copy()
    display["churn_probability"] = (display["churn_probability"] * 100).round(1)
    display["monthly_revenue"] = display["monthly_revenue"].round(0).astype(int)

    display = display.rename(columns={
        "customer_id": "ID",
        "customer_name": "Customer",
        "segment": "Segment",
        "account_owner": "Owner",
        "monthly_revenue": "MRR ($)",
        "churn_probability": "Churn % ",
        "risk_tier": "Risk",
        "contract_months_remaining": "Months Left",
        "support_tickets_90d": "Tickets 90d",
        "product_usage_score": "Usage",
        "days_since_login": "Days Idle",
        "nps_score": "NPS",
        "payment_delays": "Pay Delays",
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn % ": st.column_config.ProgressColumn(
                "Churn %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "MRR ($)": st.column_config.NumberColumn("MRR ($)", format="$%d"),
            "Risk": st.column_config.TextColumn("Risk", width="small"),
        },
    )


def _render_customer_detail_cards(filtered_df: pd.DataFrame) -> None:
    """Expandable per-customer AI insight cards for top-risk accounts."""
    _ensure_llm_cache()

    top_risk = filtered_df[filtered_df["risk_tier"].isin(["Critical", "High"])].head(20)
    if top_risk.empty:
        return

    st.subheader("AI Customer Insights")

    ai_enabled = _llm_available()
    if not ai_enabled:
        st.warning(
            "Set **ANTHROPIC_API_KEY** in your .env file to enable AI-generated explanations, "
            "retention actions, and talking points.",
            icon="🔑",
        )

    n_to_analyze = st.slider(
        "Customers to analyze with AI",
        min_value=1,
        max_value=min(20, len(top_risk)),
        value=min(5, len(top_risk)),
        disabled=not ai_enabled,
    )

    if ai_enabled:
        if st.button("🤖 Generate AI Insights", type="primary"):
            for _, row in top_risk.head(n_to_analyze).iterrows():
                _get_or_generate_explanation(row.to_dict())
            st.success(f"AI insights generated for {n_to_analyze} customers.")

    st.divider()

    for _, row in top_risk.head(n_to_analyze).iterrows():
        cid = row["customer_id"]
        tier_color = RISK_TIER_COLORS.get(str(row["risk_tier"]), "#888")

        with st.expander(
            f"**{row['customer_name']}** — {str(row['risk_tier'])} Risk "
            f"({row['churn_probability']:.1%}) | MRR: {_fmt_currency(row['monthly_revenue'])} | {row['account_owner']}"
        ):
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown("**Risk Factors (SHAP)**")
                for factor in row.get("top_risk_factors", []):
                    st.markdown(f"- {factor}")

                st.markdown("**Customer Metrics**")
                metrics_md = (
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Segment | {row['segment']} |\n"
                    f"| Contract Remaining | {row['contract_months_remaining']} months |\n"
                    f"| Support Tickets 90d | {row['support_tickets_90d']} |\n"
                    f"| Product Usage | {row['product_usage_score']}/100 |\n"
                    f"| Days Since Login | {row['days_since_login']} |\n"
                    f"| NPS Score | {row['nps_score']} |\n"
                    f"| Payment Delays | {row['payment_delays']} |"
                )
                st.markdown(metrics_md)

            with col_b:
                explanation = st.session_state["llm_cache"].get(cid)
                if explanation:
                    st.markdown("**AI Explanation**")
                    st.info(explanation.plain_english_explanation)

                    st.markdown("**Recommended Action**")
                    st.success(explanation.retention_action)

                    urgency_color = RISK_TIER_COLORS.get(explanation.urgency_level, "#888")
                    st.markdown(
                        f"**Urgency:** "
                        f'<span style="color:{urgency_color};font-weight:700">'
                        f"{explanation.urgency_level}</span>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Account Manager Talking Points**")
                    for point in explanation.talking_points:
                        st.markdown(f"- {point}")
                elif ai_enabled:
                    st.caption("Click 'Generate AI Insights' above to analyze this customer.")
                else:
                    st.caption("Configure ANTHROPIC_API_KEY to see AI insights.")


# ── Export ────────────────────────────────────────────────────────────────────

def _render_export(filtered_df: pd.DataFrame) -> None:
    st.subheader("Export")
    col1, col2 = st.columns(2)

    with col1:
        # Build exportable DataFrame with LLM results from session state
        export_df = filtered_df.copy()
        if "llm_cache" in st.session_state and st.session_state["llm_cache"]:
            export_df["explanation"] = export_df["customer_id"].map(
                st.session_state["llm_cache"]
            )
        else:
            export_df["explanation"] = None

        export_df = flatten_llm_columns(export_df)

        # Build in-memory CSV
        buf = io.StringIO()

        # Flatten list columns
        out = export_df.copy()
        if "top_risk_factors" in out.columns:
            out["top_risk_factors"] = out["top_risk_factors"].apply(
                lambda x: " | ".join(x) if isinstance(x, list) else x
            )
        if "ai_talking_points" in out.columns:
            out["ai_talking_points"] = out["ai_talking_points"].apply(
                lambda x: " | ".join(x) if isinstance(x, list) else (x or "")
            )

        export_cols = [
            "customer_id", "customer_name", "segment", "revenue_tier",
            "monthly_revenue", "account_owner", "churn_probability", "risk_tier",
            "contract_months_remaining", "support_tickets_90d", "product_usage_score",
            "days_since_login", "nps_score", "payment_delays",
        ]
        for c in ["top_risk_factors", "ai_explanation", "ai_retention_action", "ai_urgency", "ai_talking_points"]:
            if c in out.columns:
                export_cols.append(c)

        out_final = out[[c for c in export_cols if c in out.columns]].copy()
        out_final["churn_probability"] = out_final["churn_probability"].map("{:.1%}".format)
        out_final.to_csv(buf, index=False)

        st.download_button(
            label="⬇ Download Retention List (CSV)",
            data=buf.getvalue(),
            file_name="retention_list.csv",
            mime="text/csv",
            type="primary",
        )

    with col2:
        exec_summary = st.session_state.get("exec_summary")
        if exec_summary:
            report_md = _build_summary_markdown(exec_summary)
            st.download_button(
                label="⬇ Download Executive Report (MD)",
                data=report_md,
                file_name="executive_summary.md",
                mime="text/markdown",
            )
        else:
            st.caption("Generate the Executive Summary below to enable report download.")


def _build_summary_markdown(summary: ExecutiveSummary) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y")
    lines = [
        "# Churn Early Warning — Executive Summary",
        f"*Generated: {now}*\n",
        "## Portfolio Overview\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total Customers | {summary.total_customers_analyzed} |",
        f"| Critical Risk | {summary.critical_risk_count} |",
        f"| High Risk | {summary.high_risk_count} |",
        f"| At-Risk MRR | ${summary.total_at_risk_mrr:,.0f} |\n",
        "## Narrative\n",
        summary.summary_narrative + "\n",
        "## Segments at Risk\n",
        *[f"- {s}" for s in summary.top_segments_at_risk],
        "\n## Recommended Priorities\n",
        *[f"{i+1}. {p}" for i, p in enumerate(summary.recommended_priorities)],
    ]
    return "\n".join(lines)


# ── Executive Summary ─────────────────────────────────────────────────────────

def _render_executive_summary(filtered_df: pd.DataFrame) -> None:
    st.subheader("Executive Summary")
    _ensure_llm_cache()

    ai_enabled = _llm_available()

    if not ai_enabled:
        st.warning("Set ANTHROPIC_API_KEY to generate an AI executive summary.", icon="🔑")
        return

    if st.button("📊 Generate Executive Summary", disabled=not ai_enabled):
        critical_n = (filtered_df["risk_tier"] == "Critical").sum()
        high_n = (filtered_df["risk_tier"] == "High").sum()
        at_risk_mrr = compute_at_risk_mrr(filtered_df, "High")
        with st.spinner("Generating executive summary…"):
            try:
                summary = generate_executive_summary(
                    filtered_df,
                    total_at_risk_mrr=at_risk_mrr,
                    critical_count=int(critical_n),
                    high_count=int(high_n),
                )
                st.session_state["exec_summary"] = summary
            except Exception as exc:
                st.error(f"Failed to generate summary: {exc}")

    summary = st.session_state.get("exec_summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers Analyzed", summary.total_customers_analyzed)
        c2.metric("Critical Risk", summary.critical_risk_count)
        c3.metric("High Risk", summary.high_risk_count)
        c4.metric("At-Risk MRR", _fmt_currency(summary.total_at_risk_mrr))

        st.markdown("#### Narrative")
        st.info(summary.summary_narrative)

        col_seg, col_pri = st.columns(2)
        with col_seg:
            st.markdown("#### Segments at Risk")
            for s in summary.top_segments_at_risk:
                st.markdown(f"- {s}")
        with col_pri:
            st.markdown("#### Recommended Priorities")
            for i, p in enumerate(summary.recommended_priorities, 1):
                st.markdown(f"{i}. {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not Path(DATA_PATH).exists():
        st.error(
            f"Data file not found: **{DATA_PATH}**\n\n"
            "Run `python generate_data.py` to create sample data."
        )
        st.stop()

    model, _ = _get_model_and_data()
    mtime = _data_mtime()
    full_df = _get_full_scored_df(model, mtime)
    importance_df = _get_global_importance(model, mtime)

    # Header
    st.title("⚠️ AI Churn Early Warning System")
    st.caption(
        "Machine learning churn prediction + Claude AI retention intelligence · "
        f"{len(full_df)} customers loaded"
    )
    st.divider()

    # Sidebar → filtered view
    filtered_df = _render_sidebar(full_df)

    # KPIs
    _render_kpis(filtered_df, full_df)
    st.divider()

    # Charts
    _render_charts(filtered_df, importance_df)
    st.divider()

    # Customer table
    _render_customer_table(filtered_df)
    st.divider()

    # AI detail cards
    _render_customer_detail_cards(filtered_df)
    st.divider()

    # Export
    _render_export(filtered_df)
    st.divider()

    # Executive summary
    _render_executive_summary(filtered_df)


if __name__ == "__main__":
    main()
