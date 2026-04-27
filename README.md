# AI Customer Churn Early Warning System

An AI-powered B2B customer churn prediction and retention intelligence platform that combines machine learning risk scoring with Claude AI-generated business explanations, retention playbooks, and account manager talking points.

---

## What it does

| Layer | Technology | Output |
|-------|-----------|--------|
| Data | pandas · CSV | Cleaned customer health metrics |
| ML Model | scikit-learn GradientBoosting | Per-customer churn probability |
| Explainability | SHAP TreeExplainer | Top risk drivers per customer |
| AI Insights | Anthropic Claude | Plain-English explanation, retention action, talking points, urgency |
| Dashboard | Streamlit + Plotly | Interactive filtering, charts, customer cards |
| Export | pandas · Markdown | Prioritized CSV retention list, executive report |

---

## Project structure

```
ai-customer-churn-warning-system/
├── app.py                      # Streamlit dashboard (main entry point)
├── generate_data.py            # One-time sample data generator
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   └── sample_customer_data.csv
├── output/                     # Generated reports and exports land here
└── src/
    ├── __init__.py
    ├── schemas.py              # Pydantic models (ChurnExplanation, ExecutiveSummary)
    ├── data_loader.py          # CSV loading, feature definitions, revenue tiers
    ├── model.py                # GradientBoostingClassifier training & persistence
    ├── scoring.py              # Churn probability scoring, risk tier assignment
    ├── explainability.py       # SHAP values → human-readable risk factor narratives
    ├── llm_recommendations.py  # Anthropic API: per-customer AI + exec summary
    └── report_generator.py     # CSV export, markdown executive report
```

---

## Quick start

### 1. Clone and install dependencies

```bash
cd ai-customer-churn-warning-system
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Generate sample data

```bash
python generate_data.py
```

This creates `data/sample_customer_data.csv` with 160 synthetic B2B customers.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Features

### Dashboard filters
- **Risk Tier** — Critical / High / Medium / Low
- **Segment** — Enterprise / Mid-Market / SMB
- **Revenue Tier** — High (>$10k) / Mid ($2k–$10k) / Low (<$2k)
- **Account Owner** — filter by individual account managers
- **Min Churn Probability** — slider to set a floor

### Visualizations
- Churn probability histogram (color-coded by risk tier)
- Global feature importance (SHAP-based bar chart)
- Risk landscape bubble chart (segment × MRR × customer count)

### AI Insights (requires API key)
- Select up to 20 top-risk customers
- One-click batch AI analysis via Claude
- Per-customer expandable cards showing:
  - Plain-English churn explanation
  - Specific retention action
  - 3–5 account manager talking points
  - Urgency level (Low / Medium / High / Critical)

### Exports
- **Retention List CSV** — all filtered customers with risk scores, SHAP factors, and AI insights
- **Executive Summary Markdown** — AI-generated portfolio briefing for leadership

---

## Customer data schema

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique identifier (CU001…) |
| `customer_name` | string | Company name |
| `segment` | string | Enterprise / Mid-Market / SMB |
| `monthly_revenue` | float | Monthly recurring revenue ($) |
| `contract_months_remaining` | int | Months until contract expiry |
| `support_tickets_90d` | int | Support tickets raised in past 90 days |
| `product_usage_score` | int | 0–100 product engagement score |
| `days_since_login` | int | Days elapsed since last login |
| `nps_score` | int | Net Promoter Score (-100 to 100) |
| `payment_delays` | int | Number of delayed payments |
| `account_owner` | string | Assigned account manager |
| `churned` | int | 1 = churned (training label), 0 = retained |

---

## Model details

- **Algorithm**: `GradientBoostingClassifier` (sklearn)  
  200 estimators · max_depth=3 · learning_rate=0.05 · subsample=0.8
- **Explainability**: SHAP `TreeExplainer` with interventional perturbation
- **Risk tiers**:

| Tier | Probability Range |
|------|------------------|
| Critical | ≥ 80% |
| High | 60–80% |
| Medium | 35–60% |
| Low | < 35% |

---

## Using your own data

Replace `data/sample_customer_data.csv` with your real customer export. The CSV must include the columns listed in the schema above (except `churned` is optional if you only want to score, not train — in that case update `model.py` to load a pre-trained model).

---

## AI cost notes

- Each customer analysis call uses ~400 input tokens (system prompt cached)
- The system prompt is cached with Anthropic's prompt caching — only charged once per session
- Typical cost per customer insight: ~$0.002 (Sonnet 4.6 pricing)
- Executive summary: ~$0.005 per generation
