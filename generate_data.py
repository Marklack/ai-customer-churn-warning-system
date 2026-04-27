"""
Generate sample_customer_data.csv with realistic B2B SaaS customer records.
Run once: python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 160

np.random.seed(SEED)

# ── Segment & revenue ──────────────────────────────────────────────────────────
SEGMENTS = np.random.choice(
    ["Enterprise", "Mid-Market", "SMB"],
    size=N,
    p=[0.20, 0.40, 0.40],
)

REVENUE_RANGE = {
    "Enterprise": (12_000, 60_000),
    "Mid-Market": (2_500, 12_000),
    "SMB": (400, 2_500),
}

monthly_revenue = np.array(
    [np.random.uniform(*REVENUE_RANGE[s]) for s in SEGMENTS]
).round(2)

# ── Account owners ─────────────────────────────────────────────────────────────
OWNERS = [
    "Sarah Chen",
    "Marcus Rodriguez",
    "Emily Thompson",
    "David Park",
    "Rachel Kim",
    "James Wilson",
    "Priya Patel",
]
account_owner = np.random.choice(OWNERS, size=N)

# ── Behavioural / health metrics ───────────────────────────────────────────────
contract_months_remaining = np.random.randint(1, 25, N)
support_tickets_90d = np.random.randint(0, 16, N)
product_usage_score = np.random.randint(5, 101, N)
days_since_login = np.random.randint(0, 121, N)
nps_score = np.random.randint(-100, 101, N)
payment_delays = np.random.randint(0, 6, N)

# ── Churn label via logistic model + noise ────────────────────────────────────
# Intercept tuned so ~25-30% of customers churn after noise is applied
churn_logit = (
    -0.8
    + 0.5 * np.clip((13 - contract_months_remaining) / 12, 0, 1)   # expiring soon
    + 0.6 * support_tickets_90d / 15                                 # high tickets
    - 0.7 * product_usage_score / 100                                # low usage
    + 0.6 * days_since_login / 120                                   # disengaged
    - 0.4 * (nps_score + 100) / 200                                  # low NPS
    + 0.7 * payment_delays / 5                                       # payment issues
    + np.random.normal(0, 0.6, N)                                    # noise
)
churn_prob_true = 1 / (1 + np.exp(-churn_logit))
churned = (churn_prob_true > 0.50).astype(int)

# ── Company names ──────────────────────────────────────────────────────────────
PREFIXES = [
    "Acme", "Apex", "Atlas", "Aurora", "Axiom", "Azure", "Beacon", "Blue",
    "Bright", "Cedar", "Chrome", "Civic", "Cobalt", "Core", "Crest", "Delta",
    "Digital", "Dynamo", "Eagle", "Echo", "Edge", "Ember", "Empire", "Endeavor",
    "Epic", "Everest", "Falcon", "Fern", "Flex", "Forge", "Frontier", "Fusion",
    "Genesis", "Global", "Granite", "Helios", "Horizon", "Hydra", "Icon",
    "Ignite", "Impact", "Iris", "Iron", "Jolt", "Keystone", "Kinetic",
    "Latitude", "Legacy", "Lumen", "Lynx", "Matrix", "Meridian", "Modus",
    "Nexus", "Nova", "Obsidian", "Onyx", "Orbit", "Origin", "Paragon",
    "Peak", "Pinnacle", "Pivot", "Prism", "Pulse", "Quantum", "Quest",
    "Radiant", "Rally", "Raven", "Relay", "Ridge", "Ripple", "Rocket",
    "Sage", "Sapphire", "Sigma", "Signal", "Solar", "Spark", "Spire",
    "Summit", "Swift", "Synapse", "Talon", "Titan", "Torque", "Traverse",
    "Trek", "Trident", "Unity", "Vector", "Vega", "Vertex", "Vibe",
    "Vista", "Vivid", "Volt", "Vortex", "Warp", "Wave", "Zenith", "Zion",
]
SUFFIXES = [
    "Corp", "Inc", "LLC", "Solutions", "Systems", "Technologies",
    "Group", "Partners", "Global", "Industries", "Ventures", "Works",
]

rng = np.random.default_rng(SEED)
# Generate N unique company names by combining prefixes × suffixes (with replacement to support N > len(PREFIXES))
chosen_prefixes = rng.choice(PREFIXES, size=N, replace=True)
chosen_suffixes = rng.choice(SUFFIXES, size=N, replace=True)
# Ensure uniqueness by appending a counter to any duplicates
raw_names = [f"{p} {s}" for p, s in zip(chosen_prefixes, chosen_suffixes)]
seen: dict[str, int] = {}
customer_names: list[str] = []
for name in raw_names:
    if name not in seen:
        seen[name] = 0
        customer_names.append(name)
    else:
        seen[name] += 1
        customer_names.append(f"{name} {seen[name] + 1}")

# ── Assemble DataFrame ─────────────────────────────────────────────────────────
df = pd.DataFrame(
    {
        "customer_id": [f"CU{str(i + 1).zfill(3)}" for i in range(N)],
        "customer_name": customer_names,
        "segment": SEGMENTS,
        "monthly_revenue": monthly_revenue,
        "contract_months_remaining": contract_months_remaining,
        "support_tickets_90d": support_tickets_90d,
        "product_usage_score": product_usage_score,
        "days_since_login": days_since_login,
        "nps_score": nps_score,
        "payment_delays": payment_delays,
        "account_owner": account_owner,
        "churned": churned,
    }
)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sample_customer_data.csv", index=False)

churn_rate = churned.mean()
print(f"Generated {N} customers -> {churned.sum()} churned ({churn_rate:.1%} churn rate)")
print(f"Segments: {pd.Series(SEGMENTS).value_counts().to_dict()}")
print("Saved: data/sample_customer_data.csv")
