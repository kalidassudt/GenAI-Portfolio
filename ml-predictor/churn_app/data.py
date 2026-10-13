"""Synthetic customer data for an offline churn prediction demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customer_data(
    n_samples: int = 600, random_state: int = 42
) -> pd.DataFrame:
    """Generate reproducible customer records with a learnable churn signal."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")

    rng = np.random.default_rng(random_state)
    contract_type = rng.choice(
        ["month-to-month", "annual", "two-year"],
        size=n_samples,
        p=[0.55, 0.30, 0.15],
    )
    tenure_months = rng.integers(1, 73, size=n_samples)
    monthly_charges = np.clip(rng.normal(78, 22, size=n_samples), 20, 150)
    support_tickets = rng.poisson(1.2, size=n_samples)
    paperless_billing = rng.choice(["yes", "no"], size=n_samples)

    churn_log_odds = (
        1.15 * (contract_type == "month-to-month")
        - 0.045 * tenure_months
        + 0.018 * (monthly_charges - 70)
        + 0.38 * support_tickets
        + 0.25 * (paperless_billing == "yes")
        - 0.7
    )
    churn_probability = 1 / (1 + np.exp(-churn_log_odds))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "support_tickets": support_tickets,
            "contract_type": contract_type,
            "paperless_billing": paperless_billing,
            "churned": churned,
        }
    )