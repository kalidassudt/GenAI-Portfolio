"""Streamlit dashboard for the customer churn model."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from churn_app.data import generate_customer_data
from churn_app.pipeline import predict_churn, train_model


@st.cache_resource
def load_demo_model():
    customers = generate_customer_data()
    return customers, train_model(customers)


st.set_page_config(page_title="Churn Signals", layout="wide")
st.title("Customer Churn Signals")
st.caption("Synthetic customer data | Random forest | Local inference")

try:
    customers, report = load_demo_model()
except Exception as error:
    st.error(f"The model could not be trained: {error}")
    st.stop()

metric_columns = st.columns(len(report.metrics))
for column, (metric_name, metric_value) in zip(metric_columns, report.metrics.items()):
    column.metric(metric_name.replace("_", " ").title(), f"{metric_value:.1%}")

st.subheader("Holdout evaluation")
evaluation_columns = st.columns([1, 1.4])
with evaluation_columns[0]:
    st.caption(f"Confusion matrix | {report.test_rows} holdout customers")
    st.dataframe(
        {
            "Actual": ["Stayed", "Churned"],
            "Predicted stayed": report.confusion[:, 0],
            "Predicted churned": report.confusion[:, 1],
        },
        hide_index=True,
        width="stretch",
    )
with evaluation_columns[1]:
    st.caption("Relative feature importance")
    st.bar_chart(report.feature_importance.set_index("feature"), y="importance")

with st.expander("Synthetic dataset preview"):
    st.caption(f"{len(customers)} generated records; seed 42")
    st.dataframe(customers.head(12), hide_index=True, width="stretch")

st.subheader("Score a customer")
with st.form("customer-form"):
    input_columns = st.columns(3)
    tenure = input_columns[0].number_input("Tenure (months)", 1, 120, 12)
    charges = input_columns[1].number_input("Monthly charges", 20.0, 250.0, 84.0)
    tickets = input_columns[2].number_input("Support tickets", 0, 30, 1)
    contract = st.selectbox(
        "Contract type", ["month-to-month", "annual", "two-year"]
    )
    paperless = st.radio("Paperless billing", ["yes", "no"], horizontal=True)
    submitted = st.form_submit_button("Calculate churn risk", type="primary")

if submitted:
    import pandas as pd

    customer = pd.DataFrame(
        [
            {
                "tenure_months": tenure,
                "monthly_charges": charges,
                "support_tickets": tickets,
                "contract_type": contract,
                "paperless_billing": paperless,
            }
        ]
    )
    probability = float(predict_churn(report.model, customer)[0])
    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.progress(probability)
    st.caption("Model output is a demonstration, not a production customer decision.")