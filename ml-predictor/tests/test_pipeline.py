from __future__ import annotations

import pytest

from churn_app.data import generate_customer_data
from churn_app.pipeline import predict_churn, train_model


def test_synthetic_data_is_reproducible_and_trainable() -> None:
    first = generate_customer_data(300, random_state=7)
    second = generate_customer_data(300, random_state=7)
    assert first.equals(second)

    report = train_model(first, random_state=7)
    assert set(report.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= score <= 1 for score in report.metrics.values())
    assert report.confusion.shape == (2, 2)
    assert report.test_rows == 75


def test_prediction_handles_unknown_categories() -> None:
    report = train_model(generate_customer_data(300))
    customer = generate_customer_data(20, random_state=1000).drop(columns="churned").iloc[[0]]
    customer.loc[:, "contract_type"] = "new-contract"

    probability = predict_churn(report.model, customer)
    assert probability.shape == (1,)
    assert 0 <= probability[0] <= 1


def test_small_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        generate_customer_data(10)