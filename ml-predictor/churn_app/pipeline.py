"""Model training, evaluation, and single-customer prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets"]
CATEGORICAL_FEATURES = ["contract_type", "paperless_billing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"


@dataclass
class ModelReport:
    """Trained pipeline and holdout evaluation outputs."""

    model: Pipeline
    metrics: dict[str, float]
    confusion: np.ndarray
    feature_importance: pd.DataFrame
    test_rows: int


def train_model(
    customers: pd.DataFrame, random_state: int = 42
) -> ModelReport:
    """Fit a preprocessing and random-forest pipeline and score its holdout."""
    missing = set(FEATURES + [TARGET]) - set(customers.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if customers[TARGET].nunique() != 2:
        raise ValueError("Target column 'churned' must contain both classes")

    features = customers[FEATURES]
    target = customers[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    classifier = model.named_steps["classifier"]
    transformer = model.named_steps["preprocessor"]
    importance = pd.DataFrame(
        {
            "feature": transformer.get_feature_names_out(),
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return ModelReport(
        model=model,
        metrics=metrics,
        confusion=confusion_matrix(y_test, predictions),
        feature_importance=importance,
        test_rows=len(y_test),
    )


def predict_churn(model: Pipeline, customer: pd.DataFrame) -> np.ndarray:
    """Return per-row churn probabilities for customer feature records."""
    missing = set(FEATURES) - set(customer.columns)
    if missing:
        raise ValueError(f"Customer data is missing required columns: {sorted(missing)}")
    return model.predict_proba(customer[FEATURES])[:, 1]