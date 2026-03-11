"""
test_probability_model.py

Unit tests for ProbabilityModel class to verify training, prediction, and evaluation.
"""

import pandas as pd
import numpy as np

from src.models.probability_model import ProbabilityModel


def test_training_and_prediction():
    """Test training and prediction probabilities are within [0, 1]."""
    # small synthetic dataset
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [0.5, 1.5, 2.5, 3.5],
    })
    y = pd.Series([1, 0, 1, 0])
    model = ProbabilityModel()
    model.fit(X, y)
    preds = model.predict_proba(X)
    assert len(preds) == len(y)
    assert np.all(preds >= 0) and np.all(preds <= 1)


def test_evaluation_returns_metrics():
    """Test that evaluation returns accuracy and log loss metrics."""
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([1, 0, 1, 0])
    model = ProbabilityModel()
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    assert "accuracy" in metrics
    assert "log_loss" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["log_loss"] >= 0
