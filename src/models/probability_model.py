"""
Probability Model

This module implements a logistic regression-based probability model for estimating win probabilities of individual legs in multi-leg combinations. It uses scikit-learn and is structured for easy extension.

The model expects preprocessed numerical features and binary target labels (1 for success/win, 0 for failure/loss). It can be replaced with any classification algorithm.

"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from dataclasses import dataclass, field


@dataclass
class ProbabilityModel:
    """A logistic regression model for estimating probabilities of success for individual legs."""
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(max_iter=1000))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilityModel":
        """Fit the probability model using training features X and target labels y."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict success probabilities for given features."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary outcomes for given features."""
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model on validation data.

        Returns a dictionary with log loss, accuracy, and ROC AUC.
        """
        y_prob = self.predict_proba(X)
        y_pred = self.predict(X)
        return {
            "log_loss": log_loss(y, y_prob),
            "accuracy": accuracy_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_prob),
        }
