#!/usr/bin/env python
"""
train_probability_model.py

This script trains a probability model to estimate the likelihood of individual event outcomes.  It loads processed feature matrices and labels from the data/processed directory, fits the ProbabilityModel defined in src/models/probability_model.py, evaluates the model, and saves the trained model and metrics.  All file paths are derived from the configuration in src/config.py.

This script is provided as a template; depending on your data schema you may need to adjust file names, feature columns and label extraction.  The goal is to show how to encapsulate training logic in a repeatable command-line script that fits within the repository's modular structure.
"""

import pandas as pd
import joblib
from pathlib import Path

from src import config
from src.models.probability_model import ProbabilityModel


def main():
    """Run model training and save artefacts."""
    # Determine file paths
    features_path = config.PROCESSED_DATA_DIR / "features.csv"
    labels_path = config.PROCESSED_DATA_DIR / "labels.csv"
    models_dir = config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading features from {features_path}")
    X = pd.read_csv(features_path)
    print(f"Loading labels from {labels_path}")
    y = pd.read_csv(labels_path).squeeze()

    # Initialize and train model
    model = ProbabilityModel()
    print("Fitting probability model...")
    model.fit(X, y)

    # Evaluate on training data (for demonstration)
    metrics = model.evaluate(X, y)
    print(f"Training metrics: {metrics}")

    # Save trained model to disk
    model_output_path = models_dir / "probability_model.pkl"
    joblib.dump(model.model, model_output_path)
    print(f"Trained model saved to {model_output_path}")

    # Optionally save metrics
    metrics_output_path = models_dir / "training_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False)
    print(f"Training metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
