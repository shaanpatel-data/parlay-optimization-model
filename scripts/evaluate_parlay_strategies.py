"""
evaluate_parlay_strategies.py

This script loads probabilities and historical results, merges them, and evaluates multi‑leg outcome combinations by generating candidate parlays using the ParlayOptimizer.
It sorts parlays by expected value and prints the top results. Extend this script to compute additional metrics or persist evaluation results.
"""

import pandas as pd

from src.optimizer.multi_leg_optimizer import ParlayOptimizer
from src.config import PROCESSED_DATA_DIR


def main():
    """Run parlay evaluation over historical data."""
    # Paths to processed datasets (adjust as needed)
    probabilities_path = PROCESSED_DATA_DIR / "probabilities.csv"
    results_path = PROCESSED_DATA_DIR / "results.csv"

    # Load predicted probabilities and actual results
    print(f"Loading probabilities from {probabilities_path}")
    probs_df = pd.read_csv(probabilities_path)
    print(f"Loading results from {results_path}")
    results_df = pd.read_csv(results_path)

    # Merge on a common key, e.g., event_id
    if "event_id" in probs_df.columns and "event_id" in results_df.columns:
        merged_df = probs_df.merge(results_df, on="event_id", suffixes=("", "_actual"))
    else:
        raise KeyError("Both probabilities and results datasets must contain an 'event_id' column for merging.")

    # Initialize optimizer and generate parlays
    optimizer = ParlayOptimizer(max_legs=4, min_ev=0.02)
    print("Generating candidate parlays...")
    parlays = optimizer.optimize_parlays(merged_df)

    # Sort parlays by expected value and display a sample
    sorted_parlays = sorted(parlays, key=lambda p: p["expected_value"], reverse=True)
    print(f"Generated {len(sorted_parlays)} candidate parlays. Displaying top 10 by expected value:")
    for parlay in sorted_parlays[:10]:
        print(parlay)

    print("Evaluation complete. Customize this script to compute additional metrics and persist results.")


if __name__ == "__main__":
    main()
