"""
Data loader utilities.

Provides helper functions to load and save datasets for the parlay optimization model. These functions simplify reading from and writing to the project's data directories.
"""

from pathlib import Path
from typing import Union

import pandas as pd

# Import constants from the config module using relative import
from ..config import DATA_RAW_DIR, DATA_PROCESSED_DIR

def load_data(file_name: str, processed: bool = False) -> pd.DataFrame:
    """
    Load a CSV file from the raw or processed data directories.

    Args:
        file_name: The name of the CSV file to load.
        processed: If True, load from the processed data directory; otherwise load from the raw data directory.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    base_dir = DATA_PROCESSED_DIR if processed else DATA_RAW_DIR
    file_path = base_dir / file_name
    return pd.read_csv(file_path)

def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save a DataFrame to the processed data directory as a CSV file.

    Args:
        df: The DataFrame to save.
        file_name: The name of the output CSV file.
    """
    file_path = DATA_PROCESSED_DIR / file_name
    # Ensure the processed directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
