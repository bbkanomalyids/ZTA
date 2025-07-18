import sys
import numpy as np
import pandas as pd
import logging

from config import SYNERGY_FEATURES

def load_online_data(
    path: str,
    final_features: list,
    drop_label: bool = False
) -> pd.DataFrame:
    """
    Load and preprocess online data from CSV.

    Only initial features (excluding synergy) are expected in the raw file.

    Args:
        path: Path to the preprocessed, filtered CSV file.
        final_features: List of final feature names (including synergy features).
        drop_label: If True, drop the 'label' column to simulate unlabeled inference.

    Returns:
        DataFrame with all initial model features and 'label'.
    """
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_csv(path, on_bad_lines='skip', engine='c')  # much faster
        logger.info(f"Loaded online dataset => shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading online data {path}: {e}")
        sys.exit(1)

    # Replace infinite values and impute missing
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Determine initial features (exclude synergy)
    initial_feats = set(final_features) - set(SYNERGY_FEATURES)
    expected = initial_feats | {'label'}
    missing = expected - set(df.columns)
    if missing:
        for col in missing:
            df[col] = 0.0
        logger.warning(f"Added missing initial columns: {missing}")

    if drop_label and 'label' in df.columns:
        df.drop(columns=['label'], inplace=True)
        logger.info("Dropped 'label' column for unlabeled inference.")

    return df


def split_into_chunks(
    df: pd.DataFrame,
    chunk_size: int
) -> list[pd.DataFrame]:
    """
    Split a DataFrame into a list of smaller DataFrames of size chunk_size.

    Args:
        df: Input DataFrame to split.
        chunk_size: Number of rows per chunk.

    Returns:
        List of DataFrame chunks.
    """
    logger = logging.getLogger(__name__)
    chunks = [df.iloc[i: i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    logger.info(f"Data split into {len(chunks)} chunks (chunk size = {chunk_size}).")
    return chunks
