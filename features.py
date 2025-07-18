# features.py

import numpy as np
import pandas as pd
import logging

def compute_kmeans_synergy(
    df_chunk: pd.DataFrame,
    kmeans,
    kmeans_features: list,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute synergy features using a pretrained KMeans model.

    Adds the following columns to df_chunk:
      - cluster_label: Assigned cluster ID
      - distance_from_centroid: Distance to assigned centroid
      - cluster_size: Number of points in the assigned cluster
      - cluster_density: Inverse of average distance to centroid for that cluster

    Args:
        df_chunk: DataFrame containing the current data chunk.
        kmeans: Pretrained sklearn KMeans instance.
        kmeans_features: List of feature names used by KMeans.
        verbose: If True, log cluster distribution for the first chunk.

    Returns:
        DataFrame with added synergy feature columns.
    """
    logger = logging.getLogger(__name__)

    # Extract feature matrix
    X = df_chunk[kmeans_features].astype(float).values

    # Predict cluster labels
    labels = kmeans.predict(X)
    df_chunk['cluster_label'] = labels

    # Compute distances to all centroids
    # kmeans.transform returns distance to each centroid
    dists = kmeans.transform(X)
    assigned_dists = dists[np.arange(len(X)), labels]
    df_chunk['distance_from_centroid'] = assigned_dists

    # Compute cluster sizes
    size_map = pd.Series(labels).value_counts().to_dict()
    df_chunk['cluster_size'] = df_chunk['cluster_label'].map(size_map)

    # Compute cluster_density = inverse of average distance
    avg_map = {
        cluster: assigned_dists[labels == cluster].mean()
        for cluster in size_map
    }
    df_chunk['cluster_density'] = df_chunk['cluster_label'].map(
        lambda c: 1.0 / avg_map[c] if avg_map[c] > 0 else np.inf
    )

    if verbose:
        logger.info(
            f"KMeans cluster distribution: {pd.Series(labels).value_counts().to_dict()}"
        )

    return df_chunk
