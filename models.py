# models.py
# ═════════════════════════════════════════════════════════════════════════════
#  Model-level helpers shared by the online pipeline
#     • load_models()               – K-Means  +  frozen Weighted-RF
#     • load_isoforest_model()      – Head-C  (unsupervised view)
#     • get_prediction_with_confidence()  – Head-A voting + agreement ratio
# ═════════════════════════════════════════════════════════════════════════════
"""
© 2025 - PhD IDS project
"""
from __future__ import annotations
import logging
from typing import Tuple, List

import joblib
import numpy as np
from scipy.stats import mode                # agreement metric

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Load frozen artefacts
# ────────────────────────────────────────────────────────────────────────────
def load_models(
    kmeans_path: str,
    rf_path: str,
    final_features: List[str],
) -> Tuple[object, object]:
    """
    Load the offline K-Means and Weighted Random-Forest that form the *frozen*
    knowledge base.

    A sanity check verifies that the RF was trained with the same number of
    input features as the online pipeline will provide.

    Returns
    -------
    (kmeans_model, rf_model)
    """
    logger.info("Loading KMeans and Random-Forest models …")
    try:
        kmeans = joblib.load(kmeans_path)
        rf_model = joblib.load(rf_path)
    except Exception as exc:
        logger.exception("Could not load model(s)")
        raise                                # let caller decide what to do

    # ── feature-count sanity check & sklearn warning silencer ───────────────
    if hasattr(rf_model, "feature_names_in_"):   # drop to avoid warnings
        del rf_model.feature_names_in_

    if hasattr(rf_model, "n_features_in_"):
        expected = len(final_features)
        actual   = rf_model.n_features_in_
        if expected != actual:
            logger.warning("RF expects %d features; pipeline provides %d",
                           actual, expected)
        else:
            logger.info("RF feature count matches expected (%d).", expected)

    return kmeans, rf_model


def load_isoforest_model(path: str):
    """Helper for Head-C (Isolation-Forest)."""
    try:
        model = joblib.load(path)
        logger.info("Isolation-Forest model loaded (%s)", path)
        return model
    except Exception as exc:
        logger.exception("Cannot load Isolation-Forest")
        raise


# ────────────────────────────────────────────────────────────────────────────
# 2.  RF majority vote + per-sample agreement
# ────────────────────────────────────────────────────────────────────────────
def get_prediction_with_confidence(
    rf_model,
    X: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    rf_model  : fitted ``RandomForestClassifier`` (Head-A)
    X         : ndarray, shape (n_samples, n_features)
    threshold : float in (0,1].  Agreement ratio ≥ threshold ⇒ keep majority
                vote; otherwise output −1 (uncertain).

    Returns
    -------
    labels       : int ndarray  {0,1,-1}
    agree_ratio  : float ndarray in [0,1]
    """
    # votes.shape == (n_trees, n_samples)
    votes = np.asarray([tree.predict(X) for tree in rf_model.estimators_])

    # use keepdims=True to silence SciPy 1.11 deprecation warning
    mode_res = mode(votes, axis=0, keepdims=True)
    majority = mode_res.mode[0]                 # shape (n_samples,)
    counts   = mode_res.count[0]

    agree_ratio = counts / votes.shape[0]
    labels = np.where(agree_ratio >= threshold, majority, -1).astype(int)

    logger.debug("Agree μ=%.3f σ=%.3f  first5=%s",
                 float(agree_ratio.mean()),
                 float(agree_ratio.std()),
                 np.round(agree_ratio[:5], 3).tolist())
    return labels, agree_ratio
