"""
unsup.py  –  Isolation-Forest helper (Head-C)
----------------------------------------------

score_chunk(df, iso_model)  ->
    returns a DataFrame that contains
        • iso_score       (float, higher = more anomalous)
        • iso_is_anomaly  (int 0/1)
    with the *same index* as the input df.

It **does not** modify df in-place – this avoids pandas
SettingWithCopy warnings when the caller passes a slice / view.
"""
from __future__ import annotations
import logging, joblib, numpy as np, pandas as pd
from pathlib    import Path
from config     import ISO_PATH, ISO_METHOD, ISO_K_MAD, ISO_K_GAUSS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_isoforest_model(path: str | Path = ISO_PATH):
    """Load the pretrained Isolation-Forest (raise if missing)."""
    try:
        model = joblib.load(path)
        logger.info("Isolation-Forest model loaded (%s)", path)
        return model
    except Exception as exc:
        logger.error("Could not load Isolation-Forest: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# chunk scorer
# ─────────────────────────────────────────────────────────────────────────────
def score_chunk(features: pd.DataFrame, iso_model) -> pd.DataFrame:
    """
    Parameters
    ----------
    features   : DataFrame with *numeric* columns only – the SAME set of
                 (scaled) FINAL_FEATURES used to train the IF.
    iso_model  : fitted sklearn IsolationForest

    Returns
    -------
    DataFrame [index = features.index] with two columns:
        iso_score        (float)
        iso_is_anomaly   (int {0,1})
    """
    # 1) raw anomaly scores (sklearn: higher score = more normal, so invert)
    X       = features.to_numpy(dtype=float, copy=False)
    scores  = -1.0 * iso_model.decision_function(X)   # higher -> more anomalous

    # 2) robust threshold
    if ISO_METHOD.upper() == "MAD":
        med   = float(np.median(scores))
        mad   = float(np.median(np.abs(scores - med))) + 1e-9
        thresh = med + ISO_K_MAD * mad
    else:  # "GAUSS"
        mu    = float(scores.mean())
        sigma = float(scores.std()) + 1e-9
        thresh = mu + ISO_K_GAUSS * sigma

    flags = (scores >= thresh).astype(int)

    return pd.DataFrame(
        {"iso_score": scores, "iso_is_anomaly": flags},
        index=features.index
    )
