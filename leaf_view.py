"""
leaf_view.py
════════════
Leaf-embedding sleeve (Head-B) for the Tri-View streaming IDS
----------------------------------------------------------------
*  RF.apply()  ➜  per-tree leaf indices
*  Feature-hash those indices into a binary bag-of-leaves vector
*  Incrementally train a 1-hidden-layer MLP (solver='sgd') or
   fallback SGD-logistic – both support partial_fit.

Public API
----------
load_leaf_nn(path, dim)         -> nn              (always returns a model)
save_leaf_nn(nn, path)          -> None
extract_leaf_embed(X, rf, dim)  -> ndarray (n, dim)
predict_leaf_nn(nn, Z)          -> (y_hat, proba)
update_leaf_nn(nn, Z, y, …)     -> nn              (partial-fit)
"""

from __future__ import annotations
import logging, joblib, os, tempfile, shutil
from pathlib import Path
from typing   import Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier      # main sleeve
from sklearn.linear_model   import SGDClassifier      # fallback if desired
from sklearn.utils          import shuffle

from config import LEAF_HIDDEN, LEAF_LR

_log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# 1 · I/O helpers
# ────────────────────────────────────────────────────────────────

def load_leaf_nn(path: str, dim: int) -> MLPClassifier:
    """
    Always returns a ready-to-use MLPClassifier.  If path is missing or
    corrupted, a fresh (unfitted) model is created.
    """
    p = Path(path)
    if p.exists():
        try:
            nn = joblib.load(p)
            _log.info("Leaf-NN loaded (%s)", p)
            return nn
        except Exception as exc:
            _log.warning("Could not load Leaf-NN (%s) – rebuilding fresh  [%s]",
                         p, exc)

    _log.info("Leaf-NN not found – initialise fresh model (%s)", p)
    nn = _new_mlp(dim)
    nn._is_fitted = False
    nn._leaf_dim  = dim
    return nn


def save_leaf_nn(nn: MLPClassifier, path: str) -> None:
    """Atomic save (tmp-file ➜ move) so we never leave a half-written file."""
    try:
        path = str(path)
        d    = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=d) as tmp:
            joblib.dump(nn, tmp.name)
        shutil.move(tmp.name, path)
        _log.debug("Leaf-NN checkpoint saved ➜ %s", path)
    except Exception as exc:
        _log.warning("Could not save Leaf-NN: %s", exc)


# ────────────────────────────────────────────────────────────────
# 2 · Leaf embedding
# ────────────────────────────────────────────────────────────────

def _hash_fn(tree_idx: int, leaf_idx: int, dim: int) -> int:
    """Cheap 32-bit mix; replace with xxhash if collisions ever matter."""
    return ((tree_idx * 31) ^ leaf_idx) % dim


def extract_leaf_embed(X: np.ndarray, rf, dim: int = 1024) -> np.ndarray:
    """
    Returns a binary bag-of-leaves embedding  shape = (n_samples, dim).
    """
    leaves = rf.apply(X)                       # (n, n_trees)
    n, T   = leaves.shape
    Z      = np.zeros((n, dim), dtype=np.uint8)

    rows = np.repeat(np.arange(n), T)
    cols = _hash_fn(np.tile(np.arange(T), n), leaves.ravel(), dim)
    Z[rows, cols] = 1
    return Z.astype(np.float32)


# ────────────────────────────────────────────────────────────────
# 3 · Sleeve definition  (MLP by default)
# ────────────────────────────────────────────────────────────────

def _new_mlp(dim: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(LEAF_HIDDEN,),
        activation="relu",
        solver="sgd",              # only solver with partial_fit
        learning_rate_init=LEAF_LR,
        alpha=1e-4,
        batch_size=128,
        max_iter=1,               # 1 epoch per call
        warm_start=True,
        shuffle=True,
        random_state=42,
        verbose=False,
    )

# If you ever revert to a plain SGD-logistic sleeve, just swap the
# _new_mlp call in load_leaf_nn for _new_sgd and everything still works.

def _new_sgd(dim: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        warm_start=True,
        random_state=42,
    )


# ────────────────────────────────────────────────────────────────
# 4 · Predict
# ────────────────────────────────────────────────────────────────

def predict_leaf_nn(nn: MLPClassifier, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe prediction that works *before* the first fit.
    Returns y_hat ∈ {-1,0,1} and confidence P(class==1).
    """
    if not getattr(nn, "_is_fitted", False):
        n = len(Z)
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=np.float32)

    proba = nn.predict_proba(Z)[:, 1]          # P(y==1)
    y_hat = (proba >= 0.5).astype(int)
    return y_hat, proba.astype(np.float32)


# ────────────────────────────────────────────────────────────────
# 5 · Incremental fit (now with *sample_weight* & *verbose*)
# ────────────────────────────────────────────────────────────────

ALL_CLASSES = np.array([0, 1], dtype=int)


def _log_update_stats(z_shape: Tuple[int, int], y: np.ndarray,
                      sample_weight: np.ndarray | None, level: int) -> None:
    """Helper that logs diagnostics for the current update batch."""
    n_samples = z_shape[0]
    counts    = np.bincount(y, minlength=2)
    if sample_weight is not None and sample_weight.size:
        w_min, w_max = float(sample_weight.min()), float(sample_weight.max())
    else:
        w_min = w_max = float('nan')

    _log.log(level, (
        "Leaf-NN update: n=%d | class balance {0:%d,1:%d} | "
        "sample_weight[min=%.3f, max=%.3f]"),
        n_samples, counts[0], counts[1], w_min, w_max)


def update_leaf_nn(
    nn: MLPClassifier | SGDClassifier | None,
    Z:  np.ndarray,
    y:  np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    max_samples:   int | None        = 5_000,
    verbose:       bool              = False,
) -> MLPClassifier | SGDClassifier:
    """
    Incremental (partial_fit) training on the sleeve.

    • Ensures both classes are present in every update  
      (adds a zero-vector dummy of the missing class).  
    • Honors optional *sample_weight* (ignored gracefully if the sklearn
      version / model does not support it).  
    • Sub-samples enormous confident pools for speed.  
    • **verbose** – when *True*, logs diagnostics: sample count, class
      distribution, and min/max sample weights (useful to debug replay &
      counter-example injections).
    """
    if Z.size == 0:
        return nn

    # down-sample huge batches (apply *before* stats to reflect true train set)
    if max_samples and len(Z) > max_samples:
        Z, y, sample_weight = shuffle(
            Z, y, sample_weight, n_samples=max_samples, random_state=0
        )

    # pad missing class if needed
    uniq = np.unique(y)
    if uniq.size == 1:
        missing = 1 - uniq[0]
        Z       = np.vstack([Z, np.zeros((1, Z.shape[1]), dtype=Z.dtype)])
        y       = np.append(y, missing)
        if sample_weight is not None:
            sample_weight = np.append(sample_weight, 1.0)

    # log diagnostics
    log_level = logging.INFO if verbose else logging.DEBUG
    _log_update_stats(Z.shape, y, sample_weight, log_level)

    # initialise model if needed
    if nn is None:
        nn = _new_mlp(Z.shape[1])
        first = True
    else:
        first = not getattr(nn, "_is_fitted", False)

    # kwargs only if model supports them
    kw: dict = {}
    if sample_weight is not None:
        from inspect import signature
        if "sample_weight" in signature(nn.partial_fit).parameters:
            kw["sample_weight"] = sample_weight

    # perform the partial fit
    if first:
        nn.partial_fit(Z, y, classes=ALL_CLASSES, **kw)
        nn._is_fitted = True
    else:
        nn.partial_fit(Z, y, **kw)

    return nn
