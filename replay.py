# replay.py ────────────────────────────────────────────────────────────
"""
Replay helpers shared by the MLP sleeve and ACTER.

• ReplayBuffer  – FIFO per-class buffer (stores ANY NumPy vectors).
• balanced_bootstrap – returns a bootstrap sample that contains at
  least k_min minority samples, borrowing from the buffer when possible.

If the replayed rows don’t match the feature dimension of the current
confident pool, we fall back to duplicating existing minority rows so
the function never raises.
"""
from __future__ import annotations
from collections import deque
from typing import Tuple

import numpy as np
from numpy.random import default_rng

from config import RF_REPLAY_K_MIN   # add this constant to config.py


class ReplayBuffer:
    """
    FIFO ring buffer keyed by class label.

    Parameters
    ----------
    dim : int
        Number of columns each stored vector must have.
    max_per_class : int
        Capacity per class (older rows are dropped first).
    """
    def __init__(self, dim: int, max_per_class: int):
        self.dim = dim
        self.buffers = {0: deque(maxlen=max_per_class),
                        1: deque(maxlen=max_per_class)}

    # ── writers ───────────────────────────────────────────────────────
    def add(self, Z: np.ndarray, y: np.ndarray) -> int:
        """Append rows to the buffer class-wise."""
        for cls in (0, 1):
            self.buffers[cls].extend(Z[y == cls])
        return len(Z)

    # ── stats ─────────────────────────────────────────────────────────
    def count(self, cls: int) -> int: return len(self.buffers[cls])
    def total(self) -> int:           return self.count(0) + self.count(1)

    # ── readers ───────────────────────────────────────────────────────
    def sample(self, cls: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Up to *k* random rows of the requested class.
        Returns empty arrays if buffer is empty.
        """
        n = min(k, self.count(cls))
        if n == 0:
            return (np.empty((0, self.dim), np.float32),
                    np.empty(0, int))
        idx = np.random.choice(self.count(cls), n, replace=False)
        Z   = np.stack([self.buffers[cls][i] for i in idx])
        y   = np.full(n, cls, int)
        return Z.astype(np.float32), y

    def sample_balanced(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """≈k rows, half per class (best-effort)."""
        half = k // 2
        Z0, y0 = self.sample(0, half)
        Z1, y1 = self.sample(1, half)
        if not (Z0.size or Z1.size):
            return np.empty((0, self.dim), np.empty(0, int))
        Z = np.vstack([Z0, Z1]) if Z0.size and Z1.size else (Z0 if Z0.size else Z1)
        y = np.hstack([y0, y1])
        return Z.astype(np.float32), y


# ─────────────────────────────────────────────────────────────────────
#  Balanced bootstrap for ACTER
# ─────────────────────────────────────────────────────────────────────
def balanced_bootstrap(
    X_conf: np.ndarray,
    y_conf: np.ndarray,
    replay: ReplayBuffer,
    *,
    k_min: int | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment the confident pool so it contains at least *k_min* rows
    of **each** class, borrowing from *replay* when shapes match or
    duplicating existing minority rows otherwise. Returns a bootstrap
    sample drawn with replacement from the augmented pool.

    Parameters
    ----------
    X_conf, y_conf : current confident subset.
    replay         : shared ReplayBuffer.
    k_min          : desired minimum per class (defaults to config).
    rng            : optional NumPy Generator.
    """
    rng = default_rng() if rng is None else rng
    k_min = RF_REPLAY_K_MIN if k_min is None else k_min

    X_pool, y_pool = X_conf, y_conf
    feat_dim = X_conf.shape[1]

    for cls in (0, 1):
        need = max(0, k_min - (y_pool == cls).sum())
        if need == 0:
            continue

        # try replay first
        Z_rep, y_rep = replay.sample(cls, need)
        if Z_rep.size and Z_rep.shape[1] == feat_dim:
            X_pool = np.vstack([X_pool, Z_rep])
            y_pool = np.hstack([y_pool, y_rep])
            continue  # satisfied with replay rows

        # fall back: duplicate existing minority rows
        dup_rows = X_conf[y_conf == cls]
        if dup_rows.size:
            dup_idx = rng.choice(len(dup_rows), need, replace=True)
            X_pool = np.vstack([X_pool, dup_rows[dup_idx]])
            y_pool = np.hstack([y_pool, np.full(need, cls, int)])

    boot_idx = rng.choice(len(X_pool), size=len(X_pool), replace=True)
    return X_pool[boot_idx], y_pool[boot_idx]
