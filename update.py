"""
update.py – ACTER-DG with replay-balanced bootstraps
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier

from replay import ReplayBuffer, balanced_bootstrap
from config import (
    FINAL_FEATURES,
    # gates & guards
    N_MIN_CONFIDENT, DISAGREE_FRACTION, DRIFT_TRIGGER_ENABLED, MIN_TREE_ERR,
    # adaptive breadth knobs
    ACTER_M, ACTER_PCTILE, ACTER_K_SIGMA, ACTER_MIN_SWAP, ACTER_MAX_FRAC,
    # bootstrap knob
    RF_REPLAY_K_MIN,
)

logger = logging.getLogger(__name__)

# ───────────────────────── helpers ──────────────────────────
def _per_sample_agreement(votes: np.ndarray) -> np.ndarray:
    m = mode(votes, axis=0, keepdims=False)
    return m.count / votes.shape[0]


def _decide_replacements(err: np.ndarray) -> Tuple[np.ndarray, float]:
    n_trees = len(err)
    thr = (np.percentile(err, ACTER_PCTILE)
           if ACTER_PCTILE is not None
           else err.mean() + ACTER_K_SIGMA * err.std(ddof=0))
    thr = max(thr, MIN_TREE_ERR)

    cand = np.where(err >= thr)[0]
    m_dyn = int(np.clip(
        cand.size,
        ACTER_MIN_SWAP,
        max(ACTER_MIN_SWAP, int(round(ACTER_MAX_FRAC * n_trees)))
    ))

    if m_dyn == 0:
        return np.empty(0, int), thr

    worst = cand[np.argsort(err[cand])[-m_dyn:]]
    return worst, thr

# ───────────────────────── main ─────────────────────────────
def acter_update(
    forest,
    chunk,
    pseudo_labels: np.ndarray,
    *,
    replay: ReplayBuffer,
    M: int | None,
    disagree_thresh: float,
    min_tree_err: float = MIN_TREE_ERR,
    allow_adapt: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "triggered": False, "gate1_pass": False, "gate2_pass": False,
        "n_conf": 0, "mean_agree": np.nan, "std_agree": np.nan,
        "frac_high_disagree": np.nan, "disagree_thresh": disagree_thresh,
        "replaced_idx": [], "replaced_err": [], "thr_error": np.nan,
        "new_tree_class_dist": {"0": 0, "1": 0}, "boot_from_replay": 0,
    }

    if not DRIFT_TRIGGER_ENABLED or not allow_adapt:
        return forest, diag

    conf_mask = pseudo_labels != -1
    n_conf = int(conf_mask.sum())
    diag["n_conf"] = n_conf
    diag["gate1_pass"] = n_conf >= N_MIN_CONFIDENT
    if not diag["gate1_pass"]:
        return forest, diag

    X_conf = chunk.loc[conf_mask, FINAL_FEATURES].astype(float).values
    y_conf = pseudo_labels[conf_mask]

    votes = np.stack([t.predict(X_conf) for t in forest.estimators_])
    agree = _per_sample_agreement(votes)
    diag["mean_agree"], diag["std_agree"] = float(agree.mean()), float(agree.std())

    frac_hi = float((agree <= disagree_thresh).sum()) / n_conf
    diag["frac_high_disagree"] = frac_hi
    diag["gate2_pass"] = frac_hi >= DISAGREE_FRACTION
    if not diag["gate2_pass"]:
        return forest, diag

    tree_err = np.mean(votes != y_conf, axis=1)
    if tree_err.max() < min_tree_err:
        return forest, diag

    # decide replacements ───────────────────────────────────────────
    worst_idx, thr_used = (_decide_replacements(tree_err)
                           if (M is None or M <= 0)
                           else (np.argsort(tree_err)[-M:], tree_err[np.argsort(tree_err)[-M:][0]]))
    if worst_idx.size == 0:
        return forest, diag

    diag.update({
        "triggered": True, "thr_error": float(thr_used),
        "replaced_idx": worst_idx.tolist(),
        "replaced_err": tree_err[worst_idx].round(4).tolist(),
    })

    logger.info("Replacing %d tree(s) %s | chunk-errors=%s",
                len(worst_idx), diag["replaced_idx"], diag["replaced_err"])

    # build replacement trees ───────────────────────────────────────
    base_params = forest.estimators_[0].get_params()
    rng = np.random.default_rng()

    # we record stats from the FIRST bootstrap (all trees use same pool size)
    first_stats_recorded = False

    for slot in worst_idx:
        X_boot, y_boot = balanced_bootstrap(
            X_conf, y_conf, replay,
            k_min=RF_REPLAY_K_MIN,
            rng=rng,
        )
        t_new = DecisionTreeClassifier(**base_params)
        t_new.fit(X_boot, y_boot)
        forest.estimators_[slot] = t_new

        if not first_stats_recorded:
            c0, c1 = np.bincount(y_boot, minlength=2)[:2]
            from_replay = int(len(y_boot) > len(y_conf))
            diag["new_tree_class_dist"] = {"0": int(c0), "1": int(c1)}
            diag["boot_from_replay"] = from_replay
            first_stats_recorded = True

    return forest, diag
