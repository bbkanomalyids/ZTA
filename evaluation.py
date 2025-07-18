# evaluation.py
"""
Per-chunk & overall metric utilities
────────────────────────────────────
* writes evaluation_metrics.csv
* always renders the pseudo-label distribution plot
* when ACTER diagnostics are supplied:
    – dumps acter_diag.json
    – generates the 3-panel research figure
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from metrics import compute_chunk_metrics, summarize_overall_metrics
from visuals import plot_pseudo_label_distribution, plot_academic_panels


def _gt_stats(y: np.ndarray) -> Dict[str, Any]:
    """Return class counts and imbalance ratio for one chunk."""
    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())
    tot = len(y)
    return {
        "gt_class_0": c0,
        "gt_class_1": c1,
        "gt_balance": c1 / tot if tot else 0.0,  # proportion of anomalies
    }


# ────────────────────────────────────────────────────────────────────
def evaluate_chunks(
    y_trues: List[np.ndarray],
    y_preds: List[np.ndarray],
    counts:  List[Dict[int, int]],
    output_dir: str | Path,
    diag: List[Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Compute & save metrics / plots; optionally dump ACTER diagnostics.

    New columns
    -----------
    gt_class_0, gt_class_1 : true label counts per chunk
    gt_balance             : fraction of class-1 (anomalies)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-chunk metrics + GT stats
    rows: List[Dict[str, Any]] = []
    for idx, (yt, yp) in enumerate(zip(y_trues, y_preds), start=1):
        m = compute_chunk_metrics(yt, yp)
        m.update(chunk_id=idx, num_samples=len(yt), **_gt_stats(yt))
        rows.append(m)

    # 2) overall metrics row (chunk_id == 'ALL')
    all_true = np.concatenate(y_trues)
    all_pred = np.concatenate(y_preds)
    overall  = summarize_overall_metrics(all_true.tolist(), all_pred.tolist())
    overall.update(chunk_id="ALL", num_samples=len(all_true), **_gt_stats(all_true))
    rows.append(overall)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "evaluation_metrics.csv", index=False)

    # 3) pseudo-label distribution plot
    plot_pseudo_label_distribution(
        counts,
        output_path=out_dir / "pseudo_label_distribution.png"
    )

    # 4) optional ACTER diagnostics + 3-panel research figure
    if diag is not None:
        with open(out_dir / "acter_diag.json", "w") as fh:
            json.dump(diag, fh, indent=2)

        plot_academic_panels(
            counts,
            diag,
            output_path=out_dir / "academic_panels.png"
        )

    return df
