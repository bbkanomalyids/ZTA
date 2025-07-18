from __future__ import annotations

import json
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Deque

import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ── config ----------------------------------------------------------------
from config import KMEANS_PATH, FINAL_FEATURES

try:
    from config import SLM_RECENT_WINDOW_SIZE
except ImportError:
    SLM_RECENT_WINDOW_SIZE = 10

_WIN = SLM_RECENT_WINDOW_SIZE

# --------------------------------------------------------------------------
# 0.  K-means model (loaded once)
# --------------------------------------------------------------------------

def _load_kmeans() -> Any:
    """Load KMeans model via joblib or fallback to load_models."""
    try:
        return joblib_load(str(KMEANS_PATH))
    except Exception:
        from models import load_models  # lazy-import to avoid cycles

        km, _ = load_models(str(KMEANS_PATH), "", FINAL_FEATURES)
        return km

_kmeans = _load_kmeans()

# --------------------------------------------------------------------------
# 1.  feature-block helper
# --------------------------------------------------------------------------

try:
    from slm.prompt_templates import build_feature_block  # type: ignore
except Exception:
    def build_feature_block(row: pd.Series) -> str:  # type: ignore
        """Fallback when prompt_templates is unavailable."""
        return ""

# --------------------------------------------------------------------------
# 2.  recent-cluster stats (in-memory, rolling window)
# --------------------------------------------------------------------------

_recent: Dict[int, Deque[bool]] = defaultdict(lambda: deque(maxlen=_WIN))


def _update_recent_stats(labels: np.ndarray, is_anom: np.ndarray) -> None:
    """Push labels of current chunk into _recent rolling buffers."""
    for lab, anom in zip(labels, is_anom):  # type: ignore
        _recent[int(lab)].append(bool(anom))


def _make_recent_summary(cluster_id: int) -> str:
    """Return 'x/y anomalies last N chunks' or empty if unknown."""
    if cluster_id < 0 or cluster_id not in _recent:
        return ""
    buf = _recent[cluster_id]
    tot = len(buf)
    if tot == 0:
        return ""
    anom = sum(buf)
    return f"{anom}/{tot} anomalies in last {_WIN} chunks"

# --------------------------------------------------------------------------
# 3.  public helpers
# --------------------------------------------------------------------------

def ensure_cluster_fields(df: pd.DataFrame) -> None:
    """
    In-place add/refresh 'cluster_id' and 'cluster_size' columns.

    Call this once per chunk after compute_kmeans_synergy.
    """
    if "cluster_id" in df.columns and "cluster_size" in df.columns:
        return

    from config import KMEANS_FEATURES
    labels = _kmeans.predict(df[KMEANS_FEATURES].to_numpy(float))

    df["cluster_id"] = labels  # type: ignore

    counts = np.bincount(labels, minlength=_kmeans.n_clusters)  # type: ignore
    df["cluster_size"] = counts[labels]  # type: ignore

    # update rolling stats using Isolation-Forest anomaly proxy
    is_anom = df.get("iso_is_anomaly", pd.Series(False, index=df.index)).to_numpy(bool)
    _update_recent_stats(labels, is_anom)


def _format_timestamp(row: pd.Series) -> str:
    # Prefer real timestamp if available
    ts = row.get("timestamp")
    if ts:
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime().astimezone(timezone.utc)
        elif isinstance(ts, datetime):
            dt = ts.astimezone(timezone.utc)
        else:
            return ""
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fallback: use cyclic encoding heuristics
    try:
        sin_val = float(row.get("day_sin", 0))
        cos_val = float(row.get("day_cos", 0))
        angle = np.arctan2(sin_val, cos_val)  # returns radians
        day_idx = int(round(((angle + np.pi) / (2 * np.pi)) * 7)) % 7
        weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_idx]
    except Exception:
        weekday_name = "Unknown day"

    if row.get("is_weekend", 0) == 1:
        return f"Weekend ({weekday_name})"
    else:
        return f"Weekday ({weekday_name})"


def build_ctx(
    df: pd.DataFrame,
    idx: int,
    *,
    sample_id: str,
    tri_label: int,
    tri_confidence: str,
) -> Dict[str, Any]:
    """
    Assemble the dict consumed by DebateEngine / templates.

    Guarantees keys with safe defaults:
      - sample_id
      - feature_block
      - tri_label
      - tri_confidence
      - cluster_id
      - cluster_size
      - recent_cluster_summary
      - timestamp_summary
      - context_log
    """
    row = df.iloc[idx]

    cluster_id = int(row.get("cluster_id", -1))
    cluster_size = int(row.get("cluster_size", -1))

    ctx: Dict[str, Any] = {
        "sample_id": sample_id,
        "feature_block": build_feature_block(row),
        "tri_label": tri_label,
        "tri_confidence": tri_confidence,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "recent_cluster_summary": _make_recent_summary(cluster_id),
        "timestamp_summary": _format_timestamp(row),
        "context_log": "",
    }

    # ── Optional insight: cluster contrast ─────────────────────────────
    insight_lines = []
    try:
        dist = float(row.get("distance_from_centroid", 0))
        dens = float(row.get("cluster_density", 0))
        cluster_diff = ""
        if dist > 0.8 and dens < 1.0:
            cluster_diff = "Sample is far from centroid and in a sparse cluster"
        elif dist > 0.8:
            cluster_diff = "Sample is far from cluster center"
        elif dens < 1.0:
            cluster_diff = "Sample belongs to a sparse cluster"
        if cluster_diff:
            insight_lines.append(f"- Cluster contrast: {cluster_diff}")
    except:
        pass

    # ── Final context block ─────────────────────────────────────────────
    ctx["context_log"] = f"""Context:
- Tri-view label: {tri_label} ({'anomaly' if tri_label == 1 else 'normal'}), confidence: {tri_confidence}
- Cluster ID: {cluster_id} → size: {cluster_size}, density: {row.get('cluster_density', '?')}, dist: {row.get('distance_from_centroid', '?')}
- Time: {ctx['timestamp_summary']}"""

    if insight_lines:
        ctx["context_log"] += "\n" + "\n".join(insight_lines)

    return ctx

# --------------------------------------------------------------------------
# quick smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import pandas as pd
    # Dummy data
    dummy = pd.DataFrame(
        np.random.randn(5, len(FINAL_FEATURES)), columns=FINAL_FEATURES
    )
    ensure_cluster_fields(dummy)
    d = build_ctx(
        dummy, 0, sample_id="0-0", tri_label=1, tri_confidence="83.4"
    )
    print(json.dumps(d, indent=2))
