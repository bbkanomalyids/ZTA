"""
config.py – single-source parameter hub for the Tri-View IDS
─────────────────────────────────────────────────────────────
Only immutable defaults live here.  Anything dynamic is passed in
at runtime by the caller.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ════════════════════════════════════════════════════════════════════
# helper – env override with type preservation
# ════════════════════════════════════════════════════════════════════
def _env(name: str, default):
    """Read env var and cast to the type of *default*."""
    val = os.getenv(name)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in {"1", "true", "yes"}
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val  # str / Path


# ════════════════════════════════════════════════════════════════════
# 1. Paths
# ════════════════════════════════════════════════════════════════════
BASE_DIR:   Final[Path] = Path(__file__).parent.resolve()
DATA_DIR:   Final[Path] = BASE_DIR / "data"
MODEL_DIR:  Final[Path] = BASE_DIR / "models"
OUTPUT_DIR: Final[Path] = BASE_DIR / "runs"
SLM_DIR:    Final[Path] = BASE_DIR / "slm"

for _p in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _p.mkdir(exist_ok=True)

ONLINE_DATA_PATH: Final[Path] = DATA_DIR  / "online_training_data_filtered.csv"
KMEANS_PATH:      Final[Path] = MODEL_DIR / "kmeans_model.joblib"
RF_PATH:          Final[Path] = MODEL_DIR / "best_offline_rf_model.joblib"
ISO_PATH:         Final[Path] = MODEL_DIR / "isoforest.joblib"
LEAF_NN_PATH:     Final[Path] = MODEL_DIR / "leaf_nn.joblib"

# ════════════════════════════════════════════════════════════════════
# 2. SLM agents & debate knobs
# ════════════════════════════════════════════════════════════════════
SLM_AGENT_PATHS: Final[dict[str, Path]] = {
    "A": SLM_DIR / "agents" / "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf",
    "B": SLM_DIR / "agents" / "phi-2.Q4_K_M.gguf",
    "C": SLM_DIR / "agents" / "gemma-2b-it.Q4_K_M.gguf",
    "D": SLM_DIR / "agents" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
}

SLM_ENABLED:              bool   = _env("SLM_ENABLED", False)
SLM_DEFAULT_GPU_LAYERS:   int    = _env("SLM_GPU_LAYERS", 35)   # 0 ⇒ CPU-only
SLM_TRIGGER_MODE:         str    = _env("SLM_TRIGGER_MODE", "low_conf_or_disagree")  # or "every_sample"
SLM_RF_MIN_AGREE:         float  = _env("SLM_RF_MIN_AGREE", 0.40)
SLM_MAX_SAMPLES_PER_CHUNK:int    = _env("SLM_MAX_SAMPLES", 10)

# Which agents to load – tweak for speed
SLM_ACTIVE_AGENT_IDS: tuple[str, ...] = ("A", "B", "C")
# SLM_ACTIVE_AGENT_IDS: tuple[str, ...] = ("B", "D")
SLM_REFEREE_ENABLED:  bool            = True   # use agent ‘D’ if present

# Per-round generation caps (smaller ⇒ faster)
SLM_MAX_TOKENS_R1   = 512
SLM_MAX_TOKENS_R2   = 512
SLM_MAX_TOKENS_R3   = 512
SLM_MAX_TOKENS_JUDGE= 512

# ════════════════════════════════════════════════════════════════════
# 3. Streaming parameters
# ════════════════════════════════════════════════════════════════════
CHUNK_SIZE:          int   = _env("CHUNK_SIZE", 4_000)
WAIT_TIME_SECONDS:   float = _env("SIM_WAIT_S", 5.0)
CONFIDENCE_THRESH:   float = 0.50

CONSENSUS_RULE:        str  = "3of3"    # "2of3" | "3of3"
CONSENSUS_FALLBACK_P:  int  = 300
WEIGHT_CONF_BY_AGREES: bool = False

# ════════════════════════════════════════════════════════════════════
# 4. ACTER-DG knobs
# ════════════════════════════════════════════════════════════════════
N_MIN_CONFIDENT:       int   = 200
DISAGREE_THRESH:       float = 0.70
DISAGREE_FRACTION:     float = 0.10
DRIFT_TRIGGER_ENABLED: bool  = True
MIN_TREE_ERR:          float = 0.05

ACTER_M:            int | None = None
ACTER_PCTILE:       int | None = 95
ACTER_K_SIGMA:      float      = 1.0
ACTER_MIN_SWAP:     int        = 1
ACTER_MAX_FRAC:     float      = 0.10
RF_REPLAY_K_MIN:    int        = 50   # balanced bootstrap target per class

# ════════════════════════════════════════════════════════════════════
# 5. Isolation-Forest head
# ════════════════════════════════════════════════════════════════════
ISO_METHOD:   str   = "MAD"   # "MAD" | "GAUSS"
ISO_K_MAD:    float = 3.0
ISO_K_GAUSS:  float = 3.0

# ════════════════════════════════════════════════════════════════════
# 6. Leaf-NN sleeve
# ════════════════════════════════════════════════════════════════════
LEAF_DIM:            int   = 1024
NN_CONF_THRESH:      float = 0.40
RETRAIN_NN_EACH_RUN: bool  = False
LEAF_HIDDEN:         int   = 256
LEAF_LR:             float = 1e-3

# ════════════════════════════════════════════════════════════════════
# 7. Replay buffer
# ════════════════════════════════════════════════════════════════════
MEMORY_MAX_PER_CLASS: int = 5_000
REPLAY_EVERY_CHUNKS:  int = 10
REPLAY_BATCH_K:       int = 1_000

# ════════════════════════════════════════════════════════════════════
# 8. Safe-mode watchdog
# ════════════════════════════════════════════════════════════════════
K_PRIOR_WINDOW:        int   = 5
SAFE_MODE_MIN_PRIOR:   float = 0.02
HIGH_DISPERSION_SIGMA: float = 0.15

# ════════════════════════════════════════════════════════════════════
# 9. Counter-example safeguard
# ════════════════════════════════════════════════════════════════════
CE_MAX_PER_CHUNK: int   = 100
CE_WEIGHT:        float = 0.50
CE_IMBALANCE_THR: float = 0.02

# ════════════════════════════════════════════════════════════════════
# 10. Feature schema  ⚠ do NOT reorder ⚠
# ════════════════════════════════════════════════════════════════════
SCALED_NUMERICAL_FEATURES = [
    "src_port", "dst_port", "duration", "max_packets_len", "skewness_packets_len",
    "variance_receiving_packets_len", "skewness_receiving_packets_len",
    "skewness_sending_packets_len", "min_receiving_packets_delta_len",
    "max_receiving_packets_delta_len", "mean_receiving_packets_delta_len",
    "median_receiving_packets_delta_len", "variance_receiving_packets_delta_len",
    "mode_receiving_packets_delta_len",
    "coefficient_of_variation_receiving_packets_delta_len",
    "skewness_receiving_packets_delta_len", "mean_sending_packets_delta_len",
    "median_sending_packets_delta_len", "mode_sending_packets_delta_len",
    "coefficient_of_variation_sending_packets_delta_len",
    "skewness_sending_packets_delta_len",
    "coefficient_of_variation_receiving_packets_delta_time",
    "skewness_sreceiving_packets_delta_time",
    "coefficient_of_variation_sending_packets_delta_time",
    "skewness_sending_packets_delta_time",
]
DNS_FEATURES      = ["dst_ip_freq", "dns_domain_name_freq"]
TEMPORAL_FEATURES = ["is_weekend", "day_sin", "day_cos"]
SYNERGY_FEATURES  = ["cluster_size", "distance_from_centroid", "cluster_density"]

FINAL_FEATURES  = (
    SCALED_NUMERICAL_FEATURES + TEMPORAL_FEATURES + DNS_FEATURES + SYNERGY_FEATURES
)
KMEANS_FEATURES = FINAL_FEATURES[:-len(SYNERGY_FEATURES)]  # everything pre-synergy

# ════════════════════════════════════════════════════════════════════
# 11. Logging
# ════════════════════════════════════════════════════════════════════
LOG_LEVEL:  str = _env("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "[%(asctime)s] %(levelname)s: %(message)s"
LOG_DATEFMT:str = "%Y-%m-%d %H:%M:%S"

# --------------------------------------------------------------------
# public export
# --------------------------------------------------------------------
__all__ = [n for n in globals() if n.isupper()]
