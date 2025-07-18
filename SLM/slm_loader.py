"""
slm_loader.py – thin wrapper around llama-cpp-python for local GGUF models
══════════════════════════════════════════════════════════════════════════
Public helpers
--------------
load_agent("A")          → cached Llama instance (A/B/C/D)
load_all_agents()        → dict of all four
verify_models_exist()    → raise early if any path is missing

Config contract
---------------
`config.py` must expose

* BASE_DIR
* SLM_AGENT_PATHS : {"A": Path(...), …}
* SLM_DEFAULT_GPU_LAYERS : int   (optional, default 0)

The caller can still pass custom **kw (n_gpu_layers, chat_format, …).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

from llama_cpp import Llama  # type: ignore

from config import SLM_AGENT_PATHS, SLM_DEFAULT_GPU_LAYERS

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Defaults used for every model (override via **kw in load_agent)
# ----------------------------------------------------------------------
_DEFAULT_KWARGS: Dict[str, object] = {
    "n_ctx": 4096,
    "n_threads": os.cpu_count() or 4,
    "n_gpu_layers": int(SLM_DEFAULT_GPU_LAYERS),  # 0 ⇒ CPU-only
    "chat_format": "chatml",
}

_MODEL_CACHE: dict[str, Llama] = {}      # letter → Llama


# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------
def _load_model(path: Path, **kw) -> Llama:
    full_kw = {**_DEFAULT_KWARGS, **kw}
    logger.info("Loading GGUF ➜ %s  (gpu_layers=%s)",
                path.name, full_kw["n_gpu_layers"])
    return Llama(model_path=str(path), **full_kw)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def load_agent(agent_id: str, **kw) -> Llama:
    """Return cached Llama instance for agent letter A/B/C/D."""
    if agent_id not in ("A", "B", "C", "D"):
        raise ValueError(f"Invalid agent ID '{agent_id}'. Must be A–D.")

    if agent_id in _MODEL_CACHE:
        return _MODEL_CACHE[agent_id]

    try:
        path = Path(SLM_AGENT_PATHS[agent_id]).expanduser()
    except KeyError as exc:
        raise KeyError(f"SLM_AGENT_PATHS missing entry for '{agent_id}'") from exc

    if not path.exists():
        raise FileNotFoundError(f"[SLM {agent_id}] model file not found: {path}")

    _MODEL_CACHE[agent_id] = _load_model(path, **kw)
    return _MODEL_CACHE[agent_id]


def load_all_agents(**kw) -> Dict[str, Llama]:
    """Eager-load and cache all four models; returns dict A→Llama … D→Llama."""
    return {aid: load_agent(aid, **kw) for aid in ("A", "B", "C", "D")}


def verify_models_exist() -> None:
    """Raise immediately if any configured GGUF is missing on disk."""
    missing = [f"[{aid}] {p}" for aid, p in SLM_AGENT_PATHS.items()
               if not Path(p).expanduser().exists()]
    if missing:
        raise FileNotFoundError("Missing GGUF file(s):\n  " + "\n  ".join(missing))


# quick smoke-test -----------------------------------------------------------
if __name__ == "__main__":
    try:
        verify_models_exist()
        logger.info("All GGUF files present ✔")
    except FileNotFoundError as err:
        logger.error(err)
