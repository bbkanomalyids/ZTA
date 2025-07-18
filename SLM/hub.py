"""
slm/hub.py ────────────────────────────────────────────────────────────
Single entry-point used by main.py so that all LLM complexity stays
inside *slm/*.

    main.py ──► SLMHub.select_indices(...)  # pick rows to debate
             ─► SLMHub.debate(ctx)          # returns 0 / 1 / None
"""

from __future__ import annotations

import json, logging, datetime as _dt
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .debate_engine import DebateEngine, verify_models_exist

logger = logging.getLogger(__name__)

# ── trace-file placeholder ───────────────────────────────────────────
SLM_DEBATE_TRACE_PATH: Path | None = None


def _log_debate_trace(entry: dict) -> None:
    """Append one JSON-line trace entry for later analysis."""
    try:
        SLM_DEBATE_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SLM_DEBATE_TRACE_PATH.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("[TRACE] Logged SLM decision ➜ %s", SLM_DEBATE_TRACE_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TRACE] failed to log SLM decision – %s", exc)


class SLMHub:
    """Thin façade around DebateEngine with runtime safeguards."""

    # ──────────────────────────────────────────────────────────────
    # constructor
    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        log_dir: Path,
        *,
        trigger_mode: str = "low_conf_or_disagree",
        rf_min_agree: float = 0.6,
        max_per_chunk: int = 10,
        use_referee: bool = True,
    ) -> None:
        verify_models_exist()

        # dynamic trace path (per run)
        global SLM_DEBATE_TRACE_PATH
        SLM_DEBATE_TRACE_PATH = log_dir / "debate_trace.jsonl"

        self.debater = DebateEngine()
        self._mode = trigger_mode
        self._rf_min = rf_min_agree
        self._cap = max_per_chunk
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "SLMHub online – mode=%s, cap=%d, trace=%s",
            self._mode,
            self._cap,
            SLM_DEBATE_TRACE_PATH,
        )

    # ──────────────────────────────────────────────────────────────
    # sample selector (once per chunk)
    # ──────────────────────────────────────────────────────────────
    def select_indices(
        self,
        rf_conf: np.ndarray,   # RF agreement 0-1
        tri_mask: np.ndarray,  # boolean – trusted by Tri-View
    ) -> np.ndarray:
        """Return row indices that should enter debate."""
        if self._mode == "every_sample":
            mask = np.ones_like(rf_conf, dtype=bool)
        else:  # low_conf_or_disagree
            mask = (rf_conf < self._rf_min) | (~tri_mask)

        idx = np.where(mask)[0]
        return idx[: self._cap] if idx.size > self._cap else idx

    # ──────────────────────────────────────────────────────────────
    # debate wrapper  (once per selected sample)
    # ──────────────────────────────────────────────────────────────
    def debate(self, ctx: Dict[str, object]) -> Optional[int]:
        """
        Run 3-round debate (+optional referee).
        Returns new label 0 / 1  or  None on failure.
        """
        try:
            result = self.debater.debate(ctx)             # multi-round SLM
            (self._log_dir / f"{ctx.get('sample_id','na')}.json").write_text(
                json.dumps(result, indent=2)
            )

            summ = result["sample_summary"]
            slm_lbl = summ.get("slm_final_vote")
            ref_lbl = summ.get("referee_label")

            # ── trace every debated sample ──────────────────────
            _log_debate_trace(
                {
                    "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "chunk_id": ctx.get("chunk_id", "unknown"),
                    "index": ctx.get("row_id", -1),
                    "tri_label": ctx.get("tri_label"),
                    "rf_conf": ctx.get("rf_tree_agree"),
                    "mlp_pred": ctx.get("mlp_pred"),
                    "if_pred": ctx.get("if_pred"),
                    "slm_final_vote": slm_lbl,
                    "referee_used": bool(ref_lbl is not None),
                    "referee_label": ref_lbl,
                    "ref_agree_with_rf": (
                        ref_lbl == ctx.get("tri_label") if ref_lbl is not None else None
                    ),
                    "slm_changed_label": (
                        slm_lbl in (0, 1) and slm_lbl != ctx.get("tri_label")
                    ),
                }
            )

            return int(slm_lbl) if slm_lbl in (0, 1) else None

        except Exception as exc:  # noqa: BLE001
            logger.warning("[SLM] debate failed for %s – %s", ctx.get("sample_id"), exc)
            return None
