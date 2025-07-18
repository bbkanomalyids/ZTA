# slm/debate_engine.py
"""
DebateEngine – agent-aware multi-round orchestrator
────────────────────────────────────────────────────
Rounds
  1  independent hypotheses
  2  peer critique / optional label change
  3  senior-analyst clarification
  0  referee (optional)

Template selection is delegated to `prompt_templates.get_template(agent, round)`.
The engine therefore works with heterogeneous prompt formats (TinyLlama,
Phi-2, Gemma, Mistral) without any branching in this file.

Compatible with llama-cpp-python 0.1 and ≥0.2.
"""
from __future__ import annotations
import tempfile
import json, re, logging
from json import JSONDecodeError
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Global DEBUG config here
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from config import (
    SLM_ACTIVE_AGENT_IDS,
    SLM_DEFAULT_GPU_LAYERS,
    SLM_MAX_TOKENS_R1,
    SLM_MAX_TOKENS_R2,
    SLM_MAX_TOKENS_R3,
    SLM_MAX_TOKENS_JUDGE,
    SLM_REFEREE_ENABLED,
)

from .slm_loader import load_agent, verify_models_exist
from .prompt_templates import get_template, fill_template


# ───────────────────────── static knobs ────────────────────────────────
AGENT_IDS: tuple[str, ...] = tuple(SLM_ACTIVE_AGENT_IDS)
REFEREE_ID = "D"                     # judge model (optional)
FALLBACK_R2_JSON = {"revise_label": {"changed": False,
                                     "new_label": None,
                                     "reason": ""}}
_JSON_FENCE = re.compile(r"```json(.*?)```", re.S | re.I)

# ───────────────────────── tiny helpers ────────────────────────────────
def _extract_json_snippet(txt: str) -> str:
    """
    Best-effort extraction of a JSON object from model output.
    1. prefer ```json … ``` fences
    2. otherwise take substring between first '{' … last '}'
    """
    m = _JSON_FENCE.search(txt)
    if m:
        return m.group(1).strip()
    start, end = txt.find("{"), txt.rfind("}")
    return txt[start:end + 1] if 0 <= start < end else ""

def clean_json_output(raw: str) -> str:
    """
    Extract the first JSON object and repair common glitches.
    • Closes missing } or ].
    • Removes stray newline escapes and stray schema lines.
    """

    # 0️⃣ normalise stray escapes:  dst\_ip  → dst_ip
    raw = raw.replace("\\_", "_")

    # 1️⃣ locate first '{'
    start = raw.find("{")
    if start == -1:
        raise ValueError("No opening '{' found")

    depth, in_string, esc = 0, False, False
    end = None
    for i, ch in enumerate(raw[start:], start):
        if in_string:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    # fallback to last '}' if truncated
    if end is None:
        end = raw.rfind("}") + 1
        if end <= start:
            raise ValueError("No matching '}' found")

    json_str = raw[start:end]

    # 2️⃣ insert missing commas between lines
    json_str = re.sub(r'([}\]"\d])\s*\n\s*(")', r'\1,\n\2', json_str)
    # 3️⃣ remove trailing commas
    json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
    # 4️⃣ true/false literals
    json_str = json_str.replace('"true"', 'true').replace('"false"', 'false')
    # 5️⃣ close unbalanced braces/brackets
    if json_str.count("{") > json_str.count("}"):
        json_str += "}"
    if json_str.count("[") > json_str.count("]"):
        json_str += "]"

    # 6️⃣ strip lines that still contain schema pipes or comments
    json_str = "\n".join(
        ln for ln in json_str.splitlines()
        if "|" not in ln and "//" not in ln
    )

    return json_str


def format_peer_r2_block(peer_list: list[dict]) -> str:
    """
    Turn peer_r2_json_list (list of dicts) into an easy-to-read block.

    Example output:

    Agent B  (label=1, conf=0.80)
      • Critique: ...text...
      • Question: ...text...

    Agent C  (label=0, conf=0.40)
      • Critique: ...
      • Question: ...
    """
    lines = []
    for p in peer_list:
        lab  = p.get("label", "–")
        conf = p.get("confidence", "–")
        lines.append(f"Agent {p['agent']}  (label={lab}, conf={conf})")
        crit = p.get("critique")
        if crit:
            lines.append(f"  • Critique: {crit}")
        q   = p.get("clarifying_question")
        if q:
            lines.append(f"  • Question: {q}")
        lines.append("")            # blank line between agents
    return "\n".join(lines).rstrip()

def escape_curly_braces(text: str) -> str:
    """
    Escapes curly braces so they don't interfere with Python .format().
    Used to safely inject raw JSON strings into templates.
    """
    return text.replace("{", "{{").replace("}", "}}")


def _safe_float(val, default=0.0) -> float:
    """Return a clean float from int/float/list/str; fallback to default."""
    if isinstance(val, (int, float)):
        return float(val)
    # [0.6]            -> 0.6
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], (int, float, str)):
        return _safe_float(val[0], default)
    # [{"agent":..}]   -> default
    return default



def _safe_json(txt: str, aid: str, fb: dict | None = None) -> dict:
    """
    Parse model text to JSON with resilience.
    • Repairs commas/braces (clean_json_output)
    • Converts numeric lists → float
    • Collapses newlines inside quoted strings
    """
    if not txt.strip():
        logger.warning("[SLM %s] EMPTY output!", aid)
        return fb.copy() if fb else {}

    try:
        cleaned = clean_json_output(txt)

        # strip any leftover schema/comment lines (pipes //)
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines()
            if "|" not in ln and "//" not in ln
        )

        parsed = json.loads(cleaned)

    except JSONDecodeError as e:
        # retry only if invalid control character (newline) inside quotes
        if "Invalid control character" in str(e):
            fixed = re.sub(r'("(?:[^"\\]|\\.)*)[\r\n]+((?:[^"\\]|\\.)*")',
                           r'\1 \2',
                           cleaned)
            try:
                parsed = json.loads(fixed)
            except Exception:
                logger.warning("[SLM %s] JSON retry failed", aid)
                return fb.copy() if fb else {}
        else:
            logger.warning("[SLM %s] JSON parse failed: %s", aid, e)
            return fb.copy() if fb else {}

    except Exception as exc:
        logger.warning("[SLM %s] JSON parse failed: %s | first 200 char → %s",
                       aid, exc, txt[:200])
        return fb.copy() if fb else {}

    # normalise numeric fields
    for key in ("confidence", "self_conf", "self_conf_after", "self_conf_before"):
        if key in parsed:
            parsed[key] = _safe_float(parsed[key], 0.0)

    logger.debug("[SLM %s] Parsed keys: %s", aid, list(parsed.keys()))
    return parsed

# ───────────────────────── core class ──────────────────────────────────
class DebateEngine:
    """
    One instance can be reused for the whole pipeline run.
    """

    def __init__(self, gpu_layers: int | None = None) -> None:
        verify_models_exist()
        self._models: dict[str, Any] = {}
        self._gpu_layers = int(gpu_layers or SLM_DEFAULT_GPU_LAYERS)

    def _call_llm(
        self,
        aid: str,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,   # ← NEW optional stop tokens
    ) -> str:
        """
        Dispatch prompt to the requested SLM and return raw text.
        If `stop` is given, it is forwarded to the model so generation halts
        when any of those strings is produced (e.g. stop=["}"] for referee).
        """
        if aid not in (*AGENT_IDS, REFEREE_ID):
            raise ValueError(f"Unknown agent id {aid!r}")

        if aid not in self._models:
            self._models[aid] = load_agent(aid, gpu_layers=self._gpu_layers)

        llm = self._models[aid]

        # ── emulate `--file` input for llama-cpp ─────────────────────
        with tempfile.NamedTemporaryFile(
            "w+", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write(prompt)
            f.flush()
            f.seek(0)
            file_prompt = f.read()

        try:
            logger.debug("[SLM %s] PROMPT START\n%s\n---", aid, file_prompt)

            # deterministic generation (temperature = 0.0)
            resp = llm(
                file_prompt,
                max_tokens=max_tokens,
                stop=stop,                # ← forward stop list
                echo=False,
                temperature=0.0,
            )

            txt = resp if isinstance(resp, str) else resp["choices"][0]["text"]
            txt = txt.strip()

            # extract JSON substring (first '{' .. last '}')
            if "{" in txt and "}" in txt:
                txt = "{" + txt.split("{", 1)[1]
                txt = txt.rsplit("}", 1)[0] + "}"
            else:
                logger.warning(
                    "[SLM %s] Output did not contain JSON braces:\n---\n%s\n---",
                    aid, txt
                )

            logger.debug("[SLM %s] RAW OUTPUT (trimmed):\n%s\n---", aid, txt)
            return txt

        except Exception as exc:
            logger.error("[SLM %s] LLM failure: %s", aid, exc)
            return "{}"

    # ─── debate rounds ────────────────────────────────────────────────
    def _round(self, n: int, ctx: dict,
               prev_r1: dict | None = None, prev_r2: dict | None = None) -> dict[str, dict]:
        """Generic round executor (n = 1,2,3)."""
        results: dict[str, dict] = {}

        # ── safety checks ────────────────────────────────────────────────
        if n == 2:
            valid = [a for a in AGENT_IDS if a in prev_r1 and prev_r1[a]]
            if not valid:  # none valid ⇒ skip
                logger.warning("⚠️ Round-2 skipped; no valid R1s")
                return {a: {} for a in AGENT_IDS}

        if n == 3:
            miss_r1 = [a for a in AGENT_IDS if a not in prev_r1 or not prev_r1[a]]
            miss_r2 = [a for a in AGENT_IDS if a not in prev_r2 or not prev_r2[a]]
            if miss_r1 or miss_r2:
                logger.warning("⚠️ Round-3 skipped; missing R1/R2s: %s + %s", miss_r1, miss_r2)
                return {a: {} for a in AGENT_IDS}

        # ── main agent loop ─────────────────────────────────────────────
        for aid in AGENT_IDS:
            logger.debug("[SLM %s] ▶ Starting Round-%d", aid, n)
            tmpl = get_template(aid, n)

            # ---------- Round-2 ----------------------------------------
            if n == 2:
                your_r1 = prev_r1.get(aid, {})
                peer_list = [
                    {
                        "agent": a,
                        "label": prev_r1[a].get("label"),
                        "confidence": prev_r1[a].get("self_conf"),
                        "rationale": prev_r1[a].get("rationale"),
                    }
                    for a in AGENT_IDS if a != aid
                ]
                prompt = fill_template(
                    tmpl,
                    **ctx,
                    your_r1_json=json.dumps(your_r1),
                    peer_json_list=json.dumps(peer_list)
                )

            # ---------- Round-3 ----------------------------------------
            elif n == 3:
                your_r1 = prev_r1.get(aid, {})
                your_r2 = prev_r2.get(aid, {})

                peer_r2_list = [
                    {
                        "agent": a,
                        "label": prev_r2[a].get("new_label",
                                                prev_r2[a].get("label")),
                        "confidence": prev_r2[a].get("self_conf_after",
                                                     prev_r2[a].get("self_conf_before")),
                        "critique": prev_r2[a].get("critique"),
                        "clarifying_question": prev_r2[a].get("clarifying_question")
                    }
                    for a in AGENT_IDS if a != aid
                ]

                peer_r2_block = format_peer_r2_block(peer_r2_list)

                prompt = fill_template(
                    tmpl,
                    **ctx,
                    your_r1_json=json.dumps(your_r1),
                    your_r2_json=json.dumps(your_r2),
                    peer_r2_block=peer_r2_block  # ← new placeholder

                )

            # ---------- Round-1 ----------------------------------------
            else:  # n == 1
                prompt = fill_template(tmpl, **ctx)

            # ── run model ───────────────────────────────────────────────
            mxtok = {1: SLM_MAX_TOKENS_R1,
                     2: SLM_MAX_TOKENS_R2,
                     3: SLM_MAX_TOKENS_R3}[n]

            txt = self._call_llm(aid, prompt, max_tokens=mxtok)
            fb = FALLBACK_R2_JSON if n == 2 else {}
            parsed = _safe_json(txt, aid, fb)
            results[aid] = parsed or {}

            # ── logging ────────────────────────────────────────────────
            if results[aid]:
                logger.info("[SLM %s] ✅ Round-%d JSON parsed.", aid, n)
                if n == 1:
                    label = results[aid].get("label", "⚠️ missing")
                    conf = results[aid].get("self_conf", "⚠️ missing")
                    logger.info("[SLM %s] ➤ Hypothesis submitted: label=%s conf=%s", aid, label, conf)
            else:
                logger.warning("[SLM %s] ❌ Round-%d returned empty or invalid JSON.", aid, n)

        return results

    def _referee(self, ctx: dict,
                 r1: dict, r2: dict, r3: dict) -> dict:
        """
        Run the referee (Round 0) if enabled and return its verdict JSON.
        Formats R1–R3 outputs into clear per-agent blocks for easier parsing.
        """
        if not SLM_REFEREE_ENABLED:
            return {}

        aid = REFEREE_ID
        tmpl = get_template(aid, 0)

        def format_agent_json_block(title: str, round_data: dict) -> str:
            return f"=== {title} ===\n" + "\n\n".join(
                f"Agent {agent}:\n{json.dumps(data, indent=2)}"
                for agent, data in round_data.items()
            )

        # Format each round as readable per-agent blocks
        r1_block = format_agent_json_block("Round-1", r1)
        r2_block = format_agent_json_block("Round-2", r2)
        r3_block = format_agent_json_block("Round-3", r3)

        # Inject these into the prompt
        prompt = fill_template(
            tmpl,
            round1_summary=r1_block,
            round2_summary=r2_block,
            round3_summary=r3_block,
            **ctx
        )

        logger.debug("[SLM %s] ▶ Starting Referee Round", aid)

        txt = self._call_llm(
            aid,
            prompt,
            max_tokens=SLM_MAX_TOKENS_JUDGE
        )

        parsed = _safe_json(txt, aid)
        if not parsed:
            parsed = {"_raw_output": txt}  # keep for later inspection
        return parsed

    # ─── public API ───────────────────────────────────────────────────
    def debate(self, ctx: dict) -> dict:
        """
        ctx must include:
            sample_id, feature_block, tri_label, tri_confidence
        """
        logger.info("──────────────[ SAMPLE START: %s ]──────────────", ctx.get("sample_id", "unknown"))

        r1 = self._round(1, ctx)
        r2 = self._round(2, ctx, prev_r1=r1)
        r3 = self._round(3, ctx, prev_r1=r1, prev_r2=r2)
        judge = self._referee(ctx, r1, r2, r3)

        # Stage-4
        if judge:
            final = judge.get("final_label", ctx["tri_label"])
            conf = _safe_float(judge.get("confidence", 0.5), 0.5)
        else:
            votes = Counter(a.get("label") for a in r1.values())
            final, _ = votes.most_common(1)[0]
            conf = max(votes.values()) / len(AGENT_IDS)

        # compute each agent's average perceived peer confidence
        peer_mu = {}
        for aid in AGENT_IDS:
            score = r2[aid].get("peer_scores")
            if isinstance(score, dict):  # legacy format
                score = sum(score.values()) / max(len(score), 1)
            peer_mu[aid] = _safe_float(score, 0.0)

        dom = max(peer_mu, key=peer_mu.get, default="")
        least = min(peer_mu, key=peer_mu.get, default="")

        return {
            "sample_summary": {
                "sample_id": ctx["sample_id"],
                "tri_view_label": ctx["tri_label"],
                "slm_final_vote": final,
                "confidence": conf,
                "referee_label": judge.get("final_label") if judge else None
            },
            "debate_dynamics": {
                "dominant_agent": dom,
                "least_agent": least,
            },
            "_rounds": {"R1": r1, "R2": r2, "R3": r3, "judge": judge},
        }


# ─── smoke-test ───────────────────────────────────────────────────────
if __name__ == "__main__":                # pragma: no cover
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    eng = DebateEngine(gpu_layers=0)
    dummy_ctx = dict(
        sample_id      = "demo-0",
        feature_block  = "duration=12, cluster_density=0.001, is_weekend=0",
        tri_label      = 1,
        tri_confidence = "80.0",
        cluster_id     = 5,
        cluster_size   = 17,
        recent_cluster_summary = "",
        timestamp_summary      = "",
    )
    print(json.dumps(eng.debate(dummy_ctx), indent=2))
