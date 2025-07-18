"""
prompt_templates.py – per-agent, per-round blue-prints
────────────────────────────────────────────────────────────────────────
A single generic prompt confused the four small models.  We now provide
dedicated templates that respect each model family’s native chat / 
instruction style:

    A  TinyLlama-1.1B-Chat     → Zephyr tags <|system|> / <|user|> …
    B  Phi-2                   → “Instruct: … Output:” format
    C  Gemma-2B-It             → <start_of_turn>user … <start_of_turn>model
    D  Mistral-7B-Instruct     → [INST] … [/INST]

Each ROUND (1-3) plus the referee (=0) has a template per agent.
"""
from __future__ import annotations
from textwrap import dedent
from typing import Dict, Iterable
import json
from textwrap import dedent

# ────────────────────── feature helpers ───────────────────────────────
SELECTED_FEATURES_R1 = [
    "dst_ip_freq",
    "dns_domain_name_freq",
    "duration",
    "max_packets_len",
]



def build_feature_block(row: Dict[str, float] | "pd.Series",
                        feats: Iterable[str] = SELECTED_FEATURES_R1) -> str:
    return ",\n".join(f'"{k}": {json.dumps(row[k])}' for k in feats if k in row)


# ────────────────────── strict fill helper ────────────────────────────
def fill_template(tmpl: str, **kw) -> str:
    try:
        return tmpl.format(**kw)
    except KeyError as e:
        raise KeyError(f"Missing placeholder {e}") from None

# ────────────────────── TEMPLATE REGISTRY ─────────────────────────────
# short aliases for agent letters
_TINY, _PHI, _GEM, _MIS = "A", "B", "C", "D"

# ---------- Round-1 ---------------------------------------------------
R1_TEMPLATE = {
    _TINY: dedent("""\
    <|system|>
    
    You are an anomaly-detection assistant. Output ONLY one JSON object, and nothing else.

    Required keys:
    - "label": 0 or 1
    - "self_conf": float between 0 and 1
    - "rationale": Explain your reasoning for the label by interpreting the features. Do not repeat key_features — explain what they imply.
    - "key_features": list of 3–5 strings like "dst_ip_freq: high", not a dictionary
    - "cluster_novel": string (e.g, "yes", "no")
    - "cluster_reason": string (e.g. "low density", "unusual size")
    - "alt_hypothesis": possible benign explanation (e.g. "burst scan", "false positive")
    - "attack_ref": known attack type (e.g. "DDoS" ..etc ) or write "unknown"

    ✱ All values must be valid JSON.
    ✱ Do NOT use booleans, nulls, or nested structures.
    ✱ Do NOT add comments, explanations, or headers.
    ✱ Begin your response DIRECTLY with the opening brace: {{
    
    <|user|>
    Input:

    {feature_block}

    {context_log}

    **Return ONLY the JSON object — DO NOT copy or repeat the Required keys **
    
    <|assistant|>"""),

# Phi-2
    _PHI: dedent("""\
    Instruct:
    You are an anomaly-detection assistant. Your task is to output exactly one valid JSON object.

    Use exactly the following flat keys (in any order):
    - "label": 0 or 1
    - "self_conf": float between 0 and 1
    - "rationale": Short explanation of whether you agree with the tri-view label, and why.
    - "key_features": comma-separated string of 3–5 signals like "dst_ip_freq: high"
    - "cluster_novel": short string like "yes", "no", or a brief reason
    - "cluster_reason": short reason (e.g. "low density", "shifted center")
    - "alt_hypothesis": short benign explanation (e.g. "burst scan", "false positive")
    - "attack_ref": known attack type (e.g. "DDoS", "DNS tunnel") or say "unknown"

    Strict instructions:
    - Do NOT include markdown, comments, or formatting.
    - Do NOT include lists, booleans, or null values.
    - Respond with ONE valid JSON object only. No text or explanation outside the object.

    Input features:
    {feature_block}

    {context_log}
    Output:
    """),

# Gemma
    _GEM: dedent("""\
    <start_of_turn>user
    You are an anomaly-detection assistant. Return ONLY one valid JSON object.

    Instructions:
    - Use exactly these flat keys (any order):
      "label", "self_conf", "rationale", "key_features",
      "cluster_novel", "cluster_reason", "alt_hypothesis", "attack_ref"
    - All keys must have valid values:
      * "label": 0 or 1 (integer)
      * "self_conf": float between 0 and 1
      * "rationale": short explanation using 1–2 features
      * "key_features": string — 3–5 comma-separated phrases like "dst_ip_freq: high"
      * "cluster_novel": string — "yes", "no", or brief reason
      * "cluster_reason": lowercase string like "low density" or "shifted center"
      * "alt_hypothesis": short benign reason or "unknown"
      * "attack_ref": known attack type or "unknown"

    Strict rules:
    - Do NOT use booleans, nulls, markdown, or nested structures
    - Output MUST be valid JSON only — no prose, no headers
    - Output MUST start with '{{' and end with '}}' — nothing before or after

    Input features:
    {feature_block}

    {context_log}
    <end_of_turn>
    <start_of_turn>model"""),

# Referee – Mistral (Agent D)
    _MIS: dedent("""\
    [INST]
    You are an anomaly-detection assistant. Output exactly one valid JSON object.

    Instructions:
    - Include exactly these keys (flat): "label", "self_conf", "rationale", "key_features", "cluster_novel", "cluster_reason", "alt_hypothesis", "attack_ref"
    - All keys must have valid values
    - Use no booleans, nulls, or nested objects
    - Output must begin with "{{" and end with "}}"
    - No markdown, no prose, no explanation outside JSON

    Input features:
    {feature_block}

    Context:
    Tri-view label = {tri_label}, confidence = {tri_confidence}%
    Cluster ID = {cluster_id}, size = {cluster_size}
    History = {recent_cluster_summary}
    Time = {timestamp_summary}
    [/INST]"""),

}

# ---------- Round-2 ---------------------------------------------------
R2_TEMPLATE = {
    a: t.replace("Round-2-MARKER", "")   # placeholder removal hack
    for a, t in {
        _TINY: dedent("""\
        <|system|>
        You are an anomaly-detection peer reviewer.
        Your task is to critically review your own previous response (R1) after reading peer predictions. You must:

        - Compare your original label and confidence with peer outputs
        - Explain if and why your original reasoning still holds
        - Decide whether to keep or revise your label
        - Update your confidence
        - Propose a clarifying question that could help disambiguate similar future cases

        Respond with exactly one valid JSON object using the following schema:

        {{
          "self_conf_before": float,
          "peer_scores": float,            # average peer confidence only
          "critique": string,
          "revise_label": {{
            "changed": true or false,
            "new_label": 0 or 1,
            "reason": string (**why** you changed or stayed with the label)
          }},
          "self_conf_after": float,
          "clarifying_question": string
        }}

        REQUIREMENTS:
        - All values must be present and non-null
        - Do not include markdown, explanations, or formatting fences
        - Output ONLY the JSON object
        </|system|>

        <|user|>
        Your R1:
        {your_r1_json}

        Peers:
        {peer_json_list}

        Now re-evaluate your label considering the peer responses.
        Respond with **exactly one valid JSON object using the above schema** 

        </|user|>
        <|assistant|>"""),

_PHI: dedent("""\
Instruct:
You are an anomaly-detection peer reviewer. Your job is to critically evaluate your original R1 decision based on peer responses.

Follow these steps:
1. Compare your label and confidence with peer predictions and rationales.
2. Reflect on whether your original reasoning still holds.
3. Decide if you should revise your label or keep it.
4. Update your confidence accordingly.
5. Suggest a clarifying question that would help disambiguate future cases.

Respond ONLY with a valid JSON object using this exact schema:

{{
  "self_conf_before": float,         # your original R1 confidence
  "peer_scores": float,              # average confidence of peers
  "critique": string,                # your reflection on peer rationales
  "changed": true or false,          # whether you changed your label
  "new_label": 0 or 1,               # your new or retained label
  "reason": string,                  # explain your decision
  "self_conf_after": float,          # your updated confidence
  "clarifying_question": string      # helpful question to reduce ambiguity
}}

STRICT RULES:
- Output only one valid JSON object
- Use correct types for all fields — no nulls, markdown, or formatting
- Do NOT include extra keys like "label", "self_conf", or any nested objects

Your R1:
{your_r1_json}

Peers:
{peer_json_list}

Now reflect and respond with one valid JSON object:
Output:
"""),

        _GEM: dedent("""\
        <start_of_turn>user
        You are an anomaly-detection peer-reviewer.

        Your task is to reflect on your original Round-1 response after reading what your peers predicted.

        Tasks:
        1. Compare your original label and confidence with peers.
        2. Assess whether your original rationale still holds.
        3. Decide to keep or revise your label.
        4. Update your confidence score.
        5. Pose one clarifying question to reduce ambiguity in future cases.

        Respond with exactly one valid JSON object using these keys (in any order):

        - "self_conf_before": float
        - "peer_scores": float
        - "critique": string (compare your rationale to peers')
        - "changed": true or false
        - "new_label": 0 or 1
        - "reason": string (why you changed or stayed)
        - "self_conf_after": float
        - "clarifying_question": string

        STRICT RULES:
        ✱ Output must begin with '{{' and end with '}}'
        ✱ Do NOT use markdown, code, or explanatory prose
        ✱ Do NOT include extra keys, nulls, or nested objects

        ---

        Your Round-1 JSON:
        {your_r1_json}

        Peer Round-1 JSONs:
        {peer_json_list}
        
        <end_of_turn>
        
        <start_of_turn>model"""),

_MIS: dedent("""\
        [INST]
        You are an anomaly-detection peer reviewer.

        **Your goal**  
        Re-evaluate your own Round-1 answer (shown below) *after* reading peer outputs.

        **Steps**  
        1. Compare your original label & confidence with peer results.  
        2. Decide whether to keep or revise your label.  
        3. Update your confidence.  
        4. Briefly justify your decision.  
        5. Pose one clarifying question that would help resolve similar cases in future.

        **Respond with exactly ONE valid JSON object** using this schema (no other keys, any order):

        {{
          "self_conf_before": float,          // your Round-1 confidence
          "peer_scores":      float,          // average peer confidence
          "critique":         string,         // short comparison of your output vs peers
          "revise_label":     string,         // "A", "B", or "C"
          "changed":          true | false,   // boolean
          "new_label":        0 | 1,          // integer
          "reason":           string,         // why you kept/revised
          "self_conf_after":  float,          // new confidence
          "clarifying_question": string       // one question to disambiguate future cases
        }}

        **Strict rules**  
        - Output **only** the JSON object (no prose, code, or markdown).  
        - Do **not** add nulls, nested objects, or extra keys.  
        - Booleans must be unquoted `true` / `false`.  
        - Begin with {{ and end with }}.

        ---
        Your Round-1 JSON:  
        {your_r1_json}

        Peer Round-1 JSON objects:  
        {peer_json_list}
        [/INST]"""),

}.items()
}

# ── Round-3 (senior-analyst clarification) templates ──────────────────
R3_TEMPLATE = {

    # TinyLlama-1.1B-Chat  (Zephyr format) ────────────────────────────
    _TINY: dedent("""\
    <|system|>
    You are a senior anomaly analyst.  Output **one** valid JSON object only.

    Schema:
    {{
      "need_more_info": true | false,   // boolean
      "questions": ["string", ...]      // up to 3 brief clarifying questions (not objects!)
    }}

    Rules:
    • Begin with "{{" and end with "}}".  No markdown, no prose.
    • Do not return list of objects. Questions must be plain strings.
    • If need_more_info == false ➜ return an empty list.

    Context:
    • Your Round-1 JSON: {your_r1_json}
    • Your Round-2 JSON: {your_r2_json}

    === Peer Round-2 Summaries ===
    {peer_r2_block}

    </|system|>
    <|user|>Provide your follow-up JSON now.<|assistant|>"""),

# Phi-2  (Instruct / Output pattern) ──────────────────────────────
    _PHI: dedent("""\
    Instruct:
    You are a senior anomaly analyst. Your task is to return one valid JSON object with the following fields:

    - "need_more_info": either true or false
    - "questions": a list of 1 to 3 short strings if more info is needed, otherwise an empty list

    Rules:
    - Output ONLY a valid JSON object, with no explanation or formatting.
    - Do NOT include nulls, booleans as strings, or extra fields.
    - If "need_more_info" is false, then "questions" must be an empty list.
    - The response must start with '{{' and end with '}}' — nothing else.

    Your R1 output:
    {your_r1_json}

    Your R2 output:
    {your_r2_json}

    === Peer Round-2 summaries ===
    {peer_r2_block}

    Output:
    """),

# Gemma-2B-IT  (native <start_of_turn> format) ────────────────────
    _GEM: dedent("""\
    <start_of_turn>user
    You are a senior anomaly analyst. Your task is to return exactly one valid JSON object.

    Instructions:
    - Use only the following two keys:
      "need_more_info": true or false
      "questions": a list of up to 3 short strings
    - If "need_more_info" is false, then "questions" must be an empty list.
    - Do not include any other keys, nulls, or explanations.
    - Output must start with '{{' and end with '}}' — no extra formatting or comments.

    Your R1 output:
    {your_r1_json}

    Your R2 output:
    {your_r2_json}

    === Peer Round-2 summaries ===
    {peer_r2_block}

    Output:
    <end_of_turn>
    <start_of_turn>model"""),

# Mistral-7B-Instruct v0.2  (INST blocks) ─────────────────────────
    _MIS: dedent("""\
        [INST]
        You are a senior anomaly analyst.

        Produce ONE valid JSON object with this schema:
        {{
          "need_more_info": true | false,
          "questions": ["string", ...]   // up to 3 brief clarifying questions
        }}

        Rules:
        * No prose, no markdown.
        * If no extra info is needed, set need_more_info to false and use an empty list.

        === Context ===
        Your Round-1 JSON: {your_r1_json}
        Your Round-2 JSON: {your_r2_json}
        
        === Peer Round-2 Summaries ===
        {peer_r2_block}

        [/INST]"""),
}

# ── Referee (Round 0) templates – final verdict  ──────────────────────
# Unified schema all agents must output:
# {
#   "final_label":   0 | 1,       # integer
#   "confidence":    float,       # 0–1
#   "winning_agent": "A" | "B" | "C" | "tie",
#   "reason":        "string"     # short justification
# }

REF_TEMPLATE = {

    # TinyLlama-1.1B-Chat  (Zephyr format) ────────────────────────────
    _TINY: dedent("""\
        <|system|>
        You are the referee.  Output **one** valid JSON object only.

        Schema:
        {{
          "final_label":   0 | 1,
          "confidence":    float,      // 0–1
          "winning_agent": "A" | "B" | "C" | "tie",
          "reason":        "string"    // brief justification
        }}

        Rules:
        • Begin with "{" and end with "}".  No prose, markdown, or code blocks.
        • No extra keys, nulls, or nested objects.

        Round-1 JSON for all agents:
        {all_agent_json[R1]}

        Round-2 JSON for all agents:
        {all_agent_json[R2]}

        Round-3 JSON for all agents:
        {all_agent_json[R3]}
        </|system|>
        <|user|>Provide the final verdict JSON now.<|assistant|>"""),

    # Phi-2  (Instruct / Output) ──────────────────────────────────────
    _PHI: dedent("""\
        Instruct:
        You are the referee. Choose the winning agent (A, B, C, or "tie") and output exactly ONE JSON object:

        {{
          "final_label":   0 | 1,
          "confidence":    float,      // 0-1
          "winning_agent": "A" | "B" | "C" | "tie",
          "reason":        "string"
        }}

        - No markdown, no extra keys.
        - If agents disagree, use majority vote; break ties by higher confidence.

        === Round-1 JSON ===
        {all_agent_json[R1]}

        === Round-2 JSON ===
        {all_agent_json[R2]}

        === Round-3 JSON ===
        {all_agent_json[R3]}

        Output:"""),

    # Gemma-2B-IT  (native chat format) ───────────────────────────────
    _GEM: dedent("""\
        <start_of_turn>user
        You are the referee. Produce a verdict JSON with this schema:
        {{
          "final_label":   0 | 1,
          "confidence":    float,
          "winning_agent": "A" | "B" | "C" | "tie",
          "reason":        "string"
        }}

        • Output only JSON. No extra keys or prose.

        R1 JSON:
        {all_agent_json[R1]}

        R2 JSON:
        {all_agent_json[R2]}

        R3 JSON:
        {all_agent_json[R3]}
        <end_of_turn>
        <start_of_turn>model"""),

    # Mistral-7B-Instruct v0.2  (INST) ────────────────────────────────
    _MIS: dedent("""\
    [INST]
    You are the final referee for an anomaly-detection debate.

    Return one JSON object that follows *exactly* this schema:

    {{    # ← braces escaped so .format() doesn’t break
      "final_label":   0 | 1,
      "confidence":    float,            # 0.0–1.0
      "winning_agent": "A" | "B" | "C" | "tie",
      "reason":        "string"          # concise justification
    }}

    Rules:
    * Output **only** the JSON object (no markdown or extra prose).
    * Decide by majority label vote; break ties using highest average confidence.
    * If no single agent clearly dominates, set `"winning_agent": "tie"`.
    * The confidence you output should reflect your certainty in the **label**.
    * The reason must justify the winning agent **or** explain the tie.

    {round1_summary}

    {round2_summary}

    {round3_summary}
    
    [/INST]"""),

}

# ────────────────────── public helper ────────────────────────────────
def get_template(agent_id: str, round_no: int) -> str:
    """
    round_no: 1,2,3  or 0 for referee
    """
    key = agent_id.upper()
    if round_no == 1:
        return R1_TEMPLATE[key]
    if round_no == 2:
        return R2_TEMPLATE[key]
    if round_no == 3:
        return R3_TEMPLATE[key]
    if round_no == 0:
        return REF_TEMPLATE[key]
    raise ValueError("round_no must be 0,1,2,3")

# ---------------------------------------------------------------------
__all__ = [
    "SELECTED_FEATURES_R1", "build_feature_block", "fill_template", "get_template"
]
