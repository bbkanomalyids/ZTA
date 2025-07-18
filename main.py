#!/usr/bin/env python3
"""
main.py – Online Tri-View IDS  (SLM hook + full telemetry)
────────────────────────────────────────────────────────────────────────
Core pipeline identical to the original Script-1; SLM logic is sandboxed
in slm/* and never breaks the main run if it’s missing or disabled.
"""
from __future__ import annotations

import argparse, logging, sys, time, warnings, json
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
from tqdm import tqdm

# ── cfg -----------------------------------------------------------------
from config import (                                                # noqa: E501
    ONLINE_DATA_PATH, KMEANS_PATH, RF_PATH, ISO_PATH, OUTPUT_DIR,
    FINAL_FEATURES, KMEANS_FEATURES,
    CHUNK_SIZE, WAIT_TIME_SECONDS,
    CONFIDENCE_THRESH, DISAGREE_THRESH, NN_CONF_THRESH,
    CONSENSUS_FALLBACK_P, MIN_TREE_ERR,
    MEMORY_MAX_PER_CLASS, REPLAY_EVERY_CHUNKS, REPLAY_BATCH_K,
    CE_MAX_PER_CHUNK, CE_WEIGHT, CE_IMBALANCE_THR,
    LEAF_DIM, LEAF_NN_PATH, RETRAIN_NN_EACH_RUN,
    LOG_FORMAT, LOG_DATEFMT, LOG_LEVEL,
    SLM_ENABLED, SLM_TRIGGER_MODE, SLM_RF_MIN_AGREE, SLM_MAX_SAMPLES_PER_CHUNK,
)
try:
    from config import HIGH_DISPERSION_SIGMA                 # optional
except ImportError:
    HIGH_DISPERSION_SIGMA = 0.15

# ── local modules -------------------------------------------------------
from data     import load_online_data, split_into_chunks
from features import compute_kmeans_synergy
from models   import load_models, load_isoforest_model, get_prediction_with_confidence
from unsup    import score_chunk
from leaf_view import extract_leaf_embed, load_leaf_nn, predict_leaf_nn, update_leaf_nn, save_leaf_nn
from replay   import ReplayBuffer
from update   import acter_update
from evaluation import evaluate_chunks

# ── optional SLM hub & context -----------------------------------------
import logging
logging.getLogger().setLevel(logging.DEBUG)

HAVE_SLM_CTX = False
try:
    if SLM_ENABLED:
        from slm.hub     import SLMHub
        from slm.context import ensure_cluster_fields, build_ctx
        HAVE_SLM_CTX = True
except Exception as e:                                            # noqa: BLE001
    print(f"[SLM] import failed – SLM disabled ({e})", file=sys.stderr)
    SLM_ENABLED = False

# ═════════════════════════ CLI / logging ═══════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Tri-View streaming IDS")
    p.add_argument("--wait", type=float, default=WAIT_TIME_SECONDS)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--log-level", default=LOG_LEVEL)
    p.add_argument("--fixed-M", type=int, default=None,
                   help="force fixed #tree swaps (override adaptive)")
    p.add_argument("--slm", action="store_true",
                   help="enable SLM debate even if disabled in config")
    p.add_argument("--no-academic-panels", action="store_true",
                   help="skip large PDF evaluation panels (faster)")
    return p.parse_args()


def setup_logging(run_dir: Path, lvl: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, lvl.upper()),
                        format=LOG_FORMAT, datefmt=LOG_DATEFMT,
                        handlers=[logging.FileHandler(run_dir / "run.log"),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)
    return logging.getLogger("ONLINE")


def strip_feature_names(forest: Any) -> None:
    for t in forest.estimators_:
        if hasattr(t, "feature_names_in_"):
            del t.feature_names_in_

# ═════════════════════════ main loop ══════════════════════════════════
def run_pipeline(args: argparse.Namespace) -> None:
    run_id  = datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    lg      = setup_logging(run_dir, args.log_level)
    lg.info("Online pipeline started – %s", run_id)

    # artefacts ---------------------------------------------------------
    kmeans, rf = load_models(str(KMEANS_PATH), str(RF_PATH), FINAL_FEATURES)
    strip_feature_names(rf)
    iso_model  = load_isoforest_model(str(ISO_PATH))
    leaf_nn    = None if RETRAIN_NN_EACH_RUN else load_leaf_nn(str(LEAF_NN_PATH), LEAF_DIM)

    # buffers -----------------------------------------------------------
    mem_leaf = ReplayBuffer(LEAF_DIM,            MEMORY_MAX_PER_CLASS)
    mem_feat = ReplayBuffer(len(FINAL_FEATURES), MEMORY_MAX_PER_CLASS)

    # SLM hub -----------------------------------------------------------
    slm: SLMHub | None = None
    if (SLM_ENABLED or args.slm) and HAVE_SLM_CTX:
        slm = SLMHub(
            run_dir / "slm_logs",
            trigger_mode  = SLM_TRIGGER_MODE,
            rf_min_agree  = SLM_RF_MIN_AGREE,
            max_per_chunk = SLM_MAX_SAMPLES_PER_CHUNK,
        )

    # stream ------------------------------------------------------------
    df     = load_online_data(str(ONLINE_DATA_PATH), FINAL_FEATURES, drop_label=False)
    chunks = split_into_chunks(df, CHUNK_SIZE)

    ewma_prior, consecutive_os = 0.5, 0
    y_true_all: List[np.ndarray] = []
    y_pred_all:  List[np.ndarray] = []
    label_hist, diag_hist = [], []

    for c_idx, chunk in enumerate(tqdm(chunks, desc="Chunks"), start=1):
        try:
            lg.info("Chunk %d/%d – n=%d", c_idx, len(chunks), len(chunk))

            # 1 ▸ synergy & cluster fields ---------------------------------
            chunk = compute_kmeans_synergy(chunk, kmeans, KMEANS_FEATURES, verbose=(c_idx == 1))
            if HAVE_SLM_CTX:
                ensure_cluster_fields(chunk)

            # 2 ▸ ISO-Forest ----------------------------------------------
            iso_df = score_chunk(chunk[FINAL_FEATURES], iso_model)
            chunk[["iso_score", "iso_is_anomaly"]] = iso_df

            # 3 ▸ RF -------------------------------------------------------
            X = chunk[FINAL_FEATURES].to_numpy(float)
            rf_lbl, rf_agree = get_prediction_with_confidence(rf, X, CONFIDENCE_THRESH)
            chunk["pseudo_label"] = rf_lbl
            rf_conf = rf_agree / len(rf.estimators_)

            # 4 ▸ Leaf-NN --------------------------------------------------
            Z = extract_leaf_embed(X, rf, LEAF_DIM)
            if (leaf_nn is None or not getattr(leaf_nn, "_is_fitted", False)) and (rf_lbl != -1).any():
                leaf_nn = update_leaf_nn(leaf_nn, Z[rf_lbl != -1], rf_lbl[rf_lbl != -1], verbose=True)
            nn_pred, nn_conf = predict_leaf_nn(leaf_nn, Z)
            chunk[["nn_pred", "nn_conf"]] = np.c_[nn_pred, nn_conf]

            # 5 ▸ tri-view consensus --------------------------------------
            rf_ok = rf_lbl != -1
            iso_m = chunk["iso_is_anomaly"].to_numpy() == rf_lbl
            nn_m  = (nn_pred == rf_lbl) & (nn_conf >= NN_CONF_THRESH)
            all3  = rf_ok & iso_m & nn_m
            tri_m, mode = (all3, "ALL-3") if all3.sum() >= CONSENSUS_FALLBACK_P \
                          else (rf_ok & (iso_m | nn_m), "2-of-3")

            # 6 ▸ safe-mode prior -----------------------------------------
            prior = (rf_lbl == 1).mean()
            ewma_prior = 0.8*ewma_prior + 0.2*prior
            one_sided  = ewma_prior <= CE_IMBALANCE_THR or ewma_prior >= 1-CE_IMBALANCE_THR
            consecutive_os = consecutive_os + 1 if one_sided else 0
            safe_mode = consecutive_os >= 5
            if safe_mode:
                lg.warning("⛔  Safe-mode ON")

            # 7 ▸ counter-example safeguard ------------------------------
            if safe_mode:
                miss = 0 if ewma_prior > 0.98 else 1
                if mem_leaf.count(miss) < CE_IMBALANCE_THR * mem_leaf.count(1-miss):
                    sel = np.argsort(chunk["iso_score"].values)
                    idx_ce = sel[:CE_MAX_PER_CHUNK] if miss == 0 else sel[::-1][:CE_MAX_PER_CHUNK]
                    if idx_ce.size:
                        Z_ce, X_ce = Z[idx_ce], X[idx_ce]
                        y_ce = np.full(len(idx_ce), miss, int)
                        leaf_nn = update_leaf_nn(leaf_nn, Z_ce, y_ce,
                                                 sample_weight=np.full(len(y_ce), CE_WEIGHT),
                                                 verbose=True)
                        mem_leaf.add(Z_ce, y_ce);  mem_feat.add(X_ce, y_ce)
                        lg.info("✚ injected %d counter-examples", len(idx_ce))

            # 8 ▸ ACTER-DG ----------------------------------------------
            dyn_thr = max(0.30, min(0.90, DISAGREE_THRESH + 0.3*(ewma_prior-0.5)))
            allow   = (not safe_mode) or rf_agree.std() > HIGH_DISPERSION_SIGMA
            tri_lbl = rf_lbl.copy();  tri_lbl[~tri_m] = -1
            rf, diag = acter_update(
                rf, chunk, tri_lbl,
                replay=mem_feat, M=args.fixed_M,
                disagree_thresh=dyn_thr,
                allow_adapt=allow, min_tree_err=MIN_TREE_ERR,
            )
            strip_feature_names(rf)

            # ––– ACTER logs (legacy) –––––––––––––––––––––––––––––––––––
            if diag["triggered"]:
                dist = diag.get("new_tree_class_dist", {})
                lg.info("ACTER: replaced %d trees | boot balanced (0:%d / 1:%d)%s",
                        len(diag["replaced_idx"]), dist.get("0", 0), dist.get("1", 0),
                        " (+replay)" if diag.get("boot_from_replay") else "")
            lg.info("μ_agree=%.3f  σ_agree=%.3f  high-disagree=%.2f%%",
                    diag.get("mean_agree", float("nan")),
                    diag.get("std_agree",  float("nan")),
                    100 * diag.get("frac_high_disagree", 0.0))

            # 9 ▸ sleeve / memory -------------------------------------
            if all3.any():
                leaf_nn = update_leaf_nn(leaf_nn, Z[all3], rf_lbl[all3], verbose=True)
                mem_leaf.add(Z[all3], rf_lbl[all3]);  mem_feat.add(X[all3], rf_lbl[all3])

            # 10 ▸ MLP replay ----------------------------------------
            if REPLAY_BATCH_K and c_idx % REPLAY_EVERY_CHUNKS == 0 and mem_leaf.total():
                Z_r, y_r = mem_leaf.sample_balanced(REPLAY_BATCH_K)
                if len(Z_r):
                    leaf_nn = update_leaf_nn(leaf_nn, Z_r, y_r, verbose=True)

            # 11 ▸ SLM debate hook -----------------------------------
            debated = 0
            if slm:
                want = slm.select_indices(rf_conf, tri_m)
                for i in want:
                    # ▸▸▸ Reason logging
                    reason = "low_conf" if rf_conf[i] < slm._rf_min else "disagree"
                    lg.debug(f"SLM triggered for sample {i} (chunk {c_idx}): {reason}")

                    ctx = build_ctx(
                        chunk, int(i),
                        sample_id=f"{c_idx}-{i}",
                        tri_label=int(rf_lbl[i]),
                        tri_confidence=f"{rf_conf[i] * 100:.1f}",
                    )

                    # ▸▸▸ dynamic trace path JSONL file ▸▸▸
                    ctx.update(chunk_id=f"chunk_{c_idx}", row_id=int(i),
                               rf_tree_agree=float(rf_conf[i]),
                               mlp_pred=int(nn_pred[i]),
                               if_pred=int(chunk["iso_is_anomaly"].iloc[i]))

                    new_lbl = slm.debate(ctx)
                    if new_lbl is not None and new_lbl != rf_lbl[i]:
                        mem_leaf.add(Z[i:i + 1], np.array([new_lbl]))
                        mem_feat.add(X[i:i + 1], np.array([new_lbl]))
                        lg.info("SLM flip injected (%s → %d)", ctx["sample_id"], new_lbl)
                    debated += 1
                lg.info("SLM-debate: %d sample(s) this chunk", debated)

            # 12 ▸ headline -----------------------------------------
            lg.info(
                "ISO/RF %.2f%% | NN/RF %.2f%% | all-3 %.2f%% (%d) | "
                "tri %d/%d [%s] | mem %d (1:%d 0:%d) / %d | prior %.3f",
                100 * iso_m.mean(),
                100 * ((nn_pred == rf_lbl) & (nn_conf >= NN_CONF_THRESH)).mean(),
                100 * all3.mean(), int(all3.sum()),
                int(tri_m.sum()), len(chunk), mode,
                mem_leaf.total(), mem_leaf.count(1), mem_leaf.count(0),
                2 * MEMORY_MAX_PER_CLASS,
                ewma_prior,
            )

            # 13 ▸ bookkeeping --------------------------------------
            y_true_all.append(chunk["label"].to_numpy());  y_pred_all.append(rf_lbl)
            label_hist.append(dict(zip(*np.unique(rf_lbl, return_counts=True))))
            diag_hist.append({**diag, "chunk": c_idx})

            time.sleep(args.wait)

        except Exception:
            lg.exception("Error in chunk %d – continuing", c_idx)

    # save / eval -------------------------------------------------------
    save_leaf_nn(leaf_nn, str(LEAF_NN_PATH))
    evaluate_chunks(
        y_true_all, y_pred_all, label_hist,
        output_dir=run_dir,
        diag=None if args.no_academic_panels else diag_hist,
    )
    lg.info("Outputs saved ➜ %s", run_dir)
    lg.info("Pipeline completed ")

# entry-point -------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Got `batch_size`", category=UserWarning,
                            module="sklearn.neural_network")
    warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning,
                            module="sklearn")
    run_pipeline(parse_args())
