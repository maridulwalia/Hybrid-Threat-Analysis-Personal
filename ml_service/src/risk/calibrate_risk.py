"""
ml_service/src/risk/calibrate_risk.py
─────────────────────────────────────
Phase 9: Risk Scoring Weight & Variant Calibration.

Calibrates:
- Weights: (w1, w2, w3, w4) on a simplex grid (step 0.05).
- Rule Bonus: Variant A vs Variant B.
- Scaling constant: N_ref (99th percentile of incident flow count on calibration half).

Split Protocol:
- Built strictly on 20% unseen test partition (3,296 incidents).
- Incident-level 50/50 stratified split (seed=42) on (source_day, majority_gt_label).
- Optimization objective: Maximize binary ROC-AUC separating attack from false incidents.
- Tie-breaking: Euclidean distance to uniform vector (0.25, 0.25, 0.25, 0.25) within 0.01 of best AUC.
- Output: artifacts/risk/risk_weights_v1.json (with SHA-256 recorded in preregistration_risk_v1.md).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil  # type: ignore[import-untyped]
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths
from ml_service.src.correlation.correlator import correlate_flows
from ml_service.src.risk.risk_scorer import IncidentRiskScorer, SEVERITY_TABLE_PRIMARY
from ml_service.src.utils.memory_guard import assert_memory_headroom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_risk")

MEMORY_LIMIT_MB = 500
SPLIT_SEED = 42


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def build_test_incidents() -> List[Dict[str, Any]]:
    """Builds correlated incidents on the unseen test partition and extracts telemetry."""
    logger.info("Loading reference table and flags for TEST partition...")
    ref_path = Paths.ARTIFACTS_DIR / "reference" / "flow_reference_v1.parquet"
    flags_path = Paths.ARTIFACTS_DIR / "correlation" / "ml_flags_v2.parquet"
    mask_path = Paths.SPLITS_DIR / "is_train_mask_v1.npy"

    is_train_mask = np.load(mask_path)
    test_indices = np.where(~is_train_mask)[0]

    # Load test partition reference & flags
    ref_table = pq.read_table(
        ref_path,
        columns=["row_id", "source_day", "source_ip", "destination_ip", "destination_port", "epoch_s", "label"],
    )
    test_ref = ref_table.take(test_indices).to_pandas()
    del ref_table

    flags_table = pq.read_table(
        flags_path,
        columns=["row_id", "predicted_class", "confidence", "p_attack", "r104_flag", "is_flagged"],
    )
    test_flags = flags_table.take(test_indices).to_pandas()
    del flags_table

    # Filter to flagged candidate flows
    flagged_mask = test_flags["is_flagged"].to_numpy(dtype=bool)
    sub_ref = test_ref[flagged_mask].reset_index(drop=True)
    sub_flags = test_flags[flagged_mask].reset_index(drop=True)
    n_raw = len(sub_ref)
    logger.info("Flagged test candidate flows: %d", n_raw)
    assert n_raw == 86_170, f"Expected 86,170 flagged test flows, got {n_raw}"

    # Correlate flows using strict ground-truth-blind input
    blind_input = pd.DataFrame({
        "row_id": sub_ref["row_id"].to_numpy(dtype=np.int64),
        "source_ip": sub_ref["source_ip"].to_numpy(),
        "destination_ip": sub_ref["destination_ip"].to_numpy(),
        "destination_port": sub_ref["destination_port"].to_numpy(dtype=np.int32),
        "epoch_s": sub_ref["epoch_s"].to_numpy(dtype=np.float64),
        "predicted_class": sub_flags["predicted_class"].to_numpy(),
        "confidence": sub_flags["confidence"].to_numpy(dtype=np.float32),
        "r104_flag": sub_flags["r104_flag"].to_numpy(dtype=bool),
    })

    t0 = time.time()
    incidents_df = correlate_flows(
        blind_input,
        key_cols=("source_ip", "destination_ip", "destination_port"),
        gap_seconds=300.0,
    )
    logger.info("Correlated %d test flows into %d incidents in %.2f s", n_raw, len(incidents_df), time.time() - t0)
    assert len(incidents_df) == 3_296, f"Expected 3,296 test incidents, got {len(incidents_df)}"

    # Map row_id to metadata for fast lookup
    row_id_to_p_attack = dict(zip(sub_flags["row_id"], sub_flags["p_attack"]))
    row_id_to_pred_class = dict(zip(sub_flags["row_id"], sub_flags["predicted_class"]))
    row_id_to_r104 = dict(zip(sub_flags["row_id"], sub_flags["r104_flag"]))
    row_id_to_label = dict(zip(sub_ref["row_id"], sub_ref["label"]))
    row_id_to_day = dict(zip(sub_ref["row_id"], sub_ref["source_day"]))

    incidents_list: List[Dict[str, Any]] = []

    for _, row in incidents_df.iterrows():
        f_ids: List[int] = row["flow_ids"]
        k = len(f_ids)

        p_att_series = [row_id_to_p_attack[rid] for rid in f_ids]
        preds = [row_id_to_pred_class[rid] for rid in f_ids]
        r104_matches = [row_id_to_r104[rid] for rid in f_ids]
        gt_labels = [row_id_to_label[rid] for rid in f_ids]
        days = [row_id_to_day[rid] for rid in f_ids]

        primary_threat = str(row["primary_threat"])
        has_ml = any(p != "BENIGN" for p in preds)
        r104_any = any(r104_matches)

        # Ground-truth classification
        counts = Counter(gt_labels)
        maj_gt_label, maj_gt_cnt = counts.most_common(1)[0]
        n_attack_gt = sum(1 for lab in gt_labels if lab != "BENIGN")

        if n_attack_gt == 0:
            inc_label = "false"
        elif maj_gt_label != "BENIGN":
            inc_label = "attack"
        else:
            inc_label = "mixed"

        grp_key = row["key"]
        if isinstance(grp_key, (tuple, list)):
            src_ip = str(grp_key[0])
            dst_ip = str(grp_key[1])
            dst_port = int(grp_key[2])
        else:
            src_ip, dst_ip, dst_port = "", "", 0

        source_day = days[0]

        inc_record: Dict[str, Any] = {
            "incident_id": int(row["incident_id"]),
            "source_day": source_day,
            "source_ip": src_ip,
            "destination_ip": dst_ip,
            "destination_port": dst_port,
            "flow_ids": f_ids,
            "n_flows": k,
            "p_attack_series": p_att_series,
            "primary_threat": primary_threat,
            "has_ml_flag": has_ml,
            "r104_any": r104_any,
            "majority_gt_label": maj_gt_label,
            "n_attack_gt": n_attack_gt,
            "incident_label": inc_label,
        }
        incidents_list.append(inc_record)

    return incidents_list


def split_50_50_stratified(incidents: List[Dict[str, Any]], seed: int = SPLIT_SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits incidents 50/50 stratified on (source_day, majority_gt_label)."""
    # Create stratum identifier
    strata = [f"{inc['source_day']}__{inc['majority_gt_label']}" for inc in incidents]
    strata_counts = Counter(strata)

    # For singleton strata (count=1), place into calibration half deterministically
    regular_indices: List[int] = []
    regular_strata: List[str] = []
    singleton_indices: List[int] = []

    for idx, s in enumerate(strata):
        if strata_counts[s] > 1:
            regular_indices.append(idx)
            regular_strata.append(s)
        else:
            singleton_indices.append(idx)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_sub_idx, report_sub_idx = next(sss.split(regular_indices, regular_strata))

    calib_indices = [regular_indices[i] for i in train_sub_idx] + singleton_indices
    report_indices = [regular_indices[i] for i in report_sub_idx]

    calib_incidents = [incidents[i] for i in calib_indices]
    report_incidents = [incidents[i] for i in report_indices]

    return calib_incidents, report_incidents


def generate_simplex_grid(step: float = 0.05) -> List[Tuple[float, float, float, float]]:
    """Generates all non-negative 4-tuples summing to 1.0 on a coarse grid."""
    n_steps = int(round(1.0 / step))
    points: List[Tuple[float, float, float, float]] = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            for k in range(n_steps + 1 - i - j):
                l = n_steps - i - j - k
                points.append((round(i * step, 2), round(j * step, 2), round(k * step, 2), round(l * step, 2)))
    return points


def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Phase 9 Step 2d -- Risk Scoring Calibration (Calibration Half Only)")
    logger.info("=" * 70)
    assert_memory_headroom(min_free_mb=200)

    # 1. Build test incidents
    incidents = build_test_incidents()
    n_inc = len(incidents)
    logger.info("Total test incidents: %d", n_inc)

    # Count overall categories
    type_counts = Counter(inc["incident_label"] for inc in incidents)
    logger.info("Test Incident Type Counts: %s", dict(type_counts))

    # 2. 50/50 Stratified Split
    calib_inc, report_inc = split_50_50_stratified(incidents, seed=SPLIT_SEED)
    logger.info("Calibration Half size: %d, Report Half size: %d", len(calib_inc), len(report_inc))
    calib_counts = Counter(inc["incident_label"] for inc in calib_inc)
    report_counts = Counter(inc["incident_label"] for inc in report_inc)
    logger.info("  Calibration Half breakdown: %s", dict(calib_counts))
    logger.info("  Report Half breakdown:      %s", dict(report_counts))

    # 3. Compute N_ref on Calibration Half (99th percentile of incident flow counts)
    calib_sizes = [inc["n_flows"] for inc in calib_inc]
    n_ref = float(np.percentile(calib_sizes, 99.0))
    logger.info("Calibration Half 99th percentile flow count (N_ref): %.2f (max: %d)", n_ref, max(calib_sizes))

    # 4. Filter Calibration incidents for AUC optimization (Exclude mixed incidents)
    eval_calib = [inc for inc in calib_inc if inc["incident_label"] in ("attack", "false")]
    y_calib = np.array([1 if inc["incident_label"] == "attack" else 0 for inc in eval_calib], dtype=np.int32)
    logger.info("Calibration AUC evaluation sample: %d incidents (%d attack, %d false)", len(y_calib), int(np.sum(y_calib == 1)), int(np.sum(y_calib == 0)))

    # Precompute raw components for speed
    comp_a_list = []
    comp_b_list = []
    for inc in eval_calib:
        p_att = inc["p_attack_series"]
        pt = inc["primary_threat"]
        nf = inc["n_flows"]
        r104 = inc["r104_any"]
        ml = inc["has_ml_flag"]

        c_ml = 100.0 * (sum(p_att) / len(p_att)) if p_att else 0.0
        sev = SEVERITY_TABLE_PRIMARY.get(pt, 0.0)
        s_vol = min(100.0, 100.0 * (math.log(1.0 + nf) / math.log(1.0 + n_ref))) if n_ref > 0 else 0.0
        b_rule_a = 100.0 if r104 else 0.0
        b_rule_b = 100.0 if (r104 and ml) else 0.0

        comp_a_list.append((c_ml, sev, s_vol, b_rule_a))
        comp_b_list.append((c_ml, sev, s_vol, b_rule_b))

    comp_a_arr = np.array(comp_a_list, dtype=np.float32)
    comp_b_arr = np.array(comp_b_list, dtype=np.float32)

    # 5. Grid Search over simplex
    grid_points = generate_simplex_grid(step=0.05)
    logger.info("Simplex grid generated: %d points per variant", len(grid_points))

    best_auc = -1.0
    all_results: List[Tuple[float, Tuple[float, float, float, float], str]] = []

    for variant_name, comp_arr in [("Variant_A", comp_a_arr), ("Variant_B", comp_b_arr)]:
        for w in grid_points:
            w_vec = np.array(w, dtype=np.float32)
            scores = comp_arr @ w_vec
            auc = float(roc_auc_score(y_calib, scores))
            all_results.append((auc, w, variant_name))
            if auc > best_auc:
                best_auc = auc

    logger.info("Grid search completed. Maximum Calibration AUC: %.5f", best_auc)

    # 6. Regularization / Tie-Breaking: Within 0.01 of best AUC, pick closest to uniform (0.25, 0.25, 0.25, 0.25)
    uniform_vec = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    threshold_auc = best_auc - 0.01

    candidates = [res for res in all_results if res[0] >= threshold_auc]
    logger.info("Candidates within 0.01 of best AUC: %d", len(candidates))

    def dist_to_uniform(w_tuple: Tuple[float, float, float, float]) -> float:
        return float(np.linalg.norm(np.array(w_tuple, dtype=np.float32) - uniform_vec))

    # Sort candidates by distance to uniform, then by higher AUC
    candidates.sort(key=lambda x: (dist_to_uniform(x[1]), -x[0]))
    chosen_auc, chosen_weights, chosen_variant = candidates[0]
    min_dist = dist_to_uniform(chosen_weights)

    logger.info("=" * 70)
    logger.info("CALIBRATION RESULT:")
    logger.info("  Chosen Rule Bonus Variant: %s", chosen_variant)
    logger.info("  Chosen Weights (w1, w2, w3, w4): %s", chosen_weights)
    logger.info("  Calibration AUC: %.5f (Best: %.5f)", chosen_auc, best_auc)
    logger.info("  Euclidean distance to uniform: %.4f", min_dist)
    logger.info("  Scaling Reference N_ref: %.2f", n_ref)
    logger.info("=" * 70)

    # 7. Write risk_weights_v1.json
    out_dir = Paths.ARTIFACTS_DIR / "risk"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_json_path = out_dir / "risk_weights_v1.json"

    calib_payload = {
        "weights": list(chosen_weights),
        "rule_variant": chosen_variant,
        "N_ref": round(n_ref, 2),
        "split_seed": SPLIT_SEED,
        "calibration_auc": round(chosen_auc, 5),
        "best_calibration_auc": round(best_auc, 5),
        "dist_to_uniform": round(min_dist, 5),
        "calib_incident_counts": {
            "attack": calib_counts.get("attack", 0),
            "false": calib_counts.get("false", 0),
            "mixed": calib_counts.get("mixed", 0),
            "total": len(calib_inc),
        },
        "report_incident_counts": {
            "attack": report_counts.get("attack", 0),
            "false": report_counts.get("false", 0),
            "mixed": report_counts.get("mixed", 0),
            "total": len(report_inc),
        },
    }

    with open(weights_json_path, "w", encoding="utf-8") as f:
        json.dump(calib_payload, f, indent=2)
    logger.info("Saved calibration weights artifact: %s", weights_json_path.name)

    # 8. Compute SHA-256 and update preregistration document
    with open(weights_json_path, "rb") as f:
        weights_sha256 = hashlib.sha256(f.read()).hexdigest()
    logger.info("SHA-256 of %s: %s", weights_json_path.name, weights_sha256)

    prereg_path = out_dir / "preregistration_risk_v1.md"
    with open(prereg_path, "r", encoding="utf-8") as f:
        prereg_content = f.read()

    # Populate Section 7 commitment table
    table_target = """| Parameter | Calibrated Value |
|---|---|
| **$w_1$ (ML Confidence Weight)** | *Pending Calibration* |
| **$w_2$ (Severity Table Weight)** | *Pending Calibration* |
| **$w_3$ (Volume Scaling Weight)** | *Pending Calibration* |
| **$w_4$ (Rule Bonus Weight)** | *Pending Calibration* |
| **Rule Bonus Variant** | *Pending Calibration* |
| **$N_{\\text{ref}}$ (99th %ile Size)** | *Pending Calibration* |
| **Calibration Set ROC-AUC** | *Pending Calibration* |
| **SHA-256 of `risk_weights_v1.json`** | *Pending Calibration* |"""

    table_replacement = f"""| Parameter | Calibrated Value |
|---|---|
| **$w_1$ (ML Confidence Weight)** | **{chosen_weights[0]:.2f}** |
| **$w_2$ (Severity Table Weight)** | **{chosen_weights[1]:.2f}** |
| **$w_3$ (Volume Scaling Weight)** | **{chosen_weights[2]:.2f}** |
| **$w_4$ (Rule Bonus Weight)** | **{chosen_weights[3]:.2f}** |
| **Rule Bonus Variant** | **`{chosen_variant}`** |
| **$N_{{\\text{{ref}}}}$ (99th %ile Size)** | **{n_ref:.2f}** |
| **Calibration Set ROC-AUC** | **{chosen_auc:.5f}** |
| **SHA-256 of `risk_weights_v1.json`** | `{weights_sha256}` |"""

    assert table_target in prereg_content, "Commitment table target not found in preregistration document"
    updated_prereg = prereg_content.replace(table_target, table_replacement)
    with open(prereg_path, "w", encoding="utf-8") as f:
        f.write(updated_prereg)
    logger.info("Committed calibrated weights into %s prior to report scoring.", prereg_path.name)

    elapsed = time.time() - t0
    peak_rss = get_rss_mb()
    logger.info("Calibration Complete in %.2f s | Peak RSS: %.2f MB", elapsed, peak_rss)
    assert peak_rss < MEMORY_LIMIT_MB, f"Peak RSS {peak_rss:.2f} MB exceeded limit {MEMORY_LIMIT_MB} MB"


if __name__ == "__main__":
    main()
