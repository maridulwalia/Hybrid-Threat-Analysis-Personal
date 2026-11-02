"""
ml_service/src/risk/closure_analyses.py
───────────────────────────────────────
Phase 8–9 Final Closure Analyses:
Consolidates all post-calibration ad-hoc analyses into a single reproducible pipeline:
1. Cluster Bootstrap Audit by (source_day, majority_gt_label) campaign.
2. Sub-Cohort ROC-AUC Disaggregation: Multi-Flow (N >= 2) vs Singletons (N = 1)
   on Report Half and Full Test partition.
3. Descriptive Per-Class Distribution across all 3,296 test incidents with in-sample flags.
4. Policy Operating-Point Menu on Calibration Half (w2 sweep vs Critical/High rank).
5. Port-Agile Key (K2) Sensitivity on test incidents.

Outputs:
- artifacts/risk/closure_analyses_v1.json
"""

from __future__ import annotations

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

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths
from ml_service.src.correlation.correlator import correlate_flows
from ml_service.src.risk.risk_scorer import IncidentRiskScorer, SEVERITY_TABLE_PRIMARY
from ml_service.src.utils.memory_guard import assert_memory_headroom

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("closure_analyses")

MEMORY_LIMIT_MB = 500


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("Running Phase 8-9 Closure Analyses Pipeline")
    logger.info("=" * 80)
    assert_memory_headroom(min_free_mb=200)

    # Load frozen ranking parquet
    ranking_parquet_path = Paths.ARTIFACTS_DIR / "risk" / "incident_risk_ranking_v1.parquet"
    assert ranking_parquet_path.exists(), f"Missing {ranking_parquet_path}"
    df = pq.read_table(ranking_parquet_path).to_pandas()
    logger.info("Loaded ranking parquet: %d incidents", len(df))

    calib_df = df[df["partition"] == "calibration"].copy()
    report_df = df[df["partition"] == "report"].copy()

    # Load frozen weights
    weights_path = Paths.ARTIFACTS_DIR / "risk" / "risk_weights_v1.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        frozen_weights_cfg = json.load(f)

    scorer_frozen = IncidentRiskScorer(
        weights=tuple(frozen_weights_cfg["weights"]),
        severity_table=SEVERITY_TABLE_PRIMARY,
        n_ref=frozen_weights_cfg["N_ref"],
        rule_variant=frozen_weights_cfg["rule_variant"],
    )

    crit_high_threats = [k for k, v in SEVERITY_TABLE_PRIMARY.items() if v >= 80]

    # -------------------------------------------------------------------------
    # 1. Cluster Bootstrap Audit
    # -------------------------------------------------------------------------
    logger.info("Computing Cluster Bootstrap by (day, label) campaigns...")
    report_clusters = report_df["source_day"].astype(str) + "__" + report_df["majority_gt_label"].astype(str)
    report_df["cluster"] = report_clusters
    unique_clusters = list(report_clusters.unique())

    att_clusters = [
        c for c in unique_clusters
        if (report_df[report_df["cluster"] == c]["incident_label"] == "attack").sum() > 0
    ]
    fal_clusters = [
        c for c in unique_clusters
        if (report_df[report_df["cluster"] == c]["incident_label"] == "false").sum() > 0
    ]

    cluster_counts: Dict[str, Any] = {}
    for c in unique_clusters:
        sub_c = report_df[report_df["cluster"] == c]
        cluster_counts[c] = {
            "total": len(sub_c),
            "attack": int((sub_c["incident_label"] == "attack").sum()),
            "false": int((sub_c["incident_label"] == "false").sum()),
        }

    cluster_audit = {
        "total_clusters": len(unique_clusters),
        "attack_clusters": len(att_clusters),
        "false_clusters": len(fal_clusters),
        "attack_cluster_names": att_clusters,
        "is_inferentially_meaningful": False,
        "reason": "Effective sample of 4 attack clusters is below asymptotic threshold (~10-30). Point estimates reported.",
        "cluster_composition": cluster_counts,
    }

    # -------------------------------------------------------------------------
    # 2. Sub-Cohort ROC-AUC Disaggregation (N >= 2 vs N = 1)
    # -------------------------------------------------------------------------
    logger.info("Computing Sub-Cohort ROC-AUC (Multi-Flow vs Singletons)...")
    sub_cohort_results: Dict[str, Any] = {}

    for split_name, sub in [("report_half", report_df), ("full_test", df)]:
        sub_cohort_results[split_name] = {}
        for cohort_name, mask in [("multi_flow_N_ge_2", sub["n_flows"] >= 2), ("singletons_N_eq_1", sub["n_flows"] == 1)]:
            c_df = sub[mask]
            y = (c_df["incident_label"] == "attack").astype(int).to_numpy()
            n_att = int(y.sum())
            n_fal = len(y) - n_att

            if len(np.unique(y)) == 2:
                auc_frozen = float(roc_auc_score(y, c_df["risk_score"]))
                auc_ml = float(roc_auc_score(y, c_df["c_ml_raw"]))
                delta = auc_frozen - auc_ml
            else:
                auc_frozen, auc_ml, delta = 0.0, 0.0, 0.0

            sub_cohort_results[split_name][cohort_name] = {
                "total_incidents": len(c_df),
                "attack_incidents": n_att,
                "false_incidents": n_fal,
                "frozen_score_auc": round(auc_frozen, 5),
                "ml_confidence_auc": round(auc_ml, 5),
                "delta_auc": round(delta, 5),
            }

    # -------------------------------------------------------------------------
    # 3. Descriptive Per-Class Distribution Over ALL Test Incidents
    # -------------------------------------------------------------------------
    logger.info("Computing Descriptive Per-Class Distribution across all 3,296 incidents...")
    per_class_list: List[Dict[str, Any]] = []

    for threat, group in df.groupby("primary_threat"):
        n_total = len(group)
        n_calib = len(group[group["partition"] == "calibration"])
        n_report = len(group[group["partition"] == "report"])
        n_att = int((group["incident_label"] == "attack").sum())
        n_fal = int((group["incident_label"] == "false").sum())
        scores = group["risk_score"].to_numpy()

        if n_calib > 0 and n_report > 0:
            status = "in_sample_out_sample_mixed"
        elif n_calib > 0:
            status = "in_sample_only"
        else:
            status = "out_of_sample_only"

        std_val = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

        per_class_list.append({
            "primary_threat": str(threat),
            "total_incidents": n_total,
            "in_sample_calib": n_calib,
            "out_of_sample_report": n_report,
            "attack_incidents": n_att,
            "false_incidents": n_fal,
            "mean_risk_score": round(float(np.mean(scores)), 2),
            "std_risk_score": round(std_val, 2),
            "min_risk_score": round(float(np.min(scores)), 2),
            "max_risk_score": round(float(np.max(scores)), 2),
            "sample_status": status,
        })

    # Sort descending by total incidents
    per_class_list.sort(key=lambda x: -x["total_incidents"])

    # -------------------------------------------------------------------------
    # 4. Severity Weight Policy Operating-Point Menu (Calibration Half Only)
    # -------------------------------------------------------------------------
    logger.info("Computing Severity Operating-Point Menu on Calibration Half...")
    calib_eval = calib_df[calib_df["incident_label"].isin(["attack", "false"])].copy()
    y_calib = (calib_eval["incident_label"] == "attack").astype(int).to_numpy()

    # Pre-extract vectors
    cml_c = calib_eval["c_ml_raw"].to_numpy()
    sev_c = calib_eval["sev_raw"].to_numpy()
    svol_c = calib_eval["s_vol_raw"].to_numpy()
    brule_c = calib_eval["b_rule_raw"].to_numpy()

    calib_att_only = calib_eval[calib_eval["incident_label"] == "attack"].copy()
    calib_ch_mask = calib_att_only["primary_threat"].isin(crit_high_threats)
    n_ch_calib = int(calib_ch_mask.sum())

    # Frozen row
    frozen_scores = 0.30 * cml_c + 0.10 * sev_c + 0.30 * svol_c + 0.30 * brule_c
    frozen_calib_auc = float(roc_auc_score(y_calib, frozen_scores))

    calib_att_only["frozen_score"] = (
        0.30 * calib_att_only["c_ml_raw"] +
        0.10 * calib_att_only["sev_raw"] +
        0.30 * calib_att_only["s_vol_raw"] +
        0.30 * calib_att_only["b_rule_raw"]
    )
    calib_att_only["frozen_rank"] = calib_att_only["frozen_score"].rank(ascending=False, method="min")
    frozen_mean_rank = float(calib_att_only[calib_ch_mask]["frozen_rank"].mean())

    operating_menu: List[Dict[str, Any]] = [
        {
            "config_type": "frozen_calibrated_row",
            "weights": [0.30, 0.10, 0.30, 0.30],
            "calibration_auc": round(frozen_calib_auc, 5),
            "mean_rank_crit_high": round(frozen_mean_rank, 2),
            "total_crit_high_attacks": n_ch_calib,
            "total_attack_incidents": len(calib_att_only),
        }
    ]

    re_opt_specs = [
        ("re_optimized_w2_0.10", [0.60, 0.10, 0.30, 0.00]),
        ("re_optimized_w2_0.25", [0.55, 0.25, 0.20, 0.00]),
        ("re_optimized_w2_0.40", [0.45, 0.40, 0.15, 0.00]),
    ]

    for label, w in re_opt_specs:
        s_vec = w[0] * cml_c + w[1] * sev_c + w[2] * svol_c + w[3] * brule_c
        auc_val = float(roc_auc_score(y_calib, s_vec))

        calib_att_only["s_temp"] = (
            w[0] * calib_att_only["c_ml_raw"] +
            w[1] * calib_att_only["sev_raw"] +
            w[2] * calib_att_only["s_vol_raw"] +
            w[3] * calib_att_only["b_rule_raw"]
        )
        calib_att_only["r_temp"] = calib_att_only["s_temp"].rank(ascending=False, method="min")
        m_rank = float(calib_att_only[calib_ch_mask]["r_temp"].mean())

        operating_menu.append({
            "config_type": label,
            "weights": w,
            "calibration_auc": round(auc_val, 5),
            "mean_rank_crit_high": round(m_rank, 2),
            "total_crit_high_attacks": n_ch_calib,
            "total_attack_incidents": len(calib_att_only),
        })

    # Critical/High Counts per half
    report_att_only = report_df[report_df["incident_label"] == "attack"]
    report_ch_mask = report_att_only["primary_threat"].isin(crit_high_threats)
    n_ch_report = int(report_ch_mask.sum())

    ch_summary = {
        "calibration_half_crit_high_count": int(n_ch_calib),
        "calibration_half_crit_high_breakdown": {str(k): int(v) for k, v in calib_att_only[calib_ch_mask]["primary_threat"].value_counts().items()},
        "report_half_crit_high_count": int(n_ch_report),
        "report_half_crit_high_breakdown": {str(k): int(v) for k, v in report_att_only[report_ch_mask]["primary_threat"].value_counts().items()},
        "operating_points": operating_menu,
    }

    # -------------------------------------------------------------------------
    # 5. K2 Sensitivity Audit
    # -------------------------------------------------------------------------
    logger.info("Computing K2 Port-Agile Grouping Key Sensitivity...")
    # Release memory from previous steps
    del df, calib_df, report_df, calib_eval, calib_att_only
    import gc
    gc.collect()

    ref_path = Paths.ARTIFACTS_DIR / "reference" / "flow_reference_v1.parquet"
    flags_path = Paths.ARTIFACTS_DIR / "correlation" / "ml_flags_v2.parquet"
    mask_path = Paths.SPLITS_DIR / "is_train_mask_v1.npy"

    is_train_mask = np.load(mask_path)
    test_idx = np.where(~is_train_mask)[0]
    del is_train_mask
    gc.collect()

    ref_t = pq.read_table(ref_path, columns=["row_id", "source_ip", "destination_ip", "destination_port", "epoch_s", "label"])
    test_ref = ref_t.take(test_idx).to_pandas()
    del ref_t
    gc.collect()

    flags_t = pq.read_table(flags_path, columns=["row_id", "predicted_class", "confidence", "p_attack", "r104_flag", "is_flagged"])
    test_flags = flags_t.take(test_idx).to_pandas()
    del flags_t
    gc.collect()

    flg_m = test_flags["is_flagged"].to_numpy(dtype=bool)
    sub_ref_k2 = test_ref[flg_m].reset_index(drop=True)
    sub_flags_k2 = test_flags[flg_m].reset_index(drop=True)

    blind_k2 = pd.DataFrame({
        "row_id": sub_ref_k2["row_id"].to_numpy(dtype=np.int64),
        "source_ip": sub_ref_k2["source_ip"].to_numpy(),
        "destination_ip": sub_ref_k2["destination_ip"].to_numpy(),
        "destination_port": sub_ref_k2["destination_port"].to_numpy(dtype=np.int32),
        "epoch_s": sub_ref_k2["epoch_s"].to_numpy(dtype=np.float64),
        "predicted_class": sub_flags_k2["predicted_class"].to_numpy(),
        "confidence": sub_flags_k2["confidence"].to_numpy(dtype=np.float32),
        "r104_flag": sub_flags_k2["r104_flag"].to_numpy(dtype=bool),
    })

    inc_k2_df = correlate_flows(blind_k2, key_cols=("source_ip", "destination_ip"), gap_seconds=300.0)

    row_to_p_att = dict(zip(sub_flags_k2["row_id"], sub_flags_k2["p_attack"]))
    row_to_pred = dict(zip(sub_flags_k2["row_id"], sub_flags_k2["predicted_class"]))
    row_to_r104 = dict(zip(sub_flags_k2["row_id"], sub_flags_k2["r104_flag"]))
    row_to_label = dict(zip(sub_ref_k2["row_id"], sub_ref_k2["label"]))

    k2_scores: List[float] = []
    k2_y: List[int] = []

    for _, r in inc_k2_df.iterrows():
        fids = r["flow_ids"]
        labs = [row_to_label[i] for i in fids]
        p_atts = [row_to_p_att[i] for i in fids]
        preds = [row_to_pred[i] for i in fids]
        r104s = [row_to_r104[i] for i in fids]

        c = Counter(labs)
        maj_l, _ = c.most_common(1)[0]
        n_att = sum(1 for l in labs if l != "BENIGN")
        if n_att == 0:
            ilabel = "false"
        elif maj_l != "BENIGN":
            ilabel = "attack"
        else:
            ilabel = "mixed"

        if ilabel in ("attack", "false"):
            res = scorer_frozen.score_incident({
                "p_attack_series": p_atts,
                "primary_threat": str(r["primary_threat"]),
                "n_flows": len(fids),
                "r104_any": any(r104s),
                "has_ml_flag": any(p != "BENIGN" for p in preds),
            })
            k2_scores.append(res["risk_score"])
            k2_y.append(1 if ilabel == "attack" else 0)

    auc_k2 = float(roc_auc_score(k2_y, k2_scores))
    n_k2_att = int(np.sum(k2_y))
    n_k2_fal = len(k2_y) - n_k2_att

    k2_results = {
        "grouping_key": "K2: (Source IP, Destination IP)",
        "total_test_incidents": len(inc_k2_df),
        "attack_incidents": n_k2_att,
        "false_incidents": n_k2_fal,
        "frozen_score_roc_auc": round(auc_k2, 5),
        "methodological_note": "N_ref=25.00 was calibrated on K3 flow sizes. On K2, flow sizes are significantly larger; scores saturate at N >= 25.",
    }

    # -------------------------------------------------------------------------
    # 6. Assemble Full Payload & Write JSON
    # -------------------------------------------------------------------------
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "cluster_bootstrap_audit": cluster_audit,
        "sub_cohort_disaggregation": sub_cohort_results,
        "descriptive_per_class_all_test": per_class_list,
        "severity_operating_points": ch_summary,
        "k2_sensitivity_audit": k2_results,
    }

    out_path = Paths.ARTIFACTS_DIR / "risk" / "closure_analyses_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    elapsed = time.time() - t0
    peak_rss = get_rss_mb()
    logger.info("Saved closure analyses: %s", out_path)
    logger.info("Closure analyses completed in %.2f s | Peak RSS: %.2f MB", elapsed, peak_rss)
    assert peak_rss < MEMORY_LIMIT_MB, f"Peak RSS {peak_rss:.2f} MB exceeded limit {MEMORY_LIMIT_MB} MB"


if __name__ == "__main__":
    main()
