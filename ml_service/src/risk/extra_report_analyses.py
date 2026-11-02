"""
ml_service/src/risk/extra_report_analyses.py
─────────────────────────────────────────────
Phase 9 Extra Analyses (Items 4 & 5).
No refitting, analysis only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths
from ml_service.src.correlation.correlator import correlate_flows
from ml_service.src.risk.calibrate_risk import (
    build_test_incidents,
    split_50_50_stratified,
    SPLIT_SEED,
)
from ml_service.src.risk.risk_scorer import (
    IncidentRiskScorer,
    SEVERITY_TABLE_PRIMARY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extra_analyses")


def main() -> None:
    # Load frozen config
    weights_path = Paths.ARTIFACTS_DIR / "risk" / "risk_weights_v1.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        calib_cfg = json.load(f)

    weights = calib_cfg["weights"]
    n_ref = calib_cfg["N_ref"]
    rule_variant = calib_cfg["rule_variant"]

    scorer_frozen = IncidentRiskScorer(
        weights=weights,
        n_ref=n_ref,
        rule_variant=rule_variant,
        severity_table=SEVERITY_TABLE_PRIMARY,
    )

    # Load ranking parquet
    df = pd.read_parquet(Paths.ARTIFACTS_DIR / "risk" / "incident_risk_ranking_v1.parquet")
    calib_df = df[df["partition"] == "calibration"].copy()
    report_df = df[df["partition"] == "report"].sort_values(by="risk_score", ascending=False).reset_index(drop=True)

    print("=" * 80)
    print("EXTRA ANALYSIS 4a: Attack incident counts & PortScan share per half")
    print("=" * 80)
    calib_attacks = calib_df[calib_df["incident_label"] == "attack"]
    report_attacks = report_df[report_df["incident_label"] == "attack"]

    calib_class_counts = Counter(calib_attacks["primary_threat"])
    report_class_counts = Counter(report_attacks["primary_threat"])

    all_classes = sorted(list(set(calib_class_counts.keys()).union(report_class_counts.keys())))
    print(f"{'Attack Class':<30} | {'Calib Half':<12} | {'Report Half':<12}")
    print("-" * 60)
    for c in all_classes:
        print(f"{c:<30} | {calib_class_counts.get(c, 0):<12} | {report_class_counts.get(c, 0):<12}")

    calib_ps_count = calib_class_counts.get("PortScan", 0)
    report_ps_count = report_class_counts.get("PortScan", 0)
    calib_ps_share = calib_ps_count / len(calib_attacks) * 100
    report_ps_share = report_ps_count / len(report_attacks) * 100

    print("-" * 60)
    print(f"Total Attack Incidents:        | {len(calib_attacks):<12} | {len(report_attacks):<12}")
    print(f"PortScan Share of Attacks:     | {calib_ps_count} ({calib_ps_share:.2f}%)   | {report_ps_count} ({report_ps_share:.2f}%)")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 4b: ROC-AUC per attack class (>=20), Macro AUC, and Excl-PortScan AUC")
    print("=" * 80)
    # Report half evaluation set (attack vs false)
    eval_rep = report_df[report_df["incident_label"].isin(["attack", "false"])].copy()
    false_rep = eval_rep[eval_rep["incident_label"] == "false"]

    per_class_auc = {}
    for c in all_classes:
        c_attacks = eval_rep[(eval_rep["incident_label"] == "attack") & (eval_rep["primary_threat"] == c)]
        k_c = len(c_attacks)
        if k_c >= 20:
            sub = pd.concat([c_attacks, false_rep])
            y_sub = (sub["incident_label"] == "attack").astype(int).to_numpy()
            auc_c = float(roc_auc_score(y_sub, sub["risk_score"]))
            per_class_auc[c] = (k_c, auc_c)
            print(f"  Class: {c:<25} | N = {k_c:<4} | ROC-AUC vs False: {auc_c:.5f}")
        else:
            per_class_auc[c] = (k_c, None)
            print(f"  Class: {c:<25} | N = {k_c:<4} | Count only (<20 incidents)")

    valid_aucs = [v[1] for v in per_class_auc.values() if v[1] is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    print(f"\n  Class-Balanced (Macro) AUC (across classes with >=20 inc): {macro_auc:.5f}")

    # AUC excluding PortScan
    no_ps_attacks = eval_rep[(eval_rep["incident_label"] == "attack") & (eval_rep["primary_threat"] != "PortScan")]
    sub_no_ps = pd.concat([no_ps_attacks, false_rep])
    y_no_ps = (sub_no_ps["incident_label"] == "attack").astype(int).to_numpy()
    auc_no_ps = float(roc_auc_score(y_no_ps, sub_no_ps["risk_score"]))
    print(f"  ROC-AUC Excluding PortScan ({len(no_ps_attacks)} non-PS attacks vs {len(false_rep)} false): {auc_no_ps:.5f}")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 4c: ROC-AUC per day on Report Half")
    print("=" * 80)
    for day, grp in eval_rep.groupby("source_day"):
        y_day = (grp["incident_label"] == "attack").astype(int).to_numpy()
        n_att = int(y_day.sum())
        n_fal = len(y_day) - n_att
        if len(np.unique(y_day)) > 1:
            auc_day = float(roc_auc_score(y_day, grp["risk_score"]))
            print(f"  Day: {day:<35} | Total: {len(grp):<4} (Att: {n_att:<4}, Fal: {n_fal:<4}) | AUC: {auc_day:.5f}")
        else:
            print(f"  Day: {day:<35} | Total: {len(grp):<4} (Att: {n_att:<4}, Fal: {n_fal:<4}) | AUC: Undefined (single class)")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 4d: Paired Bootstrap 95% CI of AUC Differences")
    print("=" * 80)
    y_rep = (eval_rep["incident_label"] == "attack").astype(int).to_numpy()
    score_hybrid = eval_rep["risk_score"].to_numpy()
    score_ml = eval_rep["c_ml_raw"].to_numpy()
    score_vol = eval_rep["s_vol_raw"].to_numpy()
    score_sev = eval_rep["sev_raw"].to_numpy()
    score_equal = 0.25 * (score_ml + score_sev + score_vol + eval_rep["b_rule_raw"].to_numpy())

    rng = np.random.default_rng(42)
    n_samples = len(y_rep)
    diffs_ml = []
    diffs_equal = []
    diffs_sev = []

    for _ in range(1000):
        b_idx = rng.integers(0, n_samples, size=n_samples)
        b_y = y_rep[b_idx]
        if len(np.unique(b_y)) < 2:
            continue
        auc_h = float(roc_auc_score(b_y, score_hybrid[b_idx]))
        auc_m = float(roc_auc_score(b_y, score_ml[b_idx]))
        auc_e = float(roc_auc_score(b_y, score_equal[b_idx]))
        auc_s = float(roc_auc_score(b_y, score_sev[b_idx]))

        diffs_ml.append(auc_h - auc_m)
        diffs_equal.append(auc_h - auc_e)
        diffs_sev.append(auc_h - auc_s)

    for name, diff_arr, base_s in [
        ("vs ML-Confidence-Only", diffs_ml, score_ml),
        ("vs Equal Weights (0.25)", diffs_equal, score_equal),
        ("vs Severity-Only", diffs_sev, score_sev),
    ]:
        delta_mean = float(roc_auc_score(y_rep, score_hybrid)) - float(roc_auc_score(y_rep, base_s))
        ci_low = float(np.percentile(diff_arr, 2.5))
        ci_high = float(np.percentile(diff_arr, 97.5))
        print(f"  {name:<25} | Delta AUC: {delta_mean:+.5f} | 95% Bootstrap CI: [{ci_low:+.5f}, {ci_high:+.5f}]")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 4e: Ablations (w4=0 and Rule Variant A vs B)")
    print("=" * 80)
    # w4 = 0 ablation: normalize other weights: w1=0.3, w2=0.1, w3=0.3 -> sum=0.7 -> [0.3/0.7, 0.1/0.7, 0.3/0.7, 0.0]
    w_no_w4 = np.array([0.30/0.70, 0.10/0.70, 0.30/0.70, 0.0], dtype=np.float32)
    scores_no_w4 = (
        w_no_w4[0] * eval_rep["c_ml_raw"]
        + w_no_w4[1] * eval_rep["sev_raw"]
        + w_no_w4[2] * eval_rep["s_vol_raw"]
    )
    auc_no_w4 = float(roc_auc_score(y_rep, scores_no_w4))
    print(f"  Ablation with w4 = 0 (weights {list(np.round(w_no_w4, 4))}): AUC = {auc_no_w4:.5f}")

    # Variant A vs Variant B at frozen weights (0.30, 0.10, 0.30, 0.30)
    # Under Variant A, b_rule is 100 whenever r104_any is True (regardless of has_ml_flag)
    # Load incidents to check r104_any
    incidents = build_test_incidents()
    calib_inc, report_inc = split_50_50_stratified(incidents, seed=SPLIT_SEED)

    scorer_var_a = IncidentRiskScorer(weights=weights, n_ref=n_ref, rule_variant="Variant_A", severity_table=SEVERITY_TABLE_PRIMARY)
    scores_var_a = []
    y_rep_inc = []
    for inc in report_inc:
        if inc["incident_label"] in ("attack", "false"):
            res = scorer_var_a.score_incident({
                "p_attack_series": inc["p_attack_series"],
                "primary_threat": inc["primary_threat"],
                "n_flows": inc["n_flows"],
                "r104_any": inc["r104_any"],
                "has_ml_flag": inc["has_ml_flag"],
            })
            scores_var_a.append(res["risk_score"])
            y_rep_inc.append(1 if inc["incident_label"] == "attack" else 0)

    auc_var_a = float(roc_auc_score(y_rep_inc, scores_var_a))
    print(f"  Rule Bonus Variant Comparison at frozen weights:")
    print(f"    Variant B (Active, requires ML flag): AUC = {calib_cfg.get('report_auc', 0.96613):.5f}")
    print(f"    Variant A (Unconditional on rule):     AUC = {auc_var_a:.5f}")
    print(f"    Delta (B - A):                        {0.96613 - auc_var_a:+.5f}")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 4f: Extended P@k up to 500, Base Rate, Rank of First False Alarm")
    print("=" * 80)
    base_rate = len(report_attacks) / len(report_df) * 100
    print(f"  Report Half Base Rate (True Attack Share): {len(report_attacks)} / {len(report_df)} ({base_rate:.2f}%)")

    k_list = [10, 25, 50, 100, 200, 300, 400, 500]
    for k in k_list:
        sub_k = report_df.head(k)
        att_k = int((sub_k["incident_label"] == "attack").sum())
        p_k = att_k / k * 100
        print(f"  P@{k:<3}: {p_k:.2f}% ({att_k} / {k})")

    # Find rank of first false alarm
    first_false_idx = int(np.where(report_df["incident_label"] == "false")[0][0])
    first_false_rank = first_false_idx + 1
    first_false_inc = report_df.iloc[first_false_idx]
    print(f"\n  First False Alarm Rank: #{first_false_rank}")
    print(f"    Incident ID: {first_false_inc['incident_id']} | Primary Threat: {first_false_inc['primary_threat']} | Risk Score: {first_false_inc['risk_score']:.2f}")
    print(f"    Sub-scores: C_ML={first_false_inc['c_ml_raw']:.2f}, Sev={first_false_inc['sev_raw']:.2f}, Vol={first_false_inc['s_vol_raw']:.2f}, Rule={first_false_inc['b_rule_raw']:.2f}")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 4g: ROC-AUC on K2 Grouping Key Incidents")
    print("=" * 80)
    ref_path = Paths.ARTIFACTS_DIR / "reference" / "flow_reference_v1.parquet"
    flags_path = Paths.ARTIFACTS_DIR / "correlation" / "ml_flags_v2.parquet"
    mask_path = Paths.SPLITS_DIR / "is_train_mask_v1.npy"
    is_train_mask = np.load(mask_path)
    test_idx = np.where(~is_train_mask)[0]

    ref_t = pq.read_table(ref_path, columns=["row_id", "source_day", "source_ip", "destination_ip", "destination_port", "epoch_s", "label"])
    test_ref = ref_t.take(test_idx).to_pandas()
    flags_t = pq.read_table(flags_path, columns=["row_id", "predicted_class", "confidence", "p_attack", "r104_flag", "is_flagged"])
    test_flags = flags_t.take(test_idx).to_pandas()

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
    print(f"  K2 Test Incidents Formed: {len(inc_k2_df)}")

    row_to_p_att = dict(zip(sub_flags_k2["row_id"], sub_flags_k2["p_attack"]))
    row_to_pred = dict(zip(sub_flags_k2["row_id"], sub_flags_k2["predicted_class"]))
    row_to_r104 = dict(zip(sub_flags_k2["row_id"], sub_flags_k2["r104_flag"]))
    row_to_label = dict(zip(sub_ref_k2["row_id"], sub_ref_k2["label"]))

    k2_scores = []
    k2_y = []
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
    print(f"  K2 Incidents ROC-AUC (attack vs false, {int(np.sum(k2_y))} attack, {len(k2_y)-int(np.sum(k2_y))} false): {auc_k2:.5f}")

    print("\n" + "=" * 80)
    print("EXTRA ANALYSIS 5: Severity Trade-Off Table on CALIBRATION Half Only")
    print("=" * 80)
    calib_eval = calib_df[calib_df["incident_label"].isin(["attack", "false"])].copy()
    y_calib = (calib_eval["incident_label"] == "attack").astype(int).to_numpy()

    # Pre-extract vectors
    cml_c = calib_eval["c_ml_raw"].to_numpy()
    sev_c = calib_eval["sev_raw"].to_numpy()
    vol_c = calib_eval["s_vol_raw"].to_numpy()
    rule_c = calib_eval["b_rule_raw"].to_numpy()

    crit_high_threats = {"Infiltration", "Heartbleed", "Web Attack - Sql Injection", "SSH-Patator", "FTP-Patator", "Web Attack - Brute Force"}
    is_crit_high_calib = calib_eval["primary_threat"].isin(crit_high_threats) & (calib_eval["incident_label"] == "attack")

    print(f"  Calibration Half Critical/High attack incidents count: {int(is_crit_high_calib.sum())}")

    # Generate grid for fixed w2
    step = 0.05
    for target_w2 in [0.10, 0.25, 0.40]:
        rem = round(1.0 - target_w2, 2)
        n_rem_steps = int(round(rem / step))

        best_w = None
        best_auc = -1.0
        best_scores = None

        for i in range(n_rem_steps + 1):
            for j in range(n_rem_steps + 1 - i):
                k = n_rem_steps - i - j
                w1 = round(i * step, 2)
                w3 = round(j * step, 2)
                w4 = round(k * step, 2)

                cand_scores = w1 * cml_c + target_w2 * sev_c + w3 * vol_c + w4 * rule_c
                auc_cand = float(roc_auc_score(y_calib, cand_scores))
                if auc_cand > best_auc:
                    best_auc = auc_cand
                    best_w = (w1, target_w2, w3, w4)
                    best_scores = cand_scores

        # Compute mean rank of High/Critical-tier incidents among attack incidents
        calib_eval["cand_score"] = best_scores
        attack_subset = calib_eval[calib_eval["incident_label"] == "attack"].sort_values(by="cand_score", ascending=False).reset_index(drop=True)
        attack_subset["attack_rank"] = np.arange(1, len(attack_subset) + 1)
        mean_crit_high_rank = float(attack_subset[attack_subset["primary_threat"].isin(crit_high_threats)]["attack_rank"].mean())

        print(f"  w2 = {target_w2:.2f} | Optimal weights: {best_w} | Calib AUC: {best_auc:.5f} | Mean Rank of Crit/High: {mean_crit_high_rank:.2f} / {len(attack_subset)}")


if __name__ == "__main__":
    main()

