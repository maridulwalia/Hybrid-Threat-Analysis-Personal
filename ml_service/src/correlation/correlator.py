"""
ml_service/src/correlation/correlator.py
────────────────────────────────────────
Phase 8: Ground-Truth-Blind Event Correlator Engine.

Implements declarative multi-flow sessionization across network telemetry:
- Strict ground-truth blindness: rejects any DataFrame containing 'label' or 'source_day'.
- Grouping keys:
    * K3: (source_ip, destination_ip, destination_port) -- Primary
    * K2: (source_ip, destination_ip) -- Ablation (PortScan collapse)
    * K4: (source_ip, destination_ip, destination_port, predicted_class) -- Secondary
    * Dest-port only: (destination_port,) -- Baseline
- Sessionizer:
    * Sorts by (epoch_s, row_id) within each key group.
    * Sliding gap: if t - t_last <= gap_seconds (inclusive), joins incident; else starts new.
    * Singletons (N=1) preserved with duration=0.
- Baseline:
    * Fixed tumbling window: floor(epoch_s / bin_seconds) within key group.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


FORBIDDEN_COLUMNS = {"label", "source_day"}


def _assert_ground_truth_blind(df: pd.DataFrame) -> None:
    """Enforces strict ground-truth blindness.
    
    Raises:
        ValueError: If any ground-truth column is present in the input DataFrame.
    """
    found = FORBIDDEN_COLUMNS.intersection(df.columns)
    if found:
        raise ValueError(
            f"Ground-truth blindness violation: input DataFrame contains forbidden columns: {sorted(found)}"
        )


def _determine_primary_threat(
    pred_classes: Sequence[str],
    confs: Union[Sequence[float], np.ndarray],
) -> Tuple[str, float]:
    """Determines the primary threat category and representative confidence.
    
    If non-BENIGN predictions exist, takes the majority non-BENIGN class.
    Ties broken by higher mean confidence, then alphabetically.
    If all predictions are 'BENIGN' (e.g. rule-only triggers), returns ('RULE_ONLY:R104', mean_conf).
    """
    non_benign = [c for c in pred_classes if c != "BENIGN"]
    if not non_benign:
        return "RULE_ONLY:R104", float(np.mean(confs)) if len(confs) > 0 else 1.0

    counts = Counter(non_benign)
    max_count = max(counts.values())
    top_candidates = [c for c, cnt in counts.items() if cnt == max_count]

    if len(top_candidates) == 1:
        chosen = top_candidates[0]
    else:
        # Tie breaker: highest mean confidence among tied candidates
        best_cand = top_candidates[0]
        best_mean_conf = -1.0
        for cand in sorted(top_candidates):
            cand_confs = [confs[i] for i, c in enumerate(pred_classes) if c == cand]
            m_conf = float(np.mean(cand_confs)) if cand_confs else 0.0
            if m_conf > best_mean_conf:
                best_mean_conf = m_conf
                best_cand = cand
        chosen = best_cand

    chosen_confs = [confs[i] for i, c in enumerate(pred_classes) if c == chosen]
    mean_conf = float(np.mean(chosen_confs)) if chosen_confs else float(np.mean(confs))
    return chosen, mean_conf


def correlate_flows(
    flows_df: pd.DataFrame,
    key_cols: Sequence[str] = ("source_ip", "destination_ip", "destination_port"),
    gap_seconds: float = 300.0,
) -> pd.DataFrame:
    """Sliding-gap event correlation sessionizer.
    
    Parameters:
        flows_df: DataFrame containing at minimum [row_id, epoch_s, predicted_class, confidence]
                  plus all columns specified in key_cols.
        key_cols: Column names defining the incident identity.
        gap_seconds: Inactivity timeout. If delta_t <= gap_seconds (inclusive), flows merge.
        
    Returns:
        DataFrame with one row per correlated incident:
        [incident_id, key, start_time, end_time, duration_s, flow_count, flow_ids,
         primary_threat, mean_confidence, has_rule_hit]
    """
    _assert_ground_truth_blind(flows_df)

    if flows_df.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "key",
                "start_time",
                "end_time",
                "duration_s",
                "flow_count",
                "flow_ids",
                "primary_threat",
                "mean_confidence",
                "has_rule_hit",
            ]
        )

    # Required column check
    required = set(key_cols) | {"row_id", "epoch_s", "predicted_class", "confidence"}
    missing = required - set(flows_df.columns)
    if missing:
        raise KeyError(f"Input DataFrame missing required columns: {sorted(missing)}")

    has_rule_col = "r104_flag" in flows_df.columns

    # Group by key_cols
    grouped = flows_df.groupby(list(key_cols), sort=False)
    incidents: List[Dict[str, Any]] = []
    incident_counter = 1

    for group_key, group_df in grouped:
        # Sort each group by (epoch_s, row_id)
        sorted_df = group_df.sort_values(by=["epoch_s", "row_id"], ascending=[True, True])
        
        row_ids = sorted_df["row_id"].to_numpy()
        epochs = sorted_df["epoch_s"].to_numpy()
        preds = sorted_df["predicted_class"].tolist()
        confs = sorted_df["confidence"].tolist()
        rule_flags = sorted_df["r104_flag"].to_numpy() if has_rule_col else np.zeros(len(sorted_df), dtype=bool)

        key_str = str(group_key)

        # Active session state
        curr_row_ids: List[int] = [int(row_ids[0])]
        curr_epochs: List[int] = [int(epochs[0])]
        curr_preds: List[str] = [preds[0]]
        curr_confs: List[float] = [float(confs[0])]
        curr_rule_hits: bool = bool(rule_flags[0])
        last_epoch = int(epochs[0])

        for i in range(1, len(sorted_df)):
            ep = int(epochs[i])
            if ep - last_epoch <= gap_seconds:
                # Flow joins active session
                curr_row_ids.append(int(row_ids[i]))
                curr_epochs.append(ep)
                curr_preds.append(preds[i])
                curr_confs.append(float(confs[i]))
                if rule_flags[i]:
                    curr_rule_hits = True
                last_epoch = ep
            else:
                # Close active session and commit
                p_threat, m_conf = _determine_primary_threat(curr_preds, curr_confs)
                s_time = curr_epochs[0]
                e_time = curr_epochs[-1]
                incidents.append({
                    "incident_id": incident_counter,
                    "key": key_str,
                    "start_time": s_time,
                    "end_time": e_time,
                    "duration_s": float(e_time - s_time),
                    "flow_count": len(curr_row_ids),
                    "flow_ids": curr_row_ids,
                    "primary_threat": p_threat,
                    "mean_confidence": round(m_conf, 4),
                    "has_rule_hit": curr_rule_hits,
                })
                incident_counter += 1

                # Start new session
                curr_row_ids = [int(row_ids[i])]
                curr_epochs = [ep]
                curr_preds = [preds[i]]
                curr_confs = [float(confs[i])]
                curr_rule_hits = bool(rule_flags[i])
                last_epoch = ep

        # Commit final session for group
        p_threat, m_conf = _determine_primary_threat(curr_preds, curr_confs)
        s_time = curr_epochs[0]
        e_time = curr_epochs[-1]
        incidents.append({
            "incident_id": incident_counter,
            "key": key_str,
            "start_time": s_time,
            "end_time": e_time,
            "duration_s": float(e_time - s_time),
            "flow_count": len(curr_row_ids),
            "flow_ids": curr_row_ids,
            "primary_threat": p_threat,
            "mean_confidence": round(m_conf, 4),
            "has_rule_hit": curr_rule_hits,
        })
        incident_counter += 1

    return pd.DataFrame(incidents)


def correlate_tumbling_bins(
    flows_df: pd.DataFrame,
    key_cols: Sequence[str] = ("source_ip", "destination_ip", "destination_port"),
    bin_seconds: float = 300.0,
) -> pd.DataFrame:
    """Fixed tumbling-window event correlation baseline.
    
    Partitions each group by bin_id = floor(epoch_s / bin_seconds).
    All flows sharing the same (key, bin_id) merge into an incident regardless of inter-flow gaps.
    """
    _assert_ground_truth_blind(flows_df)

    if flows_df.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "key",
                "start_time",
                "end_time",
                "duration_s",
                "flow_count",
                "flow_ids",
                "primary_threat",
                "mean_confidence",
                "has_rule_hit",
            ]
        )

    required = set(key_cols) | {"row_id", "epoch_s", "predicted_class", "confidence"}
    missing = required - set(flows_df.columns)
    if missing:
        raise KeyError(f"Input DataFrame missing required columns: {sorted(missing)}")

    has_rule_col = "r104_flag" in flows_df.columns

    df = flows_df.copy()
    df["_bin_id"] = np.floor(df["epoch_s"] / bin_seconds).astype(np.int64)

    group_keys = list(key_cols) + ["_bin_id"]
    grouped = df.groupby(group_keys, sort=False)
    incidents: List[Dict[str, Any]] = []
    incident_counter = 1

    for group_key, group_df in grouped:
        sorted_df = group_df.sort_values(by=["epoch_s", "row_id"], ascending=[True, True])
        row_ids = sorted_df["row_id"].tolist()
        epochs = sorted_df["epoch_s"].to_numpy()
        preds = sorted_df["predicted_class"].tolist()
        confs = sorted_df["confidence"].tolist()
        rule_flags = sorted_df["r104_flag"].to_numpy() if has_rule_col else np.zeros(len(sorted_df), dtype=bool)

        key_tuple = group_key[:-1]  # exclude _bin_id from reported key
        p_threat, m_conf = _determine_primary_threat(preds, confs)
        s_time = int(epochs[0])
        e_time = int(epochs[-1])

        incidents.append({
            "incident_id": incident_counter,
            "key": str(key_tuple),
            "start_time": s_time,
            "end_time": e_time,
            "duration_s": float(e_time - s_time),
            "flow_count": len(row_ids),
            "flow_ids": row_ids,
            "primary_threat": p_threat,
            "mean_confidence": round(m_conf, 4),
            "has_rule_hit": bool(np.any(rule_flags)),
        })
        incident_counter += 1

    return pd.DataFrame(incidents)

