"""
ml_service/src/rules/rule_engine.py
───────────────────────────────────
Phase 7: Deterministic Security Rule Engine.

Implements declarative, vectorized security rule evaluation operating
independently of machine learning models per AGENTS.md:
- Pure deterministic boundaries (no ML dependencies).
- Operates on raw (unscaled) flow metrics and Destination Port.
- Evaluates individual flow records (dict) or vectorized batches (DataFrame).
- Outputs matched rule IDs, highest severity, and boolean violation status.
- Decoupled from numeric score points (weights w1-w4 deferred to Phase 9).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("rule_engine")


@dataclass
class RuleDefinition:
    rule_id: str
    name: str
    target_threat: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    is_active: bool  # True if passed precision floor (>= 0.50 on train), else INFORMATIONAL
    description: str
    thresholds: Dict[str, Any]
    precision_train: Optional[float] = None
    recall_train: Optional[float] = None
    notes: str = ""


# Default candidate rule definitions (threshold hypotheses)
DEFAULT_RULES: List[RuleDefinition] = [
    RuleDefinition(
        rule_id="R101_SYN_HALF_OPEN",
        name="TCP Half-Open / SYN Flood Anomaly",
        target_threat="DoS / PortScan",
        severity="HIGH",
        is_active=True,
        description="SYN flag present with zero ACK, short flow duration (< 1s), and minimal packets",
        thresholds={
            "syn_flag_min": 1,
            "ack_flag_max": 0,
            "duration_max_us": 1_000_000,
            "fwd_pkts_max": 3,
        },
        notes="Identifies incomplete TCP handshake floods and rapid SYN sweeps.",
    ),
    RuleDefinition(
        rule_id="R102_HIGH_RATE_BURST",
        name="Volumetric Flow Rate Exhaustion",
        target_threat="DoS Hulk / DDoS",
        severity="CRITICAL",
        is_active=True,
        description="Extremely high packet rate (> 50,000 pkts/s) with established duration and packet volume",
        thresholds={
            "rate_min_pkts_s": 50_000.0,
            "duration_min_us": 1_000,
            "fwd_pkts_min": 10,
        },
        notes="Duration floor prevents division-by-near-zero rate spikes on 2-packet benign queries.",
    ),
    RuleDefinition(
        rule_id="R103_SLOW_EXHAUSTION",
        name="Slow Connection Resource Starvation",
        target_threat="DoS Slowloris / Slowhttptest",
        severity="HIGH",
        is_active=True,
        description="Extended flow duration (> 10s) with low packet volume and small payload lengths",
        thresholds={
            "duration_min_us": 10_000_000,
            "fwd_pkts_max": 10,
            "pkt_len_mean_max": 100.0,
        },
        notes="Captures slow HTTP header starvation patterns holding server sockets open.",
    ),
    RuleDefinition(
        rule_id="R104_PORT_BRUTE_FORCE",
        name="Administrative Port Credential Attack",
        target_threat="FTP / SSH Brute Force",
        severity="HIGH",
        is_active=True,
        description="Targeting ports 21 or 22 with fixed 32-byte segment size and multiple exchange packets",
        thresholds={
            "target_ports": [21, 22],
            "min_seg_size_fwd": 32,
            "bwd_pkts_min": 5,
        },
        notes="Capture-specific artifact in CIC-IDS2017 reflecting automated dictionary attack handshakes.",
    ),
    RuleDefinition(
        rule_id="R105_ZERO_WINDOW_TEARDOWN",
        name="Anomalous Window Teardown Anomaly",
        target_threat="TCP Evasion / Scan",
        severity="MEDIUM",
        is_active=True,
        description="Zero backward initial window combined with RST flag on rapid termination",
        thresholds={
            "init_win_bwd_max": 0,
            "rst_flag_min": 1,
            "duration_max_us": 100_000,
        },
        notes="Detects anomalous immediate connection aborts without window negotiation.",
    ),
]


class SecurityRuleEngine:
    """
    Deterministic rule engine that validates network flow telemetry.
    Can evaluate single flow dictionaries or vectorized pandas DataFrames.
    """

    SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(
        self,
        rules: Optional[Sequence[RuleDefinition]] = None,
        schema_path: Optional[Path] = None,
    ) -> None:
        self.rules: List[RuleDefinition] = list(rules if rules is not None else DEFAULT_RULES)
        self.rules_by_id: Dict[str, RuleDefinition] = {r.rule_id: r for r in self.rules}

        if schema_path is not None and schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            self.valid_columns = set(
                schema_data.get("ml_feature_columns", [])
                + schema_data.get("identifier_columns", [])
            )
        else:
            self.valid_columns = set()

    def get_required_columns(self) -> List[str]:
        """Returns the minimal set of raw CSV column names required by all rules."""
        return [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Flow Packets/s",
            "Packet Length Mean",
            "SYN Flag Count",
            "ACK Flag Count",
            "RST Flag Count",
            "Init_Win_bytes_backward",
            "min_seg_size_forward",
        ]

    def evaluate_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Vectorized evaluation of all rules across a batch of raw flow records.
        Returns a dictionary with boolean hit arrays per rule, an ensemble mask,
        and per-row matched rule lists.
        """
        n_rows = len(df)
        rule_hits: Dict[str, np.ndarray] = {}

        # Rule 1: R101_SYN_HALF_OPEN
        r1 = self.rules_by_id.get("R101_SYN_HALF_OPEN")
        if r1 is not None:
            t = r1.thresholds
            hit_r1 = (
                (df["SYN Flag Count"].to_numpy() >= t["syn_flag_min"])
                & (df["ACK Flag Count"].to_numpy() <= t["ack_flag_max"])
                & (df["Flow Duration"].to_numpy() < t["duration_max_us"])
                & (df["Total Fwd Packets"].to_numpy() <= t["fwd_pkts_max"])
            )
            rule_hits["R101_SYN_HALF_OPEN"] = hit_r1

        # Rule 2: R102_HIGH_RATE_BURST
        r2 = self.rules_by_id.get("R102_HIGH_RATE_BURST")
        if r2 is not None:
            t = r2.thresholds
            hit_r2 = (
                (df["Flow Packets/s"].to_numpy() >= t["rate_min_pkts_s"])
                & (df["Flow Duration"].to_numpy() >= t["duration_min_us"])
                & (df["Total Fwd Packets"].to_numpy() >= t["fwd_pkts_min"])
            )
            rule_hits["R102_HIGH_RATE_BURST"] = hit_r2

        # Rule 3: R103_SLOW_EXHAUSTION
        r3 = self.rules_by_id.get("R103_SLOW_EXHAUSTION")
        if r3 is not None:
            t = r3.thresholds
            hit_r3 = (
                (df["Flow Duration"].to_numpy() >= t["duration_min_us"])
                & (df["Total Fwd Packets"].to_numpy() <= t["fwd_pkts_max"])
                & (df["Packet Length Mean"].to_numpy() <= t["pkt_len_mean_max"])
            )
            rule_hits["R103_SLOW_EXHAUSTION"] = hit_r3

        # Rule 4: R104_PORT_BRUTE_FORCE
        r4 = self.rules_by_id.get("R104_PORT_BRUTE_FORCE")
        if r4 is not None:
            t = r4.thresholds
            port_arr = df["Destination Port"].to_numpy()
            port_match = np.isin(port_arr, t["target_ports"])
            hit_r4 = (
                port_match
                & (df["min_seg_size_forward"].to_numpy() == t["min_seg_size_fwd"])
                & (df["Total Backward Packets"].to_numpy() >= t["bwd_pkts_min"])
            )
            rule_hits["R104_PORT_BRUTE_FORCE"] = hit_r4

        # Rule 5: R105_ZERO_WINDOW_TEARDOWN
        r5 = self.rules_by_id.get("R105_ZERO_WINDOW_TEARDOWN")
        if r5 is not None:
            t = r5.thresholds
            hit_r5 = (
                (df["Init_Win_bytes_backward"].to_numpy() <= t["init_win_bwd_max"])
                & (df["RST Flag Count"].to_numpy() >= t["rst_flag_min"])
                & (df["Flow Duration"].to_numpy() < t["duration_max_us"])
            )
            rule_hits["R105_ZERO_WINDOW_TEARDOWN"] = hit_r5

        # Active ensemble violation (only active rules count toward active alert)
        active_ids = [r.rule_id for r in self.rules if r.is_active and r.rule_id in rule_hits]
        if active_ids:
            ensemble_violation = np.logical_or.reduce([rule_hits[rid] for rid in active_ids])
        else:
            ensemble_violation = np.zeros(n_rows, dtype=bool)

        return {
            "rule_hits": rule_hits,
            "ensemble_violation": ensemble_violation,
            "n_rows": n_rows,
        }

    def evaluate_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single raw flow record dictionary.
        Returns matched rule IDs, violation boolean, and highest severity tag.
        """
        matched_rules: List[str] = []
        highest_severity = "NONE"

        port = flow.get("Destination Port", 0)
        duration = flow.get("Flow Duration", 0)
        fwd_pkts = flow.get("Total Fwd Packets", 0)
        bwd_pkts = flow.get("Total Backward Packets", 0)
        flow_pkts_s = flow.get("Flow Packets/s", 0.0)
        pkt_len_mean = flow.get("Packet Length Mean", 0.0)
        syn_flags = flow.get("SYN Flag Count", 0)
        ack_flags = flow.get("ACK Flag Count", 0)
        rst_flags = flow.get("RST Flag Count", 0)
        init_win_bwd = flow.get("Init_Win_bytes_backward", -1)
        min_seg_fwd = flow.get("min_seg_size_forward", 0)

        for rule in self.rules:
            hit = False
            t = rule.thresholds

            if rule.rule_id == "R101_SYN_HALF_OPEN":
                hit = (
                    syn_flags >= t["syn_flag_min"]
                    and ack_flags <= t["ack_flag_max"]
                    and duration < t["duration_max_us"]
                    and fwd_pkts <= t["fwd_pkts_max"]
                )
            elif rule.rule_id == "R102_HIGH_RATE_BURST":
                hit = (
                    flow_pkts_s >= t["rate_min_pkts_s"]
                    and duration >= t["duration_min_us"]
                    and fwd_pkts >= t["fwd_pkts_min"]
                )
            elif rule.rule_id == "R103_SLOW_EXHAUSTION":
                hit = (
                    duration >= t["duration_min_us"]
                    and fwd_pkts <= t["fwd_pkts_max"]
                    and pkt_len_mean <= t["pkt_len_mean_max"]
                )
            elif rule.rule_id == "R104_PORT_BRUTE_FORCE":
                hit = (
                    port in t["target_ports"]
                    and min_seg_fwd == t["min_seg_size_fwd"]
                    and bwd_pkts >= t["bwd_pkts_min"]
                )
            elif rule.rule_id == "R105_ZERO_WINDOW_TEARDOWN":
                hit = (
                    init_win_bwd <= t["init_win_bwd_max"]
                    and rst_flags >= t["rst_flag_min"]
                    and duration < t["duration_max_us"]
                )

            if hit:
                matched_rules.append(rule.rule_id)
                if rule.is_active:
                    current_idx = self.SEVERITY_ORDER.index(highest_severity)
                    rule_idx = self.SEVERITY_ORDER.index(rule.severity)
                    if rule_idx > current_idx:
                        highest_severity = rule.severity

        has_active_violation = any(
            self.rules_by_id[rid].is_active for rid in matched_rules if rid in self.rules_by_id
        )

        return {
            "matched_rules": matched_rules,
            "has_violation": has_active_violation,
            "highest_severity": highest_severity,
        }

    def save_definitions(self, filepath: Path) -> None:
        """Saves rule catalog to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "engine_version": "v1.0",
            "phase": 7,
            "rules": [asdict(r) for r in self.rules],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d rule definitions to %s", len(self.rules), filepath.name)

    @classmethod
    def load_definitions(cls, filepath: Path) -> SecurityRuleEngine:
        """Loads rule catalog from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        rules = [RuleDefinition(**r) for r in data.get("rules", [])]
        return cls(rules=rules)

