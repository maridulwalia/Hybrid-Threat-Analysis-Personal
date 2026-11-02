"""
ml_service/src/risk/risk_scorer.py
──────────────────────────────────
Phase 9: Transparent Incident Risk Scoring Engine.

Implements the formal risk scoring formulation:
    Score = w1 * C_ML + w2 * Sev + w3 * S_vol + w4 * B_rule

Constraints:
- Each component strictly bounded in [0, 100].
- w_k >= 0 and sum(w_k) == 1.0 -> Score in [0, 100].
- Decomposes score into 4 explicit additive sub-score contributions.
- Strict ground-truth blindness: receives only retrospective correlation
  telemetry (constituent flow predictions, rule flags, flow counts, primary threat);
  never receives ground-truth labels.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# 1. Analyst-assigned Severity Tables (Assumptions, not derived from NIST/MITRE)
SEVERITY_TABLE_PRIMARY: Dict[str, float] = {
    # Critical Tier (100)
    "Infiltration": 100.0,
    "Heartbleed": 100.0,
    "Web Attack - Sql Injection": 100.0,
    # High Tier (80)
    "SSH-Patator": 80.0,
    "FTP-Patator": 80.0,
    "Web Attack - Brute Force": 80.0,
    # Medium Tier (60)
    "Bot": 60.0,
    "Web Attack - XSS": 60.0,
    # Low / Volumetric Availability Tier (40)
    "DDoS": 40.0,
    "DoS Hulk": 40.0,
    "DoS GoldenEye": 40.0,
    "DoS slowloris": 40.0,
    "DoS Slowhttptest": 40.0,
    # Reconnaissance Tier (20)
    "PortScan": 20.0,
    # Zero Tier (0)
    "BENIGN": 0.0,
    "RULE_ONLY:R104": 0.0,
}

SEVERITY_TABLE_ALT1: Dict[str, float] = {
    # Linear 25-step Tiers
    "Infiltration": 100.0,
    "Heartbleed": 100.0,
    "Web Attack - Sql Injection": 100.0,
    "SSH-Patator": 75.0,
    "FTP-Patator": 75.0,
    "Web Attack - Brute Force": 75.0,
    "Bot": 50.0,
    "Web Attack - XSS": 50.0,
    "DDoS": 25.0,
    "DoS Hulk": 25.0,
    "DoS GoldenEye": 25.0,
    "DoS slowloris": 25.0,
    "DoS Slowhttptest": 25.0,
    "PortScan": 10.0,
    "BENIGN": 0.0,
    "RULE_ONLY:R104": 0.0,
}

SEVERITY_TABLE_ALT2: Dict[str, float] = {
    # Escalation & Exfiltration Focus
    "Infiltration": 100.0,
    "Web Attack - Sql Injection": 100.0,
    "Heartbleed": 90.0,
    "SSH-Patator": 70.0,
    "FTP-Patator": 70.0,
    "Web Attack - Brute Force": 70.0,
    "DDoS": 50.0,
    "DoS Hulk": 50.0,
    "DoS GoldenEye": 50.0,
    "DoS slowloris": 50.0,
    "DoS Slowhttptest": 50.0,
    "Bot": 40.0,
    "Web Attack - XSS": 40.0,
    "PortScan": 15.0,
    "BENIGN": 0.0,
    "RULE_ONLY:R104": 0.0,
}


class IncidentRiskScorer:
    """
    Transparent, explainable incident risk scorer.
    Computes bounded composite risk score in [0, 100] alongside additive sub-score contributions.
    """

    def __init__(
        self,
        weights: Sequence[float],
        n_ref: float,
        rule_variant: str = "Variant_A",
        severity_table: Optional[Dict[str, float]] = None,
    ) -> None:
        if len(weights) != 4:
            raise ValueError(f"Expected 4 weights (w1, w2, w3, w4), got {len(weights)}")
        sum_w = sum(weights)
        if not math.isclose(sum_w, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {sum_w:.6f}")
        for idx, w in enumerate(weights):
            if w < 0.0:
                raise ValueError(f"Weight w{idx+1} must be non-negative, got {w}")

        if n_ref <= 0.0:
            raise ValueError(f"n_ref must be positive, got {n_ref}")

        if rule_variant not in ("Variant_A", "Variant_B"):
            raise ValueError(f"Invalid rule_variant: {rule_variant}. Expected Variant_A or Variant_B.")

        self.w1, self.w2, self.w3, self.w4 = float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])
        self.n_ref = float(n_ref)
        self.rule_variant = rule_variant
        self.severity_table = severity_table if severity_table is not None else SEVERITY_TABLE_PRIMARY
        self._ln_n_ref_plus_one = math.log(1.0 + self.n_ref)

    def _assert_ground_truth_blind(self, telemetry: Dict[str, Any]) -> None:
        """Enforces that the scoring engine never accesses evaluation labels."""
        forbidden = {"label", "labels", "ground_truth", "gt_label", "true_label", "is_attack_true"}
        found = forbidden.intersection(telemetry.keys())
        if found:
            raise ValueError(f"Ground-truth telemetry leak detected: {found}. Scorer must remain strictly blind.")

    def compute_components(
        self,
        p_attack_series: Sequence[float],
        primary_threat: str,
        n_flows: int,
        r104_any: bool,
        has_ml_flag: bool,
    ) -> Tuple[float, float, float, float]:
        """
        Computes the 4 raw score components, each in [0, 100]:
        1. C_ML: 100 * mean(p_attack over constituent flows)
        2. Sev: Table lookup for primary threat
        3. S_vol: 100 * ln(1 + N) / ln(1 + N_ref), clipped at 100
        4. B_rule: 100 if rule matched (under Variant A or B), else 0
        """
        # 1. C_ML
        if len(p_attack_series) == 0:
            c_ml = 0.0
        else:
            c_ml = 100.0 * float(sum(p_attack_series) / len(p_attack_series))
        c_ml = max(0.0, min(100.0, c_ml))

        # 2. Sev
        sev = float(self.severity_table.get(primary_threat, 0.0))
        sev = max(0.0, min(100.0, sev))

        # 3. S_vol
        if n_flows <= 0:
            s_vol = 0.0
        elif self._ln_n_ref_plus_one <= 0.0:
            s_vol = 100.0
        else:
            raw_vol = 100.0 * (math.log(1.0 + n_flows) / self._ln_n_ref_plus_one)
            s_vol = max(0.0, min(100.0, raw_vol))

        # 4. B_rule
        if self.rule_variant == "Variant_A":
            b_rule = 100.0 if r104_any else 0.0
        else:  # Variant_B
            b_rule = 100.0 if (r104_any and has_ml_flag) else 0.0

        return c_ml, sev, s_vol, b_rule

    def score_incident(
        self,
        incident_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Scores a single correlated incident from its retrospective telemetry.
        Returns:
            {
                "risk_score": float [0, 100],
                "c_ml_raw": float [0, 100],
                "sev_raw": float [0, 100],
                "s_vol_raw": float [0, 100],
                "b_rule_raw": float [0, 100],
                "contrib_cml": float,
                "contrib_sev": float,
                "contrib_svol": float,
                "contrib_brule": float,
            }
        """
        self._assert_ground_truth_blind(incident_data)

        p_attack_series = incident_data["p_attack_series"]
        primary_threat = incident_data["primary_threat"]
        n_flows = incident_data.get("n_flows", len(p_attack_series))
        r104_any = bool(incident_data.get("r104_any", False))
        has_ml_flag = bool(incident_data.get("has_ml_flag", False))

        c_ml, sev, s_vol, b_rule = self.compute_components(
            p_attack_series=p_attack_series,
            primary_threat=primary_threat,
            n_flows=n_flows,
            r104_any=r104_any,
            has_ml_flag=has_ml_flag,
        )

        contrib_cml = self.w1 * c_ml
        contrib_sev = self.w2 * sev
        contrib_svol = self.w3 * s_vol
        contrib_brule = self.w4 * b_rule

        total_score = contrib_cml + contrib_sev + contrib_svol + contrib_brule
        total_score = max(0.0, min(100.0, total_score))

        return {
            "risk_score": round(total_score, 4),
            "c_ml_raw": round(c_ml, 4),
            "sev_raw": round(sev, 4),
            "s_vol_raw": round(s_vol, 4),
            "b_rule_raw": round(b_rule, 4),
            "contrib_cml": round(contrib_cml, 4),
            "contrib_sev": round(contrib_sev, 4),
            "contrib_svol": round(contrib_svol, 4),
            "contrib_brule": round(contrib_brule, 4),
        }

