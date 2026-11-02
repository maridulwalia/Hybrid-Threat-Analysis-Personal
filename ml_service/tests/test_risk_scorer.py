"""
ml_service/tests/test_risk_scorer.py
────────────────────────────────────
Unit tests for Phase 9 IncidentRiskScorer.

Asserts:
1. Bounded output in [0, 100].
2. Exact additivity of the 4 sub-score contributions.
3. Monotonicity per component (C_ML, Sev, S_vol, B_rule).
4. Determinism.
5. Strict ground-truth blindness assertion.
6. Variant A vs Variant B behavior on rule-only incidents.
"""

import unittest
from typing import Dict, Any

from ml_service.src.risk.risk_scorer import (
    IncidentRiskScorer,
    SEVERITY_TABLE_PRIMARY,
)


class TestIncidentRiskScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = [0.35, 0.30, 0.20, 0.15]
        self.n_ref = 50.0
        self.scorer_a = IncidentRiskScorer(
            weights=self.weights,
            n_ref=self.n_ref,
            rule_variant="Variant_A",
        )
        self.scorer_b = IncidentRiskScorer(
            weights=self.weights,
            n_ref=self.n_ref,
            rule_variant="Variant_B",
        )

    def test_bounds(self) -> None:
        """Asserts risk score stays strictly within [0, 100] across extreme cases."""
        min_incident: Dict[str, Any] = {
            "p_attack_series": [0.0, 0.0],
            "primary_threat": "BENIGN",
            "n_flows": 0,
            "r104_any": False,
            "has_ml_flag": False,
        }
        res_min = self.scorer_a.score_incident(min_incident)
        self.assertEqual(res_min["risk_score"], 0.0)

        max_incident: Dict[str, Any] = {
            "p_attack_series": [1.0, 1.0, 1.0],
            "primary_threat": "Infiltration",
            "n_flows": 1000,
            "r104_any": True,
            "has_ml_flag": True,
        }
        res_max = self.scorer_a.score_incident(max_incident)
        self.assertAlmostEqual(res_max["risk_score"], 100.0, places=3)
        self.assertLessEqual(res_max["risk_score"], 100.0)

    def test_additivity(self) -> None:
        """Asserts composite score is exactly the sum of its four contributions."""
        incident: Dict[str, Any] = {
            "p_attack_series": [0.85, 0.95],
            "primary_threat": "FTP-Patator",
            "n_flows": 12,
            "r104_any": True,
            "has_ml_flag": True,
        }
        res = self.scorer_a.score_incident(incident)
        sum_contribs = (
            res["contrib_cml"]
            + res["contrib_sev"]
            + res["contrib_svol"]
            + res["contrib_brule"]
        )
        self.assertAlmostEqual(res["risk_score"], sum_contribs, places=3)

    def test_monotonicity(self) -> None:
        """Asserts increasing each individual component monotonically increases total score."""
        base: Dict[str, Any] = {
            "p_attack_series": [0.50],
            "primary_threat": "PortScan",  # Sev = 20
            "n_flows": 5,
            "r104_any": False,
            "has_ml_flag": True,
        }
        base_score = self.scorer_a.score_incident(base)["risk_score"]

        # 1. Higher p_attack
        more_conf = dict(base, p_attack_series=[0.90])
        self.assertGreater(self.scorer_a.score_incident(more_conf)["risk_score"], base_score)

        # 2. Higher Severity class
        more_sev = dict(base, primary_threat="SSH-Patator")  # Sev = 80
        self.assertGreater(self.scorer_a.score_incident(more_sev)["risk_score"], base_score)

        # 3. Higher volume
        more_vol = dict(base, n_flows=40)
        self.assertGreater(self.scorer_a.score_incident(more_vol)["risk_score"], base_score)

        # 4. Rule match
        more_rule = dict(base, r104_any=True)
        self.assertGreater(self.scorer_a.score_incident(more_rule)["risk_score"], base_score)

    def test_determinism(self) -> None:
        """Asserts scoring is completely deterministic across repeated evaluations."""
        inc: Dict[str, Any] = {
            "p_attack_series": [0.77, 0.88, 0.99],
            "primary_threat": "DoS Hulk",
            "n_flows": 3,
            "r104_any": False,
            "has_ml_flag": True,
        }
        score1 = self.scorer_a.score_incident(inc)
        score2 = self.scorer_a.score_incident(inc)
        self.assertEqual(score1, score2)

    def test_ground_truth_blindness(self) -> None:
        """Asserts passing any ground truth label raises ValueError."""
        leaky_incident: Dict[str, Any] = {
            "p_attack_series": [0.9],
            "primary_threat": "DDoS",
            "label": "DDoS",
        }
        with self.assertRaises(ValueError):
            self.scorer_a.score_incident(leaky_incident)

    def test_rule_variants(self) -> None:
        """Asserts Variant B suppresses rule bonus when ML predicted BENIGN."""
        rule_only_inc: Dict[str, Any] = {
            "p_attack_series": [0.01, 0.02],
            "primary_threat": "RULE_ONLY:R104",
            "n_flows": 2,
            "r104_any": True,
            "has_ml_flag": False,  # ML did not flag this incident
        }
        res_a = self.scorer_a.score_incident(rule_only_inc)
        res_b = self.scorer_b.score_incident(rule_only_inc)

        # Variant A grants the rule bonus
        self.assertEqual(res_a["b_rule_raw"], 100.0)
        self.assertGreater(res_a["contrib_brule"], 0.0)

        # Variant B suppresses the rule bonus for rule-only false alarms
        self.assertEqual(res_b["b_rule_raw"], 0.0)
        self.assertEqual(res_b["contrib_brule"], 0.0)
        self.assertLess(res_b["risk_score"], res_a["risk_score"])


if __name__ == "__main__":
    unittest.main()

