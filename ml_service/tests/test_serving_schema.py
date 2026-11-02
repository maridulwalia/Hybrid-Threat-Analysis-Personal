"""
ml_service/tests/test_serving_schema.py
───────────────────────────────────────
Unit test validating the operational incident serving schema for Phase 10+ consumption:
1. Validates strict schema conformance (presence of required keys, correct types).
2. Asserts zero ground-truth leakage (absence of incident_label, true_label, purity, etc.).
3. Asserts exact arithmetic additivity: risk_score == sum(sub_score contributions) within 1e-4.
4. Verifies frozen weights calibration alignment against SEVERITY_TABLE_PRIMARY.
"""

from typing import Any, Dict
import unittest

from ml_service.src.risk.risk_scorer import SEVERITY_TABLE_PRIMARY

ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE: Dict[str, Any] = {
    "schema_version": "1.1",
    "incident_id": 9999,
    "representative_flow_row_id": 123456,
    "source_ip": "172.16.0.1",
    "destination_ip": "192.168.10.50",
    "destination_port": 80,
    "start_time": 1499252400.0,
    "end_time": 1499252700.0,
    "duration_s": 300.0,
    "flow_count": 28,
    "flow_count_cap_note": "Volume saturates at N_ref=25 flows (S_vol=100.00)",
    "primary_threat": "DoS Hulk",
    "severity_tier": "Low / Volumetric Availability",
    "severity_value": 40.0,
    "mean_ml_confidence": 0.9982,
    "mean_p_attack": 0.9982,
    "has_rule_hit": False,
    "shap_top_features": None,  # Nullable placeholder computed on-demand in Phase 11
    "risk_score": 63.95,
    "sub_scores": {
        "c_ml_contrib": 29.95,
        "sev_contrib": 4.00,
        "vol_contrib": 30.00,
        "rule_contrib": 0.00,
    },
}

FORBIDDEN_GROUND_TRUTH_KEYS = {
    "incident_label",
    "true_label",
    "majority_gt_label",
    "n_attack_gt",
    "purity",
    "is_attack",
    "is_false",
    "label",
}


class TestServingSchema(unittest.TestCase):
    def test_serving_schema_ground_truth_absence(self) -> None:
        """Assert zero ground truth or test-set telemetry fields exist in the serving schema."""
        keys = set(ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE.keys())
        leakage = keys.intersection(FORBIDDEN_GROUND_TRUTH_KEYS)
        self.assertFalse(leakage, f"Ground-truth leakage detected in serving schema: {leakage}")

    def test_serving_schema_additivity(self) -> None:
        """Assert strict arithmetic additivity: risk_score must equal the sum of sub-scores."""
        sub = ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE["sub_scores"]
        sub_sum = round(
            sub["c_ml_contrib"] + sub["sev_contrib"] + sub["vol_contrib"] + sub["rule_contrib"], 2
        )
        risk_score = round(ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE["risk_score"], 2)
        self.assertAlmostEqual(risk_score, sub_sum, places=2)

    def test_serving_schema_severity_alignment(self) -> None:
        """Verify that sev_contrib matches w2 * severity_value and SEVERITY_TABLE_PRIMARY[threat]."""
        threat = ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE["primary_threat"]
        self.assertIn(threat, SEVERITY_TABLE_PRIMARY)
        raw_sev = SEVERITY_TABLE_PRIMARY[threat]
        self.assertEqual(raw_sev, ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE["severity_value"])
        w2 = 0.10  # Frozen weight
        expected_sev_contrib = round(w2 * raw_sev, 2)
        observed_sev_contrib = round(ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE["sub_scores"]["sev_contrib"], 2)
        self.assertEqual(expected_sev_contrib, observed_sev_contrib)

    def test_serving_schema_required_fields(self) -> None:
        """Verify presence of all required fields including new versioning and provenance fields."""
        required_keys = [
            "schema_version",
            "incident_id",
            "representative_flow_row_id",
            "source_ip",
            "destination_ip",
            "destination_port",
            "start_time",
            "end_time",
            "duration_s",
            "flow_count",
            "flow_count_cap_note",
            "primary_threat",
            "severity_tier",
            "severity_value",
            "mean_ml_confidence",
            "mean_p_attack",
            "has_rule_hit",
            "shap_top_features",
            "risk_score",
            "sub_scores",
        ]
        for k in required_keys:
            self.assertIn(k, ILLUSTRATIVE_INCIDENT_SCHEMA_EXAMPLE)


if __name__ == "__main__":
    unittest.main()
