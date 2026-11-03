"""
ml_service/tests/test_validator_and_fallback.py
───────────────────────────────────────────────
Unit tests for DecisionValidator and Fallback Template Generator.
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_service.src.llm.bundle_builder import build_evidence_bundle
from ml_service.src.llm.fallback_template import generate_fallback_decision
from ml_service.src.llm.validator import DecisionValidator


def test_fallback_template_and_validator():
    sample_incident = {
        "incident_id": 9999,
        "source_ip": "172.16.0.1",
        "destination_ip": "192.168.10.50",
        "destination_port": 80,
        "duration_s": 300.0,
        "flow_count": 28,
        "primary_threat": "DoS Hulk",
        "severity_tier": "Low / Volumetric Availability",
        "risk_score": 63.95,
        "sub_scores": {
            "c_ml_contrib": 29.95,
            "sev_contrib": 4.00,
            "vol_contrib": 30.00,
            "rule_contrib": 0.00,
        },
        "mean_ml_confidence": 0.9982,
        "has_rule_hit": False,
    }

    # 1. Build bundle
    bundle = build_evidence_bundle(sample_incident)
    assert "bundle_hash" in bundle
    assert "INC-01" in bundle["items"]
    assert "PRED-1" in bundle["items"]
    assert "KNOW-1" in bundle["items"]

    # 2. Generate fallback decision
    decision = generate_fallback_decision(bundle)
    assert decision.incident_id == 9999
    assert decision.primary_hypothesis.predicted_class == "DoS Hulk"

    # 3. Validate fallback decision (MUST PASS 100%)
    validator = DecisionValidator(bundle)
    res = validator.validate(decision)
    assert res.valid, f"Fallback output failed validation: {res.errors}"
    assert res.output is not None

    # 4. Test Entity Hallucination Detection: Insert bogus IP
    bad_decision_dict = decision.model_dump()
    bad_decision_dict["grounded_claims"][0]["claim_text"] = "Attacker from 10.99.99.99 initiated connection."
    res_bad_ip = validator.validate(bad_decision_dict)
    assert not res_bad_ip.valid
    assert any("ClaimEntityError" in err and "10.99.99.99" in err for err in res_bad_ip.errors)

    # 5. Test Missing Citation Detection: Cite nonexistent ID
    bad_cite_dict = decision.model_dump()
    bad_cite_dict["grounded_claims"][0]["evidence_citations"] = ["NONEXISTENT-ID-999"]
    res_bad_cite = validator.validate(bad_cite_dict)
    assert not res_bad_cite.valid
    assert any("UnresolvedCitationsError" in err for err in res_bad_cite.errors)

    # 6. Test Forbidden Assertion Check
    bad_assert_dict = decision.model_dump()
    bad_assert_dict["assessment_summary"]["text"] += " This is a confirmed attacker with proven compromise."
    # Set low probability to trigger forbidden check
    bundle_low = dict(bundle)
    bundle_low["items"] = dict(bundle["items"])
    bundle_low["items"]["PRED-1"] = {"field": "primary_class", "value": "DoS Hulk", "probability": 0.40}
    val_low = DecisionValidator(bundle_low)
    res_bad_assert = val_low.validate(bad_assert_dict)
    assert not res_bad_assert.valid
    assert any("ForbiddenAssertionError" in err for err in res_bad_assert.errors)


if __name__ == "__main__":
    test_fallback_template_and_validator()
    print("ALL VALIDATOR & FALLBACK TESTS PASSED!")
