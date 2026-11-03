"""
ml_service/tests/test_client_key_sanitization.py
────────────────────────────────────────────────
Tests for client exception sanitization, key redaction, and fail-safe template fallback.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_service.src.llm.bundle_builder import build_evidence_bundle
from ml_service.src.llm.client import GeminiTransportError, LLMClientAdapter, sanitize_text
from ml_service.src.llm.schemas import GenerationMode


def test_key_sanitization():
    fake_secret = "AIzaSyB1234567890abcdefghijklmnopqrstuv"
    leak_string = f"Error connecting to https://generativelanguage.googleapis.com with key={fake_secret}"
    sanitized = sanitize_text(leak_string)
    assert fake_secret not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized

    # Exception sanitization
    err = GeminiTransportError(f"HTTP 401 using key {fake_secret}")
    err_str = str(err)
    assert fake_secret not in err_str
    assert "[REDACTED_API_KEY]" in err_str


def test_client_fallback_without_key():
    sample_incident = {
        "incident_id": 1001,
        "source_ip": "172.16.0.1",
        "destination_ip": "192.168.10.50",
        "destination_port": 22,
        "duration_s": 5.0,
        "flow_count": 32,
        "primary_threat": "SSH-Patator",
        "severity_tier": "High",
        "risk_score": 97.98,
        "sub_scores": {
            "c_ml_contrib": 29.98,
            "sev_contrib": 8.00,
            "vol_contrib": 30.00,
            "rule_contrib": 30.00,
        },
        "mean_ml_confidence": 0.9995,
        "has_rule_hit": True,
    }
    bundle = build_evidence_bundle(sample_incident)

    # Initialize client without key to verify graceful fallback
    client = LLMClientAdapter(api_key=None, timeout_seconds=1.0)
    record = client.generate_decision(bundle, use_cache=False)

    assert record.generation_mode == GenerationMode.TEMPLATE_FALLBACK
    assert record.payload.incident_id == 1001
    assert record.payload.primary_hypothesis.predicted_class == "SSH-Patator"
    assert len(record.payload.grounded_claims) >= 3


if __name__ == "__main__":
    test_key_sanitization()
    test_client_fallback_without_key()
    print("ALL CLIENT KEY SANITIZATION & FALLBACK TESTS PASSED!")

