"""
ml_service/src/llm/bundle_builder.py
────────────────────────────────────
Constructs deterministic Evidence Bundles from operational incident records,
risk decompositions, SHAP feature attributions, and retrieved knowledge chunks.
Zero ground-truth fields are ever exposed to the bundle.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
from config.paths import Paths

logger = logging.getLogger("bundle_builder")

# Global caches for static mappings and rules
_TECH_MAP_CACHE: Optional[Dict[str, Any]] = None
_RULES_CACHE: Optional[Dict[str, Any]] = None


def _load_mappings() -> Dict[str, Any]:
    global _TECH_MAP_CACHE
    if _TECH_MAP_CACHE is None:
        map_path = Paths.ARTIFACTS_DIR / "rag" / "label_technique_map_v2.json"
        with open(map_path, "r", encoding="utf-8") as f:
            _TECH_MAP_CACHE = json.load(f)["mappings"]
    return _TECH_MAP_CACHE


def _load_rules() -> Dict[str, Any]:
    global _RULES_CACHE
    if _RULES_CACHE is None:
        rules_path = Paths.ARTIFACTS_DIR / "rules" / "rule_definitions_v1.json"
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_list = json.load(f)["rules"]
            _RULES_CACHE = {r["rule_id"]: r for r in rules_list}
    return _RULES_CACHE


def get_knowledge_items(primary_threat: str) -> Dict[str, Dict[str, Any]]:
    """Retrieve knowledge items for a given primary threat label using label_technique_map_v2."""
    mappings = _load_mappings()

    entry = mappings.get(primary_threat)
    is_generic = False
    if entry is None:
        entry = mappings.get("GENERIC_FALLBACK", {})
        is_generic = True
    elif primary_threat == "RULE_ONLY:R104":
        is_generic = True

    items: Dict[str, Dict[str, Any]] = {}
    tech_id = entry.get("primary_technique_id", "NIST-IR")
    tech_name = entry.get("primary_technique_name", "Incident Response")

    # Add primary mitigation
    mitigations = entry.get("primary_mitigations", [])
    mit_text = " ".join(mitigations) if mitigations else "Isolate host and filter perimeter traffic."
    items["KNOW-1"] = {
        "chunk_id": f"chunk_{tech_id}_mitigation",
        "technique_id": tech_id,
        "technique_name": tech_name,
        "section_type": "mitigation",
        "source_url": f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}/" if tech_id.startswith("T") else "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        "is_generic": is_generic,
        "text": mit_text,
    }

    # Add detection analytics or rationale
    detect_text = entry.get("rationale", "")
    if entry.get("detection_analytics"):
        detect_text += " " + " ".join(entry.get("detection_analytics", []))

    items["KNOW-2"] = {
        "chunk_id": f"chunk_{tech_id}_detection",
        "technique_id": tech_id,
        "technique_name": tech_name,
        "section_type": "containment" if is_generic else "investigation",
        "source_url": f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}/" if tech_id.startswith("T") else "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        "is_generic": is_generic,
        "text": detect_text.strip(),
    }

    return items


def compute_shap_top_k(
    explainer: Any,
    scaler: Any,
    flow_features_scaled: np.ndarray,
    predicted_class_idx: int,
    feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Compute top-k SHAP feature attributions on a single flow."""
    # Ensure shape (1, n_features)
    x_input = np.expand_dims(flow_features_scaled, axis=0) if flow_features_scaled.ndim == 1 else flow_features_scaled
    shap_vals = explainer(x_input)
    
    # Extract values for the target class
    # TreeExplainer on multiclass returns shap_values of shape (1, 76, 15) or array of (1, 76)
    if hasattr(shap_vals, "values"):
        vals = shap_vals.values
        if vals.ndim == 3:
            class_contribs = vals[0, :, predicted_class_idx]
        elif vals.ndim == 2:
            class_contribs = vals[0, :]
        else:
            class_contribs = vals
    else:
        class_contribs = np.zeros(len(feature_names))

    # Unscale flow features via scaler inverse
    unscaled_flow = scaler.inverse_transform(x_input)[0]

    # Sort by absolute magnitude
    abs_indices = np.argsort(np.abs(class_contribs))[::-1][:top_k]

    shap_items: Dict[str, Dict[str, Any]] = {}
    for rank, idx in enumerate(abs_indices, start=1):
        attr = float(class_contribs[idx])
        direction = "increases_threat_probability" if attr > 0 else "decreases_threat_probability"
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        raw_val = float(unscaled_flow[idx])

        shap_items[f"SHAP-{rank}"] = {
            "feature": feat_name,
            "raw_value": round(raw_val, 4),
            "scaled_value": round(float(flow_features_scaled[idx]), 4),
            "attribution": round(attr, 4),
            "direction": direction,
            "rank": rank,
        }

    return shap_items


def build_evidence_bundle(
    incident: Dict[str, Any],
    top2_predictions: Optional[Tuple[str, float, str, float]] = None,
    shap_items: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Construct a complete, versioned, canonically hashed Evidence Bundle.
    
    Parameters:
        incident: Operational incident record (schema v1.1).
        top2_predictions: Optional tuple of (pred1_class, pred1_prob, pred2_class, pred2_prob).
        shap_items: Optional precomputed SHAP items.
    """
    inc_id = int(incident.get("incident_id", 0))
    src_ip = str(incident.get("source_ip", "0.0.0.0"))
    dst_ip = str(incident.get("destination_ip", "0.0.0.0"))
    dst_port = int(incident.get("destination_port", 0))
    duration_s = float(incident.get("duration_s", 0.0))
    flow_count = int(incident.get("flow_count", incident.get("n_flows", 1)))
    flow_cap_note = str(incident.get("flow_count_cap_note", "Volume saturates at N_ref=25 flows (S_vol=100.00)"))

    primary_threat = str(incident.get("primary_threat", "BENIGN"))
    risk_score = float(incident.get("risk_score", 0.0))
    sev_tier = str(incident.get("severity_tier", "Unknown"))
    
    # Sub-scores
    sub_scores = incident.get("sub_scores", {})
    c_ml_contrib = float(sub_scores.get("c_ml_contrib", incident.get("contrib_cml", 0.0)))
    sev_contrib = float(sub_scores.get("sev_contrib", incident.get("contrib_sev", 0.0)))
    vol_contrib = float(sub_scores.get("vol_contrib", incident.get("contrib_svol", 0.0)))
    rule_contrib = float(sub_scores.get("rule_contrib", incident.get("contrib_brule", 0.0)))

    items: Dict[str, Any] = {
        # Incident Telemetry
        "INC-01": {"field": "source_ip", "value": src_ip},
        "INC-02": {"field": "destination_ip", "value": dst_ip},
        "INC-03": {"field": "destination_port", "value": dst_port},
        "INC-04": {"field": "duration_seconds", "value": round(duration_s, 2)},
        "INC-05": {"field": "flow_count", "value": flow_count},
        "INC-06": {"field": "flow_count_cap_note", "value": flow_cap_note},
        # Risk Components
        "RISK-COMPOSITE": {"field": "risk_score", "value": round(risk_score, 2), "scale": "[0, 100]"},
        "RISK-TIER": {"field": "severity_tier", "value": sev_tier},
        "RISK-CML": {"field": "c_ml_contribution", "value": round(c_ml_contrib, 2)},
        "RISK-SEV": {"field": "sev_contribution", "value": round(sev_contrib, 2)},
        "RISK-VOL": {"field": "vol_contribution", "value": round(vol_contrib, 2)},
        "RISK-RULE": {"field": "rule_contribution", "value": round(rule_contrib, 2)},
    }

    # Top-2 ML Predictions
    if top2_predictions:
        p1_cls, p1_pr, p2_cls, p2_pr = top2_predictions
    else:
        p1_cls = primary_threat
        p1_pr = float(incident.get("mean_ml_confidence", 0.99))
        p2_cls = "BENIGN" if p1_cls != "BENIGN" else "PortScan"
        p2_pr = round(max(0.0001, 1.0 - p1_pr), 4)

    margin = round(p1_pr - p2_pr, 4)
    items["PRED-1"] = {"field": "primary_predicted_class", "value": p1_cls, "probability": round(p1_pr, 4)}
    items["PRED-2"] = {"field": "runner_up_predicted_class", "value": p2_cls, "probability": round(p2_pr, 4)}
    items["PRED-MARGIN"] = {"field": "top2_margin", "value": margin, "is_ambiguous": bool(margin < 0.15)}

    # Rule Hit Evaluation
    rules_dict = _load_rules()
    has_rule = bool(incident.get("has_rule_hit", False) or rule_contrib > 0.0)
    rule_desc = rules_dict.get("R104_PORT_BRUTE_FORCE", {}).get("description", "Port brute-force threshold")
    items["RULE-01"] = {
        "rule_name": "R104_PORT_BRUTE_FORCE",
        "value": has_rule,
        "description": rule_desc if has_rule else "No heuristic security rules triggered",
    }
    items["RULE-ONLY-FLAG"] = {
        "field": "is_rule_only_candidate",
        "value": bool(primary_threat == "RULE_ONLY:R104" or (has_rule and p1_pr < 0.50)),
    }

    # SHAP Attributions
    if shap_items:
        items.update(shap_items)

    # Knowledge Chunks
    know_items = get_knowledge_items(primary_threat)
    items.update(know_items)

    # Build Bundle Dictionary
    bundle_dict = {
        "bundle_version": "1.0",
        "incident_id": inc_id,
        "items": items,
    }

    # Compute deterministic canonical SHA-256 hash
    canonical_json = json.dumps(bundle_dict, sort_keys=True)
    bundle_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    bundle_dict["bundle_hash"] = bundle_hash

    return bundle_dict

