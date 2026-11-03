"""
ml_service/src/llm/fallback_template.py
───────────────────────────────────────
Deterministic, zero-LLM template generator.
Produces a 100% compliant, fully grounded DecisionOutput directly from an Evidence Bundle.
Used as the fail-safe operational fallback during timeouts, 429 rate limits,
missing credentials, or validation rejections.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from ml_service.src.llm.schemas import (
    AssessmentSummary,
    ConfidenceAndCaveats,
    DecisionOutput,
    GroundedClaim,
    IncidentPhase,
    PrimaryHypothesis,
    RecommendedAction,
)

IP_EXTRACT = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}")
HTML_COMMENT_STRIP = re.compile(r"<!--.*?-->", re.DOTALL)


def generate_fallback_decision(bundle: Dict[str, Any]) -> DecisionOutput:
    """Generate a deterministic DecisionOutput directly from the evidence bundle."""
    incident_id = int(bundle.get("incident_id", 0))
    items = bundle.get("items", {})

    # Extract and sanitize incident telemetry
    raw_src_ip = str(items.get("INC-01", {}).get("value", "0.0.0.0"))
    raw_dst_ip = str(items.get("INC-02", {}).get("value", "0.0.0.0"))
    
    src_match = IP_EXTRACT.search(raw_src_ip)
    src_ip = src_match.group(0) if src_match else "0.0.0.0"
    
    dst_match = IP_EXTRACT.search(raw_dst_ip)
    dst_ip = dst_match.group(0) if dst_match else "0.0.0.0"
    dst_port = items.get("INC-03", {}).get("value", 0)
    duration_s = float(items.get("INC-04", {}).get("value", 0.0))
    flow_count = int(items.get("INC-05", {}).get("value", 1))

    # Extract predictions & risk
    pred1_item = items.get("PRED-1", {})
    primary_class = str(pred1_item.get("value", "UNKNOWN"))
    confidence = float(pred1_item.get("probability", 0.0))

    pred2_item = items.get("PRED-2", {})
    runner_up_class = str(pred2_item.get("value", "UNKNOWN"))
    runner_up_conf = float(pred2_item.get("probability", 0.0))

    margin_item = items.get("PRED-MARGIN", {})
    margin_val = float(margin_item.get("value", 1.0))
    margin_flag = bool(margin_item.get("is_ambiguous", False))

    risk_comp = float(items.get("RISK-COMPOSITE", {}).get("value", 0.0))
    sev_tier = str(items.get("RISK-TIER", {}).get("value", "Unknown Tier"))

    # Extract rules
    rule_item = items.get("RULE-01", {})
    rule_hit = bool(rule_item.get("value", False))
    rule_only_flag = bool(items.get("RULE-ONLY-FLAG", {}).get("value", False)) or (rule_hit and confidence < 0.50)

    # Check for generic knowledge
    generic_knowledge_flag = False
    know_items: List[tuple[str, Dict[str, Any]]] = []
    for k, v in items.items():
        if k.startswith("KNOW-"):
            know_items.append((k, v))
            if v.get("is_generic", False):
                generic_knowledge_flag = True

    # 1. Assessment Summary
    summary_text = (
        f"Automated baseline assessment: Incident #{incident_id} involving communication between {src_ip} "
        f"and {dst_ip}:{dst_port} ({flow_count} flows across {duration_s:.1f}s). "
        f"Primary classification is {primary_class} with confidence {confidence:.4f} "
        f"and composite risk score {risk_comp:.2f}."
    )
    summary_citations = ["INC-01", "INC-02", "INC-03", "INC-04", "INC-05", "PRED-1", "RISK-COMPOSITE"]

    # 2. Primary Hypothesis
    hyp_citations = ["PRED-1", "RISK-COMPOSITE", "RISK-TIER"]
    hypothesis = PrimaryHypothesis(
        predicted_class=primary_class,
        confidence=confidence,
        citations=hyp_citations,
    )

    # 3. Confidence & Caveats
    caveat_notes: List[str] = []
    caveat_citations: List[str] = []

    if margin_flag:
        caveat_notes.append(
            f"Classification margin is narrow ({margin_val:.4f} margin between {primary_class} "
            f"and {runner_up_class} at {runner_up_conf:.4f})."
        )
        caveat_citations.append("PRED-MARGIN")

    if rule_only_flag:
        caveat_notes.append(
            "Incident flagged exclusively via heuristic rules without ML threat detection; "
            "rule-only detections carry high false-alarm rates."
        )
        caveat_citations.append("RULE-01")

    if generic_knowledge_flag:
        caveat_notes.append(
            "Retrieved guidance contains generic NIST SP 800-61 baseline procedures rather than threat-tailored mitigations."
        )
        for kid, kv in know_items:
            if kv.get("is_generic", False):
                caveat_citations.append(kid)

    if not caveat_notes:
        caveat_text = f"Standard classification confidence; no ambiguous margin or heuristic rule caveats flagged [PRED-1]."
        caveat_citations.append("PRED-1")
    else:
        caveat_text = " ".join(caveat_notes)

    confidence_and_caveats = ConfidenceAndCaveats(
        text=caveat_text,
        margin_flag=margin_flag,
        rule_only_flag=rule_only_flag,
        generic_knowledge_flag=generic_knowledge_flag,
        citations=list(dict.fromkeys(caveat_citations)),
    )

    # 4. Grounded Claims
    claims: List[GroundedClaim] = [
        GroundedClaim(
            claim_text=f"Traffic observed from source IP {src_ip} targeting destination IP {dst_ip} on port {dst_port}.",
            evidence_citations=["INC-01", "INC-02", "INC-03"],
        ),
        GroundedClaim(
            claim_text=f"Incident session encompasses {flow_count} constituent flows spanning {duration_s:.1f} seconds duration.",
            evidence_citations=["INC-04", "INC-05"],
        ),
        GroundedClaim(
            claim_text=f"Risk engine calculated a composite score of {risk_comp:.2f} classified under the {sev_tier} impact tier.",
            evidence_citations=["RISK-COMPOSITE", "RISK-TIER"],
        ),
    ]

    # Add SHAP claim if available
    shap1_item = items.get("SHAP-1")
    if shap1_item:
        feat_name = str(shap1_item.get("feature", ""))
        raw_val = float(shap1_item.get("raw_value", 0.0))
        direction = str(shap1_item.get("direction", "influences_prediction"))
        claims.append(
            GroundedClaim(
                claim_text=f"Primary attribution feature '{feat_name}' (raw value {raw_val:.2f}) {direction}.",
                evidence_citations=["SHAP-1"],
            )
        )

    # 5. Recommended Actions
    actions: List[RecommendedAction] = []
    for kid, kv in know_items:
        t_id = kv.get("technique_id", "IR-PROC")
        s_type = kv.get("section_type", "containment")
        txt = HTML_COMMENT_STRIP.sub("", kv.get("text", "")).strip()
        
        # Determine appropriate phase
        s_lower = s_type.lower()
        if "contain" in s_lower:
            phase = IncidentPhase.CONTAINMENT
        elif "mitigat" in s_lower:
            phase = IncidentPhase.MITIGATION
        elif "eradicat" in s_lower:
            phase = IncidentPhase.ERADICATION
        else:
            phase = IncidentPhase.INVESTIGATION

        action_text = f"Execute {t_id} {s_type} protocol: {txt}"
        actions.append(
            RecommendedAction(
                action_text=action_text,
                phase=phase,
                knowledge_citations=[kid],
            )
        )

    if not actions:
        actions.append(
            RecommendedAction(
                action_text="Isolate affected host and conduct forensic PCAP packet inspection.",
                phase=IncidentPhase.INVESTIGATION,
                knowledge_citations=["KNOW-DEFAULT"] if "KNOW-DEFAULT" in items else list(items.keys())[:1],
            )
        )

    insufficient_evidence = bool(confidence < 0.20 and not rule_hit)

    return DecisionOutput(
        incident_id=incident_id,
        insufficient_evidence=insufficient_evidence,
        assessment_summary=AssessmentSummary(text=summary_text, citations=summary_citations),
        primary_hypothesis=hypothesis,
        confidence_and_caveats=confidence_and_caveats,
        grounded_claims=claims,
        recommended_actions=actions,
    )
