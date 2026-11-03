"""
ml_service/src/llm/schemas.py
──────────────────────────────
Pydantic contracts and schemas for Phase 11 Decision Support Integration.
All LLM output schemas conform strictly to Gemini's supported OpenAPI 3.0 subset.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IncidentPhase(str, Enum):
    CONTAINMENT = "Containment"
    MITIGATION = "Mitigation"
    ERADICATION = "Eradication"
    INVESTIGATION = "Investigation"


class AssessmentSummary(BaseModel):
    text: str = Field(description="Summary of the incident telemetry and observed findings")
    citations: List[str] = Field(description="List of Citation IDs supporting the summary (e.g. INC-01, PRED-1)")


class PrimaryHypothesis(BaseModel):
    predicted_class: str = Field(description="Primary threat category predicted by ML or rule")
    confidence: float = Field(description="ML prediction confidence probability in [0, 1]")
    citations: List[str] = Field(description="Citation IDs for the prediction (e.g. PRED-1, RISK-COMPOSITE)")


class ConfidenceAndCaveats(BaseModel):
    text: str = Field(description="Detailed discussion of uncertainty, margins, and operational caveats")
    margin_flag: bool = Field(description="True if top-2 prediction margin is < 0.15")
    rule_only_flag: bool = Field(description="True if incident was flagged exclusively by heuristic rules without ML threat detection")
    generic_knowledge_flag: bool = Field(description="True if guidance is based on generic fallback incident response text")
    citations: List[str] = Field(description="Citation IDs supporting the caveats (e.g. PRED-MARGIN, RULE-01, KNOW-1)")


class GroundedClaim(BaseModel):
    claim_text: str = Field(description="A specific factual statement regarding the incident")
    evidence_citations: List[str] = Field(description="List of specific Citation IDs directly supporting this claim")


class RecommendedAction(BaseModel):
    action_text: str = Field(description="Specific containment, mitigation, or investigation step")
    phase: IncidentPhase = Field(description="Incident handling lifecycle phase")
    knowledge_citations: List[str] = Field(description="List of KNOW-* Citation IDs from which this action was derived")


class DecisionOutput(BaseModel):
    incident_id: int = Field(description="Unique integer identifier of the correlated incident")
    insufficient_evidence: bool = Field(description="True if evidence is insufficient, contradictory, or inconclusive")
    assessment_summary: AssessmentSummary = Field(description="Overall grounded assessment summary")
    primary_hypothesis: PrimaryHypothesis = Field(description="Primary threat classification hypothesis")
    confidence_and_caveats: ConfidenceAndCaveats = Field(description="Mandatory discussion of uncertainty and caveats")
    grounded_claims: List[GroundedClaim] = Field(description="Collection of individual supported claims")
    recommended_actions: List[RecommendedAction] = Field(description="Actionable guidance derived from retrieved knowledge")


class GenerationMode(str, Enum):
    LLM = "llm"
    TEMPLATE_FALLBACK = "template_fallback"


class EvaluatedDecisionRecord(BaseModel):
    generation_mode: GenerationMode
    model_id: str
    bundle_hash: str
    validator_version: str
    payload: DecisionOutput
    raw_response: Optional[str] = None
    validation_attempts: int = 1
    latency_ms: float = 0.0

