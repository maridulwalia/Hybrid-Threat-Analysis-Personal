"""
ml_service/src/llm/validator.py
───────────────────────────────
Deterministic claim-level validator and entity hallucination scanner.
Enforces strict epistemic grounding, normalized numerical tolerances,
and claim-level citation matching against the Evidence Bundle.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from ml_service.src.llm.schemas import DecisionOutput

IP_REGEX = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
PORT_REGEX = re.compile(r"\bport\s+(\d{1,5})\b", re.IGNORECASE)
TECHNIQUE_REGEX = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
CVE_REGEX = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
PERCENT_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUM_REGEX = re.compile(r"\b\d+(?:\.\d+)?\b")

FORBIDDEN_ASSERTIONS_REGEX = re.compile(
    r"\b(confirmed attack|confirmed attacker|proven compromise|guaranteed breach|malicious host identified|100% certain malicious)\b",
    re.IGNORECASE,
)


class ValidationResult:
    def __init__(self, valid: bool, errors: List[str], output: Optional[DecisionOutput] = None) -> None:
        self.valid = valid
        self.errors = errors
        self.output = output

    def __repr__(self) -> str:
        return f"ValidationResult(valid={self.valid}, error_count={len(self.errors)})"


class DecisionValidator:
    """Deterministic validator for Phase 11 decision support records."""

    VERSION = "v1.0"

    def __init__(self, bundle: Dict[str, Any]) -> None:
        self.bundle = bundle
        self.items: Dict[str, Any] = bundle.get("items", {})
        self.item_ids: Set[str] = set(self.items.keys())

    def _get_item_entities(self, item_ids: List[str]) -> Tuple[Set[str], Set[int], Set[str], Set[str], List[float]]:
        """Extract IPs, ports, techniques, CVEs, and numeric values from cited items."""
        ips: Set[str] = set()
        ports: Set[int] = set()
        techniques: Set[str] = set()
        cves: Set[str] = set()
        numbers: List[float] = []

        for iid in item_ids:
            item = self.items.get(iid)
            if not item:
                continue

            # Convert item values and text to strings for scanning
            field_name = str(item.get("field", ""))
            val = item.get("value")
            text = str(item.get("text", ""))
            prob = item.get("probability")
            raw_val = item.get("raw_value")

            combined_str = f"{field_name} {val} {text} {prob} {raw_val} {item.get('rule_name', '')}"

            for ip in IP_REGEX.findall(combined_str):
                ips.add(ip)
            for tech in TECHNIQUE_REGEX.findall(combined_str):
                techniques.add(tech)
            for cve in CVE_REGEX.findall(combined_str):
                cves.add(cve.upper())

            # Specific known numeric fields
            if isinstance(val, (int, float)):
                numbers.append(float(val))
                if field_name == "destination_port":
                    ports.add(int(val))
            if isinstance(prob, (int, float)):
                numbers.append(float(prob))
            if isinstance(raw_val, (int, float)):
                numbers.append(float(raw_val))

            # Scan any port from text
            for p in PORT_REGEX.findall(text):
                try:
                    ports.add(int(p))
                except ValueError:
                    pass

        return ips, ports, techniques, cves, numbers

    def validate(self, candidate_json: Any) -> ValidationResult:
        """Validate candidate JSON against Pydantic schema and bundle grounding."""
        errors: List[str] = []

        # 1. Schema Validation
        try:
            if isinstance(candidate_json, dict):
                output = DecisionOutput.model_validate(candidate_json)
            elif isinstance(candidate_json, str):
                output = DecisionOutput.model_validate_json(candidate_json)
            elif isinstance(candidate_json, DecisionOutput):
                output = candidate_json
            else:
                return ValidationResult(False, [f"Invalid input type: {type(candidate_json)}"])
        except ValidationError as e:
            return ValidationResult(False, [f"SchemaValidationError: {str(e)}"])

        # 2. Global Citation Existence Check
        all_cited_ids: Set[str] = set()
        all_cited_ids.update(output.assessment_summary.citations)
        all_cited_ids.update(output.primary_hypothesis.citations)
        all_cited_ids.update(output.confidence_and_caveats.citations)

        for c in output.grounded_claims:
            all_cited_ids.update(c.evidence_citations)

        for a in output.recommended_actions:
            all_cited_ids.update(a.knowledge_citations)

        unresolved = all_cited_ids - self.item_ids
        if unresolved:
            errors.append(f"UnresolvedCitationsError: Cited IDs do not exist in bundle: {sorted(list(unresolved))}")

        # 3. Action Grounding: Every action must cite at least one KNOW-* ID
        for idx, action in enumerate(output.recommended_actions):
            know_cites = [cid for cid in action.knowledge_citations if cid.startswith("KNOW-")]
            if not know_cites:
                errors.append(f"ActionGroundingError: Action #{idx} does not cite any valid KNOW-* chunk ID.")

        # 4. Claim-Level Entity & Numeric Support
        for idx, claim in enumerate(output.grounded_claims):
            c_text = claim.claim_text
            c_cites = claim.evidence_citations

            if not c_cites:
                errors.append(f"ClaimGroundingError: Claim #{idx} has an empty evidence_citations list.")
                continue

            ips, ports, techs, cves, nums = self._get_item_entities(c_cites)

            # Check IPs in claim text
            claim_ips = IP_REGEX.findall(c_text)
            for ip in claim_ips:
                if ip not in ips:
                    errors.append(
                        f"ClaimEntityError: IP address '{ip}' in Claim #{idx} is not supported by cited items {c_cites}."
                    )

            # Check explicit port references
            claim_ports = PORT_REGEX.findall(c_text)
            for cp in claim_ports:
                try:
                    p_int = int(cp)
                    if p_int not in ports:
                        errors.append(
                            f"ClaimPortError: Port '{p_int}' in Claim #{idx} is not supported by cited items {c_cites}."
                        )
                except ValueError:
                    pass

            # Check technique IDs
            claim_techs = TECHNIQUE_REGEX.findall(c_text)
            for t in claim_techs:
                if t not in techs:
                    errors.append(
                        f"ClaimTechniqueError: Technique '{t}' in Claim #{idx} is not supported by cited items {c_cites}."
                    )

            # Check numeric values in claim text (normalized)
            # Find percentages and convert to ratios
            normalized_nums: List[float] = []
            for pct in PERCENT_REGEX.findall(c_text):
                try:
                    normalized_nums.append(float(pct) / 100.0)
                except ValueError:
                    pass

            # Whitelist derived values (e.g. margin, minutes)
            # Accept if close to any cited number within 1% relative tolerance
            for c_num in normalized_nums:
                matches_cited = False
                for b_num in nums:
                    tol = max(0.01, 0.01 * abs(b_num))
                    if abs(c_num - b_num) <= tol:
                        matches_cited = True
                        break
                if not matches_cited:
                    # Check whitelisted differences or sums across all cited numbers
                    for i in range(len(nums)):
                        for j in range(len(nums)):
                            if abs(c_num - (nums[i] - nums[j])) <= 0.01:
                                matches_cited = True
                                break
                            if abs(c_num - (nums[i] + nums[j])) <= 0.01:
                                matches_cited = True
                                break
                        if matches_cited:
                            break

                if not matches_cited and abs(c_num) > 0.001:
                    errors.append(
                        f"ClaimNumericError: Numeric value '{c_num}' in Claim #{idx} has no support in cited items {c_cites}."
                    )

        # 5. Secondary Forbidden Assertion Check
        pred_item = self.items.get("PRED-1", {})
        p_conf = float(pred_item.get("probability", 0.0))
        rule_only = bool(self.items.get("RULE-ONLY-FLAG", {}).get("value", False))

        full_text = f"{output.assessment_summary.text} {output.primary_hypothesis.predicted_class} {' '.join(c.claim_text for c in output.grounded_claims)}"
        forbidden_matches = FORBIDDEN_ASSERTIONS_REGEX.findall(full_text)
        if forbidden_matches and (p_conf < 0.80 or rule_only):
            errors.append(
                f"ForbiddenAssertionError: High-certainty term '{forbidden_matches[0]}' used on low-confidence or rule-only incident."
            )

        if errors:
            return ValidationResult(False, errors, output=None)

        return ValidationResult(True, [], output=output)
