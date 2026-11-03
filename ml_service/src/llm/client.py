"""
ml_service/src/llm/client.py
─────────────────────────────
HTTPX-based client adapter for Gemini API with header-only key transmission,
sanitized exception handling, deterministic local disk caching,
single retry on validation failure, and fail-safe template fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
from config.paths import Paths
from ml_service.src.llm.fallback_template import generate_fallback_decision
from ml_service.src.llm.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from ml_service.src.llm.schemas import (
    DecisionOutput,
    EvaluatedDecisionRecord,
    GenerationMode,
)
from ml_service.src.llm.validator import DecisionValidator

logger = logging.getLogger("llm_client")

API_KEY_REGEX = re.compile(r"AIzaSy[A-Za-z0-9_-]{33}")


def sanitize_text(text: str) -> str:
    """Scrub potential Gemini API keys from logs and error strings."""
    return API_KEY_REGEX.sub("[REDACTED_API_KEY]", text)


class GeminiTransportError(Exception):
    """Sanitized transport error that guarantees zero secret leakage."""
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        clean_msg = sanitize_text(message)
        super().__init__(clean_msg)
        self.status_code = status_code


class LLMClientAdapter:
    """Production client adapter for Gemini decision support."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_seconds: float = 10.0,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_id = model_id or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.timeout_seconds = timeout_seconds
        self.cache_dir = cache_dir or (Paths.ARTIFACTS_DIR / "llm" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_cache_key(self, bundle_hash: str) -> str:
        """Compute composite deterministic cache key."""
        schema_hash = hashlib.sha256(json.dumps(DecisionOutput.model_json_schema(), sort_keys=True).encode()).hexdigest()[:8]
        composite = f"{bundle_hash}_{self.model_id}_{DecisionValidator.VERSION}_{schema_hash}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    @staticmethod
    def _dereference_schema(raw_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Inline all $ref definitions to conform to Gemini's supported OpenAPI 3.0 subset."""
        defs = raw_schema.get("$defs", raw_schema.get("definitions", {}))

        def resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_key = node["$ref"].split("/")[-1]
                    resolved = defs.get(ref_key, {})
                    return resolve(dict(resolved))
                return {k: resolve(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [resolve(elem) for elem in node]
            return node

        cleaned = resolve(raw_schema)
        cleaned.pop("$defs", None)
        cleaned.pop("definitions", None)
        cleaned.pop("title", None)
        return cleaned

    def _get_generation_config(self) -> Dict[str, Any]:
        """Return model-specific generation parameters."""
        raw_schema = DecisionOutput.model_json_schema()
        schema = self._dereference_schema(raw_schema)

        config: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        if "3.8-flash" in self.model_id:
            config["temperature"] = 1.0
            config["thinking_config"] = {"thinking_level": "low"}
        else:
            config["temperature"] = 0.0
            config["seed"] = 42

        return config

    def _call_gemini_api(self, prompt: str, retry_feedback: Optional[str] = None) -> Tuple[str, float]:
        """Execute a secure REST request to Google Gemini API via HTTPX."""
        if not self.api_key:
            raise GeminiTransportError("GEMINI_API_KEY not configured or empty.", status_code=401)

        endpoint = f"{self.BASE_URL}/{self.model_id}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        if retry_feedback:
            contents.append({"role": "model", "parts": [{"text": "Prior output was rejected by validator."}]})
            contents.append({"role": "user", "parts": [{"text": f"VALIDATION ERROR FEEDBACK:\n{retry_feedback}\n\nPlease regenerate the JSON adhering strictly to all citation constraints."}]})

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": contents,
            "generationConfig": self._get_generation_config(),
        }

        t0 = time.time()
        try:
            with httpx.Client(timeout=self.timeout_seconds, verify=True) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                elapsed_ms = (time.time() - t0) * 1000.0

                if resp.status_code == 429:
                    raise GeminiTransportError("Gemini API rate limit exceeded (HTTP 429).", status_code=429)
                if resp.status_code != 200:
                    raise GeminiTransportError(
                        f"Gemini API returned HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                    )

                data = resp.json()
                text_output = data["candidates"][0]["content"]["parts"][0]["text"]
                return text_output, elapsed_ms

        except httpx.RequestError as exc:
            elapsed_ms = (time.time() - t0) * 1000.0
            raise GeminiTransportError(f"Network transport error: {type(exc).__name__}") from None

    def generate_decision(
        self,
        bundle: Dict[str, Any],
        use_cache: bool = True,
    ) -> EvaluatedDecisionRecord:
        """Generate a validated decision support record for an evidence bundle.
        
        Attempts API generation with validator gating.
        Safely falls back to the deterministic template generator on any failure.
        """
        bundle_hash = bundle.get("bundle_hash", "")
        cache_key = self._compute_cache_key(bundle_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check Cache
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info("Retrieved decision from local cache for incident %s", bundle.get("incident_id"))
                return EvaluatedDecisionRecord.model_validate(cached_data)
            except Exception as e:
                logger.warning("Failed to load cached decision: %s. Re-running...", e)

        validator = DecisionValidator(bundle)
        prompt = build_user_prompt(bundle)

        # Attempt 1: Initial LLM Generation
        raw_response: Optional[str] = None
        latency_ms = 0.0
        validation_attempts = 0

        try:
            raw_response, latency_ms = self._call_gemini_api(prompt)
            validation_attempts = 1
            v_res = validator.validate(raw_response)

            # Attempt 2: Single Retry on Validation Error
            if not v_res.valid:
                logger.warning("Validation attempt 1 failed: %s. Initiating single retry...", v_res.errors)
                err_text = "\n".join(v_res.errors)
                raw_response, lat2 = self._call_gemini_api(prompt, retry_feedback=err_text)
                latency_ms += lat2
                validation_attempts = 2
                v_res = validator.validate(raw_response)

            if v_res.valid and v_res.output is not None:
                record = EvaluatedDecisionRecord(
                    generation_mode=GenerationMode.LLM,
                    model_id=self.model_id,
                    bundle_hash=bundle_hash,
                    validator_version=DecisionValidator.VERSION,
                    payload=v_res.output,
                    raw_response=raw_response,
                    validation_attempts=validation_attempts,
                    latency_ms=round(latency_ms, 2),
                )
                if use_cache:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(record.model_dump(), f, indent=2)
                return record

            logger.warning("Validation rejected output after %d attempts. Triggering fallback.", validation_attempts)

        except GeminiTransportError as te:
            logger.warning("LLM Transport error: %s. Triggering fallback template generator.", te)
        except Exception as ex:
            logger.error("Unexpected error during LLM generation: %s. Triggering fallback template generator.", sanitize_text(str(ex)))

        # Fallback Generator Engagement
        fallback_output = generate_fallback_decision(bundle)
        record = EvaluatedDecisionRecord(
            generation_mode=GenerationMode.TEMPLATE_FALLBACK,
            model_id="deterministic_template",
            bundle_hash=bundle_hash,
            validator_version=DecisionValidator.VERSION,
            payload=fallback_output,
            raw_response=raw_response,
            validation_attempts=validation_attempts,
            latency_ms=round(latency_ms, 2),
        )

        if use_cache:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record.model_dump(), f, indent=2)

        return record
