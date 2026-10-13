import re
from datetime import date
from typing import Any

from pydantic import BaseModel

from underwriting.agents.enrichment import EnrichmentResult
from underwriting.agents.intake import IntakeResult
from underwriting.agents.recommendation import RecommendationResult
from underwriting.agents.risk_scoring import RiskScoringResult

PII_FIELDS = frozenset(
    {
        "ssn",
        "first_name",
        "last_name",
        "date_of_birth",
    }
)

REDACTED = "[REDACTED]"
REDACTED_SSN = "[REDACTED-SSN]"

_SSN_PATTERN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
_ISO_DATE_PATTERN = re.compile(r"^(\d{4})-\d{2}-\d{2}")


def scrub_text(text: str) -> str:
    """Mask anything shaped like an SSN, wherever it appears in free text."""

    return _SSN_PATTERN.sub(REDACTED_SSN, text)


def _mask_field(key: str, value: Any) -> str:
    """
    Mask one identifier field.

    Date of birth keeps only the year so age-related decisions stay
    debuggable from the trace alone.
    """

    if key == "date_of_birth":
        if isinstance(value, date):
            return f"{value.year}-XX-XX"

        if isinstance(value, str):
            match = _ISO_DATE_PATTERN.match(value)

            if match:
                return f"{match.group(1)}-XX-XX"

    return REDACTED


def redact_pii(value: Any) -> Any:
    """
    Return a copy of a JSON-like payload that is safe to write to traces.

    Known identifier fields are masked at any depth, and SSN-shaped text
    is scrubbed from every string as a backstop for unexpected fields.
    """

    if isinstance(value, dict):
        return {
            key: (
                _mask_field(key, item)
                if key in PII_FIELDS
                else redact_pii(item)
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [redact_pii(item) for item in value]

    if isinstance(value, str):
        return scrub_text(value)

    return value


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, list):
        return [_dump(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _dump(item)
            for key, item in value.items()
        }

    return value


def produced_payload(result: Any) -> dict[str, Any]:
    """
    Summarize what an agent produced for a workflow trace.

    LLM request/response bodies stay on llm_response; traces keep
    the business outputs the next stage consumes.
    """

    if isinstance(result, IntakeResult):
        return {
            "status": result.status.value,
            "application": _dump(result.application),
            "review": _dump(result.review),
            "validation_errors": list(
                result.validation_errors
            ),
        }

    if isinstance(result, EnrichmentResult):
        return {
            "status": result.status.value,
            "evidence": _dump(result.evidence),
            "discrepancies": list(result.discrepancies),
            "missing_evidence": list(
                result.missing_evidence
            ),
            "review": _dump(result.review),
        }

    if isinstance(result, RiskScoringResult):
        return {
            "assessment": _dump(result.assessment),
            "review": _dump(result.review),
        }

    if isinstance(result, RecommendationResult):
        return {
            "recommendation": _dump(
                result.recommendation
            ),
            "review": _dump(result.review),
        }

    raise TypeError(
        f"Unsupported agent result: {type(result)!r}"
    )
