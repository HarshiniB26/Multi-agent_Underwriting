import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from underwriting.domain.application import Application
from underwriting.llm.client import LLMClient, LLMResponse


class IntakeStatus(StrEnum):
    """Outcome of the intake processing stage."""

    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"


class IntakeReview(BaseModel):
    """
    Structured interpretation of the LLM intake review.

    The LLM may flag concerns but cannot modify applicant-provided facts.
    """

    needs_review: bool
    summary: str
    concerns: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    """Result returned by the standalone intake agent."""

    status: IntakeStatus

    application: Application | None = None
    review: IntakeReview | None = None

    validation_errors: list[str] = Field(default_factory=list)

    llm_response: LLMResponse | None = None


INTAKE_SYSTEM_PROMPT = """
You are the intake review component of a synthetic educational
life-insurance underwriting system.

Your responsibility is limited to reviewing an already validated
application for obvious internal inconsistencies or information that
may require clarification.

You must not:
- approve or deny insurance coverage
- calculate an underwriting risk score
- invent applicant facts
- modify applicant-provided information
- assume missing external evidence
- perform external data enrichment

Return JSON only using exactly this structure:

{
  "needs_review": true or false,
  "summary": "brief explanation",
  "concerns": ["concern 1", "concern 2"]
}

Use an empty concerns list when no clarification concern is identified.
""".strip()


def _build_review_prompt(application: Application) -> str:
    """Build the case-specific prompt for the intake LLM review."""

    application_json = application.model_dump_json(indent=2)

    return (
        "Review the following validated synthetic application.\n\n"
        f"{application_json}"
    )


def _parse_intake_review(content: str) -> IntakeReview:
    """Validate untrusted LLM output against the intake review contract."""

    try:
        parsed = json.loads(content)
        return IntakeReview.model_validate(parsed)

    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            "LLM returned an invalid intake review."
        ) from exc


def _format_validation_errors(
    error: ValidationError,
) -> list[str]:
    """Convert Pydantic validation errors into a stable agent result format."""

    formatted_errors: list[str] = []

    for item in error.errors():
        location = ".".join(
            str(part)
            for part in item["loc"]
        )
        message = item["msg"]

        formatted_errors.append(
            f"{location}: {message}"
        )

    return formatted_errors


def intake_agent(
    raw_application: dict[str, Any],
    llm_client: LLMClient,
) -> IntakeResult:
    """
    Validate applicant-provided data and perform a bounded LLM review.

    Deterministic validation remains authoritative. The LLM reviews only
    successfully validated data and cannot repair or override validation.
    """

    try:
        application = Application.model_validate(
            raw_application
        )

    except ValidationError as exc:
        return IntakeResult(
            status=IntakeStatus.INVALID,
            validation_errors=_format_validation_errors(exc),
        )

    llm_response = llm_client.generate(
        system_prompt=INTAKE_SYSTEM_PROMPT,
        user_prompt=_build_review_prompt(application),
    )

    review = _parse_intake_review(
        llm_response.content
    )

    status = (
        IntakeStatus.NEEDS_REVIEW
        if review.needs_review
        else IntakeStatus.ACCEPTED
    )

    return IntakeResult(
        status=status,
        application=application,
        review=review,
        llm_response=llm_response,
    )