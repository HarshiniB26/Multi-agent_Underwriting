import json
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
)

from underwriting.domain.application import Application
from underwriting.llm.client import LLMClient, LLMResponse


class IntakeStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"


class IntakeReview(BaseModel):
    """
    Structured semantic review produced by the LLM.

    The LLM reviews only semantic clarity and completeness.
    It does not make underwriting decisions.
    """

    needs_review: bool
    summary: str = Field(min_length=1)
    concerns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review_consistency(self):
        """
        Prevent internally contradictory LLM output.

        A review requiring clarification must identify at least one
        specific concern. A review that does not require clarification
        must not contain concerns.
        """

        if self.needs_review and not self.concerns:
            raise ValueError(
                "A review requiring clarification must include concerns."
            )

        if not self.needs_review and self.concerns:
            raise ValueError(
                "A review without clarification must not include concerns."
            )

        return self


class IntakeResult(BaseModel):
    status: IntakeStatus
    application: Application | None = None
    review: IntakeReview | None = None
    validation_errors: list[str] = Field(default_factory=list)
    llm_response: LLMResponse | None = None


INTAKE_SYSTEM_PROMPT = """
You are the semantic intake-review component of a synthetic educational
life-insurance underwriting system.

The application supplied to you has already passed deterministic schema
validation.

Your responsibility is to determine whether the applicant-provided
information is semantically clear and sufficiently specific for the
automated underwriting workflow to continue.

You may flag clarification needs such as:

- vague or non-specific occupation descriptions
- ambiguous applicant-provided medical condition descriptions
- ambiguous medication descriptions
- information that is technically valid but insufficiently specific for
  downstream processing

You must distinguish semantic ambiguity from underwriting risk.

You must not:

- approve or deny the application
- calculate or estimate a risk score
- assign a risk tier
- apply underwriting thresholds
- decide whether coverage is financially appropriate
- infer medical conditions not explicitly supplied
- infer medications not explicitly supplied
- invent applicant facts
- perform external-data enrichment
- treat age, tobacco use, medical history, occupation, income, or coverage
  as reasons for rejection merely because they may represent risk factors

Set needs_review=true only when a specific clarification or semantic
completeness issue exists in the supplied application.

Every concern must be grounded in information actually present in the
application.

Return JSON only using exactly this structure:

{
  "needs_review": true,
  "summary": "brief semantic intake assessment",
  "concerns": [
    "specific evidence-grounded clarification concern"
  ]
}

If no clarification is required, return:

{
  "needs_review": false,
  "summary": "brief semantic intake assessment",
  "concerns": []
}
""".strip()


def _build_review_prompt(
    application: Application,
) -> str:
    """
    Build the LLM prompt only from the validated application.

    The LLM does not receive external enrichment data or underwriting
    policy because those belong to later stages.
    """

    return (
        "Review the following validated applicant-provided information "
        "for semantic clarity and clarification needs only.\n\n"
        + json.dumps(
            application.model_dump(mode="json"),
            indent=2,
        )
    )


def _parse_intake_review(
    content: str,
) -> IntakeReview:
    """
    Parse and validate the structured LLM response.
    """

    try:
        parsed = json.loads(content)

        return IntakeReview.model_validate(parsed)

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise ValueError(
            "LLM returned an invalid intake review."
        ) from exc


def _format_validation_errors(
    error: ValidationError,
) -> list[str]:
    """
    Convert Pydantic validation errors into stable, readable messages.
    """

    errors: list[str] = []

    for item in error.errors():
        location = ".".join(
            str(part) for part in item["loc"]
        )

        message = item["msg"]

        if location:
            errors.append(
                f"{location}: {message}"
            )
        else:
            errors.append(message)

    return errors


def intake_agent(
    raw_application: dict[str, Any],
    llm_client: LLMClient,
) -> IntakeResult:
    """
    Validate and semantically review a raw application.

    Responsibility boundary:

    Deterministic code:
    - validates schema
    - validates field types and constraints
    - creates the trusted Application object

    LLM:
    - reviews semantic clarity
    - identifies clarification needs
    - cannot score risk or make underwriting decisions
    """

    try:
        application = Application.model_validate(
            raw_application
        )

    except ValidationError as exc:
        return IntakeResult(
            status=IntakeStatus.INVALID,
            application=None,
            review=None,
            validation_errors=(
                _format_validation_errors(exc)
            ),
            llm_response=None,
        )

    llm_response = llm_client.generate(
        system_prompt=INTAKE_SYSTEM_PROMPT,
        user_prompt=_build_review_prompt(
            application
        ),
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
        validation_errors=[],
        llm_response=llm_response,
    )