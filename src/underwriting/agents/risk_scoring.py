import json
from datetime import date

from pydantic import BaseModel, Field, ValidationError

from underwriting.domain.application import Application
from underwriting.enrichment.models import EnrichmentEvidence
from underwriting.llm.client import LLMClient, LLMResponse
from underwriting.risk.models import RiskAssessment
from underwriting.risk.policy import evaluate_policy


class RiskScoringReview(BaseModel):
    """
    Structured LLM explanation of an authoritative deterministic
    risk assessment.
    """

    summary: str = Field(min_length=1)
    observations: list[str] = Field(default_factory=list)


class RiskScoringResult(BaseModel):
    """
    Result returned by the standalone Risk Scoring Agent.

    The assessment is authoritative and comes from the deterministic
    policy engine. The review is explanatory only.
    """

    assessment: RiskAssessment
    review: RiskScoringReview
    llm_response: LLMResponse


RISK_SCORING_SYSTEM_PROMPT = """
You are the risk-assessment explanation component of a synthetic
educational life-insurance underwriting system.

You will receive:
1. validated applicant information,
2. validated synthetic external evidence, and
3. an authoritative risk assessment calculated by a deterministic
   synthetic underwriting policy engine.

Your responsibility is limited to explaining the supplied risk
assessment clearly and concisely.

The deterministic policy assessment is authoritative.

You must not:
- calculate a new risk score
- change the supplied risk score
- change the supplied risk tier
- add or remove risk factors
- add or remove review flags
- invent applicant, medical, financial, or external facts
- create new underwriting rules
- approve or deny insurance coverage
- make a final underwriting recommendation
- treat missing or conflicting evidence as favorable evidence

Return JSON only using exactly this structure:

{
  "summary": "brief explanation of the supplied risk assessment",
  "observations": ["observation 1", "observation 2"]
}

Use only facts supplied in the prompt.
""".strip()


def _build_review_prompt(
    application: Application,
    evidence: EnrichmentEvidence,
    assessment: RiskAssessment,
) -> str:
    payload = {
        "application": application.model_dump(
            mode="json"
        ),
        "external_evidence": evidence.model_dump(
            mode="json"
        ),
        "authoritative_risk_assessment": (
            assessment.model_dump(mode="json")
        ),
    }

    return (
        "Explain the following deterministic synthetic "
        "risk assessment.\n\n"
        + json.dumps(payload, indent=2)
    )


def _parse_risk_scoring_review(
    content: str,
) -> RiskScoringReview:
    try:
        parsed = json.loads(content)

        return RiskScoringReview.model_validate(
            parsed
        )

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise ValueError(
            "LLM returned an invalid risk-scoring review."
        ) from exc


def risk_scoring_agent(
    application: Application,
    evidence: EnrichmentEvidence,
    evaluation_date: date,
    llm_client: LLMClient,
) -> RiskScoringResult:
    """
    Run the standalone Risk Scoring Agent.

    The deterministic policy engine calculates the authoritative
    assessment before the LLM is called. The LLM can explain the
    assessment but cannot modify it.
    """

    assessment = evaluate_policy(
        application=application,
        evidence=evidence,
        evaluation_date=evaluation_date,
    )

    llm_response = llm_client.generate(
        system_prompt=RISK_SCORING_SYSTEM_PROMPT,
        user_prompt=_build_review_prompt(
            application,
            evidence,
            assessment,
        ),
    )

    review = _parse_risk_scoring_review(
        llm_response.content
    )

    return RiskScoringResult(
        assessment=assessment,
        review=review,
        llm_response=llm_response,
    )