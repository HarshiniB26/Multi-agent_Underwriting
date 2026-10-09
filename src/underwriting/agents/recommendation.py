import json

from pydantic import BaseModel, Field, ValidationError

from underwriting.llm.client import LLMClient, LLMResponse
from underwriting.recommendation.models import Recommendation
from underwriting.recommendation.policy import evaluate_recommendation
from underwriting.risk.models import RiskAssessment


class RecommendationReview(BaseModel):
    """
    Structured LLM explanation of an authoritative deterministic
    underwriting recommendation.
    """

    summary: str = Field(min_length=1)
    observations: list[str] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    """
    Result returned by the standalone Recommendation Agent.

    The recommendation is authoritative and comes from the deterministic
    recommendation policy. The LLM review is explanatory only.
    """

    recommendation: Recommendation
    review: RecommendationReview
    llm_response: LLMResponse


RECOMMENDATION_SYSTEM_PROMPT = """
You are the recommendation explanation component of a synthetic
educational life-insurance underwriting system.

You will receive:
1. an authoritative deterministic risk assessment, and
2. an authoritative recommendation produced by a deterministic
   synthetic recommendation policy.

Your responsibility is limited to explaining the supplied recommendation
clearly and concisely.

The supplied recommendation is authoritative.

You must not:
- change the recommendation
- change APPROVE to DENY or REFER
- change DENY to APPROVE or REFER
- change REFER to APPROVE or DENY
- change the risk score
- change the risk tier
- remove or modify review requirements
- invent applicant, medical, financial, or external facts
- invent underwriting rules
- calculate a new recommendation

Return JSON only using exactly this structure:

{
  "summary": "brief explanation of the supplied recommendation",
  "observations": ["observation 1", "observation 2"]
}

Use only facts supplied in the prompt.
""".strip()


def _build_review_prompt(
    assessment: RiskAssessment,
    recommendation: Recommendation,
) -> str:
    payload = {
        "authoritative_risk_assessment": (
            assessment.model_dump(mode="json")
        ),
        "authoritative_recommendation": (
            recommendation.model_dump(mode="json")
        ),
    }

    return (
        "Explain the following deterministic synthetic "
        "underwriting recommendation.\n\n"
        + json.dumps(payload, indent=2)
    )


def _parse_recommendation_review(
    content: str,
) -> RecommendationReview:
    try:
        parsed = json.loads(content)

        return RecommendationReview.model_validate(
            parsed
        )

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise ValueError(
            "LLM returned an invalid recommendation review."
        ) from exc


def recommendation_agent(
    assessment: RiskAssessment,
    llm_client: LLMClient,
) -> RecommendationResult:
    """
    Run the standalone Recommendation Agent.

    The deterministic recommendation policy produces the authoritative
    APPROVE, DENY, or REFER decision before the LLM is called.

    The LLM may explain the decision but cannot modify it.
    """

    recommendation = evaluate_recommendation(
        assessment
    )

    llm_response = llm_client.generate(
        system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
        user_prompt=_build_review_prompt(
            assessment,
            recommendation,
        ),
    )

    review = _parse_recommendation_review(
        llm_response.content
    )

    return RecommendationResult(
        recommendation=recommendation,
        review=review,
        llm_response=llm_response,
    )