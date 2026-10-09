import json

from pydantic import BaseModel, Field, ValidationError

from underwriting.llm.client import LLMClient, LLMResponse
from underwriting.recommendation.models import Recommendation
from underwriting.recommendation.policy import evaluate_recommendation
from underwriting.risk.models import RiskAssessment


class RecommendationReview(BaseModel):
    """
    Structured LLM synthesis of an authoritative deterministic
    underwriting recommendation.

    The LLM explains the basis and policy precedence but cannot
    modify the authoritative recommendation.
    """

    summary: str = Field(min_length=1)

    decision_basis: list[str] = Field(
        default_factory=list
    )

    precedence_explanation: str | None = None

    human_review_focus: list[str] = Field(
        default_factory=list
    )


class RecommendationResult(BaseModel):
    """
    Result returned by the standalone Recommendation Agent.

    The recommendation is authoritative and comes from the
    deterministic recommendation policy. The LLM review provides
    semantic rationale only.
    """

    recommendation: Recommendation
    review: RecommendationReview
    llm_response: LLMResponse


RECOMMENDATION_SYSTEM_PROMPT = """
You are the recommendation-rationale synthesis component of a synthetic
educational life-insurance underwriting system.

You will receive:

1. an authoritative deterministic risk assessment, and
2. an authoritative recommendation produced by a deterministic
   synthetic recommendation policy.

The supplied recommendation is authoritative.

Your responsibility is to synthesize why the supplied recommendation
follows from the supplied risk assessment and recommendation policy
result.

Explain:

- the supplied factors supporting the authoritative recommendation
- which supplied condition or rule is decisive
- when applicable, why a mandatory human-review requirement takes
  precedence over an otherwise possible automated outcome
- what supplied issues a human reviewer should focus on when the
  authoritative recommendation is REFER

You may reason about relationships among facts already present in the
supplied risk assessment and recommendation.

You must not:

- change the recommendation
- change APPROVE to DENY or REFER
- change DENY to APPROVE or REFER
- change REFER to APPROVE or DENY
- change the risk score
- change the risk tier
- add or remove risk factors
- add or remove review flags
- change whether human review is required
- invent applicant facts
- invent medical facts
- invent financial facts
- invent external evidence
- invent underwriting rules
- calculate a new recommendation
- independently approve or deny insurance coverage

For a REFER recommendation, focus human_review_focus on the supplied
review flags or evidence issues requiring manual resolution.

For APPROVE or DENY, human_review_focus should normally be empty unless
the supplied authoritative information explicitly indicates otherwise.

Return JSON only using exactly this structure:

{
  "summary": "brief rationale for the authoritative recommendation",
  "decision_basis": [
    "supplied factor supporting the recommendation"
  ],
  "precedence_explanation": "explanation of applicable policy precedence or null",
  "human_review_focus": [
    "supplied issue requiring human attention"
  ]
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
        "Synthesize the rationale for the following authoritative "
        "deterministic synthetic underwriting recommendation.\n\n"
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
    Execute deterministic recommendation policy followed by
    semantic LLM rationale synthesis.

    The deterministic recommendation remains authoritative.
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