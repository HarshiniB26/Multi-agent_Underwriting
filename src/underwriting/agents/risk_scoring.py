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
    Structured LLM synthesis of an authoritative deterministic
    risk assessment.

    The LLM may interpret relationships among supplied factors and
    evidence, but it cannot modify the authoritative assessment.
    """

    summary: str = Field(min_length=1)

    primary_drivers: list[str] = Field(
        default_factory=list
    )

    factor_interactions: list[str] = Field(
        default_factory=list
    )

    evidence_context: list[str] = Field(
        default_factory=list
    )

    review_context: list[str] = Field(
        default_factory=list
    )


class RiskScoringResult(BaseModel):
    """
    Result returned by the standalone Risk Scoring Agent.

    The assessment is authoritative and comes from the deterministic
    policy engine. The LLM review provides semantic synthesis only.
    """

    assessment: RiskAssessment
    review: RiskScoringReview
    llm_response: LLMResponse


RISK_SCORING_SYSTEM_PROMPT = """
You are the risk-factor synthesis component of a synthetic educational
life-insurance underwriting system.

You will receive:

1. validated applicant information,
2. validated synthetic external evidence, and
3. an authoritative risk assessment calculated by a deterministic
   synthetic underwriting policy engine.

The deterministic risk assessment is authoritative.

Your responsibility is to synthesize the supplied assessment and
evidence so that a reviewer can understand:

- which supplied risk factors are the primary drivers of the assessment
- how multiple supplied risk factors relate to one another
- whether supplied external evidence corroborates relevant risk factors
- what supplied review flags or evidence limitations require attention

You may reason about relationships among facts that are explicitly
present in the supplied application, external evidence, and
authoritative risk assessment.

For example, if the authoritative assessment contains a diabetes risk
factor and the supplied prescription evidence contains an active
medication associated with the supplied diabetes evidence, you may
describe that evidence as corroborating the supplied medical evidence.

However, you must not create new authoritative underwriting findings.

You must not:

- calculate a new risk score
- change the supplied risk score
- change the supplied risk tier
- add or remove authoritative risk factors
- add or remove authoritative review flags
- change whether human review is required
- invent applicant facts
- invent medical conditions
- invent medications
- invent financial information
- invent external evidence
- create new underwriting rules
- assign additional risk points
- infer that missing evidence means no risk exists
- approve or deny insurance coverage
- make the final underwriting recommendation

Do not double-count related evidence.

For example, if a prescription record corroborates a supplied medical
condition, describe the relationship as corroborating evidence rather
than treating the medication as a new independent authoritative risk
factor unless the deterministic assessment explicitly identifies it as
one.

Return JSON only using exactly this structure:

{
  "summary": "brief synthesis of the supplied authoritative assessment",
  "primary_drivers": [
    "important supplied risk factor"
  ],
  "factor_interactions": [
    "relationship among supplied risk factors"
  ],
  "evidence_context": [
    "relationship between supplied evidence and supplied risk factors"
  ],
  "review_context": [
    "supplied review flag or evidence limitation requiring attention"
  ]
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
        "Synthesize the following authoritative deterministic "
        "synthetic risk assessment and its supporting evidence.\n\n"
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
    llm_client: LLMClient,
    evaluation_date: date | None = None,
) -> RiskScoringResult:
    """
    Execute deterministic risk scoring followed by semantic
    LLM synthesis.

    The deterministic assessment remains authoritative.
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