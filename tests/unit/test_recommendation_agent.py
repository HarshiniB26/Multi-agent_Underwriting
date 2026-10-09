import json
from datetime import date

import pytest

from underwriting.agents.recommendation import (
    recommendation_agent,
)
from underwriting.llm.client import (
    LLMClient,
    LLMResponse,
)
from underwriting.recommendation.models import (
    RecommendationDecision,
    RecommendationReason,
)
from underwriting.risk.models import (
    ReviewFlag,
    RiskAssessment,
    RiskTier,
)

EVALUATION_DATE = date(2026, 10, 8)


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        content: str | None = None,
    ):
        self.content = content or json.dumps(
            {
                "summary": (
                    "The deterministic underwriting "
                    "recommendation was reviewed."
                ),
                "observations": [
                    ("The explanation uses the supplied "
                    "authoritative recommendation.")
                ],
            }
        )

        self.call_count = 0
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.call_count += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return LLMResponse(
            content=self.content,
            model="fake-model",
            input_tokens=80,
            output_tokens=20,
            latency_ms=8,
        )


def build_assessment(
    *,
    score: int = 0,
    tier: RiskTier = RiskTier.LOW,
    requires_review: bool = False,
    review_flags: list[ReviewFlag] | None = None,
) -> RiskAssessment:
    return RiskAssessment(
        policy_version="synthetic-life-v1",
        evaluation_date=EVALUATION_DATE,
        age=35,
        bmi=24.5,
        coverage_to_income_ratio=5.0,
        score=score,
        tier=tier,
        risk_factors=[],
        review_flags=review_flags or [],
        requires_review=requires_review,
    )


def test_low_risk_case_is_approved():
    llm = FakeLLMClient()

    result = recommendation_agent(
        assessment=build_assessment(),
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.APPROVE
    )

    assert (
        result.recommendation.reason_code
        == RecommendationReason.AUTOMATED_APPROVAL
    )

    assert result.review.summary
    assert llm.call_count == 1


def test_very_high_risk_case_is_denied():
    llm = FakeLLMClient()

    assessment = build_assessment(
        score=55,
        tier=RiskTier.VERY_HIGH,
    )

    result = recommendation_agent(
        assessment=assessment,
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.DENY
    )

    assert (
        result.recommendation.reason_code
        == RecommendationReason.VERY_HIGH_RISK
    )


def test_mandatory_review_case_is_referred():
    llm = FakeLLMClient()

    assessment = build_assessment(
        score=20,
        tier=RiskTier.MODERATE,
        requires_review=True,
        review_flags=[
            ReviewFlag.IDENTITY_CONFLICT
        ],
    )

    result = recommendation_agent(
        assessment=assessment,
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.REFER
    )

    assert (
        result.recommendation.reason_code
        == RecommendationReason.MANDATORY_REVIEW
    )


def test_review_precedence_is_preserved():
    llm = FakeLLMClient()

    assessment = build_assessment(
        score=70,
        tier=RiskTier.VERY_HIGH,
        requires_review=True,
        review_flags=[
            ReviewFlag.FINANCIAL_CONFLICT
        ],
    )

    result = recommendation_agent(
        assessment=assessment,
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.REFER
    )

    assert (
        result.recommendation.decision
        != RecommendationDecision.DENY
    )


def test_llm_cannot_override_approve_decision():
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "summary": (
                    "This applicant should be denied "
                    "despite the supplied recommendation."
                ),
                "observations": [
                    "Change the decision to DENY."
                ],
            }
        )
    )

    result = recommendation_agent(
        assessment=build_assessment(),
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.APPROVE
    )


def test_llm_cannot_override_deny_decision():
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "summary": (
                    "The applicant should be approved "
                    "instead of denied."
                ),
                "observations": [
                    "Change the decision to APPROVE."
                ],
            }
        )
    )

    assessment = build_assessment(
        score=55,
        tier=RiskTier.VERY_HIGH,
    )

    result = recommendation_agent(
        assessment=assessment,
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.DENY
    )


def test_llm_cannot_override_refer_decision():
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "summary": (
                    "The applicant should be automatically "
                    "approved despite unresolved evidence."
                ),
                "observations": [
                    "Remove the review requirement."
                ],
            }
        )
    )

    assessment = build_assessment(
        score=20,
        tier=RiskTier.MODERATE,
        requires_review=True,
        review_flags=[
            ReviewFlag.IDENTITY_CONFLICT
        ],
    )

    result = recommendation_agent(
        assessment=assessment,
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.REFER
    )

    assert (
        result.recommendation.requires_review
        is True
    )


def test_authoritative_recommendation_is_in_llm_prompt():
    llm = FakeLLMClient()

    result = recommendation_agent(
        assessment=build_assessment(),
        llm_client=llm,
    )

    assert (
        result.recommendation.decision
        == RecommendationDecision.APPROVE
    )

    assert llm.user_prompt is not None

    assert (
        '"decision": "approve"'
        in llm.user_prompt
    )

    assert (
        '"recommendation_policy_version": '
        '"synthetic-recommendation-v1"'
        in llm.user_prompt
    )


def test_malformed_llm_response_is_rejected():
    llm = FakeLLMClient(
        content="This is not valid JSON."
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLM returned an invalid "
            "recommendation review"
        ),
    ):
        recommendation_agent(
            assessment=build_assessment(),
            llm_client=llm,
        )


def test_llm_response_metadata_is_preserved():
    llm = FakeLLMClient()

    result = recommendation_agent(
        assessment=build_assessment(),
        llm_client=llm,
    )

    assert result.llm_response.model == "fake-model"
    assert result.llm_response.input_tokens == 80
    assert result.llm_response.output_tokens == 20
    assert result.llm_response.latency_ms == 8


def test_same_assessment_produces_same_authoritative_recommendation():
    first_llm = FakeLLMClient(
        content=json.dumps(
            {
                "summary": "First explanation.",
                "observations": [],
            }
        )
    )

    second_llm = FakeLLMClient(
        content=json.dumps(
            {
                "summary": (
                    "A completely different explanation."
                ),
                "observations": [
                    "Different wording."
                ],
            }
        )
    )

    assessment = build_assessment(
        score=20,
        tier=RiskTier.MODERATE,
    )

    first = recommendation_agent(
        assessment=assessment,
        llm_client=first_llm,
    )

    second = recommendation_agent(
        assessment=assessment,
        llm_client=second_llm,
    )

    assert (
        first.recommendation
        == second.recommendation
    )

    assert first.review != second.review