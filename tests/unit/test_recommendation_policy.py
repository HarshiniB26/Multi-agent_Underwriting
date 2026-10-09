from datetime import date

import pytest

from underwriting.recommendation.models import (
    RecommendationDecision,
    RecommendationReason,
)
from underwriting.recommendation.policy import (
    RECOMMENDATION_POLICY_VERSION,
    evaluate_recommendation,
)
from underwriting.risk.models import (
    ReviewFlag,
    RiskAssessment,
    RiskTier,
)

EVALUATION_DATE = date(2026, 10, 8)


def build_assessment(
    *,
    score: int,
    tier: RiskTier,
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


@pytest.mark.parametrize(
    ("score", "tier"),
    [
        (0, RiskTier.LOW),
        (20, RiskTier.MODERATE),
        (35, RiskTier.HIGH),
    ],
)
def test_supported_non_review_tiers_are_approved(
    score: int,
    tier: RiskTier,
):
    assessment = build_assessment(
        score=score,
        tier=tier,
    )

    result = evaluate_recommendation(
        assessment
    )

    assert (
        result.decision
        == RecommendationDecision.APPROVE
    )

    assert (
        result.reason_code
        == RecommendationReason.AUTOMATED_APPROVAL
    )


def test_very_high_risk_is_denied():
    assessment = build_assessment(
        score=55,
        tier=RiskTier.VERY_HIGH,
    )

    result = evaluate_recommendation(
        assessment
    )

    assert (
        result.decision
        == RecommendationDecision.DENY
    )

    assert (
        result.reason_code
        == RecommendationReason.VERY_HIGH_RISK
    )


@pytest.mark.parametrize(
    ("score", "tier"),
    [
        (0, RiskTier.LOW),
        (20, RiskTier.MODERATE),
        (35, RiskTier.HIGH),
        (55, RiskTier.VERY_HIGH),
    ],
)
def test_mandatory_review_always_refers(
    score: int,
    tier: RiskTier,
):
    assessment = build_assessment(
        score=score,
        tier=tier,
        requires_review=True,
        review_flags=[
            ReviewFlag.IDENTITY_CONFLICT
        ],
    )

    result = evaluate_recommendation(
        assessment
    )

    assert (
        result.decision
        == RecommendationDecision.REFER
    )

    assert (
        result.reason_code
        == RecommendationReason.MANDATORY_REVIEW
    )


def test_review_precedence_overrides_very_high_risk_denial():
    """
    This is the critical precedence test.

    A VERY_HIGH case with unresolved evidence issues must be REFER,
    not automatically DENY.
    """

    assessment = build_assessment(
        score=70,
        tier=RiskTier.VERY_HIGH,
        requires_review=True,
        review_flags=[
            ReviewFlag.FINANCIAL_CONFLICT
        ],
    )

    result = evaluate_recommendation(
        assessment
    )

    assert (
        result.decision
        == RecommendationDecision.REFER
    )

    assert (
        result.decision
        != RecommendationDecision.DENY
    )

    assert (
        result.reason_code
        == RecommendationReason.MANDATORY_REVIEW
    )


def test_policy_versions_are_preserved():
    assessment = build_assessment(
        score=20,
        tier=RiskTier.MODERATE,
    )

    result = evaluate_recommendation(
        assessment
    )

    assert (
        result.recommendation_policy_version
        == RECOMMENDATION_POLICY_VERSION
    )

    assert (
        result.risk_policy_version
        == "synthetic-life-v1"
    )


def test_risk_context_is_preserved():
    assessment = build_assessment(
        score=35,
        tier=RiskTier.HIGH,
    )

    result = evaluate_recommendation(
        assessment
    )

    assert result.risk_score == 35
    assert result.risk_tier == RiskTier.HIGH
    assert result.requires_review is False


def test_same_assessment_produces_same_recommendation():
    assessment = build_assessment(
        score=20,
        tier=RiskTier.MODERATE,
    )

    first = evaluate_recommendation(
        assessment
    )

    second = evaluate_recommendation(
        assessment
    )

    assert first == second