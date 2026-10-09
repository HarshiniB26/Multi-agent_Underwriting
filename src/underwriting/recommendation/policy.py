from underwriting.recommendation.models import (
    Recommendation,
    RecommendationDecision,
    RecommendationReason,
)
from underwriting.risk.models import (
    RiskAssessment,
    RiskTier,
)

RECOMMENDATION_POLICY_VERSION = (
    "synthetic-recommendation-v1"
)


def evaluate_recommendation(
    assessment: RiskAssessment,
) -> Recommendation:
    """
    Evaluate the synthetic recommendation policy deterministically.

    Decision precedence:

    1. mandatory review -> REFER
    2. very-high risk -> DENY
    3. all remaining supported tiers -> APPROVE

    The same RiskAssessment and recommendation policy version always
    produce the same recommendation.
    """

    if assessment.requires_review:
        decision = RecommendationDecision.REFER
        reason = RecommendationReason.MANDATORY_REVIEW

    elif assessment.tier == RiskTier.VERY_HIGH:
        decision = RecommendationDecision.DENY
        reason = RecommendationReason.VERY_HIGH_RISK

    else:
        decision = RecommendationDecision.APPROVE
        reason = RecommendationReason.AUTOMATED_APPROVAL

    return Recommendation(
        recommendation_policy_version=(
            RECOMMENDATION_POLICY_VERSION
        ),
        risk_policy_version=assessment.policy_version,
        decision=decision,
        reason_code=reason,
        risk_score=assessment.score,
        risk_tier=assessment.tier,
        requires_review=assessment.requires_review,
    )