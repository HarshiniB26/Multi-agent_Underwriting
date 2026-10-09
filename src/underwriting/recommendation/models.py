from enum import StrEnum

from pydantic import BaseModel, Field

from underwriting.risk.models import RiskTier


class RecommendationDecision(StrEnum):
    """Authoritative synthetic underwriting recommendation."""

    APPROVE = "approve"
    DENY = "deny"
    REFER = "refer"


class RecommendationReason(StrEnum):
    """Primary deterministic reason for the recommendation."""

    MANDATORY_REVIEW = "mandatory_review"
    VERY_HIGH_RISK = "very_high_risk"
    AUTOMATED_APPROVAL = "automated_approval"


class Recommendation(BaseModel):
    """
    Authoritative recommendation produced by the deterministic
    recommendation policy.
    """

    recommendation_policy_version: str = Field(min_length=1)
    risk_policy_version: str = Field(min_length=1)

    decision: RecommendationDecision
    reason_code: RecommendationReason

    risk_score: int = Field(ge=0)
    risk_tier: RiskTier

    requires_review: bool