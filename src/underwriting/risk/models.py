from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class RiskTier(StrEnum):
    """Synthetic underwriting risk classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReviewFlag(StrEnum):
    """Conditions that prevent fully automated risk assessment."""

    IDENTITY_CONFLICT = "identity_conflict"
    FINANCIAL_CONFLICT = "financial_conflict"

    MISSING_IDENTITY_EVIDENCE = "missing_identity_evidence"
    MISSING_MEDICAL_EVIDENCE = "missing_medical_evidence"
    MISSING_PRESCRIPTION_EVIDENCE = "missing_prescription_evidence"
    MISSING_FINANCIAL_EVIDENCE = "missing_financial_evidence"

    UNDECLARED_MEDICAL_CONDITION = "undeclared_medical_condition"
    UNDECLARED_ACTIVE_MEDICATION = "undeclared_active_medication"

    UNSUPPORTED_MEDICAL_CONDITION = "unsupported_medical_condition"
    EXCESSIVE_COVERAGE_RATIO = "excessive_coverage_ratio"


class RiskFactor(BaseModel):
    """One deterministic factor contributing points to the risk score."""

    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    points: int = Field(ge=0)


class RiskAssessment(BaseModel):
    """Authoritative result produced by the deterministic policy engine."""

    policy_version: str = Field(min_length=1)
    evaluation_date: date

    age: int = Field(ge=0)
    bmi: float = Field(gt=0)
    coverage_to_income_ratio: float = Field(gt=0)

    score: int = Field(ge=0)
    tier: RiskTier

    risk_factors: list[RiskFactor] = Field(
        default_factory=list
    )

    review_flags: list[ReviewFlag] = Field(
        default_factory=list
    )

    requires_review: bool