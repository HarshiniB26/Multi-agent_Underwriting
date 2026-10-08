from datetime import date

from underwriting.domain.application import (
    Application,
    TobaccoUse,
)
from underwriting.enrichment.models import (
    EnrichmentEvidence,
    EvidenceStatus,
)
from underwriting.risk.models import (
    ReviewFlag,
    RiskAssessment,
    RiskFactor,
    RiskTier,
)

POLICY_VERSION = "synthetic-life-v1"


_MEDICAL_CONDITION_POINTS: dict[str, int] = {
    "asthma": 5,
    "hypothyroidism": 5,
    "high cholesterol": 5,
    "hypertension": 10,
    "type 2 diabetes": 20,
}


def _normalize_text(value: str) -> str:
    return value.strip().casefold()


def calculate_age(
    date_of_birth: date,
    evaluation_date: date,
) -> int:
    """Calculate age as of the supplied policy evaluation date."""

    if date_of_birth > evaluation_date:
        raise ValueError(
            "Date of birth cannot be after the evaluation date."
        )

    years = evaluation_date.year - date_of_birth.year

    birthday_has_occurred = (
        evaluation_date.month,
        evaluation_date.day,
    ) >= (
        date_of_birth.month,
        date_of_birth.day,
    )

    if not birthday_has_occurred:
        years -= 1

    return years


def calculate_bmi(
    height_cm: int,
    weight_kg: float,
) -> float:
    """Calculate BMI from validated height and weight."""

    height_m = height_cm / 100

    return weight_kg / (height_m**2)


def calculate_coverage_to_income_ratio(
    coverage_amount: int,
    annual_income: int,
) -> float:
    """Calculate requested coverage relative to annual income."""

    return coverage_amount / annual_income


def _age_factor(age: int) -> RiskFactor | None:
    if age >= 70:
        return RiskFactor(
            code="AGE_70_PLUS",
            description="Applicant age is 70 or older.",
            points=30,
        )

    if age >= 60:
        return RiskFactor(
            code="AGE_60_69",
            description="Applicant age is between 60 and 69.",
            points=20,
        )

    if age >= 50:
        return RiskFactor(
            code="AGE_50_59",
            description="Applicant age is between 50 and 59.",
            points=10,
        )

    if age >= 40:
        return RiskFactor(
            code="AGE_40_49",
            description="Applicant age is between 40 and 49.",
            points=5,
        )

    return None


def _bmi_factor(bmi: float) -> RiskFactor | None:
    if bmi < 18.5:
        return RiskFactor(
            code="BMI_UNDER_18_5",
            description="BMI is below 18.5.",
            points=10,
        )

    if bmi >= 40:
        return RiskFactor(
            code="BMI_40_PLUS",
            description="BMI is 40 or greater.",
            points=20,
        )

    if bmi >= 35:
        return RiskFactor(
            code="BMI_35_39_9",
            description="BMI is between 35.0 and 39.9.",
            points=10,
        )

    if bmi >= 30:
        return RiskFactor(
            code="BMI_30_34_9",
            description="BMI is between 30.0 and 34.9.",
            points=5,
        )

    return None


def _tobacco_factor(
    tobacco_use: TobaccoUse,
) -> RiskFactor | None:
    if tobacco_use == TobaccoUse.CURRENT:
        return RiskFactor(
            code="TOBACCO_CURRENT",
            description="Applicant reports current tobacco use.",
            points=20,
        )

    if tobacco_use == TobaccoUse.FORMER:
        return RiskFactor(
            code="TOBACCO_FORMER",
            description="Applicant reports former tobacco use.",
            points=5,
        )

    return None


def _coverage_factor(
    ratio: float,
) -> RiskFactor | None:
    if ratio > 10:
        return None

    if ratio > 8:
        return RiskFactor(
            code="COVERAGE_RATIO_8_TO_10",
            description=(
                "Coverage-to-income ratio is greater than 8 "
                "and no greater than 10."
            ),
            points=10,
        )

    if ratio > 5:
        return RiskFactor(
            code="COVERAGE_RATIO_5_TO_8",
            description=(
                "Coverage-to-income ratio is greater than 5 "
                "and no greater than 8."
            ),
            points=5,
        )

    return None


def _medical_factors(
    evidence: EnrichmentEvidence,
) -> tuple[list[RiskFactor], list[ReviewFlag]]:
    factors: list[RiskFactor] = []
    flags: list[ReviewFlag] = []

    if evidence.medical.status != EvidenceStatus.AVAILABLE:
        return factors, flags

    seen_conditions: set[str] = set()

    for condition in evidence.medical.conditions:
        normalized = _normalize_text(
            condition.condition
        )

        if normalized in seen_conditions:
            continue

        seen_conditions.add(normalized)

        points = _MEDICAL_CONDITION_POINTS.get(
            normalized
        )

        if points is None:
            flags.append(
                ReviewFlag.UNSUPPORTED_MEDICAL_CONDITION
            )
            continue

        factors.append(
            RiskFactor(
                code=(
                    "MEDICAL_"
                    + normalized.upper()
                    .replace(" ", "_")
                ),
                description=(
                    "External medical evidence contains "
                    f"{condition.condition}."
                ),
                points=points,
            )
        )

    return factors, flags


def _evidence_review_flags(
    application: Application,
    evidence: EnrichmentEvidence,
    coverage_ratio: float,
) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []

    if evidence.identity.status == EvidenceStatus.CONFLICT:
        flags.append(
            ReviewFlag.IDENTITY_CONFLICT
        )

    if evidence.financial.status == EvidenceStatus.CONFLICT:
        flags.append(
            ReviewFlag.FINANCIAL_CONFLICT
        )

    if evidence.identity.status == EvidenceStatus.NOT_FOUND:
        flags.append(
            ReviewFlag.MISSING_IDENTITY_EVIDENCE
        )

    if evidence.medical.status == EvidenceStatus.NOT_FOUND:
        flags.append(
            ReviewFlag.MISSING_MEDICAL_EVIDENCE
        )

    if evidence.prescription.status == EvidenceStatus.NOT_FOUND:
        flags.append(
            ReviewFlag.MISSING_PRESCRIPTION_EVIDENCE
        )

    if evidence.financial.status == EvidenceStatus.NOT_FOUND:
        flags.append(
            ReviewFlag.MISSING_FINANCIAL_EVIDENCE
        )

    declared_conditions = {
        _normalize_text(condition)
        for condition in application.medical_conditions_declared
    }

    if evidence.medical.status == EvidenceStatus.AVAILABLE:
        for condition in evidence.medical.conditions:
            if (
                _normalize_text(condition.condition)
                not in declared_conditions
            ):
                flags.append(
                    ReviewFlag.UNDECLARED_MEDICAL_CONDITION
                )
                break

    declared_medications = {
        _normalize_text(medication)
        for medication in application.medications_declared
    }

    if evidence.prescription.status == EvidenceStatus.AVAILABLE:
        for prescription in evidence.prescription.prescriptions:
            if (
                prescription.active
                and _normalize_text(
                    prescription.medication
                )
                not in declared_medications
            ):
                flags.append(
                    ReviewFlag.UNDECLARED_ACTIVE_MEDICATION
                )
                break

    if coverage_ratio > 10:
        flags.append(
            ReviewFlag.EXCESSIVE_COVERAGE_RATIO
        )

    return flags


def _risk_tier(score: int) -> RiskTier:
    if score >= 50:
        return RiskTier.VERY_HIGH

    if score >= 30:
        return RiskTier.HIGH

    if score >= 15:
        return RiskTier.MODERATE

    return RiskTier.LOW


def evaluate_policy(
    application: Application,
    evidence: EnrichmentEvidence,
    evaluation_date: date,
) -> RiskAssessment:
    """
    Evaluate the synthetic underwriting policy deterministically.

    The same application, evidence, evaluation date, and policy version
    always produce the same assessment.
    """

    age = calculate_age(
        application.date_of_birth,
        evaluation_date,
    )

    bmi = calculate_bmi(
        application.height_cm,
        application.weight_kg,
    )

    coverage_ratio = (
        calculate_coverage_to_income_ratio(
            application.coverage_amount,
            application.annual_income,
        )
    )

    risk_factors: list[RiskFactor] = []

    age_factor = _age_factor(age)
    if age_factor is not None:
        risk_factors.append(age_factor)

    bmi_factor = _bmi_factor(bmi)
    if bmi_factor is not None:
        risk_factors.append(bmi_factor)

    tobacco_factor = _tobacco_factor(
        application.tobacco_use
    )
    if tobacco_factor is not None:
        risk_factors.append(tobacco_factor)

    coverage_factor = _coverage_factor(
        coverage_ratio
    )
    if coverage_factor is not None:
        risk_factors.append(coverage_factor)

    medical_factors, medical_flags = (
        _medical_factors(evidence)
    )

    risk_factors.extend(
        medical_factors
    )

    review_flags = _evidence_review_flags(
        application,
        evidence,
        coverage_ratio,
    )

    review_flags.extend(
        medical_flags
    )

    review_flags = list(
        dict.fromkeys(review_flags)
    )

    score = sum(
        factor.points
        for factor in risk_factors
    )

    return RiskAssessment(
        policy_version=POLICY_VERSION,
        evaluation_date=evaluation_date,
        age=age,
        bmi=round(bmi, 2),
        coverage_to_income_ratio=round(
            coverage_ratio,
            2,
        ),
        score=score,
        tier=_risk_tier(score),
        risk_factors=risk_factors,
        review_flags=review_flags,
        requires_review=bool(review_flags),
    )