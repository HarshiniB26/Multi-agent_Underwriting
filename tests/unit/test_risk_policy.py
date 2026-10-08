from datetime import date

import pytest

from underwriting.domain.application import Application
from underwriting.enrichment.models import (
    EnrichmentEvidence,
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
    MedicalCondition,
    MedicalEvidence,
    PrescriptionEvidence,
    PrescriptionRecord,
)
from underwriting.risk.models import (
    ReviewFlag,
    RiskTier,
)
from underwriting.risk.policy import (
    POLICY_VERSION,
    calculate_age,
    calculate_bmi,
    evaluate_policy,
)

EVALUATION_DATE = date(2026, 10, 8)


def build_application(
    **overrides,
) -> Application:
    data = {
        "first_name": "Synthetic",
        "last_name": "Applicant",
        "date_of_birth": date(1990, 5, 15),
        "state": "PA",
        "annual_income": 100000,
        "coverage_amount": 500000,
        "occupation": "Software Engineer",
        "tobacco_use": "never",
        "height_cm": 175,
        "weight_kg": 75,
        "medical_conditions_declared": [],
        "medications_declared": [],
    }

    data.update(overrides)

    return Application(**data)


def clean_evidence() -> EnrichmentEvidence:
    return EnrichmentEvidence(
        identity=IdentityEvidence(
            status=EvidenceStatus.AVAILABLE,
            identity_verified=True,
            name_match=True,
            date_of_birth_match=True,
            state_match=True,
        ),
        medical=MedicalEvidence(
            status=EvidenceStatus.AVAILABLE,
            conditions=[],
        ),
        prescription=PrescriptionEvidence(
            status=EvidenceStatus.AVAILABLE,
            prescriptions=[],
        ),
        financial=FinancialEvidence(
            status=EvidenceStatus.AVAILABLE,
            income_verified=True,
            verified_annual_income=100000,
        ),
    )


def test_age_calculation_before_and_after_birthday():
    dob = date(1986, 10, 9)

    assert calculate_age(
        dob,
        date(2026, 10, 8),
    ) == 39

    assert calculate_age(
        dob,
        date(2026, 10, 9),
    ) == 40


def test_future_date_of_birth_is_rejected():
    with pytest.raises(
        ValueError,
        match="Date of birth cannot be after",
    ):
        calculate_age(
            date(2030, 1, 1),
            EVALUATION_DATE,
        )


def test_bmi_calculation():
    bmi = calculate_bmi(
        height_cm=175,
        weight_kg=75,
    )

    assert bmi == pytest.approx(
        24.49,
        abs=0.01,
    )


def test_clean_case_is_low_risk():
    assessment = evaluate_policy(
        build_application(),
        clean_evidence(),
        EVALUATION_DATE,
    )

    assert assessment.policy_version == POLICY_VERSION
    assert assessment.score == 0
    assert assessment.tier == RiskTier.LOW
    assert assessment.requires_review is False
    assert assessment.review_flags == []


def test_age_40_adds_five_points():
    application = build_application(
        date_of_birth=date(1986, 10, 8)
    )

    assessment = evaluate_policy(
        application,
        clean_evidence(),
        EVALUATION_DATE,
    )

    assert assessment.age == 40
    assert assessment.score == 5


def test_bmi_30_adds_five_points():
    application = build_application(
        height_cm=170,
        weight_kg=86.7,
    )

    assessment = evaluate_policy(
        application,
        clean_evidence(),
        EVALUATION_DATE,
    )

    assert assessment.bmi >= 30
    assert assessment.score == 5


def test_current_tobacco_adds_twenty_points():
    application = build_application(
        tobacco_use="current"
    )

    assessment = evaluate_policy(
        application,
        clean_evidence(),
        EVALUATION_DATE,
    )

    assert assessment.score == 20
    assert assessment.tier == RiskTier.MODERATE


def test_supported_medical_condition_adds_policy_points():
    application = build_application(
        medical_conditions_declared=[
            "Hypertension"
        ]
    )

    evidence = clean_evidence()
    evidence.medical.conditions = [
        MedicalCondition(
            condition="Hypertension",
            status="managed",
        )
    ]

    assessment = evaluate_policy(
        application,
        evidence,
        EVALUATION_DATE,
    )

    assert assessment.score == 10
    assert assessment.requires_review is False


def test_medication_does_not_double_count_condition():
    application = build_application(
        medical_conditions_declared=[
            "Hypertension"
        ],
        medications_declared=[
            "Lisinopril"
        ],
    )

    evidence = clean_evidence()

    evidence.medical.conditions = [
        MedicalCondition(
            condition="Hypertension",
            status="managed",
        )
    ]

    evidence.prescription.prescriptions = [
        PrescriptionRecord(
            medication="Lisinopril",
            active=True,
        )
    ]

    assessment = evaluate_policy(
        application,
        evidence,
        EVALUATION_DATE,
    )

    assert assessment.score == 10
    assert assessment.requires_review is False


def test_undeclared_condition_requires_review():
    evidence = clean_evidence()

    evidence.medical.conditions = [
        MedicalCondition(
            condition="Type 2 Diabetes",
            status="managed",
        )
    ]

    assessment = evaluate_policy(
        build_application(),
        evidence,
        EVALUATION_DATE,
    )

    assert assessment.score == 20
    assert (
        ReviewFlag.UNDECLARED_MEDICAL_CONDITION
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_undeclared_active_medication_requires_review():
    evidence = clean_evidence()

    evidence.prescription.prescriptions = [
        PrescriptionRecord(
            medication="Metformin",
            active=True,
        )
    ]

    assessment = evaluate_policy(
        build_application(),
        evidence,
        EVALUATION_DATE,
    )

    assert (
        ReviewFlag.UNDECLARED_ACTIVE_MEDICATION
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_missing_medical_evidence_requires_review():
    evidence = clean_evidence()

    evidence.medical = MedicalEvidence(
        status=EvidenceStatus.NOT_FOUND
    )

    assessment = evaluate_policy(
        build_application(),
        evidence,
        EVALUATION_DATE,
    )

    assert (
        ReviewFlag.MISSING_MEDICAL_EVIDENCE
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_identity_conflict_requires_review():
    evidence = clean_evidence()

    evidence.identity = IdentityEvidence(
        status=EvidenceStatus.CONFLICT,
        identity_verified=False,
        name_match=True,
        date_of_birth_match=True,
        state_match=False,
    )

    assessment = evaluate_policy(
        build_application(),
        evidence,
        EVALUATION_DATE,
    )

    assert (
        ReviewFlag.IDENTITY_CONFLICT
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_financial_conflict_requires_review():
    evidence = clean_evidence()

    evidence.financial = FinancialEvidence(
        status=EvidenceStatus.CONFLICT,
        income_verified=False,
        verified_annual_income=50000,
    )

    assessment = evaluate_policy(
        build_application(),
        evidence,
        EVALUATION_DATE,
    )

    assert (
        ReviewFlag.FINANCIAL_CONFLICT
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_coverage_ratio_above_ten_requires_review():
    application = build_application(
        annual_income=50000,
        coverage_amount=600000,
    )

    assessment = evaluate_policy(
        application,
        clean_evidence(),
        EVALUATION_DATE,
    )

    assert assessment.coverage_to_income_ratio == 12
    assert (
        ReviewFlag.EXCESSIVE_COVERAGE_RATIO
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_unknown_medical_condition_requires_review():
    application = build_application(
        medical_conditions_declared=[
            "Synthetic Unknown Condition"
        ]
    )

    evidence = clean_evidence()

    evidence.medical.conditions = [
        MedicalCondition(
            condition="Synthetic Unknown Condition",
            status="managed",
        )
    ]

    assessment = evaluate_policy(
        application,
        evidence,
        EVALUATION_DATE,
    )

    assert (
        ReviewFlag.UNSUPPORTED_MEDICAL_CONDITION
        in assessment.review_flags
    )
    assert assessment.requires_review is True


def test_same_inputs_produce_same_assessment():
    application = build_application(
        tobacco_use="former"
    )

    evidence = clean_evidence()

    first = evaluate_policy(
        application,
        evidence,
        EVALUATION_DATE,
    )

    second = evaluate_policy(
        application,
        evidence,
        EVALUATION_DATE,
    )

    assert first == second