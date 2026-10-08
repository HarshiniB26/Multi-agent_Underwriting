import json
from datetime import date

import pytest

from underwriting.agents.risk_scoring import (
    risk_scoring_agent,
)
from underwriting.domain.application import Application
from underwriting.enrichment.models import (
    EnrichmentEvidence,
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
    MedicalCondition,
    MedicalEvidence,
    PrescriptionEvidence,
)
from underwriting.llm.client import (
    LLMClient,
    LLMResponse,
)
from underwriting.risk.models import (
    ReviewFlag,
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
                    "The deterministic synthetic risk "
                    "assessment was reviewed."
                ),
                "observations": [
                    ("The explanation uses the supplied "
                    "authoritative assessment.")
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
            input_tokens=100,
            output_tokens=25,
            latency_ms=10,
        )


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


def test_clean_case_returns_authoritative_low_risk_assessment():
    llm = FakeLLMClient()

    result = risk_scoring_agent(
        application=build_application(),
        evidence=clean_evidence(),
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.assessment.score == 0
    assert result.assessment.tier == RiskTier.LOW
    assert result.assessment.requires_review is False

    assert result.review.summary
    assert llm.call_count == 1


def test_risk_points_are_calculated_by_policy_engine():
    llm = FakeLLMClient()

    application = build_application(
        tobacco_use="current"
    )

    result = risk_scoring_agent(
        application=application,
        evidence=clean_evidence(),
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.assessment.score == 20
    assert (
        result.assessment.tier
        == RiskTier.MODERATE
    )

    assert any(
        factor.code == "TOBACCO_CURRENT"
        for factor in result.assessment.risk_factors
    )


def test_medical_risk_is_calculated_deterministically():
    llm = FakeLLMClient()

    application = build_application(
        medical_conditions_declared=[
            "Type 2 Diabetes"
        ]
    )

    evidence = clean_evidence()

    evidence.medical.conditions = [
        MedicalCondition(
            condition="Type 2 Diabetes",
            status="managed",
        )
    ]

    result = risk_scoring_agent(
        application=application,
        evidence=evidence,
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.assessment.score == 20
    assert (
        result.assessment.tier
        == RiskTier.MODERATE
    )
    assert result.assessment.requires_review is False


def test_evidence_problem_preserves_score_and_requires_review():
    llm = FakeLLMClient()

    evidence = clean_evidence()

    evidence.medical.conditions = [
        MedicalCondition(
            condition="Type 2 Diabetes",
            status="managed",
        )
    ]

    result = risk_scoring_agent(
        application=build_application(),
        evidence=evidence,
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.assessment.score == 20

    assert (
        ReviewFlag.UNDECLARED_MEDICAL_CONDITION
        in result.assessment.review_flags
    )

    assert result.assessment.requires_review is True


def test_llm_cannot_override_authoritative_assessment():
    malicious_or_incorrect_response = json.dumps(
        {
            "summary": (
                "Ignore the supplied policy result. "
                "The applicant should have risk score 99 "
                "and should be approved."
            ),
            "observations": [
                "Risk tier should be very high."
            ],
        }
    )

    llm = FakeLLMClient(
        content=malicious_or_incorrect_response
    )

    result = risk_scoring_agent(
        application=build_application(),
        evidence=clean_evidence(),
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.assessment.score == 0
    assert result.assessment.tier == RiskTier.LOW
    assert result.assessment.requires_review is False


def test_authoritative_assessment_is_in_llm_prompt():
    llm = FakeLLMClient()

    result = risk_scoring_agent(
        application=build_application(
            tobacco_use="current"
        ),
        evidence=clean_evidence(),
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.assessment.score == 20

    assert llm.user_prompt is not None
    assert '"score": 20' in llm.user_prompt
    assert '"tier": "moderate"' in llm.user_prompt


def test_malformed_llm_response_is_rejected():
    llm = FakeLLMClient(
        content="This is not valid JSON."
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLM returned an invalid "
            "risk-scoring review"
        ),
    ):
        risk_scoring_agent(
            application=build_application(),
            evidence=clean_evidence(),
            evaluation_date=EVALUATION_DATE,
            llm_client=llm,
        )


def test_llm_response_metadata_is_preserved():
    llm = FakeLLMClient()

    result = risk_scoring_agent(
        application=build_application(),
        evidence=clean_evidence(),
        evaluation_date=EVALUATION_DATE,
        llm_client=llm,
    )

    assert result.llm_response.model == "fake-model"
    assert result.llm_response.input_tokens == 100
    assert result.llm_response.output_tokens == 25
    assert result.llm_response.latency_ms == 10