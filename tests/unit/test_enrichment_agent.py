from datetime import date
from uuid import UUID

from underwriting.agents.enrichment import (
    EnrichmentStatus,
    enrichment_agent,
)
from underwriting.domain.application import Application
from underwriting.enrichment.models import (
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
    MedicalCondition,
    MedicalEvidence,
    PrescriptionEvidence,
    PrescriptionRecord,
)
from underwriting.enrichment.providers import (
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)
from underwriting.llm.client import (
    LLMClient,
    LLMResponse,
)


class FakeIdentityProvider(IdentityProvider):
    def __init__(
        self,
        evidence: IdentityEvidence,
    ) -> None:
        self.evidence = evidence

    def get_identity_evidence(
        self,
        application: Application,
    ) -> IdentityEvidence:
        return self.evidence


class FakeMedicalProvider(MedicalHistoryProvider):
    def __init__(
        self,
        evidence: MedicalEvidence,
    ) -> None:
        self.evidence = evidence

    def get_medical_evidence(
        self,
        application: Application,
    ) -> MedicalEvidence:
        return self.evidence


class FakePrescriptionProvider(PrescriptionProvider):
    def __init__(
        self,
        evidence: PrescriptionEvidence,
    ) -> None:
        self.evidence = evidence

    def get_prescription_evidence(
        self,
        application: Application,
    ) -> PrescriptionEvidence:
        return self.evidence


class FakeFinancialProvider(FinancialProvider):
    def __init__(
        self,
        evidence: FinancialEvidence,
    ) -> None:
        self.evidence = evidence

    def get_financial_evidence(
        self,
        application: Application,
    ) -> FinancialEvidence:
        return self.evidence


class FakeLLMClient(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.calls += 1

        return LLMResponse(
            content=(
                '{"summary":"Synthetic evidence reviewed.",'
                '"observations":[]}'
            ),
            model="fake-model",
            input_tokens=150,
            output_tokens=30,
            latency_ms=25,
        )


def build_application() -> Application:
    return Application(
        application_id=UUID(
            "11111111-1111-4111-8111-111111111111"
        ),
        first_name="Aarav",
        last_name="Sharma",
        date_of_birth=date(1990, 5, 15),
        state="PA",
        annual_income=90000,
        coverage_amount=500000,
        occupation="Software Engineer",
        tobacco_use="never",
        height_cm=175,
        weight_kg=75,
        medical_conditions_declared=[],
        medications_declared=[],
    )


def clean_identity() -> IdentityEvidence:
    return IdentityEvidence(
        status=EvidenceStatus.AVAILABLE,
        identity_verified=True,
        name_match=True,
        date_of_birth_match=True,
        state_match=True,
    )


def clean_medical() -> MedicalEvidence:
    return MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[],
    )


def clean_prescription() -> PrescriptionEvidence:
    return PrescriptionEvidence(
        status=EvidenceStatus.AVAILABLE,
        prescriptions=[],
    )


def clean_financial() -> FinancialEvidence:
    return FinancialEvidence(
        status=EvidenceStatus.AVAILABLE,
        income_verified=True,
        verified_annual_income=90000,
    )


def run_agent(
    *,
    identity: IdentityEvidence | None = None,
    medical: MedicalEvidence | None = None,
    prescription: PrescriptionEvidence | None = None,
    financial: FinancialEvidence | None = None,
):
    llm = FakeLLMClient()

    result = enrichment_agent(
        application=build_application(),
        identity_provider=FakeIdentityProvider(
            identity or clean_identity()
        ),
        medical_provider=FakeMedicalProvider(
            medical or clean_medical()
        ),
        prescription_provider=FakePrescriptionProvider(
            prescription or clean_prescription()
        ),
        financial_provider=FakeFinancialProvider(
            financial or clean_financial()
        ),
        llm_client=llm,
    )

    return result, llm


def test_clean_evidence_completes_enrichment():
    result, llm = run_agent()

    assert result.status == EnrichmentStatus.COMPLETED
    assert result.discrepancies == []
    assert result.missing_evidence == []
    assert result.review.summary
    assert llm.calls == 1


def test_undeclared_medical_condition_requires_review():
    result, _ = run_agent(
        medical=MedicalEvidence(
            status=EvidenceStatus.AVAILABLE,
            conditions=[
                MedicalCondition(
                    condition="Type 2 Diabetes",
                    status="managed",
                )
            ],
        )
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW
    assert (
        "External medical evidence contains an undeclared "
        "condition: Type 2 Diabetes."
        in result.discrepancies
    )


def test_undeclared_active_medication_requires_review():
    result, _ = run_agent(
        prescription=PrescriptionEvidence(
            status=EvidenceStatus.AVAILABLE,
            prescriptions=[
                PrescriptionRecord(
                    medication="Metformin",
                    active=True,
                )
            ],
        )
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW
    assert (
        "External prescription evidence contains an undeclared "
        "active medication: Metformin."
        in result.discrepancies
    )


def test_identity_conflict_requires_review():
    result, _ = run_agent(
        identity=IdentityEvidence(
            status=EvidenceStatus.CONFLICT,
            identity_verified=False,
            name_match=True,
            date_of_birth_match=True,
            state_match=False,
        )
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW
    assert (
        "External identity evidence has a conflict."
        in result.discrepancies
    )


def test_financial_conflict_requires_review():
    result, _ = run_agent(
        financial=FinancialEvidence(
            status=EvidenceStatus.CONFLICT,
            income_verified=False,
            verified_annual_income=55000,
        )
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW
    assert (
        "External financial evidence has a conflict."
        in result.discrepancies
    )


def test_missing_evidence_is_not_treated_as_clean():
    result, _ = run_agent(
        medical=MedicalEvidence(
            status=EvidenceStatus.NOT_FOUND,
        )
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW
    assert "medical" in result.missing_evidence


def test_matching_declared_condition_is_not_a_discrepancy():
    application = build_application()
    application.medical_conditions_declared = [
        "Hypertension"
    ]

    llm = FakeLLMClient()

    result = enrichment_agent(
        application=application,
        identity_provider=FakeIdentityProvider(
            clean_identity()
        ),
        medical_provider=FakeMedicalProvider(
            MedicalEvidence(
                status=EvidenceStatus.AVAILABLE,
                conditions=[
                    MedicalCondition(
                        condition="Hypertension",
                        status="managed",
                    )
                ],
            )
        ),
        prescription_provider=FakePrescriptionProvider(
            clean_prescription()
        ),
        financial_provider=FakeFinancialProvider(
            clean_financial()
        ),
        llm_client=llm,
    )

    assert result.status == EnrichmentStatus.COMPLETED
    assert result.discrepancies == []