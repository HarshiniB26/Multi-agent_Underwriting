import json
from datetime import date

import pytest

from underwriting.agents.enrichment import (
    EnrichmentStatus,
    EvidenceConsistency,
    enrichment_agent,
)
from underwriting.domain.application import (
    Application,
    TobaccoUse,
)
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

APPLICATION_ID = "11111111-1111-4111-8111-111111111111"


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class FakeIdentityProvider(IdentityProvider):
    def __init__(
        self,
        evidence: IdentityEvidence,
    ):
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
    ):
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
    ):
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
    ):
        self.evidence = evidence

    def get_financial_evidence(
        self,
        application: Application,
    ) -> FinancialEvidence:
        return self.evidence


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        content: str | None = None,
    ):
        self.content = content or json.dumps(
            {
                "evidence_consistency": "consistent",
                "summary": (
                    "The supplied external evidence is "
                    "consistent with the application."
                ),
                "material_concerns": [],
                "corroborating_evidence": [],
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
            input_tokens=200,
            output_tokens=50,
            latency_ms=15,
        )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def application() -> Application:
    return Application(
        application_id=APPLICATION_ID,
        first_name="Aarav",
        last_name="Sharma",
        date_of_birth=date(1988, 4, 15),
        state="PA",
        annual_income=90000,
        coverage_amount=500000,
        occupation="Software Engineer",
        tobacco_use=TobaccoUse.NEVER,
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


def clean_financial(
    income: int = 90000,
) -> FinancialEvidence:
    return FinancialEvidence(
        status=EvidenceStatus.AVAILABLE,
        income_verified=True,
        verified_annual_income=income,
    )


def run_agent(
    *,
    application: Application,
    llm: FakeLLMClient | None = None,
    identity: IdentityEvidence | None = None,
    medical: MedicalEvidence | None = None,
    prescription: PrescriptionEvidence | None = None,
    financial: FinancialEvidence | None = None,
):
    return enrichment_agent(
        application=application,
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
            financial
            or clean_financial(
                application.annual_income
            )
        ),
        llm_client=llm or FakeLLMClient(),
    )


# ---------------------------------------------------------------------------
# Basic enrichment behavior
# ---------------------------------------------------------------------------


def test_clean_evidence_is_completed(
    application,
):
    result = run_agent(
        application=application,
    )

    assert result.status == EnrichmentStatus.COMPLETED
    assert result.discrepancies == []
    assert result.missing_evidence == []

    assert (
        result.review.evidence_consistency
        == EvidenceConsistency.CONSISTENT
    )


def test_undeclared_medical_condition_requires_review(
    application,
):
    medical = MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[
            MedicalCondition(
                condition="Type 2 Diabetes",
                status="managed",
            )
        ],
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "External medical evidence conflicts "
                    "with the applicant declaration."
                ),
                "material_concerns": [
                    (
                        "Type 2 Diabetes appears in "
                        "external medical evidence but "
                        "was not declared."
                    )
                ],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        medical=medical,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW

    assert any(
        "Type 2 Diabetes" in discrepancy
        for discrepancy in result.discrepancies
    )


def test_undeclared_active_medication_requires_review(
    application,
):
    prescription = PrescriptionEvidence(
        status=EvidenceStatus.AVAILABLE,
        prescriptions=[
            PrescriptionRecord(
                medication="Metformin",
                active=True,
            )
        ],
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "External prescription evidence "
                    "conflicts with the application."
                ),
                "material_concerns": [
                    (
                        "Active Metformin appears in "
                        "external evidence but was not "
                        "declared."
                    )
                ],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        prescription=prescription,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW

    assert any(
        "Metformin" in discrepancy
        for discrepancy in result.discrepancies
    )


# ---------------------------------------------------------------------------
# Meaningful LLM cross-source reasoning
# ---------------------------------------------------------------------------


def test_cross_source_medical_and_prescription_synthesis(
    application,
):
    medical = MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[
            MedicalCondition(
                condition="Type 2 Diabetes",
                status="managed",
            )
        ],
    )

    prescription = PrescriptionEvidence(
        status=EvidenceStatus.AVAILABLE,
        prescriptions=[
            PrescriptionRecord(
                medication="Metformin",
                active=True,
            )
        ],
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "External medical and prescription "
                    "evidence corroborate one another "
                    "but conflict with the applicant "
                    "declarations."
                ),
                "material_concerns": [
                    (
                        "Type 2 Diabetes and active "
                        "Metformin were not declared."
                    )
                ],
                "corroborating_evidence": [
                    (
                        "Active Metformin is consistent "
                        "with the external Type 2 "
                        "Diabetes evidence."
                    )
                ],
            }
        )
    )

    result = run_agent(
        application=application,
        medical=medical,
        prescription=prescription,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW

    assert len(result.discrepancies) == 2

    assert (
        result.review.evidence_consistency
        == EvidenceConsistency.INCONSISTENT
    )

    assert result.review.corroborating_evidence

    corroboration = (
        result.review.corroborating_evidence[0]
    )

    assert "Metformin" in corroboration
    assert "Type 2 Diabetes" in corroboration


# ---------------------------------------------------------------------------
# Identity and financial conflicts
# ---------------------------------------------------------------------------


def test_identity_conflict_requires_review(
    application,
):
    identity = IdentityEvidence(
        status=EvidenceStatus.CONFLICT,
        identity_verified=False,
        name_match=True,
        date_of_birth_match=True,
        state_match=False,
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "Identity evidence conflicts with "
                    "the application."
                ),
                "material_concerns": [
                    "Identity evidence contains a conflict."
                ],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        identity=identity,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW

    assert any(
        "Identity evidence conflicts"
        in discrepancy
        for discrepancy in result.discrepancies
    )


def test_financial_conflict_requires_review(
    application,
):
    financial = FinancialEvidence(
        status=EvidenceStatus.CONFLICT,
        income_verified=False,
        verified_annual_income=55000,
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "Financial evidence conflicts with "
                    "the application."
                ),
                "material_concerns": [
                    "Declared income is not verified."
                ],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        financial=financial,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW

    assert any(
        "Financial evidence conflicts"
        in discrepancy
        for discrepancy in result.discrepancies
    )


# ---------------------------------------------------------------------------
# Missing evidence
# ---------------------------------------------------------------------------


def test_missing_medical_evidence_requires_review(
    application,
):
    medical = MedicalEvidence(
        status=EvidenceStatus.NOT_FOUND,
        conditions=[],
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "incomplete",
                "summary": (
                    "Medical evidence is unavailable, "
                    "so the evidence picture is incomplete."
                ),
                "material_concerns": [
                    "Medical evidence is unavailable."
                ],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        medical=medical,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.NEEDS_REVIEW
    assert "medical" in result.missing_evidence

    assert (
        result.review.evidence_consistency
        == EvidenceConsistency.INCOMPLETE
    )


# ---------------------------------------------------------------------------
# Matching declarations
# ---------------------------------------------------------------------------


def test_declared_condition_matching_external_evidence_is_completed():
    application = Application(
        application_id=APPLICATION_ID,
        first_name="Maya",
        last_name="Patel",
        date_of_birth=date(1982, 7, 20),
        state="NJ",
        annual_income=120000,
        coverage_amount=750000,
        occupation="Project Manager",
        tobacco_use=TobaccoUse.NEVER,
        height_cm=165,
        weight_kg=68,
        medical_conditions_declared=[
            "Hypertension"
        ],
        medications_declared=[
            "Lisinopril"
        ],
    )

    medical = MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[
            MedicalCondition(
                condition="Hypertension",
                status="managed",
            )
        ],
    )

    prescription = PrescriptionEvidence(
        status=EvidenceStatus.AVAILABLE,
        prescriptions=[
            PrescriptionRecord(
                medication="Lisinopril",
                active=True,
            )
        ],
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "consistent",
                "summary": (
                    "External evidence is consistent "
                    "with the applicant declarations."
                ),
                "material_concerns": [],
                "corroborating_evidence": [
                    (
                        "Medical and prescription evidence "
                        "are consistent with the declared "
                        "hypertension treatment."
                    )
                ],
            }
        )
    )

    result = run_agent(
        application=application,
        medical=medical,
        prescription=prescription,
        llm=llm,
    )

    assert result.status == EnrichmentStatus.COMPLETED
    assert result.discrepancies == []
    assert result.missing_evidence == []

    assert result.review.corroborating_evidence


# ---------------------------------------------------------------------------
# LLM authority boundaries
# ---------------------------------------------------------------------------


def test_llm_cannot_remove_deterministic_discrepancy(
    application,
):
    medical = MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[
            MedicalCondition(
                condition="Type 2 Diabetes",
                status="managed",
            )
        ],
    )

    # Deliberately incorrect LLM interpretation.
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "consistent",
                "summary": (
                    "Everything appears consistent."
                ),
                "material_concerns": [],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        medical=medical,
        llm=llm,
    )

    # Authoritative result still comes from deterministic code.
    assert result.status == EnrichmentStatus.NEEDS_REVIEW

    assert result.discrepancies

    assert any(
        "Type 2 Diabetes" in discrepancy
        for discrepancy in result.discrepancies
    )


def test_llm_cannot_create_authoritative_discrepancy(
    application,
):
    # Deliberately incorrect LLM interpretation.
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "The applicant may have an issue."
                ),
                "material_concerns": [
                    (
                        "The applicant appears to have "
                        "an unspecified concern."
                    )
                ],
                "corroborating_evidence": [],
            }
        )
    )

    result = run_agent(
        application=application,
        llm=llm,
    )

    # LLM prose cannot create authoritative discrepancies.
    assert result.status == EnrichmentStatus.COMPLETED
    assert result.discrepancies == []
    assert result.missing_evidence == []


# ---------------------------------------------------------------------------
# Structured-output validation
# ---------------------------------------------------------------------------


def test_consistent_review_cannot_contain_material_concerns(
    application,
):
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "consistent",
                "summary": "Evidence is consistent.",
                "material_concerns": [
                    (
                        "This concern contradicts the "
                        "consistent classification."
                    )
                ],
                "corroborating_evidence": [],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="LLM returned an invalid enrichment review",
    ):
        run_agent(
            application=application,
            llm=llm,
        )


def test_malformed_llm_response_is_rejected(
    application,
):
    llm = FakeLLMClient(
        content="not-json"
    )

    with pytest.raises(
        ValueError,
        match="LLM returned an invalid enrichment review",
    ):
        run_agent(
            application=application,
            llm=llm,
        )


# ---------------------------------------------------------------------------
# Prompt / observability checks
# ---------------------------------------------------------------------------


def test_authoritative_findings_are_sent_to_llm(
    application,
):
    medical = MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[
            MedicalCondition(
                condition="Type 2 Diabetes",
                status="managed",
            )
        ],
    )

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "evidence_consistency": "inconsistent",
                "summary": (
                    "Medical evidence conflicts with "
                    "the application."
                ),
                "material_concerns": [
                    "An undeclared condition exists."
                ],
                "corroborating_evidence": [],
            }
        )
    )

    run_agent(
        application=application,
        medical=medical,
        llm=llm,
    )

    assert llm.user_prompt is not None

    assert (
        "authoritative_discrepancies"
        in llm.user_prompt
    )

    assert (
        "authoritative_missing_evidence"
        in llm.user_prompt
    )

    assert "Type 2 Diabetes" in llm.user_prompt


def test_llm_prompt_limits_model_authority(
    application,
):
    llm = FakeLLMClient()

    run_agent(
        application=application,
        llm=llm,
    )

    assert llm.system_prompt is not None

    prompt = llm.system_prompt.lower()

    assert "risk score" in prompt
    assert "approve or deny" in prompt
    assert "invent medical conditions" in prompt


def test_llm_response_metadata_is_preserved(
    application,
):
    result = run_agent(
        application=application,
    )

    assert result.llm_response.model == "fake-model"
    assert result.llm_response.input_tokens == 200
    assert result.llm_response.output_tokens == 50
    assert result.llm_response.latency_ms == 15