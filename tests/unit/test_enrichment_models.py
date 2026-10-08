import pytest
from pydantic import ValidationError

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


def test_complete_enrichment_evidence_can_be_created():
    evidence = EnrichmentEvidence(
        identity=IdentityEvidence(
            status=EvidenceStatus.AVAILABLE,
            identity_verified=True,
            name_match=True,
            date_of_birth_match=True,
            state_match=True,
        ),
        medical=MedicalEvidence(
            status=EvidenceStatus.AVAILABLE,
            conditions=[
                MedicalCondition(
                    condition="Hypertension",
                    status="managed",
                )
            ],
        ),
        prescription=PrescriptionEvidence(
            status=EvidenceStatus.AVAILABLE,
            prescriptions=[
                PrescriptionRecord(
                    medication="Lisinopril",
                    active=True,
                )
            ],
        ),
        financial=FinancialEvidence(
            status=EvidenceStatus.AVAILABLE,
            income_verified=True,
            verified_annual_income=90000,
        ),
    )

    assert evidence.identity.identity_verified is True
    assert len(evidence.medical.conditions) == 1
    assert len(evidence.prescription.prescriptions) == 1
    assert evidence.financial.verified_annual_income == 90000


def test_not_found_evidence_is_distinct_from_available_empty_evidence():
    available = MedicalEvidence(
        status=EvidenceStatus.AVAILABLE,
        conditions=[],
    )

    not_found = MedicalEvidence(
        status=EvidenceStatus.NOT_FOUND,
        conditions=[],
    )

    assert available.status == EvidenceStatus.AVAILABLE
    assert not_found.status == EvidenceStatus.NOT_FOUND
    assert available.status != not_found.status


def test_financial_income_must_be_positive_when_present():
    with pytest.raises(ValidationError):
        FinancialEvidence(
            status=EvidenceStatus.AVAILABLE,
            income_verified=True,
            verified_annual_income=-50000,
        )


def test_medical_condition_requires_non_empty_name():
    with pytest.raises(ValidationError):
        MedicalCondition(
            condition="",
            status="managed",
        )