import json
from datetime import date
from pathlib import Path
from uuid import UUID

import pytest

from underwriting.domain.application import (
    Application,
)
from underwriting.enrichment.models import (
    EvidenceStatus,
)
from underwriting.enrichment.providers import (
    EnrichmentProviderResponseError,
)
from underwriting.enrichment.synthetic_providers import (
    SyntheticFinancialProvider,
    SyntheticIdentityProvider,
    SyntheticMedicalHistoryProvider,
    SyntheticPrescriptionProvider,
)
from underwriting.enrichment.synthetic_store import (
    SyntheticDataStore,
)

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
)


def build_application(
    application_id: str,
) -> Application:
    return Application(
        application_id=UUID(application_id),
        first_name="Synthetic",
        last_name="Applicant",
        date_of_birth=date(1990, 1, 1),
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


def test_identity_provider_returns_available_evidence():
    provider = SyntheticIdentityProvider(
        SyntheticDataStore(
            DATA_DIR / "identity_records.json"
        )
    )

    application = build_application(
        "11111111-1111-4111-8111-111111111111"
    )

    evidence = provider.get_identity_evidence(
        application
    )

    assert evidence.status == EvidenceStatus.AVAILABLE
    assert evidence.identity_verified is True


def test_medical_provider_returns_condition():
    provider = SyntheticMedicalHistoryProvider(
        SyntheticDataStore(
            DATA_DIR / "medical_records.json"
        )
    )

    application = build_application(
        "22222222-2222-4222-8222-222222222222"
    )

    evidence = provider.get_medical_evidence(
        application
    )

    assert evidence.status == EvidenceStatus.AVAILABLE
    assert len(evidence.conditions) == 1
    assert (
        evidence.conditions[0].condition
        == "Hypertension"
    )


def test_prescription_provider_returns_prescription():
    provider = SyntheticPrescriptionProvider(
        SyntheticDataStore(
            DATA_DIR / "prescription_records.json"
        )
    )

    application = build_application(
        "22222222-2222-4222-8222-222222222222"
    )

    evidence = provider.get_prescription_evidence(
        application
    )

    assert evidence.status == EvidenceStatus.AVAILABLE
    assert len(evidence.prescriptions) == 1
    assert (
        evidence.prescriptions[0].medication
        == "Lisinopril"
    )


def test_financial_provider_returns_conflict():
    provider = SyntheticFinancialProvider(
        SyntheticDataStore(
            DATA_DIR / "financial_records.json"
        )
    )

    application = build_application(
        "33333333-3333-4333-8333-333333333333"
    )

    evidence = provider.get_financial_evidence(
        application
    )

    assert evidence.status == EvidenceStatus.CONFLICT
    assert evidence.income_verified is False
    assert evidence.verified_annual_income == 55000


def test_missing_record_returns_not_found():
    provider = SyntheticMedicalHistoryProvider(
        SyntheticDataStore(
            DATA_DIR / "medical_records.json"
        )
    )

    application = build_application(
        "22222220-2222-4222-8222-222222222220"
    )

    evidence = provider.get_medical_evidence(
        application
    )

    assert evidence.status == EvidenceStatus.NOT_FOUND
    assert evidence.conditions == []

def test_malformed_medical_record_raises_provider_error(
    tmp_path,
):
    data_file = tmp_path / "medical_records.json"

    application_id = (
        "11111111-1111-4111-8111-111111111111"
    )

    malformed_data = {
        application_id: {
            "status": "available",
            "conditions": [
                {
                    "condition": "",
                    "status": "managed",
                }
            ],
        }
    }

    data_file.write_text(
        json.dumps(malformed_data),
        encoding="utf-8",
    )

    provider = SyntheticMedicalHistoryProvider(
        SyntheticDataStore(data_file)
    )

    application = build_application(
        application_id
    )

    with pytest.raises(
        EnrichmentProviderResponseError
    ):
        provider.get_medical_evidence(
            application
        )