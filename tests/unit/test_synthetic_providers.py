import json
from datetime import date
from pathlib import Path

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

SSN = "901-00-0001"
DATE_OF_BIRTH = date(1990, 1, 1)


def build_application(
    ssn: str = SSN,
    date_of_birth: date = DATE_OF_BIRTH,
) -> Application:
    return Application(
        ssn=ssn,
        first_name="Synthetic",
        last_name="Applicant",
        date_of_birth=date_of_birth,
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


def write_records(tmp_path, records: dict) -> Path:
    data_file = tmp_path / "records.json"
    data_file.write_text(
        json.dumps(records),
        encoding="utf-8",
    )
    return data_file


def test_identity_provider_returns_available_evidence():
    provider = SyntheticIdentityProvider(
        SyntheticDataStore(
            DATA_DIR / "identity_records.json"
        )
    )

    evidence = provider.get_identity_evidence(
        build_application(
            "901-00-0001",
            date(1990, 5, 15),
        )
    )

    assert evidence.status == EvidenceStatus.AVAILABLE
    assert evidence.identity_verified is True


def test_medical_provider_returns_condition():
    provider = SyntheticMedicalHistoryProvider(
        SyntheticDataStore(
            DATA_DIR / "medical_records.json"
        )
    )

    evidence = provider.get_medical_evidence(
        build_application(
            "902-00-0002",
            date(1982, 8, 21),
        )
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

    evidence = provider.get_prescription_evidence(
        build_application(
            "902-00-0002",
            date(1982, 8, 21),
        )
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

    evidence = provider.get_financial_evidence(
        build_application(
            "903-00-0003",
            date(1978, 3, 10),
        )
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

    evidence = provider.get_medical_evidence(
        build_application("999-00-9999")
    )

    assert evidence.status == EvidenceStatus.NOT_FOUND
    assert evidence.conditions == []


ALL_PROVIDERS = [
    (SyntheticIdentityProvider, "get_identity_evidence"),
    (SyntheticMedicalHistoryProvider, "get_medical_evidence"),
    (SyntheticPrescriptionProvider, "get_prescription_evidence"),
    (SyntheticFinancialProvider, "get_financial_evidence"),
]

DATE_OF_BIRTH_CHECKED_PROVIDERS = [
    (
        SyntheticMedicalHistoryProvider,
        "get_medical_evidence",
        {"conditions": []},
    ),
    (
        SyntheticPrescriptionProvider,
        "get_prescription_evidence",
        {"prescriptions": []},
    ),
    (
        SyntheticFinancialProvider,
        "get_financial_evidence",
        {
            "income_verified": True,
            "verified_annual_income": 90000,
        },
    ),
]


@pytest.mark.parametrize(
    ("provider_class", "method_name", "fields"),
    DATE_OF_BIRTH_CHECKED_PROVIDERS,
)
def test_record_with_matching_ssn_and_date_of_birth_is_returned(
    tmp_path,
    provider_class,
    method_name,
    fields,
):
    data_file = write_records(
        tmp_path,
        {
            SSN: {
                "subject_date_of_birth": DATE_OF_BIRTH.isoformat(),
                "status": "available",
                **fields,
            }
        },
    )

    provider = provider_class(SyntheticDataStore(data_file))

    evidence = getattr(provider, method_name)(
        build_application()
    )

    assert evidence.status == EvidenceStatus.AVAILABLE


@pytest.mark.parametrize(
    ("provider_class", "method_name", "fields"),
    DATE_OF_BIRTH_CHECKED_PROVIDERS,
)
def test_record_for_different_date_of_birth_is_conflict(
    tmp_path,
    provider_class,
    method_name,
    fields,
):
    data_file = write_records(
        tmp_path,
        {
            SSN: {
                "subject_date_of_birth": "1975-06-30",
                "status": "available",
                **fields,
            }
        },
    )

    provider = provider_class(SyntheticDataStore(data_file))

    evidence = getattr(provider, method_name)(
        build_application()
    )

    assert evidence == type(evidence)(
        status=EvidenceStatus.CONFLICT
    )


def test_identity_provider_reports_vendor_match_results(
    tmp_path,
):
    data_file = write_records(
        tmp_path,
        {
            SSN: {
                "status": "available",
                "identity_verified": True,
                "name_match": True,
                "date_of_birth_match": False,
                "state_match": True,
            }
        },
    )

    provider = SyntheticIdentityProvider(
        SyntheticDataStore(data_file)
    )

    evidence = provider.get_identity_evidence(
        build_application()
    )

    assert evidence.status == EvidenceStatus.AVAILABLE
    assert evidence.date_of_birth_match is False


@pytest.mark.parametrize(
    ("provider_class", "method_name"),
    ALL_PROVIDERS,
)
def test_every_provider_returns_not_found_for_missing_record(
    tmp_path,
    provider_class,
    method_name,
):
    data_file = write_records(tmp_path, {})

    provider = provider_class(SyntheticDataStore(data_file))

    evidence = getattr(provider, method_name)(
        build_application()
    )

    assert evidence.status == EvidenceStatus.NOT_FOUND


@pytest.mark.parametrize(
    ("provider_class", "method_name"),
    ALL_PROVIDERS,
)
def test_every_provider_wraps_invalid_record(
    tmp_path,
    provider_class,
    method_name,
):
    data_file = write_records(
        tmp_path,
        {
            SSN: {
                "subject_date_of_birth": DATE_OF_BIRTH.isoformat(),
                "status": "unknown",
            }
        },
    )

    provider = provider_class(SyntheticDataStore(data_file))

    with pytest.raises(
        EnrichmentProviderResponseError,
        match="returned invalid evidence",
    ):
        getattr(provider, method_name)(
            build_application()
        )


@pytest.mark.parametrize(
    ("provider_class", "method_name"),
    ALL_PROVIDERS,
)
def test_every_provider_wraps_unreadable_dataset(
    tmp_path,
    provider_class,
    method_name,
):
    provider = provider_class(
        SyntheticDataStore(tmp_path / "missing.json")
    )

    with pytest.raises(
        EnrichmentProviderResponseError,
        match="returned invalid evidence",
    ):
        getattr(provider, method_name)(
            build_application()
        )


def test_malformed_medical_record_raises_provider_error(
    tmp_path,
):
    data_file = write_records(
        tmp_path,
        {
            SSN: {
                "subject_date_of_birth": DATE_OF_BIRTH.isoformat(),
                "status": "available",
                "conditions": [
                    {
                        "condition": "",
                        "status": "managed",
                    }
                ],
            }
        },
    )

    provider = SyntheticMedicalHistoryProvider(
        SyntheticDataStore(data_file)
    )

    with pytest.raises(
        EnrichmentProviderResponseError
    ):
        provider.get_medical_evidence(
            build_application()
        )
