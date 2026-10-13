from typing import Any

from pydantic import ValidationError

from underwriting.domain.application import Application
from underwriting.enrichment.models import (
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
    MedicalEvidence,
    PrescriptionEvidence,
)
from underwriting.enrichment.providers import (
    EnrichmentProviderResponseError,
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)
from underwriting.enrichment.synthetic_store import (
    SyntheticDataStore,
    SyntheticDataStoreError,
)

SUBJECT_DATE_OF_BIRTH = "subject_date_of_birth"


def _find_subject_record(
    store: SyntheticDataStore,
    application: Application,
    *,
    require_date_of_birth_match: bool,
) -> dict[str, Any] | None:
    """
    Look up a vendor record by SSN.

    Returns None when the vendor has no record for the SSN. When the
    record belongs to someone with a different date of birth, the
    returned record has CONFLICT status so the mismatch is reviewed
    rather than silently attributed to the applicant.
    """

    record = store.get_record(application.ssn)

    if record is None:
        return None

    record = dict(record)
    subject_date_of_birth = record.pop(
        SUBJECT_DATE_OF_BIRTH,
        None,
    )

    if (
        require_date_of_birth_match
        and subject_date_of_birth
        != application.date_of_birth.isoformat()
    ):
        return {"status": EvidenceStatus.CONFLICT}

    return record


class SyntheticIdentityProvider(IdentityProvider):
    """
    Identity provider backed by synthetic JSON records.

    The vendor is searched by SSN and itself reports whether the
    applicant's name, date of birth, and state match its records.
    """

    def __init__(
        self,
        store: SyntheticDataStore,
    ) -> None:
        self.store = store

    def get_identity_evidence(
        self,
        application: Application,
    ) -> IdentityEvidence:
        try:
            record = _find_subject_record(
                self.store,
                application,
                require_date_of_birth_match=False,
            )

            if record is None:
                return IdentityEvidence(
                    status=EvidenceStatus.NOT_FOUND
                )

            return IdentityEvidence.model_validate(
                record
            )

        except (
            ValidationError,
            SyntheticDataStoreError,
        ) as exc:
            raise EnrichmentProviderResponseError(
                "Identity provider returned invalid evidence."
            ) from exc


class SyntheticMedicalHistoryProvider(
    MedicalHistoryProvider
):
    """Medical provider backed by synthetic JSON records."""

    def __init__(
        self,
        store: SyntheticDataStore,
    ) -> None:
        self.store = store

    def get_medical_evidence(
        self,
        application: Application,
    ) -> MedicalEvidence:
        try:
            record = _find_subject_record(
                self.store,
                application,
                require_date_of_birth_match=True,
            )

            if record is None:
                return MedicalEvidence(
                    status=EvidenceStatus.NOT_FOUND
                )

            return MedicalEvidence.model_validate(
                record
            )

        except (
            ValidationError,
            SyntheticDataStoreError,
        ) as exc:
            raise EnrichmentProviderResponseError(
                "Medical provider returned invalid evidence."
            ) from exc


class SyntheticPrescriptionProvider(
    PrescriptionProvider
):
    """Prescription provider backed by synthetic JSON records."""

    def __init__(
        self,
        store: SyntheticDataStore,
    ) -> None:
        self.store = store

    def get_prescription_evidence(
        self,
        application: Application,
    ) -> PrescriptionEvidence:
        try:
            record = _find_subject_record(
                self.store,
                application,
                require_date_of_birth_match=True,
            )

            if record is None:
                return PrescriptionEvidence(
                    status=EvidenceStatus.NOT_FOUND
                )

            return PrescriptionEvidence.model_validate(
                record
            )

        except (
            ValidationError,
            SyntheticDataStoreError,
        ) as exc:
            raise EnrichmentProviderResponseError(
                "Prescription provider returned invalid evidence."
            ) from exc


class SyntheticFinancialProvider(FinancialProvider):
    """Financial provider backed by synthetic JSON records."""

    def __init__(
        self,
        store: SyntheticDataStore,
    ) -> None:
        self.store = store

    def get_financial_evidence(
        self,
        application: Application,
    ) -> FinancialEvidence:
        try:
            record = _find_subject_record(
                self.store,
                application,
                require_date_of_birth_match=True,
            )

            if record is None:
                return FinancialEvidence(
                    status=EvidenceStatus.NOT_FOUND
                )

            return FinancialEvidence.model_validate(
                record
            )

        except (
            ValidationError,
            SyntheticDataStoreError,
        ) as exc:
            raise EnrichmentProviderResponseError(
                "Financial provider returned invalid evidence."
            ) from exc
