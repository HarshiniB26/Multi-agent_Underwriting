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


class SyntheticIdentityProvider(IdentityProvider):
    """Identity provider backed by synthetic JSON records."""

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
            record = self.store.get_record(
                application.application_id
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
            record = self.store.get_record(
                application.application_id
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
            record = self.store.get_record(
                application.application_id
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
            record = self.store.get_record(
                application.application_id
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