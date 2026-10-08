from abc import ABC, abstractmethod

from underwriting.domain.application import Application
from underwriting.enrichment.models import (
    FinancialEvidence,
    IdentityEvidence,
    MedicalEvidence,
    PrescriptionEvidence,
)


class EnrichmentProviderError(Exception):
    """Base exception for enrichment-provider failures."""


class EnrichmentProviderTimeoutError(
    EnrichmentProviderError
):
    """Raised when an enrichment provider times out."""


class EnrichmentProviderResponseError(
    EnrichmentProviderError
):
    """Raised when a provider returns an invalid response."""


class IdentityProvider(ABC):
    """Contract for retrieving identity evidence."""

    @abstractmethod
    def get_identity_evidence(
        self,
        application: Application,
    ) -> IdentityEvidence:
        raise NotImplementedError


class MedicalHistoryProvider(ABC):
    """Contract for retrieving medical-history evidence."""

    @abstractmethod
    def get_medical_evidence(
        self,
        application: Application,
    ) -> MedicalEvidence:
        raise NotImplementedError


class PrescriptionProvider(ABC):
    """Contract for retrieving prescription evidence."""

    @abstractmethod
    def get_prescription_evidence(
        self,
        application: Application,
    ) -> PrescriptionEvidence:
        raise NotImplementedError


class FinancialProvider(ABC):
    """Contract for retrieving financial-verification evidence."""

    @abstractmethod
    def get_financial_evidence(
        self,
        application: Application,
    ) -> FinancialEvidence:
        raise NotImplementedError