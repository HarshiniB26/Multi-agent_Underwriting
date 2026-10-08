from enum import StrEnum

from pydantic import BaseModel, Field


class EvidenceStatus(StrEnum):
    """Availability and quality status of external evidence."""

    AVAILABLE = "available"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"


class IdentityEvidence(BaseModel):
    """Synthetic identity-verification evidence."""

    status: EvidenceStatus

    identity_verified: bool | None = None
    name_match: bool | None = None
    date_of_birth_match: bool | None = None
    state_match: bool | None = None


class MedicalCondition(BaseModel):
    """Synthetic condition returned by a medical-history provider."""

    condition: str = Field(min_length=1)
    status: str = Field(min_length=1)


class MedicalEvidence(BaseModel):
    """Synthetic medical-history evidence."""

    status: EvidenceStatus

    conditions: list[MedicalCondition] = Field(
        default_factory=list
    )


class PrescriptionRecord(BaseModel):
    """Synthetic prescription record."""

    medication: str = Field(min_length=1)
    active: bool


class PrescriptionEvidence(BaseModel):
    """Synthetic prescription-history evidence."""

    status: EvidenceStatus

    prescriptions: list[PrescriptionRecord] = Field(
        default_factory=list
    )


class FinancialEvidence(BaseModel):
    """Synthetic financial-verification evidence."""

    status: EvidenceStatus

    income_verified: bool | None = None
    verified_annual_income: int | None = Field(
        default=None,
        gt=0,
    )


class EnrichmentEvidence(BaseModel):
    """
    Complete external evidence gathered for an underwriting case.

    All data is synthetic and intended only for the educational project.
    """

    identity: IdentityEvidence
    medical: MedicalEvidence
    prescription: PrescriptionEvidence
    financial: FinancialEvidence