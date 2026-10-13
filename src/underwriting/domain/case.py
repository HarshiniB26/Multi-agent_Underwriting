from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from underwriting.domain.application import Application


class CaseStatus(StrEnum):
    """Lifecycle states for an underwriting case."""

    RECEIVED = "received"
    INTAKE_COMPLETED = "intake_completed"
    ENRICHMENT_COMPLETED = "enrichment_completed"
    RISK_ASSESSED = "risk_assessed"
    RECOMMENDATION_COMPLETED = "recommendation_completed"
    HUMAN_REVIEW = "human_review"
    PROCESSING_FAILED = "processing_failed"


CLOSED_CASE_STATUSES = frozenset(
    {
        CaseStatus.RECOMMENDATION_COMPLETED,
        CaseStatus.PROCESSING_FAILED,
    }
)
"""
Statuses that end a case for duplicate-detection purposes.

HUMAN_REVIEW is deliberately not closed: an underwriter has not yet
decided the case, so a new application from the same person must be
reviewed alongside it.
"""


class CaseMetadata(BaseModel):
    """Metadata used to identify, trace, and version a case."""

    case_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID = Field(default_factory=uuid4)

    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    applicant_key: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    version: int = Field(default=1, ge=1)


class CaseState(BaseModel):
    """
    Domain representation of a single underwriting case.

    Each processing stage owns a specific portion of this state.
    """

    metadata: CaseMetadata = Field(default_factory=CaseMetadata)

    status: CaseStatus = CaseStatus.RECEIVED

    application: Application | None = None
    enrichment: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None

    possible_duplicate_of: list[UUID] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)