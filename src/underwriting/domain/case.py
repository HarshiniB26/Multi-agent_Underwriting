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


class CaseMetadata(BaseModel):
    """Metadata used to identify, trace, and version a case."""

    case_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID = Field(default_factory=uuid4)

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

    errors: list[str] = Field(default_factory=list)