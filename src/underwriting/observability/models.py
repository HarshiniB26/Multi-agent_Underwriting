from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class TraceStatus(StrEnum):
    """Execution outcome for one workflow stage attempt."""

    SUCCESS = "success"
    FAILED = "failed"


class TraceRecord(BaseModel):
    """
    Structured telemetry for one workflow-stage attempt.

    Business payloads are intentionally excluded to avoid placing
    applicant information in operational telemetry.
    """

    case_id: UUID
    trace_id: UUID

    stage: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    status: TraceStatus

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    duration_ms: float = Field(ge=0)

    llm_latency_ms: float = Field(
        default=0.0,
        ge=0,
    )

    model: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0,
    )

    error_type: str | None = None
    error_message: str | None = None