from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TraceStatus(StrEnum):
    """Execution outcome for one workflow stage attempt."""

    SUCCESS = "success"
    FAILED = "failed"


class TraceRecord(BaseModel):
    """
    Structured telemetry for one workflow-stage attempt.

    One record answers: which agent ran, when, what it read from shared
    state and where that input came from, what it wrote back, how long
    it took, what it cost, and whether it failed or was retried.

    received and produced are scrubbed of direct identifiers before
    they reach a trace store (see observability/payloads.py).

    estimated_cost_usd is the equivalent commercial-API cost of the
    call, not what local inference actually cost; see
    observability/cost.py.
    """

    case_id: UUID
    trace_id: UUID

    stage: str = Field(min_length=1)
    component: str | None = None
    attempt: int = Field(ge=1)
    status: TraceStatus

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    finished_at: datetime | None = None
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

    input_sources: dict[str, str] = Field(
        default_factory=dict
    )
    tools_used: list[str] = Field(
        default_factory=list
    )
    received: dict[str, Any] = Field(
        default_factory=dict
    )
    produced: dict[str, Any] = Field(
        default_factory=dict
    )
    state_writes: list[str] = Field(
        default_factory=list
    )

    error_type: str | None = None
    error_message: str | None = None
