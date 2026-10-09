from abc import ABC, abstractmethod
from uuid import UUID

from underwriting.observability.models import TraceRecord


class TraceStore(ABC):
    """Persistence contract for structured workflow traces."""

    @abstractmethod
    def record(
        self,
        trace: TraceRecord,
    ) -> None:
        """Persist one workflow trace record."""
        raise NotImplementedError

    @abstractmethod
    def get_by_trace_id(
        self,
        trace_id: UUID,
    ) -> list[TraceRecord]:
        """Return all records belonging to one workflow trace."""
        raise NotImplementedError


class InMemoryTraceStore(TraceStore):
    """
    In-memory trace store for local execution and isolated tests.

    A durable implementation can later use SQLite, PostgreSQL,
    or an external observability platform without changing
    orchestration code.
    """

    def __init__(self) -> None:
        self._records: list[TraceRecord] = []

    def record(
        self,
        trace: TraceRecord,
    ) -> None:
        self._records.append(
            trace.model_copy(deep=True)
        )

    def get_by_trace_id(
        self,
        trace_id: UUID,
    ) -> list[TraceRecord]:
        return [
            record.model_copy(deep=True)
            for record in self._records
            if record.trace_id == trace_id
        ]