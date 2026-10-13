import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from underwriting.observability.models import (
    TraceRecord,
    TraceStatus,
)
from underwriting.observability.trace_store import TraceStore


class SQLiteTraceStore(TraceStore):
    """SQLite implementation of the structured trace store."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = str(database_path)
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection that commits on success and always closes."""

        connection = sqlite3.connect(
            self.database_path
        )
        connection.row_factory = sqlite3.Row

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    component TEXT,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL NOT NULL,
                    llm_latency_ms REAL NOT NULL DEFAULT 0.0,
                    model TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    input_sources_json TEXT NOT NULL DEFAULT '{}',
                    tools_used_json TEXT NOT NULL DEFAULT '[]',
                    received_json TEXT NOT NULL DEFAULT '{}',
                    produced_json TEXT NOT NULL DEFAULT '{}',
                    state_writes_json TEXT NOT NULL DEFAULT '[]',
                    error_type TEXT,
                    error_message TEXT
                )
                """
            )

            self._ensure_optional_columns(
                connection
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_workflow_traces_trace_id
                ON workflow_traces (trace_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_workflow_traces_case_id
                ON workflow_traces (case_id)
                """
            )

    _OPTIONAL_COLUMNS: ClassVar[dict[str, str]] = {
        "component": "TEXT",
        "finished_at": "TEXT",
        "llm_latency_ms": "REAL NOT NULL DEFAULT 0.0",
        "input_sources_json": "TEXT NOT NULL DEFAULT '{}'",
        "tools_used_json": "TEXT NOT NULL DEFAULT '[]'",
        "received_json": "TEXT NOT NULL DEFAULT '{}'",
        "produced_json": "TEXT NOT NULL DEFAULT '{}'",
        "state_writes_json": "TEXT NOT NULL DEFAULT '[]'",
    }

    @classmethod
    def _ensure_optional_columns(
        cls,
        connection: sqlite3.Connection,
    ) -> None:
        """
        Add columns introduced after the original table schema.
        """

        column_names = {
            column["name"]
            for column in connection.execute(
                "PRAGMA table_info(workflow_traces)"
            ).fetchall()
        }

        for column_name, definition in cls._OPTIONAL_COLUMNS.items():
            if column_name not in column_names:
                connection.execute(
                    "ALTER TABLE workflow_traces "
                    f"ADD COLUMN {column_name} {definition}"
                )

    def record(
        self,
        trace: TraceRecord,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_traces (
                    case_id,
                    trace_id,
                    stage,
                    component,
                    attempt,
                    status,
                    started_at,
                    finished_at,
                    duration_ms,
                    llm_latency_ms,
                    model,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
                    input_sources_json,
                    tools_used_json,
                    received_json,
                    produced_json,
                    state_writes_json,
                    error_type,
                    error_message
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    str(trace.case_id),
                    str(trace.trace_id),
                    trace.stage,
                    trace.component,
                    trace.attempt,
                    trace.status.value,
                    trace.started_at.isoformat(),
                    (
                        trace.finished_at.isoformat()
                        if trace.finished_at is not None
                        else None
                    ),
                    trace.duration_ms,
                    trace.llm_latency_ms,
                    trace.model,
                    trace.input_tokens,
                    trace.output_tokens,
                    trace.estimated_cost_usd,
                    json.dumps(trace.input_sources),
                    json.dumps(trace.tools_used),
                    json.dumps(trace.received),
                    json.dumps(trace.produced),
                    json.dumps(trace.state_writes),
                    trace.error_type,
                    trace.error_message,
                ),
            )

    def get_by_trace_id(
        self,
        trace_id: UUID,
    ) -> list[TraceRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    case_id,
                    trace_id,
                    stage,
                    component,
                    attempt,
                    status,
                    started_at,
                    finished_at,
                    duration_ms,
                    llm_latency_ms,
                    model,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
                    input_sources_json,
                    tools_used_json,
                    received_json,
                    produced_json,
                    state_writes_json,
                    error_type,
                    error_message
                FROM workflow_traces
                WHERE trace_id = ?
                ORDER BY id
                """,
                (str(trace_id),),
            ).fetchall()

        return [
            self._row_to_trace(row)
            for row in rows
        ]

    @staticmethod
    def _row_to_trace(
        row: sqlite3.Row,
    ) -> TraceRecord:
        return TraceRecord(
            case_id=UUID(row["case_id"]),
            trace_id=UUID(row["trace_id"]),
            stage=row["stage"],
            component=row["component"],
            attempt=row["attempt"],
            status=TraceStatus(row["status"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_ms=row["duration_ms"],
            llm_latency_ms=row["llm_latency_ms"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            estimated_cost_usd=(
                row["estimated_cost_usd"]
            ),
            input_sources=json.loads(
                row["input_sources_json"] or "{}"
            ),
            tools_used=json.loads(
                row["tools_used_json"] or "[]"
            ),
            received=json.loads(
                row["received_json"] or "{}"
            ),
            produced=json.loads(
                row["produced_json"] or "{}"
            ),
            state_writes=json.loads(
                row["state_writes_json"] or "[]"
            ),
            error_type=row["error_type"],
            error_message=row["error_message"],
        )
