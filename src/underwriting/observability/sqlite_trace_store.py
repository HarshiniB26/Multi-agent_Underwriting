import sqlite3
from pathlib import Path
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    llm_latency_ms REAL NOT NULL DEFAULT 0.0,
                    model TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    error_type TEXT,
                    error_message TEXT
                )
                """
            )

            self._ensure_llm_latency_column(
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

    @staticmethod
    def _ensure_llm_latency_column(
        connection: sqlite3.Connection,
    ) -> None:
        """
        Add llm_latency_ms to databases created by older versions.
        """

        columns = connection.execute(
            "PRAGMA table_info(workflow_traces)"
        ).fetchall()

        column_names = {
            column["name"]
            for column in columns
        }

        if "llm_latency_ms" not in column_names:
            connection.execute(
                """
                ALTER TABLE workflow_traces
                ADD COLUMN llm_latency_ms
                    REAL NOT NULL DEFAULT 0.0
                """
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
                    attempt,
                    status,
                    started_at,
                    duration_ms,
                    llm_latency_ms,
                    model,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
                    error_type,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(trace.case_id),
                    str(trace.trace_id),
                    trace.stage,
                    trace.attempt,
                    trace.status.value,
                    trace.started_at.isoformat(),
                    trace.duration_ms,
                    trace.llm_latency_ms,
                    trace.model,
                    trace.input_tokens,
                    trace.output_tokens,
                    trace.estimated_cost_usd,
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
                    attempt,
                    status,
                    started_at,
                    duration_ms,
                    llm_latency_ms,
                    model,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
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
            attempt=row["attempt"],
            status=TraceStatus(row["status"]),
            started_at=row["started_at"],
            duration_ms=row["duration_ms"],
            llm_latency_ms=row["llm_latency_ms"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            estimated_cost_usd=(
                row["estimated_cost_usd"]
            ),
            error_type=row["error_type"],
            error_message=row["error_message"],
        )