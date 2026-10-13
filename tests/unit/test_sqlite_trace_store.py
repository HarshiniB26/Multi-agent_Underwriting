import sqlite3
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from underwriting.observability.models import (
    TraceRecord,
    TraceStatus,
)
from underwriting.observability.sqlite_trace_store import (
    SQLiteTraceStore,
)


def test_sqlite_trace_store_persists_trace(
    tmp_path,
):
    database_path = tmp_path / "underwriting.db"

    store = SQLiteTraceStore(database_path)

    case_id = uuid4()
    trace_id = uuid4()

    trace = TraceRecord(
        case_id=case_id,
        trace_id=trace_id,
        stage="intake",
        attempt=1,
        status=TraceStatus.SUCCESS,
        duration_ms=125.5,
        llm_latency_ms=110.25,
        model="qwen3:8b",
        input_tokens=200,
        output_tokens=50,
        estimated_cost_usd=0.00006,
        received={"raw_application": {"first_name": "Aarav"}},
        produced={"status": "accepted"},
    )

    store.record(trace)

    records = store.get_by_trace_id(
        trace_id
    )

    assert len(records) == 1

    persisted = records[0]

    assert persisted.case_id == case_id
    assert persisted.trace_id == trace_id
    assert persisted.stage == "intake"
    assert persisted.attempt == 1
    assert persisted.status == TraceStatus.SUCCESS
    assert persisted.duration_ms == 125.5
    assert persisted.llm_latency_ms == 110.25
    assert persisted.model == "qwen3:8b"
    assert persisted.input_tokens == 200
    assert persisted.output_tokens == 50
    assert persisted.estimated_cost_usd == 0.00006
    assert persisted.received == {
        "raw_application": {"first_name": "Aarav"}
    }
    assert persisted.produced == {
        "status": "accepted"
    }


def test_sqlite_trace_store_persists_failure(
    tmp_path,
):
    database_path = tmp_path / "underwriting.db"

    store = SQLiteTraceStore(database_path)

    trace_id = uuid4()

    store.record(
        TraceRecord(
            case_id=uuid4(),
            trace_id=trace_id,
            stage="enrichment",
            attempt=1,
            status=TraceStatus.FAILED,
            duration_ms=50,
            error_type="TimeoutError",
            error_message="Provider timed out.",
        )
    )

    records = store.get_by_trace_id(
        trace_id
    )

    assert len(records) == 1
    assert records[0].status == TraceStatus.FAILED
    assert records[0].error_type == "TimeoutError"
    assert (
        records[0].error_message
        == "Provider timed out."
    )


def test_sqlite_trace_store_preserves_attempt_order(
    tmp_path,
):
    database_path = tmp_path / "underwriting.db"

    store = SQLiteTraceStore(database_path)

    case_id = uuid4()
    trace_id = uuid4()

    store.record(
        TraceRecord(
            case_id=case_id,
            trace_id=trace_id,
            stage="intake",
            attempt=1,
            status=TraceStatus.FAILED,
            duration_ms=50,
            error_type="ConnectionError",
            error_message="Temporary failure.",
        )
    )

    store.record(
        TraceRecord(
            case_id=case_id,
            trace_id=trace_id,
            stage="intake",
            attempt=2,
            status=TraceStatus.SUCCESS,
            duration_ms=75,
            model="qwen3:8b",
            input_tokens=100,
            output_tokens=25,
        )
    )

    records = store.get_by_trace_id(
        trace_id
    )

    assert len(records) == 2

    assert records[0].attempt == 1
    assert records[0].status == TraceStatus.FAILED

    assert records[1].attempt == 2
    assert records[1].status == TraceStatus.SUCCESS


def test_sqlite_trace_store_isolates_traces(
    tmp_path,
):
    database_path = tmp_path / "underwriting.db"

    store = SQLiteTraceStore(database_path)

    first_trace_id = uuid4()
    second_trace_id = uuid4()

    store.record(
        TraceRecord(
            case_id=uuid4(),
            trace_id=first_trace_id,
            stage="intake",
            attempt=1,
            status=TraceStatus.SUCCESS,
            duration_ms=100,
        )
    )

    store.record(
        TraceRecord(
            case_id=uuid4(),
            trace_id=second_trace_id,
            stage="intake",
            attempt=1,
            status=TraceStatus.SUCCESS,
            duration_ms=100,
        )
    )

    records = store.get_by_trace_id(
        first_trace_id
    )

    assert len(records) == 1
    assert records[0].trace_id == first_trace_id


def test_legacy_schema_is_migrated_with_optional_columns(
    tmp_path,
):
    database_path = tmp_path / "underwriting.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE workflow_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                model TEXT,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                error_type TEXT,
                error_message TEXT
            )
            """
        )
    connection.close()

    store = SQLiteTraceStore(database_path)

    with sqlite3.connect(database_path) as connection:
        column_names = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(workflow_traces)"
            )
        }
    connection.close()

    assert set(SQLiteTraceStore._OPTIONAL_COLUMNS) <= column_names

    trace_id = uuid4()

    store.record(
        TraceRecord(
            case_id=uuid4(),
            trace_id=trace_id,
            stage="intake",
            attempt=1,
            status=TraceStatus.SUCCESS,
            duration_ms=10,
            received={"raw_application": {}},
        )
    )

    records = store.get_by_trace_id(trace_id)

    assert records[0].received == {"raw_application": {}}


def test_sqlite_trace_store_persists_lineage_fields(
    tmp_path,
):
    store = SQLiteTraceStore(tmp_path / "underwriting.db")

    trace_id = uuid4()
    finished_at = datetime(2026, 10, 13, 9, 0, 1, tzinfo=UTC)

    store.record(
        TraceRecord(
            case_id=uuid4(),
            trace_id=trace_id,
            stage="enrichment",
            component="enrichment_agent",
            attempt=1,
            status=TraceStatus.SUCCESS,
            finished_at=finished_at,
            duration_ms=10,
            input_sources={"application": "intake_agent"},
            tools_used=["SyntheticIdentityProvider"],
            state_writes=["enrichment"],
        )
    )

    persisted = store.get_by_trace_id(trace_id)[0]

    assert persisted.component == "enrichment_agent"
    assert persisted.finished_at == finished_at
    assert persisted.input_sources == {"application": "intake_agent"}
    assert persisted.tools_used == ["SyntheticIdentityProvider"]
    assert persisted.state_writes == ["enrichment"]


def test_legacy_rows_read_back_with_empty_lineage(
    tmp_path,
):
    database_path = tmp_path / "underwriting.db"
    store = SQLiteTraceStore(database_path)
    trace_id = uuid4()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO workflow_traces (
                case_id, trace_id, stage, attempt, status,
                started_at, duration_ms, input_tokens,
                output_tokens, estimated_cost_usd
            )
            VALUES (?, ?, 'intake', 1, 'success', ?, 5, 0, 0, 0)
            """,
            (
                str(uuid4()),
                str(trace_id),
                datetime.now(UTC).isoformat(),
            ),
        )
    connection.close()

    persisted = store.get_by_trace_id(trace_id)[0]

    assert persisted.component is None
    assert persisted.finished_at is None
    assert persisted.input_sources == {}
    assert persisted.tools_used == []
    assert persisted.state_writes == []


def test_sqlite_stores_close_every_connection(
    tmp_path,
    monkeypatch,
):
    from underwriting.domain.case import CaseState
    from underwriting.repositories.sqlite_case_repository import (
        SQLiteCaseRepository,
    )

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)

    database_path = tmp_path / "underwriting.db"
    case_repository = SQLiteCaseRepository(database_path)
    trace_store = SQLiteTraceStore(database_path)

    case = CaseState()
    case_repository.create(case)
    case_repository.save(case_repository.get(case.metadata.case_id))
    trace_store.record(
        TraceRecord(
            case_id=case.metadata.case_id,
            trace_id=case.metadata.trace_id,
            stage="intake",
            attempt=1,
            status=TraceStatus.SUCCESS,
            duration_ms=1,
        )
    )
    trace_store.get_by_trace_id(case.metadata.trace_id)

    assert len(opened) == 7

    for connection in opened:
        with pytest.raises(
            sqlite3.ProgrammingError,
            match="closed",
        ):
            connection.execute("SELECT 1")


def test_case_and_trace_tables_can_share_database(
    tmp_path,
):
    from underwriting.domain.case import CaseState
    from underwriting.repositories.sqlite_case_repository import (
        SQLiteCaseRepository,
    )

    database_path = tmp_path / "underwriting.db"

    case_repository = SQLiteCaseRepository(
        database_path
    )
    trace_store = SQLiteTraceStore(
        database_path
    )

    case = CaseState()
    case_repository.create(case)

    trace_store.record(
        TraceRecord(
            case_id=case.metadata.case_id,
            trace_id=case.metadata.trace_id,
            stage="intake",
            attempt=1,
            status=TraceStatus.SUCCESS,
            duration_ms=100,
        )
    )

    persisted_case = case_repository.get(
        case.metadata.case_id
    )

    traces = trace_store.get_by_trace_id(
        case.metadata.trace_id
    )

    assert persisted_case == case
    assert len(traces) == 1
    assert (
        traces[0].case_id
        == case.metadata.case_id
    )