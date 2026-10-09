from uuid import uuid4

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