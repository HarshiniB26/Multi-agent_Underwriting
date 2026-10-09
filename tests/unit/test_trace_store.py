from uuid import uuid4

from underwriting.observability.models import (
    TraceRecord,
    TraceStatus,
)
from underwriting.observability.trace_store import (
    InMemoryTraceStore,
)


def test_trace_record_stores_execution_metrics():
    case_id = uuid4()
    trace_id = uuid4()

    record = TraceRecord(
        case_id=case_id,
        trace_id=trace_id,
        stage="intake",
        attempt=1,
        status=TraceStatus.SUCCESS,
        duration_ms=125.5,
        model="qwen3:8b",
        input_tokens=200,
        output_tokens=40,
    )

    assert record.case_id == case_id
    assert record.trace_id == trace_id
    assert record.stage == "intake"
    assert record.attempt == 1
    assert record.status == TraceStatus.SUCCESS
    assert record.duration_ms == 125.5
    assert record.input_tokens == 200
    assert record.output_tokens == 40


def test_trace_store_returns_records_for_trace():
    store = InMemoryTraceStore()

    trace_id = uuid4()
    case_id = uuid4()

    first = TraceRecord(
        case_id=case_id,
        trace_id=trace_id,
        stage="intake",
        attempt=1,
        status=TraceStatus.SUCCESS,
        duration_ms=100,
    )

    second = TraceRecord(
        case_id=case_id,
        trace_id=trace_id,
        stage="enrichment",
        attempt=1,
        status=TraceStatus.SUCCESS,
        duration_ms=200,
    )

    store.record(first)
    store.record(second)

    records = store.get_by_trace_id(
        trace_id
    )

    assert len(records) == 2
    assert records[0].stage == "intake"
    assert records[1].stage == "enrichment"


def test_trace_store_isolates_different_traces():
    store = InMemoryTraceStore()

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


def test_trace_store_preserves_retry_attempts():
    store = InMemoryTraceStore()

    trace_id = uuid4()
    case_id = uuid4()

    store.record(
        TraceRecord(
            case_id=case_id,
            trace_id=trace_id,
            stage="enrichment",
            attempt=1,
            status=TraceStatus.FAILED,
            duration_ms=50,
            error_type="TimeoutError",
            error_message="Provider timed out.",
        )
    )

    store.record(
        TraceRecord(
            case_id=case_id,
            trace_id=trace_id,
            stage="enrichment",
            attempt=2,
            status=TraceStatus.SUCCESS,
            duration_ms=80,
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