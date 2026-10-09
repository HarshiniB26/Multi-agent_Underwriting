import json
from datetime import date
from pathlib import Path

import pytest

from underwriting.domain.application import Application
from underwriting.domain.case import CaseStatus
from underwriting.enrichment.models import (
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
    MedicalEvidence,
    PrescriptionEvidence,
)
from underwriting.enrichment.providers import (
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)
from underwriting.llm.client import LLMClient, LLMResponse
from underwriting.observability.models import TraceStatus
from underwriting.observability.trace_store import InMemoryTraceStore
from underwriting.repositories.sqlite_case_repository import (
    SQLiteCaseRepository,
)
from underwriting.workflow.orchestrator import (
    RetryPolicy,
    UnderwritingOrchestrator,
)

APPLICATION_ID = "11111111-1111-4111-8111-111111111111"
EVALUATION_DATE = date(2026, 10, 8)


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------


class SequentialFakeLLMClient(LLMClient):
    """
    Return deterministic responses in workflow order.

    Expected order:
    1. intake
    2. enrichment
    3. risk scoring
    4. recommendation
    """

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = responses
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        if self.call_count >= len(self.responses):
            raise AssertionError(
                "Unexpected extra LLM call."
            )

        content = self.responses[self.call_count]
        self.call_count += 1

        return LLMResponse(
            content=content,
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            latency_ms=10,
        )


class FailingLLMClient(LLMClient):
    """LLM that always raises a transient connection error."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.call_count += 1
        raise ConnectionError(
            "Synthetic LLM connection failure."
        )


class RecoveringLLMClient(LLMClient):
    """
    Fail transiently on the first call, then execute the normal
    workflow responses.
    """

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = responses
        self.call_count = 0
        self.successful_response_index = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.call_count += 1

        if self.call_count == 1:
            raise ConnectionError(
                "Synthetic transient LLM failure."
            )

        if (
            self.successful_response_index
            >= len(self.responses)
        ):
            raise AssertionError(
                "Unexpected extra LLM call."
            )

        content = self.responses[
            self.successful_response_index
        ]
        self.successful_response_index += 1

        return LLMResponse(
            content=content,
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            latency_ms=10,
        )


class ProgrammingErrorLLMClient(LLMClient):
    """LLM test double that simulates a programming defect."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.call_count += 1
        raise TypeError(
            "Synthetic programming error."
        )


# ---------------------------------------------------------------------------
# Fake enrichment providers
# ---------------------------------------------------------------------------


class FakeIdentityProvider(IdentityProvider):
    def get_identity_evidence(
        self,
        application: Application,
    ) -> IdentityEvidence:
        return IdentityEvidence(
            status=EvidenceStatus.AVAILABLE,
            identity_verified=True,
            name_match=True,
            date_of_birth_match=True,
            state_match=True,
        )


class FakeMedicalProvider(MedicalHistoryProvider):
    def get_medical_evidence(
        self,
        application: Application,
    ) -> MedicalEvidence:
        return MedicalEvidence(
            status=EvidenceStatus.AVAILABLE,
            conditions=[],
        )


class FakePrescriptionProvider(PrescriptionProvider):
    def get_prescription_evidence(
        self,
        application: Application,
    ) -> PrescriptionEvidence:
        return PrescriptionEvidence(
            status=EvidenceStatus.AVAILABLE,
            prescriptions=[],
        )


class FakeFinancialProvider(FinancialProvider):
    def get_financial_evidence(
        self,
        application: Application,
    ) -> FinancialEvidence:
        return FinancialEvidence(
            status=EvidenceStatus.AVAILABLE,
            income_verified=True,
            verified_annual_income=(
                application.annual_income
            ),
        )


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def raw_application() -> dict:
    return {
        "application_id": APPLICATION_ID,
        "first_name": "Aarav",
        "last_name": "Sharma",
        "date_of_birth": "1988-04-15",
        "state": "PA",
        "annual_income": 90000,
        "coverage_amount": 500000,
        "occupation": "Software Engineer",
        "tobacco_use": "never",
        "height_cm": 175,
        "weight_kg": 75,
        "medical_conditions_declared": [],
        "medications_declared": [],
    }


def happy_path_responses() -> list[str]:
    return [
        json.dumps(
            {
                "needs_review": False,
                "summary": (
                    "The application is semantically "
                    "clear and sufficiently specific."
                ),
                "concerns": [],
            }
        ),
        json.dumps(
            {
                "evidence_consistency": "consistent",
                "summary": (
                    "External evidence is consistent "
                    "with the application."
                ),
                "material_concerns": [],
                "corroborating_evidence": [],
            }
        ),
        json.dumps(
            {
                "summary": (
                    "The authoritative deterministic "
                    "risk assessment was synthesized."
                ),
                "primary_drivers": [],
                "factor_interactions": [],
                "evidence_context": [],
                "review_context": [],
            }
        ),
        json.dumps(
            {
                "summary": (
                    "The authoritative recommendation "
                    "is supported by the supplied "
                    "risk assessment."
                ),
                "decision_basis": [],
                "precedence_explanation": None,
                "human_review_focus": [],
            }
        ),
    ]


def build_orchestrator(
    *,
    tmp_path: Path,
    llm_client: LLMClient,
    retry_policy: RetryPolicy | None = None,
    trace_store: InMemoryTraceStore | None = None,
) -> UnderwritingOrchestrator:
    repository = SQLiteCaseRepository(
        tmp_path / "underwriting.db"
    )

    return UnderwritingOrchestrator(
        repository=repository,
        llm_client=llm_client,
        identity_provider=FakeIdentityProvider(),
        medical_provider=FakeMedicalProvider(),
        prescription_provider=FakePrescriptionProvider(),
        financial_provider=FakeFinancialProvider(),
        trace_store=trace_store or InMemoryTraceStore(),
        retry_policy=retry_policy,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_full_workflow_completes_successfully(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    assert (
        result.status
        == CaseStatus.RECOMMENDATION_COMPLETED
    )

    assert result.application is not None
    assert result.enrichment is not None
    assert result.risk_assessment is not None
    assert result.recommendation is not None

    assert result.errors == []
    assert llm.call_count == 4


def test_full_workflow_persists_final_case(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    persisted = orchestrator.repository.get(
        result.metadata.case_id
    )

    assert persisted == result

    assert (
        persisted.status
        == CaseStatus.RECOMMENDATION_COMPLETED
    )


def test_successful_workflow_advances_version(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    # Initial create uses version 1.
    #
    # Successful workflow saves after:
    # 1. intake
    # 2. enrichment
    # 3. risk assessment
    # 4. recommendation
    #
    # Final version should therefore be 5.
    assert result.metadata.version == 5


# ---------------------------------------------------------------------------
# Business review routing
# ---------------------------------------------------------------------------


def test_intake_semantic_review_routes_to_human_review(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        [
            json.dumps(
                {
                    "needs_review": True,
                    "summary": (
                        "Occupation requires "
                        "clarification."
                    ),
                    "concerns": [
                        (
                            "Occupation is too vague "
                            "for automated processing."
                        )
                    ],
                }
            )
        ]
    )

    application = raw_application()
    application["occupation"] = "Other"

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )

    result = orchestrator.start_case(
        application,
        evaluation_date=EVALUATION_DATE,
    )

    assert result.status == CaseStatus.HUMAN_REVIEW
    assert result.application is not None
    assert result.enrichment is None
    assert result.risk_assessment is None
    assert result.recommendation is None

    assert any(
        "Occupation" in error
        for error in result.errors
    )

    assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Invalid intake
# ---------------------------------------------------------------------------


def test_invalid_application_becomes_processing_failed(
    tmp_path,
):
    llm = SequentialFakeLLMClient([])

    application = raw_application()
    del application["annual_income"]

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )

    result = orchestrator.start_case(
        application,
        evaluation_date=EVALUATION_DATE,
    )

    assert (
        result.status
        == CaseStatus.PROCESSING_FAILED
    )

    assert result.application is None
    assert result.errors

    # Deterministic schema validation fails before
    # the LLM is called.
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


def test_transient_failure_is_retried_and_recovers(
    tmp_path,
):
    llm = RecoveringLLMClient(
        happy_path_responses()
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    assert (
        result.status
        == CaseStatus.RECOMMENDATION_COMPLETED
    )

    # One failed call plus the four successful
    # agent LLM calls.
    assert llm.call_count == 5


def test_retry_exhaustion_becomes_processing_failed(
    tmp_path,
):
    llm = FailingLLMClient()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    assert (
        result.status
        == CaseStatus.PROCESSING_FAILED
    )

    assert llm.call_count == 3

    assert any(
        "failed after 3 attempts" in error
        for error in result.errors
    )


def test_programming_error_is_not_retried(
    tmp_path,
):
    llm = ProgrammingErrorLLMClient()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )

    with pytest.raises(
        TypeError,
        match="Synthetic programming error",
    ):
        orchestrator.start_case(
            raw_application(),
            evaluation_date=EVALUATION_DATE,
        )

    assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_successful_workflow_records_all_stage_traces(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        trace_store=trace_store,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    assert len(traces) == 4

    assert [
        trace.stage
        for trace in traces
    ] == [
        "intake",
        "enrichment",
        "risk_scoring",
        "recommendation",
    ]

    assert all(
        trace.status == TraceStatus.SUCCESS
        for trace in traces
    )

    assert all(
        trace.attempt == 1
        for trace in traces
    )


def test_successful_traces_capture_llm_metrics(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        trace_store=trace_store,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    for trace in traces:
        assert trace.model == "fake-model"
        assert trace.input_tokens == 100
        assert trace.output_tokens == 25
        assert trace.duration_ms >= 0
        assert trace.llm_latency_ms == 10
        assert trace.error_type is None
        assert trace.error_message is None


def test_workflow_traces_share_case_and_trace_ids(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        trace_store=trace_store,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    assert all(
        trace.case_id == result.metadata.case_id
        for trace in traces
    )

    assert all(
        trace.trace_id == result.metadata.trace_id
        for trace in traces
    )


def test_retry_records_failed_then_successful_attempt(
    tmp_path,
):
    llm = RecoveringLLMClient(
        happy_path_responses()
    )
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        trace_store=trace_store,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    intake_traces = [
        trace
        for trace in traces
        if trace.stage == "intake"
    ]

    assert len(intake_traces) == 2

    assert intake_traces[0].attempt == 1
    assert (
        intake_traces[0].status
        == TraceStatus.FAILED
    )
    assert (
        intake_traces[0].error_type
        == "ConnectionError"
    )

    assert intake_traces[1].attempt == 2
    assert (
        intake_traces[1].status
        == TraceStatus.SUCCESS
    )


def test_retry_exhaustion_records_every_attempt(
    tmp_path,
):
    llm = FailingLLMClient()
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        trace_store=trace_store,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    assert len(traces) == 3

    assert [
        trace.attempt
        for trace in traces
    ] == [1, 2, 3]

    assert all(
        trace.stage == "intake"
        for trace in traces
    )

    assert all(
        trace.status == TraceStatus.FAILED
        for trace in traces
    )

    assert all(
        trace.error_type == "ConnectionError"
        for trace in traces
    )


def test_trace_records_do_not_contain_application_payload(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        trace_store=trace_store,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    serialized_traces = json.dumps(
        [
            trace.model_dump(
                mode="json"
            )
            for trace in traces
        ]
    )

    assert "Aarav" not in serialized_traces
    assert "Sharma" not in serialized_traces
    assert APPLICATION_ID not in serialized_traces
    assert "90000" not in serialized_traces


# ---------------------------------------------------------------------------
# Retry policy validation
# ---------------------------------------------------------------------------


def test_retry_policy_rejects_zero_attempts():
    with pytest.raises(
        ValueError,
        match="max_attempts must be at least 1",
    ):
        RetryPolicy(
            max_attempts=0,
        )


def test_retry_policy_rejects_negative_backoff():
    with pytest.raises(
        ValueError,
        match="backoff_seconds cannot be negative",
    ):
        RetryPolicy(
            backoff_seconds=-1,
        )