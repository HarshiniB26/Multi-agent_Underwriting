import json
from datetime import date
from pathlib import Path

import pytest

from underwriting.agents.intake import IntakeResult, IntakeStatus
from underwriting.domain.application import Application
from underwriting.domain.case import (
    CaseMetadata,
    CaseState,
    CaseStatus,
)
from underwriting.enrichment.models import (
    EvidenceStatus,
    FinancialEvidence,
    IdentityEvidence,
    MedicalEvidence,
    PrescriptionEvidence,
)
from underwriting.enrichment.providers import (
    EnrichmentProviderTimeoutError,
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)
from underwriting.llm.client import LLMClient, LLMResponse
from underwriting.observability.models import TraceStatus
from underwriting.observability.trace_store import InMemoryTraceStore
from underwriting.repositories.case_repository import (
    CaseAlreadyExistsError,
)
from underwriting.repositories.sqlite_case_repository import (
    SQLiteCaseRepository,
)
from underwriting.workflow import orchestrator as orchestrator_module
from underwriting.workflow.orchestrator import (
    RetryPolicy,
    UnderwritingOrchestrator,
)

SSN = "901-00-0001"
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


class TimeoutOnRiskScoringLLMClient(LLMClient):
    """
    Complete intake and enrichment, then raise TimeoutError
    on every risk-scoring call.
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
        self.call_count += 1

        # Intake and enrichment succeed. Risk scoring is
        # the third LLM call and every retry of it fails.
        if self.call_count >= 3:
            raise TimeoutError(
                "Synthetic risk-scoring timeout."
            )

        content = self.responses[self.call_count - 1]

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
        "ssn": SSN,
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


def test_programming_error_is_not_retried_and_fails_case(
    tmp_path,
):
    llm = ProgrammingErrorLLMClient()
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

    assert llm.call_count == 1
    assert (
        result.status
        == CaseStatus.PROCESSING_FAILED
    )
    assert any(
        "intake failed: TypeError: Synthetic programming error"
        in error
        for error in result.errors
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    assert len(traces) == 1
    assert traces[0].status == TraceStatus.FAILED
    assert traces[0].error_type == "TypeError"


def test_failed_case_is_persisted(
    tmp_path,
):
    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(["not-json"]),
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    persisted = orchestrator.repository.get(
        result.metadata.case_id
    )

    assert persisted.status == CaseStatus.PROCESSING_FAILED
    assert persisted == result


@pytest.mark.parametrize(
    ("responses", "stage"),
    [
        (["not-json"], "intake"),
        (happy_path_responses()[:1] + ["not-json"], "enrichment"),
        (happy_path_responses()[:2] + ["not-json"], "risk_scoring"),
        (happy_path_responses()[:3] + ["not-json"], "recommendation"),
    ],
)
def test_invalid_llm_output_at_any_stage_fails_case(
    tmp_path,
    responses,
    stage,
):
    llm = SequentialFakeLLMClient(responses)
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

    assert (
        result.status
        == CaseStatus.PROCESSING_FAILED
    )
    assert llm.call_count == len(responses)
    assert any(
        f"{stage} failed: ValueError: LLM returned an invalid"
        in error
        for error in result.errors
    )

    failed_traces = [
        trace
        for trace in trace_store.get_by_trace_id(
            result.metadata.trace_id
        )
        if trace.status == TraceStatus.FAILED
    ]

    assert [trace.stage for trace in failed_traces] == [stage]
    assert failed_traces[0].error_type == "ValueError"


def test_enrichment_provider_timeout_is_retried(
    tmp_path,
):
    class FlakyIdentityProvider(FakeIdentityProvider):
        def __init__(self) -> None:
            self.call_count = 0

        def get_identity_evidence(
            self,
            application: Application,
        ) -> IdentityEvidence:
            self.call_count += 1

            if self.call_count == 1:
                raise EnrichmentProviderTimeoutError(
                    "Synthetic identity provider timeout."
                )

            return super().get_identity_evidence(application)

    identity_provider = FlakyIdentityProvider()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(
            happy_path_responses()
        ),
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )
    orchestrator.identity_provider = identity_provider

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    assert identity_provider.call_count == 2
    assert (
        result.status
        == CaseStatus.RECOMMENDATION_COMPLETED
    )


def test_missing_evaluation_date_defaults_to_today(
    tmp_path,
):
    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(
            happy_path_responses()
        ),
    )

    result = orchestrator.start_case(raw_application())

    assert (
        result.status
        == CaseStatus.RECOMMENDATION_COMPLETED
    )
    assert (
        result.risk_assessment["evaluation_date"]
        == date.today().isoformat()
    )


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
        assert trace.estimated_cost_usd == 0.00003
        assert trace.received
        assert trace.produced
        assert trace.error_type is None
        assert trace.error_message is None

    assert "raw_application" in traces[0].received
    assert traces[0].produced["status"] == "accepted"
    assert "application" in traces[1].received
    assert "evidence" in traces[1].produced
    assert "evidence" in traces[2].received
    assert "assessment" in traces[2].produced
    assert "assessment" in traces[3].received
    assert "recommendation" in traces[3].produced


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


def serialized_traces_for(
    tmp_path: Path,
    responses: list[str],
) -> str:
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(responses),
        trace_store=trace_store,
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    assert traces

    return json.dumps(
        [
            trace.model_dump(
                mode="json"
            )
            for trace in traces
        ]
    )


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param(happy_path_responses(), id="success"),
        pytest.param(["not json"], id="intake_failure"),
        pytest.param(
            happy_path_responses()[:1] + ["not json"],
            id="enrichment_failure",
        ),
        pytest.param(
            happy_path_responses()[:3] + ["not json"],
            id="recommendation_failure",
        ),
    ],
)
def test_no_applicant_pii_reaches_trace_store(
    tmp_path,
    responses,
):
    serialized_traces = serialized_traces_for(
        tmp_path,
        responses,
    )

    applicant = raw_application()

    for field in ("first_name", "last_name", "ssn"):
        assert applicant[field] not in serialized_traces

    assert str(applicant["date_of_birth"]) not in serialized_traces
    assert "901000001" not in serialized_traces
    assert "[REDACTED]" in serialized_traces


def test_trace_keeps_birth_year_for_age_debugging(
    tmp_path,
):
    serialized_traces = serialized_traces_for(
        tmp_path,
        happy_path_responses(),
    )

    assert '"date_of_birth": "1988-XX-XX"' in serialized_traces


# ---------------------------------------------------------------------------
# Retry policy validation
# ---------------------------------------------------------------------------


def test_risk_scoring_timeout_retries_then_escalates(
    tmp_path,
):
    llm = TimeoutOnRiskScoringLLMClient(
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

    assert (
        result.status
        == CaseStatus.PROCESSING_FAILED
    )
    assert result.recommendation is None
    assert any(
        "risk_scoring failed after 3 attempts"
        in error
        for error in result.errors
    )

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )
    risk_traces = [
        trace
        for trace in traces
        if trace.stage == "risk_scoring"
    ]

    assert len(risk_traces) == 3
    assert all(
        trace.status == TraceStatus.FAILED
        for trace in risk_traces
    )
    assert all(
        trace.error_type == "TimeoutError"
        for trace in risk_traces
    )
    assert all(
        trace.received.get("evidence")
        for trace in risk_traces
    )


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


def test_retry_backoff_sleeps_between_attempts(
    tmp_path,
    monkeypatch,
):
    sleeps: list[float] = []
    monkeypatch.setattr(
        orchestrator_module,
        "sleep",
        sleeps.append,
    )

    llm = FailingLLMClient()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0.5,
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

    assert sleeps == [0.5, 1.0]


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


def test_recommendation_requiring_review_routes_to_human_review(
    tmp_path,
):
    application = raw_application()
    application["annual_income"] = 40000

    llm = SequentialFakeLLMClient(
        happy_path_responses()
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )

    result = orchestrator.start_case(
        application,
        evaluation_date=EVALUATION_DATE,
    )

    assert result.status == CaseStatus.HUMAN_REVIEW
    assert result.recommendation is not None
    assert (
        result.recommendation["recommendation"]["requires_review"]
        is True
    )
    assert llm.call_count == 4


def test_identity_mismatch_is_referred_not_approved(
    tmp_path,
):
    class NameMismatchIdentityProvider(FakeIdentityProvider):
        def get_identity_evidence(
            self,
            application: Application,
        ) -> IdentityEvidence:
            return super().get_identity_evidence(
                application
            ).model_copy(update={"name_match": False})

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(
            happy_path_responses()
        ),
    )
    orchestrator.identity_provider = NameMismatchIdentityProvider()

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    assert result.status == CaseStatus.HUMAN_REVIEW
    assert (
        result.recommendation["recommendation"]["decision"]
        == "refer"
    )
    assert result.risk_assessment["review_flags"] == [
        "identity_mismatch"
    ]


def test_accepted_intake_without_application_fails_case(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        orchestrator_module,
        "intake_agent",
        lambda **kwargs: IntakeResult(
            status=IntakeStatus.ACCEPTED,
        ),
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient([]),
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    assert (
        result.status
        == CaseStatus.PROCESSING_FAILED
    )
    assert any(
        "did not contain a validated application" in error
        for error in result.errors
    )


# ---------------------------------------------------------------------------
# Idempotent submission
# ---------------------------------------------------------------------------


def test_same_idempotency_key_returns_existing_case(
    tmp_path,
):
    first_llm = SequentialFakeLLMClient(
        happy_path_responses()
    )
    first = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=first_llm,
    ).start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
        idempotency_key="submit-123",
    )

    retry_llm = SequentialFakeLLMClient([])
    retried = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=retry_llm,
    ).start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
        idempotency_key="submit-123",
    )

    assert retried == first
    assert retry_llm.call_count == 0


def test_different_idempotency_keys_start_separate_cases(
    tmp_path,
):
    first = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(
            happy_path_responses()
        ),
    ).start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
        idempotency_key="submit-1",
    )

    second = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(
            happy_path_responses()
        ),
    ).start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
        idempotency_key="submit-2",
    )

    assert first.metadata.case_id != second.metadata.case_id


class LateIdempotencyRepository(SQLiteCaseRepository):
    """Simulates another worker creating the case between lookup and insert."""

    def __init__(self, database_path: Path) -> None:
        super().__init__(database_path)
        self.lookups = 0

    def find_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> CaseState | None:
        self.lookups += 1

        if self.lookups == 1:
            return None

        return super().find_by_idempotency_key(idempotency_key)


def test_concurrent_submission_with_same_key_returns_winner(
    tmp_path,
):
    winner = CaseState(
        metadata=CaseMetadata(idempotency_key="submit-123")
    )
    SQLiteCaseRepository(tmp_path / "underwriting.db").create(winner)

    llm = SequentialFakeLLMClient([])
    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )
    orchestrator.repository = LateIdempotencyRepository(
        tmp_path / "underwriting.db"
    )

    result = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
        idempotency_key="submit-123",
    )

    assert result == winner
    assert llm.call_count == 0


def test_case_id_collision_without_idempotency_key_is_raised(
    tmp_path,
):
    class CollidingRepository(SQLiteCaseRepository):
        def create(self, case: CaseState) -> None:
            raise CaseAlreadyExistsError("Synthetic collision.")

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient([]),
    )
    orchestrator.repository = CollidingRepository(
        tmp_path / "underwriting.db"
    )

    with pytest.raises(
        CaseAlreadyExistsError,
        match="Synthetic collision",
    ):
        orchestrator.start_case(raw_application())


# ---------------------------------------------------------------------------
# Repeat applications from the same person
# ---------------------------------------------------------------------------


def needs_review_responses() -> list[str]:
    return [
        json.dumps(
            {
                "needs_review": True,
                "summary": "Occupation requires clarification.",
                "concerns": ["Occupation is too vague."],
            }
        )
    ]


def run_case(
    tmp_path: Path,
    responses: list[str],
    application: dict | None = None,
):
    return build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(responses),
    ).start_case(
        application or raw_application(),
        evaluation_date=EVALUATION_DATE,
    )


def test_same_applicant_with_open_case_is_flagged_for_review(
    tmp_path,
):
    first = run_case(tmp_path, needs_review_responses())
    assert first.status == CaseStatus.HUMAN_REVIEW

    second = run_case(tmp_path, happy_path_responses())

    assert second.status == CaseStatus.HUMAN_REVIEW
    assert second.possible_duplicate_of == [
        first.metadata.case_id
    ]
    assert any(
        f"Possible duplicate of open case(s): {first.metadata.case_id}"
        in error
        for error in second.errors
    )
    assert second.recommendation is not None
    assert (
        second.metadata.applicant_key
        == first.metadata.applicant_key
    )


def test_duplicate_flag_is_persisted_when_intake_needs_review(
    tmp_path,
):
    first = run_case(tmp_path, needs_review_responses())

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(
            needs_review_responses()
        ),
    )
    second = orchestrator.start_case(
        raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    persisted = orchestrator.repository.get(
        second.metadata.case_id
    )

    assert persisted.possible_duplicate_of == [
        first.metadata.case_id
    ]
    assert persisted == second


def application_with_coverage(coverage_amount: int) -> dict:
    application = raw_application()
    application["coverage_amount"] = coverage_amount
    return application


def test_same_applicant_after_closed_case_is_not_flagged(
    tmp_path,
):
    first = run_case(
        tmp_path,
        happy_path_responses(),
        application_with_coverage(200_000),
    )
    assert first.status == CaseStatus.RECOMMENDATION_COMPLETED

    second = run_case(
        tmp_path,
        happy_path_responses(),
        application_with_coverage(200_000),
    )

    assert second.status == CaseStatus.RECOMMENDATION_COMPLETED
    assert second.possible_duplicate_of == []


def test_existing_approved_coverage_counts_toward_income_limit(
    tmp_path,
):
    first = run_case(tmp_path, happy_path_responses())
    assert first.recommendation["recommendation"]["decision"] == "approve"

    second = run_case(tmp_path, happy_path_responses())

    assert second.status == CaseStatus.HUMAN_REVIEW
    assert second.possible_duplicate_of == []
    assert second.risk_assessment["existing_coverage"] == 500_000
    assert second.risk_assessment["coverage_to_income_ratio"] == 11.11
    assert "excessive_coverage_ratio" in (
        second.risk_assessment["review_flags"]
    )
    assert second.recommendation["recommendation"]["decision"] == "refer"


def test_existing_coverage_moves_ratio_into_a_higher_band(
    tmp_path,
):
    run_case(
        tmp_path,
        happy_path_responses(),
        application_with_coverage(200_000),
    )

    second = run_case(
        tmp_path,
        happy_path_responses(),
        application_with_coverage(300_000),
    )

    assert second.status == CaseStatus.RECOMMENDATION_COMPLETED
    assert second.risk_assessment["coverage_to_income_ratio"] == 5.56
    assert "COVERAGE_RATIO_5_TO_8" in [
        factor["code"]
        for factor in second.risk_assessment["risk_factors"]
    ]


def test_cases_that_were_not_approved_do_not_count_as_coverage(
    tmp_path,
):
    referred = run_case(tmp_path, needs_review_responses())
    failed = run_case(
        tmp_path,
        happy_path_responses()[:3] + ["not json"],
    )

    assert referred.status == CaseStatus.HUMAN_REVIEW
    assert failed.status == CaseStatus.PROCESSING_FAILED

    third = run_case(
        tmp_path,
        happy_path_responses(),
        application_with_coverage(200_000),
    )

    assert third.risk_assessment["existing_coverage"] == 0


def test_risk_trace_shows_which_cases_supplied_existing_coverage(
    tmp_path,
):
    first = run_case(
        tmp_path,
        happy_path_responses(),
        application_with_coverage(200_000),
    )

    trace_store = InMemoryTraceStore()
    second = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=SequentialFakeLLMClient(happy_path_responses()),
        trace_store=trace_store,
    ).start_case(
        application_with_coverage(200_000),
        evaluation_date=EVALUATION_DATE,
    )

    risk_trace = next(
        trace
        for trace in trace_store.get_by_trace_id(
            second.metadata.trace_id
        )
        if trace.stage == "risk_scoring"
    )

    assert risk_trace.received["existing_coverage"] == {
        "amount": 200_000,
        "case_ids": [str(first.metadata.case_id)],
    }
    assert (
        risk_trace.input_sources["existing_coverage"]
        == "case_repository"
    )


def test_different_applicant_is_not_flagged(
    tmp_path,
):
    run_case(tmp_path, needs_review_responses())

    other_applicant = raw_application()
    other_applicant["ssn"] = "902-00-0002"

    second = run_case(
        tmp_path,
        happy_path_responses(),
        other_applicant,
    )

    assert second.status == CaseStatus.RECOMMENDATION_COMPLETED
    assert second.possible_duplicate_of == []


class FailingTerminalSaveRepository(SQLiteCaseRepository):
    """Repository that loses its connection when saving a terminal case."""

    def save(self, case: CaseState) -> CaseState:
        if case.status == CaseStatus.HUMAN_REVIEW:
            raise ConnectionError(
                "Synthetic database connection failure."
            )

        return super().save(case)


def test_failure_after_terminal_transition_is_not_overwritten(
    tmp_path,
):
    llm = SequentialFakeLLMClient(
        [
            json.dumps(
                {
                    "needs_review": True,
                    "summary": "Occupation requires clarification.",
                    "concerns": ["Occupation is too vague."],
                }
            )
        ]
    )

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm,
    )
    orchestrator.repository = FailingTerminalSaveRepository(
        tmp_path / "underwriting.db"
    )

    with pytest.raises(
        ConnectionError,
        match="Synthetic database connection failure",
    ):
        orchestrator.start_case(
            raw_application(),
            evaluation_date=EVALUATION_DATE,
        )

# ---------------------------------------------------------------------------
# Trace lineage
# ---------------------------------------------------------------------------


def run_traced_case(
    tmp_path: Path,
    llm_client: LLMClient,
    application: dict | None = None,
):
    trace_store = InMemoryTraceStore()

    orchestrator = build_orchestrator(
        tmp_path=tmp_path,
        llm_client=llm_client,
        trace_store=trace_store,
    )

    result = orchestrator.start_case(
        application or raw_application(),
        evaluation_date=EVALUATION_DATE,
    )

    return result, trace_store.get_by_trace_id(
        result.metadata.trace_id
    )


def test_traces_record_which_agent_ran_and_data_lineage(
    tmp_path,
):
    _, traces = run_traced_case(
        tmp_path,
        SequentialFakeLLMClient(happy_path_responses()),
    )

    assert [
        (trace.component, trace.input_sources, trace.state_writes)
        for trace in traces
    ] == [
        (
            "intake_agent",
            {"raw_application": "user_input"},
            ["application"],
        ),
        (
            "enrichment_agent",
            {"application": "intake_agent"},
            ["enrichment"],
        ),
        (
            "risk_scoring_agent",
            {
                "application": "intake_agent",
                "evidence": "enrichment_agent",
                "existing_coverage": "case_repository",
            },
            ["risk_assessment"],
        ),
        (
            "recommendation_agent",
            {"assessment": "risk_scoring_agent"},
            ["recommendation"],
        ),
    ]

    for trace in traces:
        assert set(trace.input_sources) == set(trace.received)
        assert trace.finished_at is not None
        assert trace.finished_at >= trace.started_at


def test_enrichment_trace_lists_the_tools_it_called(
    tmp_path,
):
    _, traces = run_traced_case(
        tmp_path,
        SequentialFakeLLMClient(happy_path_responses()),
    )

    tools_by_stage = {
        trace.stage: trace.tools_used
        for trace in traces
    }

    assert tools_by_stage == {
        "intake": [],
        "enrichment": [
            "FakeIdentityProvider",
            "FakeMedicalProvider",
            "FakePrescriptionProvider",
            "FakeFinancialProvider",
        ],
        "risk_scoring": [],
        "recommendation": [],
    }


def test_invalid_intake_trace_writes_no_state(
    tmp_path,
):
    application = raw_application()
    del application["annual_income"]

    _, traces = run_traced_case(
        tmp_path,
        SequentialFakeLLMClient([]),
        application,
    )

    assert len(traces) == 1
    assert traces[0].status == TraceStatus.SUCCESS
    assert traces[0].produced["status"] == "invalid"
    assert traces[0].state_writes == []
    assert traces[0].model is None
    assert traces[0].estimated_cost_usd == 0


def test_failed_attempt_still_records_llm_usage_and_cost(
    tmp_path,
):
    _, traces = run_traced_case(
        tmp_path,
        SequentialFakeLLMClient(["not json"]),
    )

    assert len(traces) == 1

    failed = traces[0]

    assert failed.status == TraceStatus.FAILED
    assert failed.state_writes == []
    assert failed.model == "fake-model"
    assert failed.input_tokens == 100
    assert failed.output_tokens == 25
    assert failed.llm_latency_ms == 10
    assert failed.estimated_cost_usd > 0


def test_retried_attempt_is_not_billed_for_earlier_attempts(
    tmp_path,
):
    _, traces = run_traced_case(
        tmp_path,
        RecoveringLLMClient(happy_path_responses()),
    )

    first_attempt, second_attempt = traces[:2]

    assert first_attempt.input_tokens == 0
    assert first_attempt.model is None
    assert second_attempt.input_tokens == 100


class LeakyErrorLLMClient(LLMClient):
    """LLM test double whose error message echoes an SSN."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        raise ValueError(f"upstream rejected subject {SSN}")


def test_error_messages_are_scrubbed_of_ssns(
    tmp_path,
):
    _, traces = run_traced_case(
        tmp_path,
        LeakyErrorLLMClient(),
    )

    assert traces[0].error_message == (
        "upstream rejected subject [REDACTED-SSN]"
    )
