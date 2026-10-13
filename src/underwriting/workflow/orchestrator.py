from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from time import perf_counter, sleep
from typing import Any, TypeVar

from underwriting.agents.enrichment import enrichment_agent
from underwriting.agents.intake import (
    IntakeStatus,
    intake_agent,
)
from underwriting.agents.recommendation import (
    recommendation_agent,
)
from underwriting.agents.risk_scoring import (
    risk_scoring_agent,
)
from underwriting.domain.application import Application
from underwriting.domain.case import (
    CaseMetadata,
    CaseState,
    CaseStatus,
)
from underwriting.enrichment.providers import (
    FinancialProvider,
    IdentityProvider,
    MedicalHistoryProvider,
    PrescriptionProvider,
)
from underwriting.llm.client import (
    LLMClient,
    LLMResponse,
)
from underwriting.observability.cost import (
    estimate_llm_cost_usd,
)
from underwriting.observability.models import (
    TraceRecord,
    TraceStatus,
)
from underwriting.observability.payloads import (
    produced_payload,
    redact_pii,
    scrub_text,
)
from underwriting.observability.trace_store import (
    TraceStore,
)
from underwriting.repositories.case_repository import (
    CaseAlreadyExistsError,
    CaseRepository,
)
from underwriting.workflow.transitions import (
    transition_case,
)

T = TypeVar("T")


class WorkflowExecutionError(Exception):
    """Raised when an underwriting workflow cannot continue."""


@dataclass(frozen=True)
class StageContract:
    """
    Data lineage for one workflow stage.

    input_sources maps each key of the stage's traced input to the
    component that produced it ("user_input" for the submission).
    state_writes lists the case-state fields a successful run fills.
    """

    name: str
    component: str
    input_sources: dict[str, str]
    state_writes: tuple[str, ...]


INTAKE_STAGE = StageContract(
    name="intake",
    component="intake_agent",
    input_sources={"raw_application": "user_input"},
    state_writes=("application",),
)

ENRICHMENT_STAGE = StageContract(
    name="enrichment",
    component="enrichment_agent",
    input_sources={"application": "intake_agent"},
    state_writes=("enrichment",),
)

RISK_SCORING_STAGE = StageContract(
    name="risk_scoring",
    component="risk_scoring_agent",
    input_sources={
        "application": "intake_agent",
        "evidence": "enrichment_agent",
    },
    state_writes=("risk_assessment",),
)

RECOMMENDATION_STAGE = StageContract(
    name="recommendation",
    component="recommendation_agent",
    input_sources={"assessment": "risk_scoring_agent"},
    state_writes=("recommendation",),
)


class _MeteredLLMClient(LLMClient):
    """
    Record every LLM call made during the current stage attempt.

    Usage is captured even when the stage later fails (for example on
    unparseable model output), so failed attempts are still costed.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls: list[LLMResponse] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        response = self.inner.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.calls.append(response)
        return response


@dataclass(frozen=True)
class RetryPolicy:
    """
    Retry configuration for transient technical failures.

    max_attempts includes the initial attempt.
    """

    max_attempts: int = 3
    backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if self.backoff_seconds < 0:
            raise ValueError(
                "backoff_seconds cannot be negative."
            )


class UnderwritingOrchestrator:
    """
    Coordinate the underwriting workflow.

    Responsibilities:
    - sequence standalone agents
    - persist case state after successful stages
    - route business-review cases
    - retry explicitly retryable technical failures
    - persist terminal technical failures
    - emit structured per-attempt execution traces

    Underwriting policy remains inside the individual deterministic
    policy components, not inside this orchestrator.
    """

    def __init__(
        self,
        *,
        repository: CaseRepository,
        llm_client: LLMClient,
        identity_provider: IdentityProvider,
        medical_provider: MedicalHistoryProvider,
        prescription_provider: PrescriptionProvider,
        financial_provider: FinancialProvider,
        trace_store: TraceStore,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self._metered_llm = _MeteredLLMClient(llm_client)
        self.identity_provider = identity_provider
        self.medical_provider = medical_provider
        self.prescription_provider = prescription_provider
        self.financial_provider = financial_provider
        self.trace_store = trace_store
        self.retry_policy = retry_policy or RetryPolicy()

    def start_case(
        self,
        raw_application: dict[str, Any],
        *,
        evaluation_date: date | None = None,
        idempotency_key: str | None = None,
    ) -> CaseState:
        """
        Create and execute a new underwriting case.

        The case is persisted immediately so failures during the first
        processing stage still have a durable case record.

        A repeated submission with the same idempotency_key returns the
        case created by the first submission instead of starting another.
        """

        if idempotency_key is not None:
            existing = self.repository.find_by_idempotency_key(
                idempotency_key
            )

            if existing is not None:
                return existing

        case = CaseState(
            metadata=CaseMetadata(
                idempotency_key=idempotency_key,
            )
        )

        try:
            self.repository.create(case)

        except CaseAlreadyExistsError:
            existing = (
                self.repository.find_by_idempotency_key(
                    idempotency_key
                )
                if idempotency_key is not None
                else None
            )

            if existing is None:
                raise

            return existing

        return self._execute(
            case=case,
            raw_application=raw_application,
            evaluation_date=evaluation_date or date.today(),
        )

    def _execute(
        self,
        *,
        case: CaseState,
        raw_application: dict[str, Any],
        evaluation_date: date,
    ) -> CaseState:
        """
        Execute the underwriting workflow from intake through
        recommendation.
        """

        try:
            intake_result = self._with_retry(
                case=case,
                stage=INTAKE_STAGE,
                received={
                    "raw_application": raw_application,
                },
                operation=lambda: intake_agent(
                    raw_application=raw_application,
                    llm_client=self._metered_llm,
                ),
                wrote_state=lambda result: (
                    result.application is not None
                ),
            )

            if intake_result.status == IntakeStatus.INVALID:
                validation_message = "; ".join(
                    intake_result.validation_errors
                )

                return self._fail_case(
                    case,
                    "Intake validation failed: "
                    f"{validation_message}",
                )

            if (
                intake_result.status
                == IntakeStatus.NEEDS_REVIEW
            ):
                if intake_result.application is not None:
                    self._attach_application(
                        case,
                        intake_result.application,
                    )

                concerns = (
                    intake_result.review.concerns
                    if intake_result.review is not None
                    else []
                )

                case.errors.extend(
                    f"Intake review: {concern}"
                    for concern in concerns
                )

                case = transition_case(
                    case,
                    CaseStatus.HUMAN_REVIEW,
                )

                case = self.repository.save(case)

                if case.application is not None and (
                    self._record_possible_duplicates(case)
                ):
                    case = self.repository.save(case)

                return case

            if intake_result.application is None:
                raise WorkflowExecutionError(
                    "Accepted intake result did not contain "
                    "a validated application."
                )

            self._attach_application(
                case,
                intake_result.application,
            )

            case = transition_case(
                case,
                CaseStatus.INTAKE_COMPLETED,
            )

            case = self.repository.save(case)

            self._record_possible_duplicates(case)

            enrichment_result = self._with_retry(
                case=case,
                stage=ENRICHMENT_STAGE,
                received={
                    "application": (
                        case.application.model_dump(
                            mode="json"
                        )
                    ),
                },
                operation=lambda: enrichment_agent(
                    application=case.application,
                    identity_provider=(
                        self.identity_provider
                    ),
                    medical_provider=(
                        self.medical_provider
                    ),
                    prescription_provider=(
                        self.prescription_provider
                    ),
                    financial_provider=(
                        self.financial_provider
                    ),
                    llm_client=self._metered_llm,
                ),
                tools_used=[
                    type(provider).__name__
                    for provider in (
                        self.identity_provider,
                        self.medical_provider,
                        self.prescription_provider,
                        self.financial_provider,
                    )
                ],
            )

            case.enrichment = (
                enrichment_result.model_dump(
                    mode="json"
                )
            )

            case = transition_case(
                case,
                CaseStatus.ENRICHMENT_COMPLETED,
            )

            case = self.repository.save(case)

            risk_result = self._with_retry(
                case=case,
                stage=RISK_SCORING_STAGE,
                received={
                    "application": (
                        case.application.model_dump(
                            mode="json"
                        )
                    ),
                    "evidence": (
                        enrichment_result.evidence.model_dump(
                            mode="json"
                        )
                    ),
                },
                operation=lambda: risk_scoring_agent(
                    application=case.application,
                    evidence=enrichment_result.evidence,
                    llm_client=self._metered_llm,
                    evaluation_date=evaluation_date,
                ),
            )

            case.risk_assessment = (
                risk_result.assessment.model_dump(
                    mode="json"
                )
            )

            case = transition_case(
                case,
                CaseStatus.RISK_ASSESSED,
            )

            case = self.repository.save(case)

            recommendation_result = self._with_retry(
                case=case,
                stage=RECOMMENDATION_STAGE,
                received={
                    "assessment": (
                        risk_result.assessment.model_dump(
                            mode="json"
                        )
                    ),
                },
                operation=lambda: recommendation_agent(
                    assessment=risk_result.assessment,
                    llm_client=self._metered_llm,
                ),
            )

            case.recommendation = (
                recommendation_result.model_dump(
                    mode="json"
                )
            )

            if (
                recommendation_result
                .recommendation
                .requires_review
                or case.possible_duplicate_of
            ):
                case = transition_case(
                    case,
                    CaseStatus.HUMAN_REVIEW,
                )

            else:
                case = transition_case(
                    case,
                    CaseStatus.RECOMMENDATION_COMPLETED,
                )

            return self.repository.save(case)

        except (
            ConnectionError,
            TimeoutError,
            WorkflowExecutionError,
        ) as exc:
            if case.status in {
                CaseStatus.RECOMMENDATION_COMPLETED,
                CaseStatus.HUMAN_REVIEW,
                CaseStatus.PROCESSING_FAILED,
            }:
                raise

            return self._fail_case(
                case,
                "Workflow execution failed: "
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _attach_application(
        case: CaseState,
        application: Application,
    ) -> None:
        case.application = application
        case.metadata.applicant_key = application.applicant_key

    def _record_possible_duplicates(
        self,
        case: CaseState,
    ) -> bool:
        """
        Flag other open cases for the same applicant.

        Runs after the case's applicant key is persisted, so two
        concurrent submissions cannot both miss each other.
        """

        other_case_ids = [
            other.metadata.case_id
            for other in self.repository.find_open_by_applicant(
                case.metadata.applicant_key
            )
            if other.metadata.case_id != case.metadata.case_id
        ]

        if not other_case_ids:
            return False

        case.possible_duplicate_of = other_case_ids
        case.errors.append(
            "Possible duplicate of open case(s): "
            + ", ".join(str(case_id) for case_id in other_case_ids)
            + "."
        )

        return True

    def _with_retry(
        self,
        *,
        case: CaseState,
        stage: StageContract,
        received: dict[str, Any],
        operation: Callable[[], T],
        tools_used: list[str] | None = None,
        wrote_state: Callable[[T], bool] = lambda _: True,
    ) -> T:
        """
        Execute one workflow stage with bounded retries and tracing.

        Every attempt produces a TraceRecord.

        Only explicitly retryable technical exceptions are retried.
        Any other stage error (invalid LLM output, provider errors,
        programming defects) is traced once and converted into a
        WorkflowExecutionError so the case is persisted as failed.
        """

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.retry_policy.max_attempts + 1,
        ):
            started_at = datetime.now(UTC)
            started_counter = perf_counter()
            self._metered_llm.calls.clear()

            record = partial(
                self._record_trace,
                case=case,
                stage=stage,
                attempt=attempt,
                started_at=started_at,
                started_counter=started_counter,
                tools_used=tools_used or [],
                received=received,
            )

            try:
                result = operation()

                record(
                    status=TraceStatus.SUCCESS,
                    produced=produced_payload(result),
                    state_writes=(
                        stage.state_writes
                        if wrote_state(result)
                        else ()
                    ),
                )

                return result

            except (
                ConnectionError,
                TimeoutError,
            ) as exc:
                last_error = exc

                record(status=TraceStatus.FAILED, error=exc)

                if (
                    attempt
                    < self.retry_policy.max_attempts
                    and self.retry_policy.backoff_seconds > 0
                ):
                    sleep(
                        self.retry_policy.backoff_seconds
                        * attempt
                    )

            except Exception as exc:
                record(status=TraceStatus.FAILED, error=exc)

                raise WorkflowExecutionError(
                    f"{stage.name} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        raise WorkflowExecutionError(
            f"{stage.name} failed after "
            f"{self.retry_policy.max_attempts} attempts: "
            f"{last_error}"
        ) from last_error

    def _record_trace(
        self,
        *,
        case: CaseState,
        stage: StageContract,
        attempt: int,
        started_at: datetime,
        started_counter: float,
        tools_used: list[str],
        received: dict[str, Any],
        status: TraceStatus,
        produced: dict[str, Any] | None = None,
        state_writes: tuple[str, ...] = (),
        error: Exception | None = None,
    ) -> None:
        """
        Record one stage attempt, with payloads scrubbed of PII.

        LLM usage covers every model call made during the attempt,
        including calls made before a failure.
        """

        duration_ms = (perf_counter() - started_counter) * 1000
        llm_calls = self._metered_llm.calls
        input_tokens = sum(call.input_tokens for call in llm_calls)
        output_tokens = sum(call.output_tokens for call in llm_calls)

        self.trace_store.record(
            TraceRecord(
                case_id=case.metadata.case_id,
                trace_id=case.metadata.trace_id,
                stage=stage.name,
                component=stage.component,
                attempt=attempt,
                status=status,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                duration_ms=duration_ms,
                llm_latency_ms=sum(
                    call.latency_ms for call in llm_calls
                ),
                model=(
                    llm_calls[-1].model
                    if llm_calls
                    else None
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimate_llm_cost_usd(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                input_sources=stage.input_sources,
                tools_used=tools_used,
                received=redact_pii(received),
                produced=redact_pii(produced or {}),
                state_writes=list(state_writes),
                error_type=(
                    type(error).__name__
                    if error is not None
                    else None
                ),
                error_message=(
                    scrub_text(str(error))
                    if error is not None
                    else None
                ),
            )
        )

    def _fail_case(
        self,
        case: CaseState,
        error_message: str,
    ) -> CaseState:
        """
        Persist a terminal technical workflow failure.
        """

        failed_case = case.model_copy(deep=True)

        failed_case.errors.append(
            error_message
        )

        failed_case = transition_case(
            failed_case,
            CaseStatus.PROCESSING_FAILED,
        )

        return self.repository.save(
            failed_case
        )