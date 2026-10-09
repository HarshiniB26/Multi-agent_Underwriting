from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from time import perf_counter, sleep
from typing import Any, Protocol, TypeVar

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
from underwriting.domain.case import (
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
from underwriting.observability.models import (
    TraceRecord,
    TraceStatus,
)
from underwriting.observability.trace_store import (
    TraceStore,
)
from underwriting.repositories.case_repository import (
    CaseRepository,
)
from underwriting.workflow.transitions import (
    transition_case,
)


class HasLLMResponse(Protocol):
    """
    Structural contract for agent results that may expose
    LLM invocation metadata.
    """

    llm_response: LLMResponse | None


T = TypeVar(
    "T",
    bound=HasLLMResponse,
)


class WorkflowExecutionError(Exception):
    """Raised when an underwriting workflow cannot continue."""


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
    ) -> CaseState:
        """
        Create and execute a new underwriting case.

        The case is persisted immediately so failures during the first
        processing stage still have a durable case record.
        """

        case = CaseState()

        self.repository.create(case)

        return self._execute(
            case=case,
            raw_application=raw_application,
            evaluation_date=evaluation_date,
        )

    def _execute(
        self,
        *,
        case: CaseState,
        raw_application: dict[str, Any],
        evaluation_date: date | None,
    ) -> CaseState:
        """
        Execute the underwriting workflow from intake through
        recommendation.
        """

        try:
            intake_result = self._with_retry(
                case=case,
                stage_name="intake",
                operation=lambda: intake_agent(
                    raw_application=raw_application,
                    llm_client=self.llm_client,
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
                    case.application = (
                        intake_result.application
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

                return self.repository.save(case)

            if intake_result.application is None:
                raise WorkflowExecutionError(
                    "Accepted intake result did not contain "
                    "a validated application."
                )

            case.application = intake_result.application

            case = transition_case(
                case,
                CaseStatus.INTAKE_COMPLETED,
            )

            case = self.repository.save(case)

            enrichment_result = self._with_retry(
                case=case,
                stage_name="enrichment",
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
                    llm_client=self.llm_client,
                ),
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
                stage_name="risk_scoring",
                operation=lambda: risk_scoring_agent(
                    application=case.application,
                    evidence=enrichment_result.evidence,
                    llm_client=self.llm_client,
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
                stage_name="recommendation",
                operation=lambda: recommendation_agent(
                    assessment=risk_result.assessment,
                    llm_client=self.llm_client,
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

    def _with_retry(
        self,
        *,
        case: CaseState,
        stage_name: str,
        operation: Callable[[], T],
    ) -> T:
        """
        Execute one workflow stage with bounded retries and tracing.

        Every attempt produces a TraceRecord.

        Only explicitly retryable technical exceptions are retried.
        Unexpected programming errors are not retried.
        """

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.retry_policy.max_attempts + 1,
        ):
            started_at = datetime.now(UTC)
            started_counter = perf_counter()

            try:
                result = operation()

                duration_ms = (
                    perf_counter() - started_counter
                ) * 1000

                self._record_success_trace(
                    case=case,
                    stage_name=stage_name,
                    attempt=attempt,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    llm_response=result.llm_response,
                )

                return result

            except (
                ConnectionError,
                TimeoutError,
            ) as exc:
                duration_ms = (
                    perf_counter() - started_counter
                ) * 1000

                last_error = exc

                self._record_failure_trace(
                    case=case,
                    stage_name=stage_name,
                    attempt=attempt,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    error=exc,
                )

                if (
                    attempt
                    < self.retry_policy.max_attempts
                    and self.retry_policy.backoff_seconds > 0
                ):
                    sleep(
                        self.retry_policy.backoff_seconds
                        * attempt
                    )

        raise WorkflowExecutionError(
            f"{stage_name} failed after "
            f"{self.retry_policy.max_attempts} attempts: "
            f"{last_error}"
        ) from last_error

    def _record_success_trace(
        self,
        *,
        case: CaseState,
        stage_name: str,
        attempt: int,
        started_at: datetime,
        duration_ms: float,
        llm_response: LLMResponse | None,
    ) -> None:
        """
        Record successful execution telemetry.

        Applicant/business payloads are intentionally excluded.
        """

        self.trace_store.record(
            TraceRecord(
                case_id=case.metadata.case_id,
                trace_id=case.metadata.trace_id,
                stage=stage_name,
                attempt=attempt,
                status=TraceStatus.SUCCESS,
                started_at=started_at,
                duration_ms=duration_ms,
                model=(
                    llm_response.model
                    if llm_response is not None
                    else None
                ),
                input_tokens=(
                    llm_response.input_tokens
                    if llm_response is not None
                    else 0
                ),
                output_tokens=(
                    llm_response.output_tokens
                    if llm_response is not None
                    else 0
                ),
            )
        )

    def _record_failure_trace(
        self,
        *,
        case: CaseState,
        stage_name: str,
        attempt: int,
        started_at: datetime,
        duration_ms: float,
        error: Exception,
    ) -> None:
        """
        Record failed technical execution telemetry.

        Only the exception type and message are stored.
        Applicant/business payloads are not included.
        """

        self.trace_store.record(
            TraceRecord(
                case_id=case.metadata.case_id,
                trace_id=case.metadata.trace_id,
                stage=stage_name,
                attempt=attempt,
                status=TraceStatus.FAILED,
                started_at=started_at,
                duration_ms=duration_ms,
                error_type=type(error).__name__,
                error_message=str(error),
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