import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from underwriting.enrichment.synthetic_providers import (
    SyntheticFinancialProvider,
    SyntheticIdentityProvider,
    SyntheticMedicalHistoryProvider,
    SyntheticPrescriptionProvider,
)
from underwriting.enrichment.synthetic_store import SyntheticDataStore
from underwriting.llm.ollama_client import OllamaLLMClient
from underwriting.observability.models import TraceRecord
from underwriting.observability.sqlite_trace_store import (
    SQLiteTraceStore,
)
from underwriting.repositories.sqlite_case_repository import (
    SQLiteCaseRepository,
)
from underwriting.workflow.orchestrator import (
    RetryPolicy,
    UnderwritingOrchestrator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATIONS_PATH = (
    PROJECT_ROOT / "data" / "applications.json"
)

IDENTITY_PATH = (
    PROJECT_ROOT / "data" / "identity_records.json"
)

MEDICAL_PATH = (
    PROJECT_ROOT / "data" / "medical_records.json"
)

PRESCRIPTION_PATH = (
    PROJECT_ROOT / "data" / "prescription_records.json"
)

FINANCIAL_PATH = (
    PROJECT_ROOT / "data" / "financial_records.json"
)

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT / "artifacts" / "underwriting.db"
)

DEFAULT_CASES = (
    "case_01",
    "case_02",
    "case_03",
)

EVALUATION_DATE = date(2026, 10, 8)


def load_applications() -> dict[str, dict[str, Any]]:
    with APPLICATIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
        "applications.json must contain a JSON object."
    )

    return data


def build_orchestrator(
    *,
    database_path: Path,
    ollama_url: str,
    model: str,
) -> UnderwritingOrchestrator:
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository = SQLiteCaseRepository(
        database_path
    )

    trace_store = SQLiteTraceStore(
        database_path
    )

    llm_client = OllamaLLMClient(
        base_url=ollama_url,
        model=model,
        timeout_seconds=60,
    )

    identity_store = SyntheticDataStore(
        IDENTITY_PATH
    )

    medical_store = SyntheticDataStore(
        MEDICAL_PATH
    )

    prescription_store = SyntheticDataStore(
        PRESCRIPTION_PATH
    )

    financial_store = SyntheticDataStore(
        FINANCIAL_PATH
    )

    return UnderwritingOrchestrator(
        repository=repository,
        llm_client=llm_client,
        identity_provider=SyntheticIdentityProvider(
            identity_store
        ),
        medical_provider=SyntheticMedicalHistoryProvider(
            medical_store
        ),
        prescription_provider=SyntheticPrescriptionProvider(
            prescription_store
        ),
        financial_provider=SyntheticFinancialProvider(
            financial_store
        ),
        trace_store=trace_store,
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=1,
        ),
    )


def print_case_result(
    *,
    case_name: str,
    result,
    trace_store: SQLiteTraceStore,
    show_trace: bool = False,
) -> None:
    print()
    print("=" * 72)
    print(f"CASE: {case_name}")
    print("=" * 72)

    print(
        f"case_id: {result.metadata.case_id}"
    )
    print(
        f"trace_id: {result.metadata.trace_id}"
    )
    print(
        f"status: {result.status.value}"
    )

    if result.application is not None:
        print(
            "application_id: "
            f"{result.application.application_id}"
        )

    if result.risk_assessment is not None:
        print(
            "risk_tier: "
            f"{result.risk_assessment.get('tier')}"
        )
        print(
            "risk_score: "
            f"{result.risk_assessment.get('score')}"
        )

        if result.risk_assessment.get("existing_coverage"):
            print(
                "existing_coverage: "
                f"{result.risk_assessment['existing_coverage']}"
            )

        print(
            "coverage_to_income_ratio: "
            f"{result.risk_assessment.get('coverage_to_income_ratio')}"
        )
        print(
            "requires_review: "
            f"{result.risk_assessment.get('requires_review')}"
        )

        if result.risk_assessment.get("review_flags"):
            print(
                "review_flags: "
                + ", ".join(result.risk_assessment["review_flags"])
            )

    if result.recommendation is not None:
        recommendation = result.recommendation.get(
            "recommendation",
            {},
        )

        print(
            "recommendation: "
            f"{recommendation.get('decision')}"
        )

        print(
            "recommendation_requires_review: "
            f"{recommendation.get('requires_review')}"
        )

    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"  - {error}")

    traces = trace_store.get_by_trace_id(
        result.metadata.trace_id
    )

    print()
    print("STAGES")

    for trace in traces:
        print(
            f"  {trace.stage:<16} "
            f"{trace.status.value:<7} "
            f"attempt={trace.attempt} "
            f"{trace.duration_ms / 1000:6.2f}s "
            f"tokens={trace.input_tokens}/{trace.output_tokens}"
            + (
                f"  error={trace.error_type}"
                if trace.error_type is not None
                else ""
            )
        )

    print(
        f"  {'total':<16} "
        f"{'':<7} "
        f"{'':<9} "
        f"{sum(t.duration_ms for t in traces) / 1000:6.2f}s "
        f"tokens={sum(t.input_tokens for t in traces)}/"
        f"{sum(t.output_tokens for t in traces)} "
        "equiv_cloud_cost_usd="
        f"{sum(t.estimated_cost_usd for t in traces):.6f}"
    )

    if show_trace:
        print_full_trace(traces)


def print_full_trace(traces: list[TraceRecord]) -> None:
    print()
    print("TRACE")

    for trace in traces:
        print(
            f"{trace.stage:<16} "
            f"component={trace.component or '-'} "
            f"attempt={trace.attempt} "
            f"status={trace.status.value:<7} "
            f"started_at={trace.started_at.isoformat()} "
            f"duration_ms={trace.duration_ms:.2f} "
            f"model={trace.model or '-'} "
            f"tokens="
            f"{trace.input_tokens}/"
            f"{trace.output_tokens} "
            f"equiv_cloud_cost_usd="
            f"{trace.estimated_cost_usd:.8f}"
        )
        print(
            "  input_sources="
            + json.dumps(trace.input_sources)
        )
        if trace.tools_used:
            print(
                "  tools_used="
                + ", ".join(trace.tools_used)
            )
        print(
            "  received="
            + json.dumps(trace.received, default=str)
        )
        print(
            "  produced="
            + json.dumps(trace.produced, default=str)
        )
        print(
            "  state_writes="
            + json.dumps(trace.state_writes)
        )

        if trace.error_type is not None:
            print(
                "  error="
                f"{trace.error_type}: "
                f"{trace.error_message}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic life-insurance underwriting "
            "cases end to end."
        )
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help=(
            "Application keys from applications.json. "
            "Default: case_01 case_02 case_03"
        ),
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database output path.",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Print each stage's full received/produced payloads. "
            "Traces are always saved to the database either way."
        ),
    )

    parser.add_argument(
        "--ollama-url",
        default="http://host.docker.internal:11434",
        help="Ollama server base URL.",
    )

    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Ollama model name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    applications = load_applications()

    orchestrator = build_orchestrator(
        database_path=args.database,
        ollama_url=args.ollama_url,
        model=args.model,
    )

    for case_name in args.cases:
        application = applications.get(
            case_name
        )

        if application is None:
            print(
                f"Skipping unknown case: {case_name}"
            )
            continue

        result = orchestrator.start_case(
            application,
            evaluation_date=EVALUATION_DATE,
        )

        print_case_result(
            case_name=case_name,
            result=result,
            trace_store=orchestrator.trace_store,
            show_trace=args.trace,
        )


if __name__ == "__main__":
    main()