"""
Capture assignment evidence: three full pipeline traces, a
timeout-then-escalate failure, and unit-test output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from underwriting.enrichment.synthetic_providers import (
    SyntheticFinancialProvider,
    SyntheticIdentityProvider,
    SyntheticMedicalHistoryProvider,
    SyntheticPrescriptionProvider,
)
from underwriting.enrichment.synthetic_store import SyntheticDataStore
from underwriting.llm.client import LLMClient, LLMResponse
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
ARTIFACTS = PROJECT_ROOT / "artifacts" / "lab_evidence"
EVALUATION_DATE = date(2026, 10, 8)
CASE_NAMES = ("case_01", "case_02", "case_03")


class SequentialFakeLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        content = self.responses[self.index]
        self.index += 1
        return LLMResponse(
            content=content,
            model="fake-model",
            input_tokens=420,
            output_tokens=90,
            latency_ms=18,
        )


class TimeoutOnRiskScoringLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.call_count += 1
        if self.call_count >= 3:
            raise TimeoutError(
                "Synthetic risk-scoring timeout."
            )
        return LLMResponse(
            content=self.responses[self.call_count - 1],
            model="fake-model",
            input_tokens=420,
            output_tokens=90,
            latency_ms=18,
        )


def happy_path_responses() -> list[str]:
    return [
        json.dumps(
            {
                "needs_review": False,
                "summary": (
                    "Applicant-provided information is clear."
                ),
                "concerns": [],
            }
        ),
        json.dumps(
            {
                "evidence_consistency": "consistent",
                "summary": (
                    "External evidence is consistent with "
                    "the application."
                ),
                "material_concerns": [],
                "corroborating_evidence": [
                    "Identity and financial records match."
                ],
            }
        ),
        json.dumps(
            {
                "summary": (
                    "Deterministic risk assessment synthesized."
                ),
                "primary_drivers": ["age", "coverage"],
                "factor_interactions": [],
                "evidence_context": [],
                "review_context": [],
            }
        ),
        json.dumps(
            {
                "summary": (
                    "Authoritative recommendation follows "
                    "the risk assessment."
                ),
                "decision_basis": ["tier", "score"],
                "precedence_explanation": None,
                "human_review_focus": [],
            }
        ),
    ]


def load_applications() -> dict:
    path = PROJECT_ROOT / "data" / "applications.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_orchestrator(
    *,
    database_path: Path,
    llm_client: LLMClient,
) -> UnderwritingOrchestrator:
    identity_store = SyntheticDataStore(
        PROJECT_ROOT / "data" / "identity_records.json"
    )
    medical_store = SyntheticDataStore(
        PROJECT_ROOT / "data" / "medical_records.json"
    )
    prescription_store = SyntheticDataStore(
        PROJECT_ROOT / "data" / "prescription_records.json"
    )
    financial_store = SyntheticDataStore(
        PROJECT_ROOT / "data" / "financial_records.json"
    )

    return UnderwritingOrchestrator(
        repository=SQLiteCaseRepository(database_path),
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
        trace_store=SQLiteTraceStore(database_path),
        retry_policy=RetryPolicy(
            max_attempts=3,
            backoff_seconds=0,
        ),
    )


def serialize_case(result, traces) -> dict:
    return {
        "case_id": str(result.metadata.case_id),
        "trace_id": str(result.metadata.trace_id),
        "status": result.status.value,
        "errors": list(result.errors),
        "recommendation": result.recommendation,
        "risk_assessment": result.risk_assessment,
        "totals": {
            "duration_ms": sum(
                trace.duration_ms for trace in traces
            ),
            "input_tokens": sum(
                trace.input_tokens for trace in traces
            ),
            "output_tokens": sum(
                trace.output_tokens for trace in traces
            ),
            "estimated_cost_usd": sum(
                trace.estimated_cost_usd for trace in traces
            ),
        },
        "traces": [
            {
                "stage": trace.stage,
                "component": trace.component,
                "attempt": trace.attempt,
                "status": trace.status.value,
                "started_at": trace.started_at.isoformat(),
                "finished_at": (
                    trace.finished_at.isoformat()
                    if trace.finished_at is not None
                    else None
                ),
                "duration_ms": trace.duration_ms,
                "llm_latency_ms": trace.llm_latency_ms,
                "model": trace.model,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "estimated_cost_usd": trace.estimated_cost_usd,
                "input_sources": trace.input_sources,
                "tools_used": trace.tools_used,
                "received": trace.received,
                "produced": trace.produced,
                "state_writes": trace.state_writes,
                "error_type": trace.error_type,
                "error_message": trace.error_message,
            }
            for trace in traces
        ],
    }


def write_text_trace(path: Path, case_name: str, payload: dict) -> None:
    lines = [
        "=" * 72,
        f"CASE: {case_name}",
        "=" * 72,
        f"case_id: {payload['case_id']}",
        f"trace_id: {payload['trace_id']}",
        f"status: {payload['status']}",
        "",
        "TRACE",
    ]

    for trace in payload["traces"]:
        lines.append(
            f"{trace['stage']:<16} "
            f"component={trace['component'] or '-'} "
            f"attempt={trace['attempt']} "
            f"status={trace['status']:<7} "
            f"started_at={trace['started_at']} "
            f"duration_ms={trace['duration_ms']:.2f} "
            f"tokens={trace['input_tokens']}/"
            f"{trace['output_tokens']} "
            f"equiv_cloud_cost_usd={trace['estimated_cost_usd']:.8f}"
        )
        lines.append(
            "  input_sources=" + json.dumps(trace["input_sources"])
        )
        if trace["tools_used"]:
            lines.append(
                "  tools_used=" + ", ".join(trace["tools_used"])
            )
        lines.append(
            "  received="
            + json.dumps(trace["received"], default=str)
        )
        lines.append(
            "  produced="
            + json.dumps(trace["produced"], default=str)
        )
        lines.append(
            "  state_writes=" + json.dumps(trace["state_writes"])
        )
        if trace["error_type"]:
            lines.append(
                f"  error={trace['error_type']}: "
                f"{trace['error_message']}"
            )

    totals = payload["totals"]
    lines.extend(
        [
            "",
            f"total_stage_duration_ms: {totals['duration_ms']:.2f}",
            f"total_input_tokens: {totals['input_tokens']}",
            f"total_output_tokens: {totals['output_tokens']}",
            (
                "total_equiv_cloud_cost_usd (local inference is free): "
                f"{totals['estimated_cost_usd']:.8f}"
            ),
        ]
    )

    if payload["errors"]:
        lines.append("errors:")
        lines.extend(f"  - {error}" for error in payload["errors"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    applications = load_applications()
    database_path = ARTIFACTS / "underwriting.db"

    if database_path.exists():
        database_path.unlink()

    summaries = []

    for case_name in CASE_NAMES:
        orchestrator = build_orchestrator(
            database_path=database_path,
            llm_client=SequentialFakeLLMClient(
                happy_path_responses()
            ),
        )
        result = orchestrator.start_case(
            applications[case_name],
            evaluation_date=EVALUATION_DATE,
        )
        traces = orchestrator.trace_store.get_by_trace_id(
            result.metadata.trace_id
        )
        payload = serialize_case(result, traces)
        summaries.append((case_name, payload))

        json_path = ARTIFACTS / f"{case_name}_trace.json"
        text_path = ARTIFACTS / f"{case_name}_trace.txt"
        json_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        write_text_trace(text_path, case_name, payload)

    failure_database_path = ARTIFACTS / "failure_underwriting.db"

    if failure_database_path.exists():
        failure_database_path.unlink()

    orchestrator = build_orchestrator(
        database_path=failure_database_path,
        llm_client=TimeoutOnRiskScoringLLMClient(
            happy_path_responses()
        ),
    )
    failure = orchestrator.start_case(
        applications["case_01"],
        evaluation_date=EVALUATION_DATE,
    )
    failure_traces = orchestrator.trace_store.get_by_trace_id(
        failure.metadata.trace_id
    )
    failure_payload = serialize_case(failure, failure_traces)
    (ARTIFACTS / "failure_trace.json").write_text(
        json.dumps(failure_payload, indent=2, default=str),
        encoding="utf-8",
    )
    write_text_trace(
        ARTIFACTS / "failure_trace.txt",
        "case_01_timeout_risk_scoring",
        failure_payload,
    )

    test_output = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov",
            "--tb=no",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (ARTIFACTS / "unit_tests.txt").write_text(
        test_output.stdout + test_output.stderr,
        encoding="utf-8",
    )

    print(f"Wrote evidence to {ARTIFACTS}")
    for case_name, payload in summaries:
        totals = payload["totals"]
        print(
            f"{case_name}: status={payload['status']} "
            f"duration_ms={totals['duration_ms']:.2f} "
            f"equiv_cloud_cost_usd={totals['estimated_cost_usd']:.8f}"
        )
    print(
        "failure: "
        f"status={failure_payload['status']} "
        f"errors={failure_payload['errors']}"
    )
    print(test_output.stdout)


if __name__ == "__main__":
    main()
