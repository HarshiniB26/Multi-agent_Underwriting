import json
import os
from pathlib import Path

import pytest

from underwriting.agents.enrichment import (
    EnrichmentStatus,
    enrichment_agent,
)
from underwriting.domain.application import Application
from underwriting.enrichment.synthetic_providers import (
    SyntheticFinancialProvider,
    SyntheticIdentityProvider,
    SyntheticMedicalHistoryProvider,
    SyntheticPrescriptionProvider,
)
from underwriting.enrichment.synthetic_store import (
    SyntheticDataStore,
)
from underwriting.llm.ollama_client import (
    OllamaLLMClient,
)

DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
)


def load_application(case_name: str) -> Application:
    """Load one applicant-submitted record from the synthetic dataset."""

    applications_path = DATA_DIR / "applications.json"

    with applications_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        applications = json.load(file)

    return Application.model_validate(
        applications[case_name]
    )


def build_identity_provider() -> SyntheticIdentityProvider:
    return SyntheticIdentityProvider(
        SyntheticDataStore(
            DATA_DIR / "identity_records.json"
        )
    )


def build_medical_provider() -> SyntheticMedicalHistoryProvider:
    return SyntheticMedicalHistoryProvider(
        SyntheticDataStore(
            DATA_DIR / "medical_records.json"
        )
    )


def build_prescription_provider() -> SyntheticPrescriptionProvider:
    return SyntheticPrescriptionProvider(
        SyntheticDataStore(
            DATA_DIR / "prescription_records.json"
        )
    )


def build_financial_provider() -> SyntheticFinancialProvider:
    return SyntheticFinancialProvider(
        SyntheticDataStore(
            DATA_DIR / "financial_records.json"
        )
    )


def build_llm_client() -> OllamaLLMClient:
    return OllamaLLMClient(
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://host.docker.internal:11434",
        ),
        model=os.getenv(
            "OLLAMA_MODEL",
            "qwen3:8b",
        ),
        timeout_seconds=120,
    )


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Ollama integration tests are disabled.",
)
@pytest.mark.parametrize(
    (
        "case_name",
        "expected_status",
        "expected_discrepancy_fragments",
    ),
    [
        (
            "case_01",
            EnrichmentStatus.COMPLETED,
            [],
        ),
        (
            "case_02",
            EnrichmentStatus.COMPLETED,
            [],
        ),
        (
            "case_03",
            EnrichmentStatus.NEEDS_REVIEW,
            [
                "identity evidence has a conflict",
                "undeclared condition: Type 2 Diabetes",
                "undeclared active medication: Metformin",
                "financial evidence has a conflict",
            ],
        ),
    ],
)
def test_enrichment_agent_with_real_providers_and_ollama(
    case_name: str,
    expected_status: EnrichmentStatus,
    expected_discrepancy_fragments: list[str],
):
    application = load_application(
        case_name
    )

    result = enrichment_agent(
        application=application,
        identity_provider=build_identity_provider(),
        medical_provider=build_medical_provider(),
        prescription_provider=build_prescription_provider(),
        financial_provider=build_financial_provider(),
        llm_client=build_llm_client(),
    )

    assert result.status == expected_status

    for fragment in expected_discrepancy_fragments:
        assert any(
            fragment.casefold()
            in discrepancy.casefold()
            for discrepancy in result.discrepancies
        )

    assert result.review.summary
    assert result.llm_response.model
    assert result.llm_response.input_tokens > 0
    assert result.llm_response.output_tokens > 0
    assert result.llm_response.latency_ms > 0