import json
import os
from datetime import date
from pathlib import Path

import pytest

from underwriting.agents.risk_scoring import (
    risk_scoring_agent,
)
from underwriting.domain.application import Application
from underwriting.enrichment.models import EnrichmentEvidence
from underwriting.enrichment.synthetic_providers import (
    SyntheticFinancialProvider,
    SyntheticIdentityProvider,
    SyntheticMedicalHistoryProvider,
    SyntheticPrescriptionProvider,
)
from underwriting.enrichment.synthetic_store import (
    SyntheticDataStore,
)
from underwriting.llm.ollama_client import OllamaLLMClient
from underwriting.risk.models import ReviewFlag
from underwriting.risk.policy import (
    POLICY_VERSION,
    evaluate_policy,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EVALUATION_DATE = date(2026, 10, 8)


def load_application(case_name: str) -> Application:
    applications_path = DATA_DIR / "applications.json"

    with applications_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        applications = json.load(file)

    return Application.model_validate(
        applications[case_name]
    )


def build_evidence(
    application: Application,
) -> EnrichmentEvidence:
    identity_provider = SyntheticIdentityProvider(
        SyntheticDataStore(
            DATA_DIR / "identity_records.json"
        )
    )

    medical_provider = SyntheticMedicalHistoryProvider(
        SyntheticDataStore(
            DATA_DIR / "medical_records.json"
        )
    )

    prescription_provider = SyntheticPrescriptionProvider(
        SyntheticDataStore(
            DATA_DIR / "prescription_records.json"
        )
    )

    financial_provider = SyntheticFinancialProvider(
        SyntheticDataStore(
            DATA_DIR / "financial_records.json"
        )
    )

    return EnrichmentEvidence(
        identity=identity_provider.get_identity_evidence(
            application
        ),
        medical=medical_provider.get_medical_evidence(
            application
        ),
        prescription=(
            prescription_provider.get_prescription_evidence(
                application
            )
        ),
        financial=financial_provider.get_financial_evidence(
            application
        ),
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
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Ollama integration tests are disabled.",
)
@pytest.mark.parametrize(
    "case_name",
    [
        "case_01",
        "case_02",
        "case_03",
    ],
)
def test_risk_scoring_agent_with_real_data_and_ollama(
    case_name: str,
):
    application = load_application(case_name)
    evidence = build_evidence(application)

    expected_assessment = evaluate_policy(
        application=application,
        evidence=evidence,
        evaluation_date=EVALUATION_DATE,
    )

    result = risk_scoring_agent(
        application=application,
        evidence=evidence,
        evaluation_date=EVALUATION_DATE,
        llm_client=build_llm_client(),
    )

    # Authoritative policy result must be preserved.
    assert result.assessment == expected_assessment

    assert (
        result.assessment.policy_version
        == POLICY_VERSION
    )

    # Real LLM explanation and usage metadata.
    assert result.review.summary
    assert result.llm_response.model
    assert result.llm_response.input_tokens > 0
    assert result.llm_response.output_tokens > 0
    assert result.llm_response.latency_ms > 0

    flags = set(
        result.assessment.review_flags
    )

    factor_codes = {
        factor.code
        for factor in result.assessment.risk_factors
    }

    if case_name == "case_01":
        assert result.assessment.requires_review is False
        assert flags == set()

    elif case_name == "case_02":
        assert "MEDICAL_HYPERTENSION" in factor_codes

        assert (
            ReviewFlag.UNDECLARED_MEDICAL_CONDITION
            not in flags
        )

        assert (
            ReviewFlag.UNDECLARED_ACTIVE_MEDICATION
            not in flags
        )

        assert result.assessment.requires_review is False

    elif case_name == "case_03":
        assert ReviewFlag.IDENTITY_CONFLICT in flags
        assert ReviewFlag.FINANCIAL_CONFLICT in flags

        assert (
            ReviewFlag.UNDECLARED_MEDICAL_CONDITION
            in flags
        )

        assert (
            ReviewFlag.UNDECLARED_ACTIVE_MEDICATION
            in flags
        )

        assert result.assessment.requires_review is True