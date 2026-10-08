import os
from datetime import date

import pytest

from underwriting.agents.intake import (
    IntakeStatus,
    intake_agent,
)
from underwriting.llm.ollama_client import OllamaLLMClient


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Ollama integration tests are disabled.",
)
def test_intake_agent_with_real_ollama():
    client = OllamaLLMClient(
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

    raw_application = {
        "first_name": "Aarav",
        "last_name": "Sharma",
        "date_of_birth": date(1990, 5, 15),
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

    result = intake_agent(
        raw_application,
        client,
    )

    assert result.status in {
        IntakeStatus.ACCEPTED,
        IntakeStatus.NEEDS_REVIEW,
    }

    assert result.application is not None
    assert result.review is not None
    assert result.llm_response is not None

    assert result.review.summary
    assert result.llm_response.model
    assert result.llm_response.input_tokens > 0
    assert result.llm_response.output_tokens > 0
    assert result.llm_response.latency_ms > 0