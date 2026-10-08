from datetime import date

import pytest

from underwriting.agents.intake import (
    IntakeStatus,
    intake_agent,
)
from underwriting.llm.client import (
    LLMClient,
    LLMResponse,
)


class FakeLLMClient(LLMClient):
    """Deterministic LLM test double for intake-agent tests."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.calls += 1

        return LLMResponse(
            content=self.content,
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            latency_ms=50,
        )


def valid_raw_application():
    return {
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


def test_valid_application_is_accepted():
    llm = FakeLLMClient(
        """
        {
          "needs_review": false,
          "summary": "Application appears internally consistent.",
          "concerns": []
        }
        """
    )

    result = intake_agent(
        valid_raw_application(),
        llm,
    )

    assert result.status == IntakeStatus.ACCEPTED
    assert result.application is not None
    assert result.review is not None
    assert result.review.needs_review is False
    assert result.validation_errors == []
    assert result.llm_response is not None
    assert llm.calls == 1


def test_llm_can_flag_application_for_review():
    llm = FakeLLMClient(
        """
        {
          "needs_review": true,
          "summary": "Clarification is recommended.",
          "concerns": [
            "Declared information requires clarification."
          ]
        }
        """
    )

    result = intake_agent(
        valid_raw_application(),
        llm,
    )

    assert result.status == IntakeStatus.NEEDS_REVIEW
    assert result.application is not None
    assert result.review is not None
    assert result.review.needs_review is True
    assert len(result.review.concerns) == 1
    assert llm.calls == 1


def test_invalid_application_does_not_call_llm():
    raw_application = valid_raw_application()
    raw_application["annual_income"] = -50000

    llm = FakeLLMClient(
        """
        {
          "needs_review": false,
          "summary": "Should never be used.",
          "concerns": []
        }
        """
    )

    result = intake_agent(
        raw_application,
        llm,
    )

    assert result.status == IntakeStatus.INVALID
    assert result.application is None
    assert result.review is None
    assert result.llm_response is None
    assert result.validation_errors
    assert llm.calls == 0


def test_malformed_llm_response_fails_safely():
    llm = FakeLLMClient(
        "This is not valid JSON."
    )

    with pytest.raises(
        ValueError,
        match="LLM returned an invalid intake review",
    ):
        intake_agent(
            valid_raw_application(),
            llm,
        )

    assert llm.calls == 1