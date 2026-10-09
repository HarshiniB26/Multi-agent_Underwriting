import json
from typing import Any

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
    def __init__(
        self,
        content: str | None = None,
    ):
        self.content = content or json.dumps(
            {
                "needs_review": False,
                "summary": (
                    "Applicant-provided information "
                    "is sufficiently clear."
                ),
                "concerns": [],
            }
        )

        self.call_count = 0
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.call_count += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return LLMResponse(
            content=self.content,
            model="fake-model",
            input_tokens=100,
            output_tokens=25,
            latency_ms=10,
        )


@pytest.fixture
def valid_raw_application() -> dict[str, Any]:
    return {
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


def test_valid_clear_application_is_accepted(
    valid_raw_application,
):
    llm = FakeLLMClient()

    result = intake_agent(
        raw_application=valid_raw_application,
        llm_client=llm,
    )

    assert result.status == IntakeStatus.ACCEPTED
    assert result.application is not None
    assert result.review is not None

    assert result.review.needs_review is False
    assert result.review.concerns == []

    assert result.validation_errors == []
    assert llm.call_count == 1


def test_semantic_ambiguity_can_require_review(
    valid_raw_application,
):
    raw_application = {
        **valid_raw_application,
        "occupation": "other",
    }

    llm = FakeLLMClient(
        content=json.dumps(
            {
                "needs_review": True,
                "summary": (
                    "Occupation requires clarification."
                ),
                "concerns": [
                    (
                        "Occupation 'other' is not "
                        "sufficiently specific."
                    )
                ],
            }
        )
    )

    result = intake_agent(
        raw_application=raw_application,
        llm_client=llm,
    )

    assert (
        result.status
        == IntakeStatus.NEEDS_REVIEW
    )

    assert result.application is not None
    assert result.review is not None

    assert result.review.needs_review is True
    assert len(result.review.concerns) == 1

    assert (
        "Occupation 'other'"
        in result.review.concerns[0]
    )

    assert llm.call_count == 1


def test_invalid_application_does_not_call_llm(
    valid_raw_application,
):
    raw_application = {
        **valid_raw_application,
        "annual_income": -100,
    }

    llm = FakeLLMClient()

    result = intake_agent(
        raw_application=raw_application,
        llm_client=llm,
    )

    assert result.status == IntakeStatus.INVALID
    assert result.application is None
    assert result.review is None

    assert result.validation_errors
    assert llm.call_count == 0


def test_missing_required_field_is_invalid(
    valid_raw_application,
):
    raw_application = dict(
        valid_raw_application
    )

    del raw_application["first_name"]

    llm = FakeLLMClient()

    result = intake_agent(
        raw_application=raw_application,
        llm_client=llm,
    )

    assert result.status == IntakeStatus.INVALID
    assert result.application is None

    assert result.validation_errors
    assert llm.call_count == 0


def test_contradictory_llm_review_is_rejected(
    valid_raw_application,
):
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "needs_review": False,
                "summary": (
                    "Application is clear."
                ),
                "concerns": [
                    "Occupation requires clarification."
                ],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLM returned an invalid "
            "intake review"
        ),
    ):
        intake_agent(
            raw_application=valid_raw_application,
            llm_client=llm,
        )


def test_review_requires_at_least_one_concern(
    valid_raw_application,
):
    llm = FakeLLMClient(
        content=json.dumps(
            {
                "needs_review": True,
                "summary": (
                    "Application needs clarification."
                ),
                "concerns": [],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLM returned an invalid "
            "intake review"
        ),
    ):
        intake_agent(
            raw_application=valid_raw_application,
            llm_client=llm,
        )


def test_malformed_llm_response_is_rejected(
    valid_raw_application,
):
    llm = FakeLLMClient(
        content="This is not valid JSON."
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLM returned an invalid "
            "intake review"
        ),
    ):
        intake_agent(
            raw_application=valid_raw_application,
            llm_client=llm,
        )


def test_valid_application_is_normalized(
    valid_raw_application,
):
    raw_application = {
        **valid_raw_application,
        "first_name": "  Aarav  ",
        "last_name": "  Sharma  ",
        "state": "pa",
        "occupation": "  Software Engineer  ",
    }

    llm = FakeLLMClient()

    result = intake_agent(
        raw_application=raw_application,
        llm_client=llm,
    )

    assert result.application is not None

    assert result.application.first_name == "Aarav"
    assert result.application.last_name == "Sharma"
    assert result.application.state == "PA"

    assert (
        result.application.occupation
        == "Software Engineer"
    )


def test_validated_application_is_sent_to_llm(
    valid_raw_application,
):
    llm = FakeLLMClient()

    intake_agent(
        raw_application=valid_raw_application,
        llm_client=llm,
    )

    assert llm.user_prompt is not None

    assert '"first_name": "Aarav"' in llm.user_prompt

    assert (
        '"occupation": "Software Engineer"'
        in llm.user_prompt
    )


def test_intake_prompt_limits_llm_authority(
    valid_raw_application,
):
    llm = FakeLLMClient()

    intake_agent(
        raw_application=valid_raw_application,
        llm_client=llm,
    )

    assert llm.system_prompt is not None

    prompt = llm.system_prompt.lower()

    assert "approve or deny" in prompt
    assert "risk score" in prompt
    assert "external-data enrichment" in prompt


def test_llm_response_metadata_is_preserved(
    valid_raw_application,
):
    llm = FakeLLMClient()

    result = intake_agent(
        raw_application=valid_raw_application,
        llm_client=llm,
    )

    assert result.llm_response is not None

    assert (
        result.llm_response.model
        == "fake-model"
    )

    assert result.llm_response.input_tokens == 100
    assert result.llm_response.output_tokens == 25
    assert result.llm_response.latency_ms == 10