import json
from datetime import date

import pytest

from underwriting.agents.intake import (
    IntakeStatus,
    intake_agent,
)
from underwriting.llm.client import LLMClient, LLMResponse
from underwriting.observability.payloads import (
    _dump,
    produced_payload,
    redact_pii,
    scrub_text,
)


class FakeLLMClient(LLMClient):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "needs_review": False,
                    "summary": "Clear application.",
                    "concerns": [],
                }
            ),
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
        )


def test_produced_payload_includes_intake_status():
    result = intake_agent(
        raw_application={
            "ssn": "901-00-0001",
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
        },
        llm_client=FakeLLMClient(),
    )

    payload = produced_payload(result)

    assert result.status == IntakeStatus.ACCEPTED
    assert payload["status"] == "accepted"
    assert payload["application"]["first_name"] == "Aarav"


def test_dump_serializes_nested_lists_and_dicts():
    response = LLMResponse(
        content="Reviewed.",
        model="fake-model",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )

    dumped = _dump(
        {
            "responses": [response],
            "count": 1,
        }
    )

    assert dumped == {
        "responses": [response.model_dump(mode="json")],
        "count": 1,
    }


def test_redact_pii_masks_identifiers_at_any_depth():
    payload = {
        "raw_application": {
            "first_name": "Aarav",
            "last_name": "Sharma",
            "date_of_birth": "1988-04-15",
            "annual_income": 90000,
        },
        "history": [{"first_name": "Aarav"}],
    }

    redacted = redact_pii(payload)

    assert redacted == {
        "raw_application": {
            "first_name": "[REDACTED]",
            "last_name": "[REDACTED]",
            "date_of_birth": "1988-XX-XX",
            "annual_income": 90000,
        },
        "history": [{"first_name": "[REDACTED]"}],
    }
    assert payload["raw_application"]["first_name"] == "Aarav"


@pytest.mark.parametrize(
    ("date_of_birth", "expected"),
    [
        (date(1988, 4, 15), "1988-XX-XX"),
        ("1988-04-15", "1988-XX-XX"),
        ("15/04/1988", "[REDACTED]"),
        (19880415, "[REDACTED]"),
        (None, "[REDACTED]"),
    ],
)
def test_date_of_birth_keeps_only_the_year(
    date_of_birth,
    expected,
):
    assert redact_pii({"date_of_birth": date_of_birth}) == {
        "date_of_birth": expected,
    }


@pytest.mark.parametrize(
    "text",
    [
        "applicant 901-00-0001 not found",
        "applicant 901000001 not found",
        "applicant 901 00 0001 not found",
    ],
)
def test_scrub_text_masks_ssn_shaped_values(text):
    assert scrub_text(text) == "applicant [REDACTED-SSN] not found"


@pytest.mark.parametrize(
    "text",
    [
        "11111111-1111-4111-8111-111111111111",
        "2026-10-13T09:00:00+00:00",
        "call 555-123-4567",
        "income 90000",
    ],
)
def test_scrub_text_leaves_non_ssn_values_alone(text):
    assert scrub_text(text) == text


def test_redact_pii_scrubs_ssn_from_unexpected_fields():
    assert redact_pii(
        {"notes": ["ssn on file: 901-00-0001"], "count": 3}
    ) == {
        "notes": ["ssn on file: [REDACTED-SSN]"],
        "count": 3,
    }


def test_unsupported_result_type_is_rejected():
    with pytest.raises(
        TypeError,
        match="Unsupported agent result",
    ):
        produced_payload({"status": "accepted"})
