import pytest

from underwriting.llm.client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    """Deterministic LLM test double."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        return self.response


def test_llm_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMClient()


def test_fake_llm_client_returns_configured_response():
    expected = LLMResponse(
        content="Application appears internally consistent.",
        model="fake-model",
        input_tokens=100,
        output_tokens=20,
        latency_ms=50,
    )

    client = FakeLLMClient(expected)

    actual = client.generate(
        system_prompt="Review the application.",
        user_prompt="Synthetic application data.",
    )

    assert actual == expected


def test_fake_llm_client_records_calls():
    response = LLMResponse(
        content="Reviewed.",
        model="fake-model",
        input_tokens=10,
        output_tokens=5,
        latency_ms=25,
    )

    client = FakeLLMClient(response)

    client.generate(
        system_prompt="System instructions",
        user_prompt="Application data",
    )

    assert len(client.calls) == 1
    assert client.calls[0]["system_prompt"] == "System instructions"
    assert client.calls[0]["user_prompt"] == "Application data"