import json
from urllib.error import HTTPError, URLError

import pytest

from underwriting.llm import ollama_client
from underwriting.llm.ollama_client import (
    LLMConnectionError,
    LLMProviderError,
    LLMServerError,
    OllamaLLMClient,
)


class FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def install_urlopen(
    monkeypatch,
    *,
    body: str | None = None,
    error: Exception | None = None,
) -> list[dict]:
    calls: list[dict] = []

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "request": request,
                "timeout": timeout,
            }
        )

        if error is not None:
            raise error

        return FakeHTTPResponse(body)

    monkeypatch.setattr(
        ollama_client,
        "urlopen",
        fake_urlopen,
    )

    return calls


def build_client() -> OllamaLLMClient:
    return OllamaLLMClient(
        base_url="http://localhost:11434/",
        model="qwen3:8b",
        timeout_seconds=5,
    )


def test_generate_posts_prompt_and_parses_response(
    monkeypatch,
):
    calls = install_urlopen(
        monkeypatch,
        body=json.dumps(
            {
                "response": "Reviewed.",
                "model": "qwen3:8b-q4",
                "prompt_eval_count": 12,
                "eval_count": 4,
            }
        ),
    )

    response = build_client().generate(
        system_prompt="System instructions",
        user_prompt="Application data",
    )

    assert response.content == "Reviewed."
    assert response.model == "qwen3:8b-q4"
    assert response.input_tokens == 12
    assert response.output_tokens == 4
    assert response.latency_ms >= 0

    request = calls[0]["request"]

    assert calls[0]["timeout"] == 5
    assert (
        request.full_url
        == "http://localhost:11434/api/generate"
    )
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {
        "model": "qwen3:8b",
        "system": "System instructions",
        "prompt": "Application data",
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }


def test_generate_defaults_missing_metadata(
    monkeypatch,
):
    install_urlopen(
        monkeypatch,
        body=json.dumps({"response": "Reviewed."}),
    )

    response = build_client().generate(
        system_prompt="System instructions",
        user_prompt="Application data",
    )

    assert response.model == "qwen3:8b"
    assert response.input_tokens == 0
    assert response.output_tokens == 0


@pytest.mark.parametrize(
    "error",
    [
        URLError("Connection refused"),
        TimeoutError("timed out"),
    ],
)
def test_unreachable_server_raises_connection_error(
    monkeypatch,
    error,
):
    install_urlopen(monkeypatch, error=error)

    with pytest.raises(
        LLMConnectionError,
        match="Unable to reach Ollama at http://localhost:11434",
    ):
        build_client().generate(
            system_prompt="System instructions",
            user_prompt="Application data",
        )


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (400, False),
        (404, False),
        (500, True),
        (503, True),
    ],
)
def test_http_error_raises_provider_error(
    monkeypatch,
    code,
    retryable,
):
    install_urlopen(
        monkeypatch,
        error=HTTPError(
            url="http://localhost:11434/api/generate",
            code=code,
            msg="HTTP error",
            hdrs=None,
            fp=None,
        ),
    )

    with pytest.raises(
        LLMProviderError,
        match=f"HTTP status {code}",
    ) as exc_info:
        build_client().generate(
            system_prompt="System instructions",
            user_prompt="Application data",
        )

    assert isinstance(exc_info.value, LLMServerError) is retryable
    assert (
        isinstance(exc_info.value, ConnectionError)
        is retryable
    )


@pytest.mark.parametrize(
    "body",
    [
        "not-json",
        json.dumps({"model": "qwen3:8b"}),
        json.dumps(["Reviewed."]),
    ],
)
def test_invalid_response_body_raises_provider_error(
    monkeypatch,
    body,
):
    install_urlopen(monkeypatch, body=body)

    with pytest.raises(
        LLMProviderError,
        match="Ollama returned an invalid response",
    ):
        build_client().generate(
            system_prompt="System instructions",
            user_prompt="Application data",
        )
