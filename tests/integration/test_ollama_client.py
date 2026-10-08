import os

import pytest

from underwriting.llm.ollama_client import OllamaLLMClient


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Ollama integration tests are disabled.",
)
def test_ollama_generates_response():
    client = OllamaLLMClient(
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://host.docker.internal:11434",
        ),
        model=os.getenv(
            "OLLAMA_MODEL",
            "qwen3:8b",
        ),
    )

    response = client.generate(
        system_prompt=(
            "You are a test assistant. "
            "Answer using only the requested word."
        ),
        user_prompt="Respond with exactly: healthy",
    )

    assert response.content
    assert response.model
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0