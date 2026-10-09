import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from underwriting.llm.client import LLMClient, LLMResponse


class LLMConnectionError(ConnectionError):
    """Raised when the configured LLM provider cannot be reached."""


class LLMProviderError(Exception):
    """Raised when the LLM provider returns an invalid or failed response."""


class OllamaLLMClient(LLMClient):
    """LLM client backed by a locally running Ollama server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
        }

        request = Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started_at = time.perf_counter()

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")

        except (URLError, TimeoutError) as exc:
            raise LLMConnectionError(
                f"Unable to reach Ollama at {self.base_url}."
            ) from exc

        except HTTPError as exc:
            raise LLMProviderError(
                f"Ollama returned HTTP status {exc.code}."
            ) from exc

        latency_ms = (time.perf_counter() - started_at) * 1000

        try:
            data = json.loads(response_body)

            content = data["response"]
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise LLMProviderError(
                "Ollama returned an invalid response."
            ) from exc

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )