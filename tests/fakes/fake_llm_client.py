from underwriting.llm.client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    """Deterministic LLM test double for isolated agent tests."""

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