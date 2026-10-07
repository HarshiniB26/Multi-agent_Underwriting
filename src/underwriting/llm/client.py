from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Provider-independent result of a single LLM invocation."""

    content: str

    model: str

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    latency_ms: float = Field(ge=0)


class LLMClient(ABC):
    """Provider-independent interface for LLM reasoning."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Execute one LLM reasoning request."""
        raise NotImplementedError