"""
Equivalent commercial LLM cost estimates for lab traces.

Local Ollama inference is $0.00. Traces still record an equivalent
cloud cost so Part 4 can compare real-time vs batch using published
per-token rates for a small chat model class (gpt-4o-mini, 2025).
"""

INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60


def estimate_llm_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate USD cost for one LLM call from token counts.
    """

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_USD_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_USD_PER_MILLION

    return round(input_cost + output_cost, 8)
