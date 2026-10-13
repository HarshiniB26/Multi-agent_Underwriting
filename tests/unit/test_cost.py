import pytest

from underwriting.observability.cost import (
    estimate_llm_cost_usd,
)


def test_estimate_matches_published_mini_rates():
    cost = estimate_llm_cost_usd(
        input_tokens=100,
        output_tokens=25,
    )

    assert cost == 0.00003


def test_estimate_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_llm_cost_usd(
            input_tokens=-1,
            output_tokens=0,
        )
