from __future__ import annotations

from ecologits.estimations import (
    PROVIDER_CONFIG_MAP,
    LLMEstimationResult,
    estimate_llm_impacts,
)

ImpactsOutput = LLMEstimationResult

__all__ = [
    "PROVIDER_CONFIG_MAP",
    "ImpactsOutput",
    "llm_impacts",
]


def llm_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: float,
    electricity_mix_zone: str | None  = None,
) -> ImpactsOutput:
    """
    High-level function to compute the impacts of an LLM generation request.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone (WOR by default).

    Returns:
        The impacts of an LLM generation request.
    """
    return estimate_llm_impacts(
        provider=provider,
        model_name=model_name,
        output_token_count=output_token_count,
        request_latency=request_latency,
        electricity_mix_zone=electricity_mix_zone,
    )
