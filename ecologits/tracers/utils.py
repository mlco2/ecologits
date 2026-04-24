from __future__ import annotations

from typing import Any

from ecologits.estimations import (
    PROVIDER_CONFIG_MAP,
    LLMEstimationResult,
    estimate_llm_impacts,
)


class ImpactsOutput(LLMEstimationResult):
    """
    Environmental impacts of an LLM generation request.

    Attributes:
        energy: Total energy consumption.
        gwp: Total Global Warming Potential (GWP) impact.
        adpe: Total Abiotic Depletion Potential for Elements (ADPe) impact.
        pe: Total Primary Energy (PE) impact.
        wcf: Usage-only Water Consumption Footprint (WCF) impact.
        usage: Impacts for the usage phase.
        embodied: Impacts for the embodied phase.
        warnings: List of warnings.
        errors: List of errors.
        details: Intermediate estimation values.
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, LLMEstimationResult):
            return self.model_dump() == other.model_dump()
        return super().__eq__(other)

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
    estimation = estimate_llm_impacts(
        provider=provider,
        model_name=model_name,
        output_token_count=output_token_count,
        request_latency=request_latency,
        electricity_mix_zone=electricity_mix_zone,
    )
    return ImpactsOutput.model_construct(
        _fields_set=estimation.model_fields_set,
        **estimation.__dict__,
    )
