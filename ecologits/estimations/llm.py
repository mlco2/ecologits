from __future__ import annotations

import math
from dataclasses import dataclass

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts, compute_llm_impacts_dag
from ecologits.log import logger
from ecologits.model_repository import ParametersMoE, models
from ecologits.status_messages import ModelNotRegisteredError, ZoneNotRegisteredError
from ecologits.utils.range_value import RangeValue, ValueOrRange

from .modeling import LLMEstimationDetails, LLMEstimationResult


@dataclass
class ProviderConfig:
    """
    Default datacenter configuration for a provider.

    Attributes:
        datacenter_location: ISO 3166-1 alpha-3 code of the datacenter electricity mix zone.
        datacenter_pue: Power Usage Effectiveness of the datacenter.
        datacenter_wue: Water Usage Effectiveness of the datacenter.
    """
    datacenter_location: str | None
    datacenter_pue: float | RangeValue
    datacenter_wue: float | RangeValue


PROVIDER_CONFIG_MAP = {
    "anthropic": ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=RangeValue(min=1.09, max=1.14),
        datacenter_wue=RangeValue(min=0.13, max=0.999),
    ),
    "cohere": ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.09,
        datacenter_wue=0.999,
    ),
    "google_genai": ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.09,
        datacenter_wue=0.999,
    ),
    "huggingface_hub": ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=RangeValue(min=1.09, max=1.14),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "mistralai": ProviderConfig(
        datacenter_location="SWE",
        datacenter_pue=1.16,
        datacenter_wue=0.09,
    ),
    "openai": ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.20,
        datacenter_wue=0.569,
    ),
}


def estimate_llm_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: float | None = None,
    electricity_mix_zone: str | None = None,
    tps: float | None = None,
    ttft: float | None = None,
    include_details: bool = False,
) -> LLMEstimationResult:
    """
    Estimate the impacts of an LLM generation request without provider tracing.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the datacenter electricity mix zone.
        tps: Number of generated tokens per second.
        ttft: Time-to-first-token latency in seconds.
        include_details: Include intermediate methodology values in the result.

    Returns:
        The estimated impacts of an LLM generation request.
    """
    model = models.find_model(provider=provider, model_name=model_name)
    if model is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return LLMEstimationResult(errors=[error])

    if isinstance(model.architecture.parameters, ParametersMoE):
        model_total_params = model.architecture.parameters.total
        model_active_params = model.architecture.parameters.active
    else:
        model_total_params = model.architecture.parameters
        model_active_params = model.architecture.parameters

    provider_config = PROVIDER_CONFIG_MAP[provider]
    resolved_electricity_mix_zone = electricity_mix_zone or provider_config.datacenter_location or "WOR"
    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=resolved_electricity_mix_zone)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(
            message=f"Could not find electricity mix for `{resolved_electricity_mix_zone}` zone."
        )
        logger.warning_once(str(error))
        return LLMEstimationResult(errors=[error])

    resolved_tps = _resolve_optional_float(tps, model.deployment.tps if model.deployment else None)
    resolved_ttft = _resolve_optional_float(ttft, model.deployment.ttft if model.deployment else None)

    impacts = compute_llm_impacts(
        model_active_parameter_count=model_active_params,
        model_total_parameter_count=model_total_params,
        output_token_count=output_token_count,
        request_latency=request_latency,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=provider_config.datacenter_pue,
        datacenter_wue=provider_config.datacenter_wue,
        tps=resolved_tps,
        ttft=resolved_ttft,
    )
    result = LLMEstimationResult.model_validate(impacts.model_dump())

    if include_details:
        result.details = _estimate_llm_details(
            provider=provider,
            model_name=model_name,
            model_active_parameter_count=model_active_params,
            model_total_parameter_count=model_total_params,
            output_token_count=output_token_count,
            request_latency=request_latency,
            electricity_mix_zone=resolved_electricity_mix_zone,
            datacenter_location=provider_config.datacenter_location,
            datacenter_pue=provider_config.datacenter_pue,
            datacenter_wue=provider_config.datacenter_wue,
            if_electricity_mix_adpe=if_electricity_mix.adpe,
            if_electricity_mix_pe=if_electricity_mix.pe,
            if_electricity_mix_gwp=if_electricity_mix.gwp,
            if_electricity_mix_wue=if_electricity_mix.wue,
            tps=resolved_tps,
            ttft=resolved_ttft,
        )

    if model.has_warnings:
        for warning in model.warnings:
            logger.warning_once(str(warning))
            result.add_warning(warning)

    return result


def _estimate_llm_details(
    provider: str,
    model_name: str,
    model_active_parameter_count: ValueOrRange,
    model_total_parameter_count: ValueOrRange,
    output_token_count: int,
    request_latency: float | None,
    electricity_mix_zone: str,
    datacenter_location: str | None,
    datacenter_pue: ValueOrRange,
    datacenter_wue: ValueOrRange,
    if_electricity_mix_adpe: float,
    if_electricity_mix_pe: float,
    if_electricity_mix_gwp: float,
    if_electricity_mix_wue: float,
    tps: float | None,
    ttft: float | None,
) -> LLMEstimationDetails:
    dag_results = compute_llm_impacts_dag(
        model_active_parameter_count=_mean_value(model_active_parameter_count),
        model_total_parameter_count=_mean_value(model_total_parameter_count),
        output_token_count=output_token_count,
        request_latency=request_latency if request_latency is not None else math.inf,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        tps=tps,
        ttft=ttft,
    )
    return LLMEstimationDetails(
        provider=provider,
        model_name=model_name,
        model_active_parameter_count=model_active_parameter_count,
        model_total_parameter_count=model_total_parameter_count,
        output_token_count=output_token_count,
        request_latency=request_latency,
        tps=tps,
        ttft=ttft,
        electricity_mix_zone=electricity_mix_zone,
        datacenter_location=datacenter_location,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        generation_latency=dag_results["generation_latency"],
        gpu_required_count=dag_results["gpu_required_count"],
        request_energy=dag_results["request_energy"],
        request_usage_gwp=dag_results["request_usage_gwp"],
        request_usage_adpe=dag_results["request_usage_adpe"],
        request_usage_pe=dag_results["request_usage_pe"],
        request_usage_wcf=dag_results["request_usage_wcf"],
        request_embodied_gwp=dag_results["request_embodied_gwp"],
        request_embodied_adpe=dag_results["request_embodied_adpe"],
        request_embodied_pe=dag_results["request_embodied_pe"],
    )


def _mean_value(value: ValueOrRange) -> float:
    if isinstance(value, RangeValue):
        return value.mean
    return float(value)


def _resolve_optional_float(value: float | None, fallback: float | None) -> float | None:
    if value is not None:
        return value
    return fallback
