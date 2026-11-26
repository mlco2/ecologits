from __future__ import annotations

import math
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from wrapt import ObjectProxy  # type: ignore[import-untyped]

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Usage
from ecologits.log import logger
from ecologits.model_repository import ParametersMoE, models
from ecologits.status_messages import ErrorMessage, ModelNotRegisteredError, WarningMessage, ZoneNotRegisteredError
from ecologits.utils.range_value import RangeValue

RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"


def is_raw_response_request(kwargs: dict[str, Any]) -> bool:
    """
    Check if a request was made with `with_raw_response` or `with_streaming_response`.

    The OpenAI and Anthropic clients add the `X-Stainless-Raw-Response` header to these requests and return the raw
    HTTP response instead of the parsed model, so there is nothing to add impacts to.

    Args:
        kwargs: Keyword arguments of the request.

    Returns:
        True if the client returns the raw HTTP response for this request.
    """
    extra_headers = kwargs.get("extra_headers") or {}
    return RAW_RESPONSE_HEADER in extra_headers


class ImpactsStream(ObjectProxy):  # type: ignore[misc]
    """
    An SDK stream object whose iteration yields the items with impacts. Other attributes go to the SDK object.
    """
    def __init__(self, stream: Any, items: Iterator[Any]) -> None:
        super().__init__(stream)
        self._self_items = items

    def __iter__(self) -> Iterator[Any]:
        return self._self_items

    def __next__(self) -> Any:
        return next(self._self_items)

    def __enter__(self) -> ImpactsStream:
        self.__wrapped__.__enter__()
        return self


class AsyncImpactsStream(ObjectProxy):  # type: ignore[misc]
    """
    An async SDK stream object whose iteration yields the items with impacts. Other attributes go to the SDK object.
    """
    def __init__(self, stream: Any, items: AsyncIterator[Any]) -> None:
        super().__init__(stream)
        self._self_items = items

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._self_items

    async def __anext__(self) -> Any:
        return await self._self_items.__anext__()

    async def __aenter__(self) -> AsyncImpactsStream:
        await self.__wrapped__.__aenter__()
        return self

    # wrapt 1.x proxies have no __aexit__.
    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self.__wrapped__.__aexit__(*exc_info)


class ImpactsOutput(BaseModel):
    """
    Impacts output data model.

    Attributes:
        energy: Total energy consumption
        gwp: Total Global Warming Potential (GWP) impact
        adpe: Total Abiotic Depletion Potential for Elements (ADPe) impact
        pe: Total Primary Energy (PE) impact
        wcf: Usage-only Water Consumption Footprint (WCF) impact
        usage: Impacts for the usage phase
        embodied: Impacts for the embodied phase
        warnings: List of warnings
        errors: List of errors
    """
    energy: Energy | None = None
    gwp: GWP | None = None
    adpe: ADPe | None = None
    pe: PE | None = None
    wcf: WCF | None = None
    usage: Usage | None = None
    embodied: Embodied | None = None
    warnings: list[WarningMessage] | None = None
    errors: list[ErrorMessage] | None = None

    @property
    def has_warnings(self) -> bool:
        return isinstance(self.warnings, list) and len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        return isinstance(self.errors, list) and len(self.errors) > 0

    def add_warning(self, warning: WarningMessage) -> None:
        if self.warnings is None:
            self.warnings = []
        if warning.code in {w.code for w in self.warnings}:
            return
        self.warnings.append(warning)

    def add_errors(self, error: ErrorMessage) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append(error)


def _validate_datacenter_override(name: str, value: float | RangeValue | None, minimum: int) -> None:
    if value is None:
        return
    values = (value.min, value.max) if isinstance(value, RangeValue) else (value,)
    for number in values:
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < minimum
        ):
            raise ValueError(f"{name} must be finite and at least {minimum}.")


def _llm_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: float | None,
    electricity_mix_zone: str | None  = None,
    *,
    datacenter_pue: float | RangeValue | None = None,
    datacenter_wue: float | RangeValue | None = None,
) -> ImpactsOutput:
    """
    High-level function to compute the impacts of an LLM generation request.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds, or None to estimate it.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone.
            Uses the provider's default location when None, falling back to WOR if unavailable.
        datacenter_pue: Power Usage Effectiveness. Uses the provider default when None.
        datacenter_wue: Water Usage Effectiveness in L/kWh. Uses the provider default when None.

    Returns:
        The impacts of an LLM generation request.
    """

    _validate_datacenter_override("datacenter_pue", datacenter_pue, 1)
    _validate_datacenter_override("datacenter_wue", datacenter_wue, 0)
    model = models.find_model(provider=provider, model_name=model_name)
    if model is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    if isinstance(model.architecture.parameters, ParametersMoE):
        model_total_params = model.architecture.parameters.total
        model_active_params = model.architecture.parameters.active
    else:
        model_total_params = model.architecture.parameters
        model_active_params = model.architecture.parameters

    datacenter_location = PROVIDER_CONFIG_MAP[provider].datacenter_location
    if datacenter_pue is None:
        datacenter_pue = PROVIDER_CONFIG_MAP[provider].datacenter_pue
    if datacenter_wue is None:
        datacenter_wue = PROVIDER_CONFIG_MAP[provider].datacenter_wue

    if electricity_mix_zone is None:
        electricity_mix_zone = datacenter_location
    if electricity_mix_zone is None:
        electricity_mix_zone = "WOR"
    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=electricity_mix_zone)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(message=f"Could not find electricity mix for `{electricity_mix_zone}` zone.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    impacts = compute_llm_impacts(
        model_active_parameter_count=model_active_params,
        model_total_parameter_count=model_total_params,
        output_token_count=output_token_count,
        request_latency=request_latency,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        tps=model.deployment.tps if model.deployment else None,
        ttft=model.deployment.ttft if model.deployment else None,
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    for warning in model.warnings + if_electricity_mix.warnings:
        logger.warning_once(str(warning))
        impacts.add_warning(warning)

    return impacts


def llm_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: float | None,
    electricity_mix_zone: str | None = None,
    *,
    datacenter_pue: float | RangeValue | None = None,
    datacenter_wue: float | RangeValue | None = None,
) -> ImpactsOutput:
    """Compute LLM impacts using the deprecated tracer utility interface.

    Deprecated:
        Use `ecologits.estimations.text_impacts` for standalone text generation
        estimates. See the text generation tutorial for argument mappings.

    Preserves the original arguments and calculations, including measured request
    latency. The replacement estimates latency and requires total input tokens and
    a namespaced model identifier, so migration may change the resulting estimates.
    """
    warnings.warn(
        "ecologits.tracers.utils.llm_impacts is deprecated; "
        "use ecologits.estimations.text_impacts instead. "
        "See the text generation tutorial for migration details.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _llm_impacts(
        provider=provider,
        model_name=model_name,
        output_token_count=output_token_count,
        request_latency=request_latency,
        electricity_mix_zone=electricity_mix_zone,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
    )


@dataclass
class _ProviderConfig:
    datacenter_location: str
    datacenter_pue: float | RangeValue
    datacenter_wue: float | RangeValue


PROVIDER_CONFIG_MAP = {
    "anthropic": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=RangeValue(min=1.09, max=1.14),
        datacenter_wue=RangeValue(min=0.13, max=0.999),
    ),
    "cohere": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.09,
        datacenter_wue=0.999,
    ),
    "google_genai": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.09,
        datacenter_wue=0.999,
    ),
    "huggingface_hub": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=RangeValue(min=1.09, max=1.14),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "mistralai": _ProviderConfig(
        datacenter_location="SWE",
        datacenter_pue=1.16,
        datacenter_wue=0.09,
    ),
    "openai": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.20,
        datacenter_wue=0.569,
    )
}
