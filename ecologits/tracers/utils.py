from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts, compute_llm_impacts_measured
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Usage
from ecologits.log import logger
from ecologits.measurement_repository import Deployment, measurements
from ecologits.model_repository import ParametersMoE, models
from ecologits.status_messages import (
    ErrorMessage,
    MeasurementAmbiguousError,
    MeasurementConcurrencyPreferredWarning,
    MeasurementNotFoundError,
    MeasurementUsedWarning,
    ModelNotRegisteredError,
    WarningMessage,
    ZoneNotRegisteredError,
)
from ecologits.utils.range_value import RangeValue


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


def _resolve_electricity_mix(zone: str | None, fallback_zone: str | None = None):
    """Resolve an electricity mix, returning `(mix, None)` or `(None, error)`."""
    if zone is None:
        zone = fallback_zone
    if zone is None:
        zone = "WOR"
    mix = electricity_mixes.find_electricity_mix(zone=zone)
    if mix is None:
        error = ZoneNotRegisteredError(message=f"Could not find electricity mix for `{zone}` zone.")
        logger.warning_once(str(error))
        return None, error
    return mix, None


def _attach_warnings(impacts: ImpactsOutput, source) -> None:
    """Log and attach the warnings carried by a repository object (model, mix, ...)."""
    if source.has_warnings:
        for w in source.warnings:
            logger.warning_once(str(w))
            impacts.add_warning(w)


def llm_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: float,
    electricity_mix_zone: str | None  = None,
    deployment: Deployment | None = None,
) -> ImpactsOutput:
    """
    High-level function to compute the impacts of an LLM generation request.

    Args:
        provider: Name of the provider. `"local"` computes impacts from a measured
            energy reference (see `EcoLogits.init(measurements=..., deployment=...)`)
            instead of the parameter-count regression.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone (WOR by default).
        deployment: Which measured deployment to use, for `provider="local"`. Overrides
            the default set with `EcoLogits.init(deployment=...)` for this call only.

    Returns:
        The impacts of an LLM generation request.
    """
    if provider == "local":
        return _llm_impacts_measured(
            model_name=model_name,
            output_token_count=output_token_count,
            request_latency=request_latency,
            electricity_mix_zone=electricity_mix_zone,
            deployment=deployment,
        )

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
    datacenter_pue = PROVIDER_CONFIG_MAP[provider].datacenter_pue
    datacenter_wue = PROVIDER_CONFIG_MAP[provider].datacenter_wue

    if_electricity_mix, mix_error = _resolve_electricity_mix(electricity_mix_zone, datacenter_location)
    if if_electricity_mix is None:
        return ImpactsOutput(errors=[mix_error])

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

    _attach_warnings(impacts, model)
    _attach_warnings(impacts, if_electricity_mix)

    return impacts


def _llm_impacts_measured(
    model_name: str,
    output_token_count: int,
    request_latency: float,
    electricity_mix_zone: str | None,
    deployment: Deployment | None,
) -> ImpactsOutput:
    # Lazy import: `ecologits._ecologits` is the package entry point and may
    # not be fully initialized yet when this module is first imported.
    from ecologits._ecologits import EcoLogits

    if deployment is None:
        deployment = EcoLogits.config.deployment

    selection = measurements.select(model_name, deployment)

    if selection.measurement is None:
        if selection.is_empty:
            error = MeasurementNotFoundError(
                message=f"Could not find a measurement for `{model_name}` matching the requested deployment."
            )
        else:
            error = MeasurementAmbiguousError(
                message=(
                    f"Multiple measurements match `{model_name}` and no deployment selector "
                    f"narrows it down to one: {selection.describe_candidates()}"
                )
            )
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    measurement = selection.measurement

    if_electricity_mix, mix_error = _resolve_electricity_mix(electricity_mix_zone)
    if if_electricity_mix is None:
        return ImpactsOutput(errors=[mix_error])

    impacts = compute_llm_impacts_measured(
        it_energy_per_output_token=measurement.it_energy_per_token,
        latency_per_output_token=measurement.latency_per_token,
        output_token_count=output_token_count,
        gpu_count=measurement.gpu_count,
        concurrency=measurement.concurrency,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        # On-premise hardware the caller measured directly: the measurement's
        # own energy figure already reflects that deployment, so there is no
        # separate provider PUE/WUE to apply on top of it.
        datacenter_pue=1.0,
        datacenter_wue=0.0,
        request_latency=request_latency,
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    warning = MeasurementUsedWarning(
        message=f"Impacts are estimated from a measured reference: {measurement.summary}."
    )
    logger.warning_once(str(warning))
    impacts.add_warning(warning)

    if selection.used_concurrency_preference:
        # Attached to the output, not just logged: the substituted concurrency
        # can change the energy figure severalfold, and programmatic consumers
        # only see `impacts.warnings`.
        preference_warning = MeasurementConcurrencyPreferredWarning(
            message=f"No concurrency was requested; picked the measurement at concurrency "
                    f"{measurement.concurrency} to match EcoLogits' own modelling assumption."
        )
        logger.warning_once(str(preference_warning))
        impacts.add_warning(preference_warning)

    _attach_warnings(impacts, if_electricity_mix)

    return impacts


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
