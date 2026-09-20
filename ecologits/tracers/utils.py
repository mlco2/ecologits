from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Training, Usage
from ecologits.log import logger
from ecologits.model_repository import Model, ParametersMoE, models
from ecologits.status_messages import (
    ErrorMessage,
    ModelNotRegisteredError,
    TrainingNotModeledWarning,
    WarningMessage,
    ZoneNotRegisteredError,
)
from ecologits.utils.range_value import RangeValue

TRAINING_ELECTRICITY_MIX_ZONE = "WOR"


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
        training: Impacts allocated from the model training (not included in totals)
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
    training: Training | None = None
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


def llm_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: float,
    electricity_mix_zone: str | None  = None,
    input_token_count: int | None = None,
) -> ImpactsOutput:
    """
    High-level function to compute the impacts of an LLM generation request.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone (WOR by default).
        input_token_count: Number of input tokens (optional, used to estimate the pre-fill latency).

    Returns:
        The impacts of an LLM generation request.
    """

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

    datacenter_pue = PROVIDER_CONFIG_MAP[provider].datacenter_pue
    datacenter_wue = PROVIDER_CONFIG_MAP[provider].datacenter_wue

    if electricity_mix_zone is None:
        electricity_mix_zone = PROVIDER_CONFIG_MAP[provider].datacenter_location or "WOR"
    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=electricity_mix_zone)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(message=f"Could not find electricity mix for `{electricity_mix_zone}` zone.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    training_inputs = _training_inputs(provider=provider, model=model)

    impacts = compute_llm_impacts(
        model_active_parameter_count=model_active_params,
        model_total_parameter_count=model_total_params,
        output_token_count=output_token_count,
        input_token_count=input_token_count,
        request_latency=request_latency,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        tps=model.deployment.tps if model.deployment else None,
        ttft=model.deployment.ttft if model.deployment else None,
        **training_inputs
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    warnings = model.warnings + if_electricity_mix.warnings
    if not training_inputs:
        warnings.append(TrainingNotModeledWarning())
    for w in warnings:
        logger.warning_once(str(w))
        impacts.add_warning(w)

    return impacts


def _training_inputs(provider: str, model: Model) -> dict[str, Any]:
    """
    Gather the inputs required to estimate the training impacts of a model (adapted from Impact'IA).

    The training phase requires the model release date, the provider AI compute capacity and at least one active
    model for the provider. The training location being unknown, the world electricity mix is used.

    Args:
        provider: Name of the provider.
        model: The model.

    Returns:
        The keyword arguments for `compute_llm_impacts`, empty when the training impacts cannot be estimated.
    """
    ai_compute_capacity = PROVIDER_CONFIG_MAP[provider].ai_compute_capacity
    if model.release_date is None or ai_compute_capacity is None:
        return {}
    active_model_count = models.count_active_models(provider=provider)
    training_electricity_mix = electricity_mixes.find_electricity_mix(zone=TRAINING_ELECTRICITY_MIX_ZONE)
    if active_model_count == 0 or training_electricity_mix is None:
        return {}
    return {
        "model_release_date": model.release_date,
        "provider_ai_compute_capacity": ai_compute_capacity,
        "provider_active_model_count": active_model_count,
        "if_training_electricity_mix_adpe": training_electricity_mix.adpe,
        "if_training_electricity_mix_pe": training_electricity_mix.pe,
        "if_training_electricity_mix_gwp": training_electricity_mix.gwp,
        "if_training_electricity_mix_wue": training_electricity_mix.wue,
    }


@dataclass
class _ProviderConfig:
    """
    Provider configuration.

    Attributes:
        datacenter_location: ISO 3166-1 alpha-3 code of the default data center location.
        datacenter_pue: Power Usage Effectiveness of the data centers.
        datacenter_wue: Water Usage Effectiveness of the data centers in L / kWh.
        ai_compute_capacity: AI compute capacity of the provider in kW (public announcements gathered by Impact'IA),
            used to estimate the training impacts. None when unknown.
    """
    datacenter_location: str
    datacenter_pue: float | RangeValue
    datacenter_wue: float | RangeValue
    ai_compute_capacity: float | None = None


PROVIDER_CONFIG_MAP = {
    "anthropic": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=RangeValue(min=1.09, max=1.14),
        datacenter_wue=RangeValue(min=0.13, max=0.999),
        ai_compute_capacity=1_400_000,
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
        ai_compute_capacity=9_000_000,
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
        ai_compute_capacity=200_000,
    ),
    "openai": _ProviderConfig(
        datacenter_location="USA",
        datacenter_pue=1.20,
        datacenter_wue=0.569,
        ai_compute_capacity=2_000_000,
    )
}
