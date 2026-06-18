from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.diffusion import compute_diffusion_impacts
from ecologits.impacts.llm import compute_llm_impacts
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Usage
from ecologits.log import logger
from ecologits.model_repository import ModalityTypes, ParametersMoE, models
from ecologits.status_messages import ErrorMessage, ModelNotRegisteredError, WarningMessage, ZoneNotRegisteredError
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

    if model.has_warnings:
        for w in model.warnings:
            logger.warning_once(str(w))
            impacts.add_warning(w)

    if if_electricity_mix.has_warnings:
        for w in if_electricity_mix.warnings:
            logger.warning_once(str(w))
            impacts.add_warning(w)

    return impacts


_EDITING_TASKS = {"img2img", "inpaint", "vid2vid"}
_IMAGE_TASKS = {"txt2img", "img2img", "inpaint"}
_VIDEO_TASKS = {"txt2vid", "vid2vid"}


def diffusion_impacts(
    provider: str,
    model_name: str,
    task: str,
    request_latency: Optional[float] = None,
    width: int = 1024,
    height: int = 1024,
    n_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    n_frames: int = 1,
    denoising_strength: float = 1.0,
    electricity_mix_zone: Optional[str] = None,
) -> ImpactsOutput:
    """
    Compute the environmental impacts of a diffusion image or video generation request.

    Args:
        provider: Name of the provider (e.g. "stabilityai").
        model_name: Name of the model (e.g. "stable-diffusion-xl").
        task: Generation task — one of txt2img, img2img, inpaint, txt2vid, vid2vid.
        request_latency: Measured wall-clock time for the request in seconds.
            Required for embodied impact computation; omit to get usage-only impacts.
        width: Output image or video width in pixels.
        height: Output image or video height in pixels.
        n_steps: Number of denoising steps (falls back to model default if None).
        guidance_scale: CFG guidance scale (falls back to model default if None).
        n_frames: Number of output video frames (1 for image generation).
        denoising_strength: Fraction of steps applied for editing tasks [0, 1].
        electricity_mix_zone: ISO 3166-1 alpha-3 electricity mix zone (defaults to provider location).
    """
    model = models.find_model(provider=provider, model_name=model_name)
    if model is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    if model.modality not in (ModalityTypes.IMAGE, ModalityTypes.VIDEO):
        error = ModelNotRegisteredError(
            message=f"Model `{model_name}` is not a diffusion model (modality={model.modality.value})."
        )
        return ImpactsOutput(errors=[error])

    diff = model.diffusion
    if diff is None:
        error = ModelNotRegisteredError(
            message=f"Model `{model_name}` has no diffusion parameters registered."
        )
        return ImpactsOutput(errors=[error])

    resolved_steps = n_steps if n_steps is not None else diff.default_steps
    resolved_guidance = guidance_scale if guidance_scale is not None else diff.default_guidance_scale
    resolved_frames = n_frames if diff.is_video else 1
    resolved_default_frames = diff.default_frames if (diff.is_video and diff.default_frames) else 1

    model_total_params = (
        model.architecture.parameters.total
        if isinstance(model.architecture.parameters, ParametersMoE)
        else model.architecture.parameters
    )

    datacenter_location = PROVIDER_CONFIG_MAP[provider].datacenter_location
    datacenter_pue = PROVIDER_CONFIG_MAP[provider].datacenter_pue
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

    impacts = compute_diffusion_impacts(
        model_total_parameter_count=model_total_params,
        flops_per_step=diff.flops_denoise_per_step,
        n_steps=resolved_steps,
        guidance_scale=resolved_guidance,
        cfg_double_pass=diff.cfg_double_pass,
        denoising_strength=denoising_strength,
        n_frames=resolved_frames,
        default_frames=resolved_default_frames,
        request_latency=request_latency,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        frame_attn_fraction=diff.frame_attn_fraction,
    )
    out = ImpactsOutput.model_validate(impacts.model_dump())

    if model.has_warnings:
        for w in model.warnings:
            logger.warning_once(str(w))
            out.add_warning(w)

    if task in _EDITING_TASKS:
        editing_warning = WarningMessage.from_code("modality-editing-unvalidated")
        logger.warning_once(str(editing_warning))
        out.add_warning(editing_warning)

    return out


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
    ),
    "stabilityai": _ProviderConfig(
        datacenter_location="WOR",
        datacenter_pue=RangeValue(min=1.09, max=1.20),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "black_forest_labs": _ProviderConfig(
        datacenter_location="WOR",
        datacenter_pue=RangeValue(min=1.09, max=1.20),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "midjourney": _ProviderConfig(
        datacenter_location="WOR",
        datacenter_pue=RangeValue(min=1.09, max=1.20),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "kling": _ProviderConfig(
        datacenter_location="WOR",
        datacenter_pue=RangeValue(min=1.09, max=1.20),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "runway": _ProviderConfig(
        datacenter_location="WOR",
        datacenter_pue=RangeValue(min=1.09, max=1.20),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    ),
    "luma": _ProviderConfig(
        datacenter_location="WOR",
        datacenter_pue=RangeValue(min=1.09, max=1.20),
        datacenter_wue=RangeValue(min=0.13, max=0.99),
    )
}
