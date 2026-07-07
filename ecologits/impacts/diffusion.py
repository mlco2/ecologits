import math
from typing import Any, Optional, Union, cast

from ecologits.impacts.dag import DAG
from ecologits.impacts.llm import (
    GPU_EMBODIED_IMPACT_ADPE,
    GPU_EMBODIED_IMPACT_GWP,
    GPU_EMBODIED_IMPACT_PE,
    GPU_MEMORY,
    HARDWARE_LIFESPAN,
    MODEL_QUANTIZATION_BITS,
    SERVER_EMBODIED_IMPACT_ADPE,
    SERVER_EMBODIED_IMPACT_GWP,
    SERVER_EMBODIED_IMPACT_PE,
    SERVER_GPUS,
    SERVER_POWER,
)
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Impacts, Usage
from ecologits.utils.range_value import RangeValue, ValueOrRange

# Lower bound: AI Energy Score U-Net calibration (SD 2.1, SDXL on A100), extended downward
# to cover LTX-Video on single H200 (implied η_mean 2.29e-18, η_min 1.72e-18).
# Upper bound: extended to cover WAN2.1-14B on DGX H200 (implied η_mean 5.01e-18).
# DiT cross-check uses modelled power from Jegham et al. (arXiv:2505.09598); treat as
# directional sanity range, not ground truth. See image_generation.md calibration section.
GPU_EFFICIENCY_KWH_PER_FLOP = RangeValue(min=1.7e-18, max=5.5e-18)

# Text-encode (CLIP / T5) + VAE-decode each run once and together contribute
# 5–15% of total pipeline FLOPs (larger for T5-XXL models such as Flux).
# This overhead is explicit in the formula rather than absorbed into η.
PIPELINE_OVERHEAD_FACTOR = 1.1

# Diffusion inference is VRAM-constrained at high resolution; typical batch size 1–4.
BATCH_SIZE = 1

dag = DAG()


@dag.asset
def effective_steps(
    n_steps: int,
    guidance_scale: float,
    cfg_double_pass: bool,
    denoising_strength: float,
) -> float:
    """
    Compute effective number of denoiser forward passes.

    Classic U-Net CFG runs two forward passes per step (conditional + unconditional).
    Flow-matching (SD3, Flux) and distilled models use a single pass regardless of guidance.
    `denoising_strength` ∈ [0, 1] scales steps for img2img / vid2vid editing tasks.
    """
    cfg_multiplier = 2.0 if (cfg_double_pass and guidance_scale > 1.0) else 1.0
    return n_steps * cfg_multiplier * denoising_strength


@dag.asset
def denoise_flops(
    flops_per_step: ValueOrRange,
    effective_steps: float,
    n_frames: int,
    default_frames: int,
    pipeline_overhead_factor: float,
    frame_attn_fraction: float,
) -> ValueOrRange:
    """
    Compute total pipeline FLOPs for the request.

    Uses a mixed linear+quadratic frame scaling:

        frame_scale = (1 - f) * x + f * x²,   x = n_frames / default_frames

    where f = ``frame_attn_fraction`` is the fraction of per-step FLOPs coming
    from full-sequence 3D attention (quadratic in frame count). MLP/FFN layers
    contribute the linear term. f=0 for image models and sparse-attention video
    models (LTX-Video); f=0.54–0.64 for full-3D-attention DiT video models
    (HunyuanVideo, WAN). See image_generation.md §"Frame attention scaling" for
    the fitting procedure and per-model calibration table.
    """
    x = n_frames / max(default_frames, 1)
    frame_scale = (1 - frame_attn_fraction) * x + frame_attn_fraction * x ** 2
    return flops_per_step * effective_steps * frame_scale * pipeline_overhead_factor


@dag.asset
def gpu_energy(
    denoise_flops: ValueOrRange,
    gpu_required_count: int,
    gpu_efficiency_kwh_per_flop: ValueOrRange,
) -> ValueOrRange:
    """
    Compute energy consumption of a single GPU during diffusion inference.

    Total compute energy (denoise_flops × η) is split evenly across all
    required GPUs so the downstream `request_it_energy` formula is consistent
    with the LLM DAG: `request_it_energy = server_energy + N_gpu × gpu_energy`.
    """
    return denoise_flops * gpu_efficiency_kwh_per_flop / gpu_required_count


@dag.asset
def generation_latency(request_latency: float) -> float:
    """
    Return the user-measured request latency as generation latency.

    No regression model for diffusion latency is available yet.
    Embodied impacts are zero when request_latency is 0.
    """
    return request_latency


@dag.asset
def model_required_memory(
    model_total_parameter_count: float,
    model_quantization_bits: int,
) -> float:
    """Compute the required memory to load the model on GPU (in GB)."""
    return 1.2 * model_total_parameter_count * model_quantization_bits / 8


@dag.asset
def gpu_required_count(
    model_required_memory: float,
    gpu_memory: float,
) -> int:
    """Compute the number of required GPUs (rounded up to next power of two)."""
    gpu_nb = math.ceil(model_required_memory / gpu_memory)
    return 2 ** math.ceil(math.log2(max(gpu_nb, 1)))


@dag.asset
def server_energy(
    generation_latency: float,
    server_power: float,
    server_gpu_count: int,
    gpu_required_count: int,
    batch_size: int,
) -> float:
    """Compute server (non-GPU) energy consumption in kWh."""
    return (generation_latency / 3600) * server_power * (gpu_required_count / server_gpu_count) * (1 / batch_size)


@dag.asset
def request_it_energy(
    server_energy: float,
    gpu_required_count: int,
    gpu_energy: ValueOrRange,
) -> ValueOrRange:
    """Compute IT energy (server + GPUs) before datacenter overhead, in kWh."""
    return server_energy + gpu_required_count * gpu_energy


@dag.asset
def request_energy(
    datacenter_pue: float,
    request_it_energy: ValueOrRange,
) -> ValueOrRange:
    """Apply PUE to obtain total datacenter energy for the request, in kWh."""
    return datacenter_pue * request_it_energy


@dag.asset
def request_usage_gwp(
    request_energy: ValueOrRange,
    if_electricity_mix_gwp: float,
) -> ValueOrRange:
    """Compute GWP usage impact in kgCO2eq."""
    return request_energy * if_electricity_mix_gwp


@dag.asset
def request_usage_adpe(
    request_energy: ValueOrRange,
    if_electricity_mix_adpe: float,
) -> ValueOrRange:
    """Compute ADPe usage impact in kgSbeq."""
    return request_energy * if_electricity_mix_adpe


@dag.asset
def request_usage_pe(
    request_energy: ValueOrRange,
    if_electricity_mix_pe: float,
) -> ValueOrRange:
    """Compute PE usage impact in MJ."""
    return request_energy * if_electricity_mix_pe


@dag.asset
def request_usage_wcf(
    request_it_energy: ValueOrRange,
    if_electricity_mix_wue: float,
    datacenter_wue: float,
    datacenter_pue: float,
) -> ValueOrRange:
    """Compute water usage impact in liters."""
    return request_it_energy * (datacenter_wue + datacenter_pue * if_electricity_mix_wue)


@dag.asset
def server_gpu_embodied_gwp(
    server_embodied_gwp: float,
    server_gpu_count: float,
    gpu_embodied_gwp: float,
    gpu_required_count: int,
) -> float:
    """Compute combined GWP embodied impact of server and GPUs in kgCO2eq."""
    return (gpu_required_count / server_gpu_count) * server_embodied_gwp + gpu_required_count * gpu_embodied_gwp


@dag.asset
def server_gpu_embodied_adpe(
    server_embodied_adpe: float,
    server_gpu_count: float,
    gpu_embodied_adpe: float,
    gpu_required_count: int,
) -> float:
    """Compute combined ADPe embodied impact of server and GPUs in kgSbeq."""
    return (gpu_required_count / server_gpu_count) * server_embodied_adpe + gpu_required_count * gpu_embodied_adpe


@dag.asset
def server_gpu_embodied_pe(
    server_embodied_pe: float,
    server_gpu_count: float,
    gpu_embodied_pe: float,
    gpu_required_count: int,
) -> float:
    """Compute combined PE embodied impact of server and GPUs in MJ."""
    return (gpu_required_count / server_gpu_count) * server_embodied_pe + gpu_required_count * gpu_embodied_pe


@dag.asset
def request_embodied_gwp(
    server_gpu_embodied_gwp: float,
    server_lifetime: float,
    generation_latency: ValueOrRange,
    batch_size: int,
) -> ValueOrRange:
    """Compute GWP embodied impact amortised over the request in kgCO2eq."""
    return generation_latency * server_gpu_embodied_gwp / (server_lifetime * batch_size)


@dag.asset
def request_embodied_adpe(
    server_gpu_embodied_adpe: float,
    server_lifetime: float,
    generation_latency: ValueOrRange,
    batch_size: int,
) -> ValueOrRange:
    """Compute ADPe embodied impact amortised over the request in kgSbeq."""
    return generation_latency * server_gpu_embodied_adpe / (server_lifetime * batch_size)


@dag.asset
def request_embodied_pe(
    server_gpu_embodied_pe: float,
    server_lifetime: float,
    generation_latency: ValueOrRange,
    batch_size: int,
) -> ValueOrRange:
    """Compute PE embodied impact amortised over the request in MJ."""
    return generation_latency * server_gpu_embodied_pe / (server_lifetime * batch_size)


def compute_diffusion_impacts_dag(
    model_total_parameter_count: ValueOrRange,
    flops_per_step: ValueOrRange,
    n_steps: int,
    guidance_scale: float,
    cfg_double_pass: bool,
    denoising_strength: float,
    n_frames: int,
    default_frames: int,
    request_latency: float,
    if_electricity_mix_adpe: float,
    if_electricity_mix_pe: float,
    if_electricity_mix_gwp: float,
    if_electricity_mix_wue: float,
    datacenter_pue: ValueOrRange,
    datacenter_wue: ValueOrRange,
    model_quantization_bits: Optional[int] = MODEL_QUANTIZATION_BITS,
    gpu_efficiency_kwh_per_flop: Optional[ValueOrRange] = GPU_EFFICIENCY_KWH_PER_FLOP,
    pipeline_overhead_factor: Optional[float] = PIPELINE_OVERHEAD_FACTOR,
    frame_attn_fraction: Optional[float] = 0.0,
    gpu_memory: Optional[float] = GPU_MEMORY,
    gpu_embodied_gwp: Optional[float] = GPU_EMBODIED_IMPACT_GWP,
    gpu_embodied_adpe: Optional[float] = GPU_EMBODIED_IMPACT_ADPE,
    gpu_embodied_pe: Optional[float] = GPU_EMBODIED_IMPACT_PE,
    server_gpu_count: Optional[int] = SERVER_GPUS,
    server_power: Optional[float] = SERVER_POWER,
    server_embodied_gwp: Optional[float] = SERVER_EMBODIED_IMPACT_GWP,
    server_embodied_adpe: Optional[float] = SERVER_EMBODIED_IMPACT_ADPE,
    server_embodied_pe: Optional[float] = SERVER_EMBODIED_IMPACT_PE,
    server_lifetime: Optional[float] = HARDWARE_LIFESPAN,
    batch_size: Optional[int] = BATCH_SIZE,
) -> dict[str, ValueOrRange]:
    """
    Compute the impacts DAG of a diffusion generation request.

    Returns all intermediate and final computed values.
    """
    return dag.execute(
        model_total_parameter_count=model_total_parameter_count,
        model_quantization_bits=model_quantization_bits,
        flops_per_step=flops_per_step,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        cfg_double_pass=cfg_double_pass,
        denoising_strength=denoising_strength,
        n_frames=n_frames,
        default_frames=default_frames,
        request_latency=request_latency,
        gpu_efficiency_kwh_per_flop=gpu_efficiency_kwh_per_flop,
        pipeline_overhead_factor=pipeline_overhead_factor,
        frame_attn_fraction=frame_attn_fraction,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_wue=datacenter_wue,
        datacenter_pue=datacenter_pue,
        gpu_memory=gpu_memory,
        gpu_embodied_gwp=gpu_embodied_gwp,
        gpu_embodied_adpe=gpu_embodied_adpe,
        gpu_embodied_pe=gpu_embodied_pe,
        server_gpu_count=server_gpu_count,
        server_power=server_power,
        server_embodied_gwp=server_embodied_gwp,
        server_embodied_adpe=server_embodied_adpe,
        server_embodied_pe=server_embodied_pe,
        server_lifetime=server_lifetime,
        batch_size=batch_size,
    )


def compute_diffusion_impacts(
    model_total_parameter_count: ValueOrRange,
    flops_per_step: ValueOrRange,
    n_steps: int,
    guidance_scale: float,
    cfg_double_pass: bool,
    denoising_strength: float,
    n_frames: int,
    default_frames: int,
    if_electricity_mix_adpe: float,
    if_electricity_mix_pe: float,
    if_electricity_mix_gwp: float,
    if_electricity_mix_wue: float,
    datacenter_pue: ValueOrRange,
    datacenter_wue: ValueOrRange,
    request_latency: Optional[float] = None,
    frame_attn_fraction: float = 0.0,
    **kwargs: Any,
) -> Impacts:
    """
    Compute the environmental impacts of a diffusion generation request.

    Supports image and video generation models. The energy model uses a
    FLOPs-based scaling law calibrated from the AI Energy Score benchmark.
    Embodied impacts require a measured `request_latency`; they are zero
    when `request_latency` is not supplied.

    Args:
        model_total_parameter_count: Total parameter count of the model (in billion).
        flops_per_step: FLOPs for one denoiser forward pass at the model's native resolution.
        n_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale value.
        cfg_double_pass: Whether the model doubles forward passes for CFG (U-Net).
        denoising_strength: Fraction of steps used for img2img / vid2vid editing [0, 1].
        n_frames: Number of video frames (1 for image generation).
        default_frames: Model-native frame count used to calibrate flops_per_step.
        if_electricity_mix_adpe: ADPe electricity mix factor in kgSbeq / kWh.
        if_electricity_mix_pe: PE electricity mix factor in MJ / kWh.
        if_electricity_mix_gwp: GWP electricity mix factor in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF electricity mix factor in L / kWh.
        datacenter_pue: Power Usage Effectiveness of the datacenter.
        datacenter_wue: Water Usage Effectiveness of the datacenter in L/kWh.
        request_latency: Measured wall-clock time for the request in seconds.
    """
    if request_latency is None:
        request_latency = 0.0  # embodied not computed without measured latency

    total_params: list[ValueOrRange] = [model_total_parameter_count]
    if isinstance(model_total_parameter_count, RangeValue):
        total_params = [model_total_parameter_count.min, model_total_parameter_count.max]

    fields = [
        "request_energy", "request_usage_gwp", "request_usage_adpe",
        "request_usage_pe", "request_usage_wcf",
        "request_embodied_gwp", "request_embodied_adpe", "request_embodied_pe",
    ]
    results: dict[str, Union[RangeValue, float, int]] = {}

    for tot_param in total_params:
        res = compute_diffusion_impacts_dag(
            model_total_parameter_count=tot_param,
            flops_per_step=flops_per_step,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            cfg_double_pass=cfg_double_pass,
            denoising_strength=denoising_strength,
            n_frames=n_frames,
            default_frames=default_frames,
            request_latency=request_latency,
            if_electricity_mix_adpe=if_electricity_mix_adpe,
            if_electricity_mix_pe=if_electricity_mix_pe,
            if_electricity_mix_gwp=if_electricity_mix_gwp,
            if_electricity_mix_wue=if_electricity_mix_wue,
            datacenter_pue=datacenter_pue,
            datacenter_wue=datacenter_wue,
            frame_attn_fraction=frame_attn_fraction,
            **kwargs,
        )
        for field in fields:
            if field in results:
                min_result = results[field]
                max_result = res[field]
                if isinstance(min_result, RangeValue):
                    min_result = cast(Union[float, int], min_result.min)
                if isinstance(max_result, RangeValue):
                    max_result = cast(Union[float, int], max_result.max)
                results[field] = RangeValue(min=min_result, max=max_result)
            else:
                results[field] = res[field]

    energy = Energy(value=results["request_energy"])
    gwp_usage = GWP(value=results["request_usage_gwp"])
    adpe_usage = ADPe(value=results["request_usage_adpe"])
    pe_usage = PE(value=results["request_usage_pe"])
    wcf_usage = WCF(value=results["request_usage_wcf"])
    gwp_embodied = GWP(value=results["request_embodied_gwp"])
    adpe_embodied = ADPe(value=results["request_embodied_adpe"])
    pe_embodied = PE(value=results["request_embodied_pe"])

    return Impacts(
        energy=energy,
        gwp=gwp_usage + gwp_embodied,
        adpe=adpe_usage + adpe_embodied,
        pe=pe_usage + pe_embodied,
        wcf=wcf_usage,
        usage=Usage(
            energy=energy,
            gwp=gwp_usage,
            adpe=adpe_usage,
            pe=pe_usage,
            wcf=wcf_usage,
        ),
        embodied=Embodied(
            gwp=gwp_embodied,
            adpe=adpe_embodied,
            pe=pe_embodied,
        ),
    )
