"""
Estimation of the training impacts of an LLM allocated to a generation request.

The methodology is adapted from the top-down approach of the Impact'IA calculator (SNCF, Wavestone, Resilio, 2026):

1. The compute required by the final training run is estimated from the model size and release date with a
   regression on Epoch AI data, then converted to energy with a compute efficiency (FLOPs per watt), the same
   infrastructure overheads as the inference phase and the data center PUE.
2. Research and development experiments (test runs, ablations, etc.) are accounted for as a multiple of the final
   training run.
3. The storage of the training dataset is estimated from the number of training tokens.
4. The training impacts are allocated to the request in proportion to the number of output tokens over the total
   number of tokens the model is expected to generate during its lifetime. The latter is estimated top-down from the
   AI compute capacity of the provider shared between its active models.

Embodied impacts of the training infrastructure are estimated by applying the embodied impact intensity of the
inference phase (embodied impacts per kWh of IT energy) to the training energy.
"""
import math
from datetime import date
from typing import Any, Optional

from ecologits.impacts.dag import DAG
from ecologits.utils.range_value import ValueOrRange

TRAINING_FLOPS_REFERENCE_DATE = date(2020, 1, 1)
TRAINING_FLOPS_ALPHA = 0.0006     # log10(FLOPs) increase per day since the reference date
TRAINING_FLOPS_BETA = 17.151      # log10(FLOPs) intercept
TRAINING_FLOPS_GAMMA = 0.541      # exponent on the number of total parameters

COMPUTE_FLOPS_PER_WATT = 1.4e12   # FLOPS / W (NVIDIA H100, dense BF16 peak, Epoch AI)
RND_TRAINING_FACTOR = 4           # R&D experiments energy as a multiple of the final training run

TRAINING_DURATION = 100 * 24      # hours (Epoch AI)
TRAINING_BYTES_PER_TOKEN = 4 * 32 / 8   # 4 characters per token encoded on 32 bits
HDD_CAPACITY = 30                 # TB (Seagate Exos 30 TB)
HDD_POWER = 0.0095                # kW
HDD_USAGE_RATIO = 0.2

INFERENCE_CAPACITY_RATIO = 0.8    # share of the provider AI compute capacity dedicated to inference
COMPUTE_EFFICIENCY_FACTOR = 0.5 * 0.5 * 0.7   # GPU utilization x memory-bound limitation x GPU sharing
MODEL_LIFETIME = 2 * 365.25 * 24 * 3600  # seconds

dag = DAG()


@dag.asset
def training_flops(
        model_total_parameter_count: float,
        model_release_days: float,
        training_flops_alpha: float,
        training_flops_beta: float,
        training_flops_gamma: float
) -> float:
    """
    Estimate the compute of the final training run of the model.

    Regression from Impact'IA on Epoch AI notable models data: at equal size, recent models are trained with more
    compute, and bigger models need more compute.

    Args:
        model_total_parameter_count: Number of parameters of the model (in billion).
        model_release_days: Number of days between the reference date (2020-01-01) and the model release date.
        training_flops_alpha: Time coefficient of the regression (in log10(FLOPs) per day).
        training_flops_beta: Intercept of the regression (in log10(FLOPs)).
        training_flops_gamma: Exponent of the number of parameters.

    Returns:
        The compute of the final training run in FLOPs.
    """
    return 10 ** (training_flops_alpha * model_release_days + training_flops_beta) \
        * (model_total_parameter_count * 1e9) ** training_flops_gamma


@dag.asset
def infrastructure_overhead_ratio(
        gpu_power: float,
        server_power: float,
        network_power: float,
        server_gpu_count: int
) -> float:
    """
    Compute the ratio between the IT power of a server (GPUs, server and network equipment) and its GPUs power.

    Used to extend the GPU energy of the training phase to the same technical scope as the inference phase.

    Args:
        gpu_power: Power consumption of a single GPU in kW.
        server_power: Power consumption of the server (without GPUs) in kW.
        network_power: Power consumption of the network equipment allocated to one server in kW.
        server_gpu_count: Number of GPUs in the server.

    Returns:
        The infrastructure overhead ratio (greater than 1).
    """
    return (server_gpu_count * gpu_power + server_power + network_power) / (server_gpu_count * gpu_power)


@dag.asset
def training_it_energy(
        training_flops: float,
        compute_flops_per_watt: float,
        infrastructure_overhead_ratio: float
) -> float:
    """
    Compute the IT energy consumption of the final training run (GPUs, servers and network equipment).

    Args:
        training_flops: Compute of the final training run in FLOPs.
        compute_flops_per_watt: Compute efficiency of the training hardware in FLOPS / W.
        infrastructure_overhead_ratio: Ratio between the IT power and the GPUs power of a server.

    Returns:
        The IT energy consumption of the final training run in kWh.
    """
    return training_flops / compute_flops_per_watt / 3600 / 1000 * infrastructure_overhead_ratio


@dag.asset
def rnd_it_energy(
        training_it_energy: float,
        rnd_training_factor: float
) -> float:
    """
    Compute the IT energy consumption of the research and development experiments preceding the final training run.

    Args:
        training_it_energy: IT energy consumption of the final training run in kWh.
        rnd_training_factor: R&D energy consumption as a multiple of the final training run.

    Returns:
        The IT energy consumption of the R&D experiments in kWh.
    """
    return training_it_energy * rnd_training_factor


@dag.asset
def training_token_count(
        training_flops: float,
        model_total_parameter_count: float
) -> float:
    """
    Estimate the number of tokens used to train the model (Kaplan et al., 2020: FLOPs = 6 x parameters x tokens).

    Args:
        training_flops: Compute of the final training run in FLOPs.
        model_total_parameter_count: Number of parameters of the model (in billion).

    Returns:
        The number of training tokens.
    """
    return training_flops / (6 * model_total_parameter_count * 1e9)


@dag.asset
def training_hdd_count(
        training_token_count: float,
        training_bytes_per_token: float,
        hdd_capacity: float
) -> int:
    """
    Estimate the number of hard disk drives required to store the training dataset.

    Args:
        training_token_count: Number of training tokens.
        training_bytes_per_token: Storage size of a token in bytes.
        hdd_capacity: Capacity of a single hard disk drive in TB.

    Returns:
        The number of hard disk drives.
    """
    dataset_size = math.ceil(training_token_count * training_bytes_per_token / 1024 ** 4)   # TB
    return max(1, math.ceil(dataset_size / hdd_capacity))


@dag.asset
def storage_it_energy(
        training_hdd_count: int,
        hdd_power: float,
        hdd_usage_ratio: float,
        training_duration: float
) -> float:
    """
    Compute the IT energy consumption of the training dataset storage during the training run.

    Args:
        training_hdd_count: Number of hard disk drives storing the training dataset.
        hdd_power: Power consumption of a single hard disk drive in kW.
        hdd_usage_ratio: Effective usage ratio of a hard disk drive.
        training_duration: Duration of the training run in hours.

    Returns:
        The IT energy consumption of the training dataset storage in kWh.
    """
    return hdd_power * training_hdd_count * hdd_usage_ratio * training_duration


@dag.asset
def model_lifetime_output_token_count(
        provider_ai_compute_capacity: float,
        inference_capacity_ratio: float,
        provider_active_model_count: int,
        compute_flops_per_watt: float,
        compute_efficiency_factor: float,
        datacenter_pue: ValueOrRange,
        model_active_parameter_count: float,
        model_lifetime: float
) -> ValueOrRange:
    """
    Estimate the total number of tokens generated by the model during its lifetime.

    Top-down estimation from Impact'IA: the AI compute capacity of the provider dedicated to inference is shared
    between its active models, converted to an effective compute (FLOPS) with a compute efficiency and utilization
    factor, then to generated tokens (2 FLOPs per active parameter per token) over the model lifetime.

    Args:
        provider_ai_compute_capacity: AI compute capacity of the provider in kW.
        inference_capacity_ratio: Share of the compute capacity dedicated to inference.
        provider_active_model_count: Number of models of the provider actively serving requests.
        compute_flops_per_watt: Compute efficiency of the inference hardware in FLOPS / W.
        compute_efficiency_factor: Effective utilization of the compute capacity.
        datacenter_pue: Power Usage Effectiveness of the data center.
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_lifetime: Lifetime of the model in seconds.

    Returns:
        The number of tokens generated by the model during its lifetime.
    """
    model_compute_capacity = provider_ai_compute_capacity * 1000 * inference_capacity_ratio \
        / provider_active_model_count    # W
    model_flops = model_compute_capacity * compute_flops_per_watt * compute_efficiency_factor / datacenter_pue
    return model_flops * model_lifetime / (2 * model_active_parameter_count * 1e9)


@dag.asset
def request_training_it_energy(
        training_it_energy: float,
        rnd_it_energy: float,
        storage_it_energy: float,
        model_lifetime_output_token_count: ValueOrRange,
        output_token_count: float
) -> ValueOrRange:
    """
    Compute the IT energy consumption of the training phase allocated to the request.

    Args:
        training_it_energy: IT energy consumption of the final training run in kWh.
        rnd_it_energy: IT energy consumption of the R&D experiments in kWh.
        storage_it_energy: IT energy consumption of the training dataset storage in kWh.
        model_lifetime_output_token_count: Number of tokens generated by the model during its lifetime.
        output_token_count: Number of generated tokens.

    Returns:
        The IT energy consumption of the training phase allocated to the request in kWh.
    """
    return (training_it_energy + rnd_it_energy + storage_it_energy) * output_token_count \
        / model_lifetime_output_token_count


@dag.asset
def request_training_energy(
        request_training_it_energy: ValueOrRange,
        datacenter_pue: ValueOrRange
) -> ValueOrRange:
    """
    Compute the energy consumption of the training phase allocated to the request.

    Args:
        request_training_it_energy: IT energy consumption of the training phase allocated to the request in kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.

    Returns:
        The energy consumption of the training phase allocated to the request in kWh.
    """
    return request_training_it_energy * datacenter_pue


@dag.asset
def inference_embodied_gwp_intensity(
        request_embodied_gwp: ValueOrRange,
        request_it_energy: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) embodied impact intensity of the inference phase.

    Args:
        request_embodied_gwp: GWP embodied impact of the request in kgCO2eq.
        request_it_energy: IT energy consumption of the request in kWh.

    Returns:
        The GWP embodied impact per kWh of IT energy in kgCO2eq / kWh.
    """
    if request_it_energy == 0:
        return 0
    return request_embodied_gwp / request_it_energy


@dag.asset
def inference_embodied_adpe_intensity(
        request_embodied_adpe: ValueOrRange,
        request_it_energy: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Abiotic Depletion Potential for Elements (ADPe) embodied impact intensity of the inference phase.

    Args:
        request_embodied_adpe: ADPe embodied impact of the request in kgSbeq.
        request_it_energy: IT energy consumption of the request in kWh.

    Returns:
        The ADPe embodied impact per kWh of IT energy in kgSbeq / kWh.
    """
    if request_it_energy == 0:
        return 0
    return request_embodied_adpe / request_it_energy


@dag.asset
def inference_embodied_pe_intensity(
        request_embodied_pe: ValueOrRange,
        request_it_energy: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Primary Energy (PE) embodied impact intensity of the inference phase.

    Args:
        request_embodied_pe: PE embodied impact of the request in MJ.
        request_it_energy: IT energy consumption of the request in kWh.

    Returns:
        The PE embodied impact per kWh of IT energy in MJ / kWh.
    """
    if request_it_energy == 0:
        return 0
    return request_embodied_pe / request_it_energy


@dag.asset
def request_training_gwp(
        request_training_energy: ValueOrRange,
        request_training_it_energy: ValueOrRange,
        if_electricity_mix_gwp: float,
        inference_embodied_gwp_intensity: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) impact of the training phase allocated to the request.

    Args:
        request_training_energy: Energy consumption of the training phase allocated to the request in kWh.
        request_training_it_energy: IT energy consumption of the training phase allocated to the request in kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        inference_embodied_gwp_intensity: GWP embodied impact per kWh of IT energy in kgCO2eq / kWh.

    Returns:
        The GWP impact (usage and embodied) of the training phase allocated to the request in kgCO2eq.
    """
    return request_training_energy * if_electricity_mix_gwp \
        + request_training_it_energy * inference_embodied_gwp_intensity


@dag.asset
def request_training_adpe(
        request_training_energy: ValueOrRange,
        request_training_it_energy: ValueOrRange,
        if_electricity_mix_adpe: float,
        inference_embodied_adpe_intensity: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Abiotic Depletion Potential for Elements (ADPe) impact of the training phase allocated to the request.

    Args:
        request_training_energy: Energy consumption of the training phase allocated to the request in kWh.
        request_training_it_energy: IT energy consumption of the training phase allocated to the request in kWh.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh.
        inference_embodied_adpe_intensity: ADPe embodied impact per kWh of IT energy in kgSbeq / kWh.

    Returns:
        The ADPe impact (usage and embodied) of the training phase allocated to the request in kgSbeq.
    """
    return request_training_energy * if_electricity_mix_adpe \
        + request_training_it_energy * inference_embodied_adpe_intensity


@dag.asset
def request_training_pe(
        request_training_energy: ValueOrRange,
        request_training_it_energy: ValueOrRange,
        if_electricity_mix_pe: float,
        inference_embodied_pe_intensity: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Primary Energy (PE) impact of the training phase allocated to the request.

    Args:
        request_training_energy: Energy consumption of the training phase allocated to the request in kWh.
        request_training_it_energy: IT energy consumption of the training phase allocated to the request in kWh.
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        inference_embodied_pe_intensity: PE embodied impact per kWh of IT energy in MJ / kWh.

    Returns:
        The PE impact (usage and embodied) of the training phase allocated to the request in MJ.
    """
    return request_training_energy * if_electricity_mix_pe \
        + request_training_it_energy * inference_embodied_pe_intensity


@dag.asset
def request_training_wcf(
        request_training_it_energy: ValueOrRange,
        if_electricity_mix_wue: float,
        datacenter_wue: ValueOrRange,
        datacenter_pue: ValueOrRange
) -> ValueOrRange:
    """
    Compute the water usage impact of the training phase allocated to the request.

    Args:
        request_training_it_energy: IT energy consumption of the training phase allocated to the request in kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.

    Returns:
        The water usage impact of the training phase allocated to the request in liters.
    """
    return request_training_it_energy * (datacenter_wue + datacenter_pue * if_electricity_mix_wue)


def compute_llm_training_impacts_dag(
        inference_results: dict[str, Any],
        model_release_date: date,
        provider_ai_compute_capacity: float,
        provider_active_model_count: int,
        if_electricity_mix_adpe: float,
        if_electricity_mix_pe: float,
        if_electricity_mix_gwp: float,
        if_electricity_mix_wue: float,
        gpu_power: float,
        training_flops_alpha: Optional[float] = TRAINING_FLOPS_ALPHA,
        training_flops_beta: Optional[float] = TRAINING_FLOPS_BETA,
        training_flops_gamma: Optional[float] = TRAINING_FLOPS_GAMMA,
        compute_flops_per_watt: Optional[float] = COMPUTE_FLOPS_PER_WATT,
        rnd_training_factor: Optional[float] = RND_TRAINING_FACTOR,
        training_duration: Optional[float] = TRAINING_DURATION,
        training_bytes_per_token: Optional[float] = TRAINING_BYTES_PER_TOKEN,
        hdd_capacity: Optional[float] = HDD_CAPACITY,
        hdd_power: Optional[float] = HDD_POWER,
        hdd_usage_ratio: Optional[float] = HDD_USAGE_RATIO,
        inference_capacity_ratio: Optional[float] = INFERENCE_CAPACITY_RATIO,
        compute_efficiency_factor: Optional[float] = COMPUTE_EFFICIENCY_FACTOR,
        model_lifetime: Optional[float] = MODEL_LIFETIME,
        **kwargs: Any
) -> dict[str, Any]:
    """
    Compute the training impacts dag of an LLM generation request.

    Args:
        inference_results: Results of the inference impacts dag (see `compute_llm_impacts_dag`).
        model_release_date: Release date of the model.
        provider_ai_compute_capacity: AI compute capacity of the provider in kW.
        provider_active_model_count: Number of models of the provider actively serving requests.
        if_electricity_mix_adpe: ADPe impact factor of the training electricity mix in kgSbeq / kWh (Antimony).
        if_electricity_mix_pe: PE impact factor of the training electricity mix in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of the training electricity mix in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF impact factor of the training electricity mix in L / kWh.
        gpu_power: Power consumption of a single GPU in kW.
        training_flops_alpha: Time coefficient of the training compute regression.
        training_flops_beta: Intercept of the training compute regression.
        training_flops_gamma: Parameters exponent of the training compute regression.
        compute_flops_per_watt: Compute efficiency of the hardware in FLOPS / W.
        rnd_training_factor: R&D energy consumption as a multiple of the final training run.
        training_duration: Duration of the training run in hours.
        training_bytes_per_token: Storage size of a training token in bytes.
        hdd_capacity: Capacity of a single hard disk drive in TB.
        hdd_power: Power consumption of a single hard disk drive in kW.
        hdd_usage_ratio: Effective usage ratio of a hard disk drive.
        inference_capacity_ratio: Share of the provider compute capacity dedicated to inference.
        compute_efficiency_factor: Effective utilization of the inference compute capacity.
        model_lifetime: Lifetime of the model in seconds.
        **kwargs: Any other parameter of the training dag (overrides intermediate results).

    Returns:
        The inference results extended with the training impacts dag intermediate states.
    """
    inputs: dict[str, Any] = dict(inference_results)
    inputs.update(
        model_release_days=(model_release_date - TRAINING_FLOPS_REFERENCE_DATE).days,
        provider_ai_compute_capacity=provider_ai_compute_capacity,
        provider_active_model_count=provider_active_model_count,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_wue=if_electricity_mix_wue,
        gpu_power=gpu_power,
        training_flops_alpha=training_flops_alpha,
        training_flops_beta=training_flops_beta,
        training_flops_gamma=training_flops_gamma,
        compute_flops_per_watt=compute_flops_per_watt,
        rnd_training_factor=rnd_training_factor,
        training_duration=training_duration,
        training_bytes_per_token=training_bytes_per_token,
        hdd_capacity=hdd_capacity,
        hdd_power=hdd_power,
        hdd_usage_ratio=hdd_usage_ratio,
        inference_capacity_ratio=inference_capacity_ratio,
        compute_efficiency_factor=compute_efficiency_factor,
        model_lifetime=model_lifetime,
    )
    inputs.update(kwargs)
    return dag.execute(**inputs)
