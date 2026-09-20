import math
from datetime import date
from typing import Any, Optional, Union, cast

from ecologits.impacts.dag import DAG
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Impacts, Training, Usage
from ecologits.impacts.training import compute_llm_training_impacts_dag
from ecologits.utils.range_value import RangeValue, ValueOrRange

MODEL_QUANTIZATION_BITS = 16

GPU_ENERGY_ALPHA = 1.1665273170451914e-06
GPU_ENERGY_BETA = -0.011205921025579175
GPU_ENERGY_GAMMA = 4.052928146734005e-05

LATENCY_ALPHA = 0.0006785088094353663
LATENCY_BETA = 0.0003119310311688259
LATENCY_GAMMA = 0.019473717579473387

# Pre-fill latency modeled from the number of input tokens (adapted from Impact'IA, based on Agrawal et al. (2024)
# measurements of Mistral 7B on a single A100 GPU: TTFT[ms] = 0.062 * input_tokens + 17.88).
PREFILL_LATENCY_REFERENCE_COEFFICIENT = 0.062 / 1000   # s per input token (Mistral 7B, A100)
PREFILL_LATENCY_REFERENCE_TTFT = 0.44                   # s, median TTFT of Mistral 7B on OpenRouter
PREFILL_LATENCY_REFERENCE_PARAMETER_COUNT = 7.3         # B parameters (Mistral 7B)
PREFILL_LATENCY_INTERCEPT = 17.88 / 1000                # s

GPU_MEMORY = 80  # GB
GPU_POWER = 0.7  # kW (NVIDIA H100 80GB SXM5)
GPU_EMBODIED_IMPACT_GWP = 273
GPU_EMBODIED_IMPACT_ADPE = 0.00895
GPU_EMBODIED_IMPACT_PE = 3721

SERVER_GPUS = 8
SERVER_POWER = 1.2  # kW
SERVER_EMBODIED_IMPACT_GWP = 5700
SERVER_EMBODIED_IMPACT_ADPE = 0.37
SERVER_EMBODIED_IMPACT_PE = 70000

# Data center network equipment attached to one server (firewall, router and switch), adapted from Impact'IA.
# Power and effective usage ratios from ADEME Base Empreinte / ADEME-ARCEP-ARCOM, embodied GWP from Resilio DB
# (Fortinet FortiGate 100E firewall, Juniper PTX10002-36QDD router, Arista 7050TX3-48C8 switch).
NETWORK_FIREWALL_POWER = 0.09           # kW
NETWORK_FIREWALL_USAGE_RATIO = 0.0358
NETWORK_FIREWALL_EMBODIED_GWP = 333.8   # kgCO2eq
NETWORK_ROUTER_POWER = 0.09             # kW
NETWORK_ROUTER_USAGE_RATIO = 0.286
NETWORK_ROUTER_EMBODIED_GWP = 403       # kgCO2eq
NETWORK_SWITCH_POWER = 0.09             # kW
NETWORK_SWITCH_USAGE_RATIO = 1.468
NETWORK_SWITCH_EMBODIED_GWP = 363       # kgCO2eq

NETWORK_POWER = (
    NETWORK_FIREWALL_POWER * NETWORK_FIREWALL_USAGE_RATIO
    + NETWORK_ROUTER_POWER * NETWORK_ROUTER_USAGE_RATIO
    + NETWORK_SWITCH_POWER * NETWORK_SWITCH_USAGE_RATIO
)   # kW, network power allocated to one server
NETWORK_EMBODIED_IMPACT_GWP = (
    NETWORK_FIREWALL_EMBODIED_GWP * NETWORK_FIREWALL_USAGE_RATIO
    + NETWORK_ROUTER_EMBODIED_GWP * NETWORK_ROUTER_USAGE_RATIO
    + NETWORK_SWITCH_EMBODIED_GWP * NETWORK_SWITCH_USAGE_RATIO
)   # kgCO2eq, network embodied impacts allocated to one server
NETWORK_LIFESPAN = 5 * 365 * 24 * 60 * 60   # seconds

# Embodied impacts of the data center building and technical environment (electrical and cooling equipment)
# allocated per kWh of IT electricity consumption (Hubblo, ADEME cloud PCR), adapted from Impact'IA.
DATACENTER_BUILDING_EMBODIED_GWP_FACTOR = 0.01  # kgCO2eq / kWh

HARDWARE_LIFESPAN = 3 * 365 * 24 * 60 * 60

BATCH_SIZE = 64

dag = DAG()


@dag.asset
def gpu_energy(
        model_active_parameter_count: float,
        output_token_count: float,
        batch_size: int,
        gpu_energy_alpha: float,
        gpu_energy_beta: float,
        gpu_energy_gamma: float,
) -> ValueOrRange:
    """
    Compute energy consumption of a single GPU.

    Args:
        model_active_parameter_count: Number of active parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        batch_size: Number of requests handled concurrently by the server.
        gpu_energy_alpha: Alpha coefficient of the energy regression.
        gpu_energy_beta: Beta coefficient of the energy regression.
        gpu_energy_gamma: Beta coefficient of the energy regression.

    Returns:
        The energy consumption of a single GPU in kWh.
    """
    gpu_energy_per_token = gpu_energy_alpha * math.exp(gpu_energy_beta * batch_size) * model_active_parameter_count + \
        gpu_energy_gamma
    gpu_energy_per_token /= 1000    # convert to kWh
    return output_token_count * gpu_energy_per_token


@dag.asset
def prefill_latency(
        model_active_parameter_count: float,
        prefill_latency_reference_coefficient: float,
        prefill_latency_reference_ttft: float,
        prefill_latency_reference_parameter_count: float,
        prefill_latency_intercept: float,
        input_token_count: Optional[float] = None,
        ttft: Optional[float] = None,
) -> float:
    """
    Compute the pre-fill latency (time-to-first-token) in seconds.

    When the number of input tokens is known, the pre-fill latency is modeled as a linear function of the number of
    input tokens (Agrawal et al., 2024). The per-token coefficient measured for Mistral 7B is scaled by the ratio
    between the median time-to-first-token of the model and the one of Mistral 7B (both measured on OpenRouter), as
    proposed by Impact'IA. When the time-to-first-token of the model is unknown, the coefficient is scaled by the
    number of active parameters instead. When the number of input tokens is unknown, the measured time-to-first-token
    is used as is.

    Args:
        model_active_parameter_count: Number of active parameters of the model (in billion).
        prefill_latency_reference_coefficient: Pre-fill latency per input token of the reference model in seconds.
        prefill_latency_reference_ttft: Median time-to-first-token of the reference model in seconds.
        prefill_latency_reference_parameter_count: Number of active parameters of the reference model (in billion).
        prefill_latency_intercept: Incompressible pre-fill latency in seconds.
        input_token_count: Number of input tokens (optional).
        ttft: Time-to-first-token latency of the model in seconds (optional).

    Returns:
        The pre-fill latency in seconds.
    """
    if input_token_count is None:
        return ttft or 0
    if ttft is not None:
        coefficient = prefill_latency_reference_coefficient * ttft / prefill_latency_reference_ttft
    else:
        coefficient = prefill_latency_reference_coefficient * model_active_parameter_count \
            / prefill_latency_reference_parameter_count
    return coefficient * input_token_count + prefill_latency_intercept


@dag.asset
def generation_latency(
        model_active_parameter_count: float,
        output_token_count: float,
        batch_size: int,
        latency_alpha: float,
        latency_beta: float,
        latency_gamma: float,
        request_latency: float,
        prefill_latency: float,
        tps: Optional[float] = None,
) -> ValueOrRange:
    """
    Compute the token generation latency in seconds.

    Args:
        model_active_parameter_count: Number of active parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        batch_size: Number of requests handled concurrently by the server.
        latency_alpha: Alpha coefficient of the latency regression.
        latency_beta: Beta coefficient of the latency regression.
        latency_gamma: Gamma coefficient of the latency regression.
        request_latency: Measured request latency in seconds.
        prefill_latency: Pre-fill latency (time-to-first-token) in seconds.
        tps: Number of tokens generated per second by the model.

    Returns:
        The token generation latency in seconds.
    """
    if tps is None:
        latency_per_token = latency_alpha * model_active_parameter_count + latency_beta * batch_size + latency_gamma
    else:
        latency_per_token = 1 / tps
    gpu_latency = output_token_count * latency_per_token + prefill_latency
    if request_latency < gpu_latency:
        return request_latency
    return gpu_latency


@dag.asset
def model_required_memory(
        model_total_parameter_count: float,
        model_quantization_bits: int,
) -> float:
    """
    Compute the required memory to load the model on GPU.

    Args:
        model_total_parameter_count: Number of parameters of the model (in billion).
        model_quantization_bits: Number of bits used to represent the model weights.

    Returns:
        The amount of required GPU memory to load the model.
    """
    return 1.2 * model_total_parameter_count * model_quantization_bits / 8


@dag.asset
def gpu_required_count(
        model_required_memory: float,
        gpu_memory: float
) -> int:
    """
    Compute the number of required GPU to store the model.

    Args:
        model_required_memory: Required memory to load the model on GPU.
        gpu_memory: Amount of memory available on a single GPU.

    Returns:
        The number of required GPUs to load the model.
    """
    gpu_nb = math.ceil(model_required_memory / gpu_memory)
    return 2 ** math.ceil(math.log2(gpu_nb))    # Round-up in base two


@dag.asset
def server_energy(
        generation_latency: float,
        server_power: float,
        server_gpu_count: int,
        gpu_required_count: int,
        batch_size: int
) -> float:
    """
    Compute the energy consumption of the server.

    Args:
        generation_latency: Token generation latency in seconds.
        server_power: Power consumption of the server in kW.
        server_gpu_count: Number of available GPUs in the server.
        gpu_required_count: Number of required GPUs to load the model.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The energy consumption of the server (GPUs are not included) in kWh.
    """
    return (generation_latency / 3600) * server_power * (gpu_required_count / server_gpu_count) * (1 / batch_size)


@dag.asset
def network_energy(
        generation_latency: float,
        network_power: float,
        server_gpu_count: int,
        gpu_required_count: int,
        batch_size: int
) -> float:
    """
    Compute the energy consumption of the data center network equipment (firewall, router, switch).

    The network equipment power allocated to one server is shared between the requests the same way as the server
    power (adapted from Impact'IA).

    Args:
        generation_latency: Token generation latency in seconds.
        network_power: Power consumption of the network equipment allocated to one server in kW.
        server_gpu_count: Number of available GPUs in the server.
        gpu_required_count: Number of required GPUs to load the model.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The energy consumption of the network equipment in kWh.
    """
    return (generation_latency / 3600) * network_power * (gpu_required_count / server_gpu_count) * (1 / batch_size)


@dag.asset
def request_energy(
        datacenter_pue: float,
        request_it_energy: ValueOrRange
) -> ValueOrRange:
    """
    Compute the energy consumption of the request.

    Args:
        datacenter_pue: Power Usage Effectiveness of the data center.
        request_it_energy: IT energy consumption of the request in kWh.

    Returns:
        The energy consumption of the request in kWh.
    """
    return datacenter_pue * request_it_energy


@dag.asset
def request_it_energy(
        server_energy: float,
        gpu_required_count: int,
        gpu_energy: ValueOrRange,
        network_energy: float
) -> ValueOrRange:
    """
    Compute the IT energy consumption of the request before data center overhead.

    Args:
        server_energy: Energy consumption of the server in kWh.
        gpu_required_count: Number of required GPUs to load the model.
        gpu_energy: Energy consumption of a single GPU in kWh.
        network_energy: Energy consumption of the network equipment in kWh.

    Returns:
        The IT energy consumption of the request in kWh.
    """
    return server_energy + gpu_required_count * gpu_energy + network_energy


@dag.asset
def request_usage_gwp(
        request_energy: ValueOrRange,
        if_electricity_mix_gwp: float
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) usage impact of the request.

    Args:
        request_energy: Energy consumption of the request in kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.

    Returns:
        The GWP usage impact of the request in kgCO2eq.
    """
    return request_energy * if_electricity_mix_gwp


@dag.asset
def request_usage_adpe(
        request_energy: ValueOrRange,
        if_electricity_mix_adpe: float
) -> ValueOrRange:
    """
    Compute the Abiotic Depletion Potential for Elements (ADPe) usage impact of the request.

    Args:
        request_energy: Energy consumption of the request in kWh.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh.

    Returns:
        The ADPe usage impact of the request in kgSbeq.
    """
    return request_energy * if_electricity_mix_adpe


@dag.asset
def request_usage_pe(
        request_energy: ValueOrRange,
        if_electricity_mix_pe: float
) -> ValueOrRange:
    """
    Compute the Primary Energy (PE) usage impact of the request.

    Args:
        request_energy: Energy consumption of the request in kWh.
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.

    Returns:
        The PE usage impact of the request in MJ.
    """
    return request_energy * if_electricity_mix_pe


@dag.asset
def request_usage_wcf(
        request_it_energy: ValueOrRange,
        if_electricity_mix_wue: float,
        datacenter_wue: float,
        datacenter_pue: float
) -> ValueOrRange:
    """
    Compute the water usage impact of the request.

    Args:
        request_it_energy: IT energy consumption of the request in kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
    Returns:
        The water usage impact of the request in liters.
    """
    return request_it_energy * (datacenter_wue + datacenter_pue * if_electricity_mix_wue)


@dag.asset
def server_gpu_embodied_gwp(
        server_embodied_gwp: float,
        server_gpu_count: float,
        gpu_embodied_gwp: float,
        gpu_required_count: int
) -> float:
    """
    Compute the Global Warming Potential (GWP) embodied impact of the server

    Args:
        server_embodied_gwp: GWP embodied impact of the server in kgCO2eq.
        server_gpu_count: Number of available GPUs in the server.
        gpu_embodied_gwp: GWP embodied impact of a single GPU in kgCO2eq.
        gpu_required_count: Number of required GPUs to load the model.

    Returns:
        The GWP embodied impact of the server and the GPUs in kgCO2eq.
    """
    return (gpu_required_count / server_gpu_count) * server_embodied_gwp + gpu_required_count * gpu_embodied_gwp


@dag.asset
def server_gpu_embodied_adpe(
        server_embodied_adpe: float,
        server_gpu_count: float,
        gpu_embodied_adpe: float,
        gpu_required_count: int
) -> float:
    """
    Compute the Abiotic Depletion Potential for Elements (ADPe) embodied impact of the server

    Args:
        server_embodied_adpe: ADPe embodied impact of the server in kgSbeq.
        server_gpu_count: Number of available GPUs in the server.
        gpu_embodied_adpe: ADPe embodied impact of a single GPU in kgSbeq.
        gpu_required_count: Number of required GPUs to load the model.

    Returns:
        The ADPe embodied impact of the server and the GPUs in kgSbeq.
    """
    return (gpu_required_count / server_gpu_count) * server_embodied_adpe + gpu_required_count * gpu_embodied_adpe


@dag.asset
def server_gpu_embodied_pe(
        server_embodied_pe: float,
        server_gpu_count: float,
        gpu_embodied_pe: float,
        gpu_required_count: int
) -> float:
    """
    Compute the Primary Energy (PE) embodied impact of the server

    Args:
        server_embodied_pe: PE embodied impact of the server in MJ.
        server_gpu_count: Number of available GPUs in the server.
        gpu_embodied_pe: PE embodied impact of a single GPU in MJ.
        gpu_required_count: Number of required GPUs to load the model.

    Returns:
        The PE embodied impact of the server and the GPUs in MJ.
    """
    return (gpu_required_count / server_gpu_count) * server_embodied_pe + gpu_required_count * gpu_embodied_pe


@dag.asset
def request_hardware_embodied_gwp(
        server_gpu_embodied_gwp: float,
        server_lifetime: float,
        generation_latency: ValueOrRange,
        batch_size: int
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) embodied impact of the server and GPUs allocated to the request.

    Args:
        server_gpu_embodied_gwp: GWP embodied impact of the server and the GPUs in kgCO2eq.
        server_lifetime: Lifetime duration of the server in seconds.
        generation_latency: Token generation latency in seconds.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The GWP embodied impact of the server and GPUs allocated to the request in kgCO2eq.
    """
    return generation_latency * server_gpu_embodied_gwp / (server_lifetime * batch_size)


@dag.asset
def request_network_embodied_gwp(
        network_embodied_gwp: float,
        network_lifetime: float,
        generation_latency: ValueOrRange,
        server_gpu_count: int,
        gpu_required_count: int,
        batch_size: int
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) embodied impact of the network equipment allocated to the request.

    The network equipment embodied impacts allocated to one server are shared between the requests the same way as
    the server embodied impacts (adapted from Impact'IA).

    Args:
        network_embodied_gwp: GWP embodied impact of the network equipment allocated to one server in kgCO2eq.
        network_lifetime: Lifetime duration of the network equipment in seconds.
        generation_latency: Token generation latency in seconds.
        server_gpu_count: Number of available GPUs in the server.
        gpu_required_count: Number of required GPUs to load the model.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The GWP embodied impact of the network equipment allocated to the request in kgCO2eq.
    """
    return generation_latency * network_embodied_gwp * (gpu_required_count / server_gpu_count) \
        / (network_lifetime * batch_size)


@dag.asset
def request_datacenter_embodied_gwp(
        request_it_energy: ValueOrRange,
        datacenter_building_embodied_gwp_factor: float
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) embodied impact of the data center building and technical environment
    (electrical and cooling equipment) allocated to the request (adapted from Impact'IA).

    Args:
        request_it_energy: IT energy consumption of the request in kWh.
        datacenter_building_embodied_gwp_factor: GWP embodied impact of the data center building and technical
            environment per kWh of IT energy consumption in kgCO2eq / kWh.

    Returns:
        The GWP embodied impact of the data center building allocated to the request in kgCO2eq.
    """
    return request_it_energy * datacenter_building_embodied_gwp_factor


@dag.asset
def request_embodied_gwp(
        request_hardware_embodied_gwp: ValueOrRange,
        request_network_embodied_gwp: ValueOrRange,
        request_datacenter_embodied_gwp: ValueOrRange
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) embodied impact of the request.

    Args:
        request_hardware_embodied_gwp: GWP embodied impact of the server and GPUs allocated to the request in kgCO2eq.
        request_network_embodied_gwp: GWP embodied impact of the network equipment allocated to the request in
            kgCO2eq.
        request_datacenter_embodied_gwp: GWP embodied impact of the data center building allocated to the request in
            kgCO2eq.

    Returns:
        The GWP embodied impact of the request in kgCO2eq.
    """
    return request_hardware_embodied_gwp + request_network_embodied_gwp + request_datacenter_embodied_gwp


@dag.asset
def request_embodied_adpe(
        server_gpu_embodied_adpe: float,
        server_lifetime: float,
        generation_latency: ValueOrRange,
        batch_size: int
) -> ValueOrRange:
    """
    Compute the Abiotic Depletion Potential for Elements (ADPe) embodied impact of the request.

    Note:
        Network equipment and data center building embodied impacts are not modeled for ADPe due to a lack of data.

    Args:
        server_gpu_embodied_adpe: ADPe embodied impact of the server and the GPUs in kgSbeq.
        server_lifetime: Lifetime duration of the server in seconds.
        generation_latency: Token generation latency in seconds.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The ADPe embodied impact of the request in kgSbeq.
    """
    return generation_latency * server_gpu_embodied_adpe / (server_lifetime * batch_size)


@dag.asset
def request_embodied_pe(
        server_gpu_embodied_pe: float,
        server_lifetime: float,
        generation_latency: ValueOrRange,
        batch_size: int
) -> ValueOrRange:
    """
    Compute the Primary Energy (PE) embodied impact of the request.

    Note:
        Network equipment and data center building embodied impacts are not modeled for PE due to a lack of data.

    Args:
        server_gpu_embodied_pe: PE embodied impact of the server and the GPUs in MJ.
        server_lifetime: Lifetime duration of the server in seconds.
        generation_latency: Token generation latency in seconds.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The PE embodied impact of the request in MJ.
    """
    return generation_latency * server_gpu_embodied_pe / (server_lifetime * batch_size)


def compute_llm_impacts_dag(
        model_active_parameter_count: ValueOrRange,
        model_total_parameter_count: ValueOrRange,
        output_token_count: float,
        request_latency: float,
        if_electricity_mix_adpe: float,
        if_electricity_mix_pe: float,
        if_electricity_mix_gwp: float,
        if_electricity_mix_wue: float,
        datacenter_pue: ValueOrRange,
        datacenter_wue: ValueOrRange,
        input_token_count: Optional[float] = None,
        model_quantization_bits: Optional[int] = MODEL_QUANTIZATION_BITS,
        gpu_energy_alpha: Optional[float] = GPU_ENERGY_ALPHA,
        gpu_energy_beta: Optional[float] = GPU_ENERGY_BETA,
        gpu_energy_gamma: Optional[float] = GPU_ENERGY_GAMMA,
        latency_alpha: Optional[float] = LATENCY_ALPHA,
        latency_beta: Optional[float] = LATENCY_BETA,
        latency_gamma: Optional[float] = LATENCY_GAMMA,
        prefill_latency_reference_coefficient: Optional[float] = PREFILL_LATENCY_REFERENCE_COEFFICIENT,
        prefill_latency_reference_ttft: Optional[float] = PREFILL_LATENCY_REFERENCE_TTFT,
        prefill_latency_reference_parameter_count: Optional[float] = PREFILL_LATENCY_REFERENCE_PARAMETER_COUNT,
        prefill_latency_intercept: Optional[float] = PREFILL_LATENCY_INTERCEPT,
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
        network_power: Optional[float] = NETWORK_POWER,
        network_embodied_gwp: Optional[float] = NETWORK_EMBODIED_IMPACT_GWP,
        network_lifetime: Optional[float] = NETWORK_LIFESPAN,
        datacenter_building_embodied_gwp_factor: Optional[float] = DATACENTER_BUILDING_EMBODIED_GWP_FACTOR,
        batch_size: Optional[float] = BATCH_SIZE,
        tps: Optional[float] = None,
        ttft: Optional[float] = None,
) -> dict[str, ValueOrRange]:
    """
    Compute the impacts dag of an LLM generation request.

    Args:
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_total_parameter_count: Number of parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh (Antimony).
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
        input_token_count: Number of input tokens (optional, used to estimate the pre-fill latency).
        model_quantization_bits: Number of bits used to represent the model weights.
        gpu_energy_alpha: Alpha coefficient of the "GPU energy" regression.
        gpu_energy_beta: Beta coefficient of the "GPU energy" regression.
        gpu_energy_gamma: Gamma coefficient of the "GPU energy" regression.
        latency_alpha: Alpha coefficient of the "Latency" regression.
        latency_beta: Beta coefficient of the "Latency" regression.
        latency_gamma: Gamma coefficient of the "Latency" regression.
        prefill_latency_reference_coefficient: Pre-fill latency per input token of the reference model in seconds.
        prefill_latency_reference_ttft: Median time-to-first-token of the reference model in seconds.
        prefill_latency_reference_parameter_count: Number of active parameters of the reference model (in billion).
        prefill_latency_intercept: Incompressible pre-fill latency in seconds.
        gpu_memory: Amount of memory available on a single GPU.
        gpu_embodied_gwp: GWP embodied impact of a single GPU.
        gpu_embodied_adpe: ADPe embodied impact of a single GPU.
        gpu_embodied_pe: PE embodied impact of a single GPU.
        server_gpu_count: Number of available GPUs in the server.
        server_power: Power consumption of the server in kW.
        server_embodied_gwp: GWP embodied impact of the server in kgCO2eq.
        server_embodied_adpe: ADPe embodied impact of the server in kgSbeq.
        server_embodied_pe: PE embodied impact of the server in MJ.
        server_lifetime: Lifetime duration of the server in seconds.
        network_power: Power consumption of the network equipment allocated to one server in kW.
        network_embodied_gwp: GWP embodied impact of the network equipment allocated to one server in kgCO2eq.
        network_lifetime: Lifetime duration of the network equipment in seconds.
        datacenter_building_embodied_gwp_factor: GWP embodied impact of the data center building per kWh of IT
            energy in kgCO2eq / kWh.
        batch_size: Number of requests handled concurrently by the server.
        tps: Number of tokens generated per second by the model (optional).
        ttft: Time-to-first-token latency in seconds (optional).
    Returns:
        The environmental impacts dag with all intermediate states.
    """
    results = dag.execute(
        model_active_parameter_count=model_active_parameter_count,
        model_total_parameter_count=model_total_parameter_count,
        model_quantization_bits=model_quantization_bits,
        output_token_count=output_token_count,
        input_token_count=input_token_count,
        request_latency=request_latency,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_wue=datacenter_wue,
        datacenter_pue=datacenter_pue,
        gpu_energy_alpha=gpu_energy_alpha,
        gpu_energy_beta=gpu_energy_beta,
        gpu_energy_gamma=gpu_energy_gamma,
        latency_alpha=latency_alpha,
        latency_beta=latency_beta,
        latency_gamma=latency_gamma,
        prefill_latency_reference_coefficient=prefill_latency_reference_coefficient,
        prefill_latency_reference_ttft=prefill_latency_reference_ttft,
        prefill_latency_reference_parameter_count=prefill_latency_reference_parameter_count,
        prefill_latency_intercept=prefill_latency_intercept,
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
        network_power=network_power,
        network_embodied_gwp=network_embodied_gwp,
        network_lifetime=network_lifetime,
        datacenter_building_embodied_gwp_factor=datacenter_building_embodied_gwp_factor,
        batch_size=batch_size,
        tps=tps,
        ttft=ttft,
    )
    return results


def _default(value: Optional[float], default: float) -> float:
    return default if value is None else value


def _parameter_bounds(
        model_active_parameter_count: ValueOrRange,
        model_total_parameter_count: ValueOrRange
) -> tuple[list[ValueOrRange], list[ValueOrRange]]:
    """
    Get the (min, max) bounds of the model parameters when at least one of them is a range, else the values as is.
    """
    if not isinstance(model_active_parameter_count, RangeValue) \
            and not isinstance(model_total_parameter_count, RangeValue):
        return [model_active_parameter_count], [model_total_parameter_count]
    if isinstance(model_active_parameter_count, RangeValue):
        active_params = [model_active_parameter_count.min, model_active_parameter_count.max]
    else:
        active_params = [model_active_parameter_count, model_active_parameter_count]
    if isinstance(model_total_parameter_count, RangeValue):
        total_params = [model_total_parameter_count.min, model_total_parameter_count.max]
    else:
        total_params = [model_total_parameter_count, model_total_parameter_count]
    return active_params, total_params


def _merge_results(
        results: dict[str, Union[RangeValue, float, int]],
        new_results: dict[str, Any],
        fields: list[str]
) -> None:
    """
    Merge the results of a dag execution into the aggregated results as (min, max) ranges.
    """
    for field in fields:
        if field not in results:
            results[field] = new_results[field]
            continue
        min_result = results[field]
        max_result = new_results[field]
        if isinstance(min_result, RangeValue):
            min_result = cast(Union[float, int], min_result.min)
        if isinstance(max_result, RangeValue):
            max_result = cast(Union[float, int], max_result.max)
        results[field] = RangeValue(min=min_result, max=max_result)


_INFERENCE_FIELDS = [
    "request_energy", "request_usage_gwp", "request_usage_adpe", "request_usage_pe", "request_usage_wcf",
    "request_embodied_gwp", "request_embodied_adpe", "request_embodied_pe"
]
_TRAINING_FIELDS = [
    "request_training_energy", "request_training_gwp", "request_training_adpe", "request_training_pe",
    "request_training_wcf"
]


def compute_llm_impacts(
        model_active_parameter_count: ValueOrRange,
        model_total_parameter_count: ValueOrRange,
        output_token_count: float,
        if_electricity_mix_adpe: float,
        if_electricity_mix_pe: float,
        if_electricity_mix_gwp: float,
        if_electricity_mix_wue: float,
        datacenter_pue: ValueOrRange,
        datacenter_wue: ValueOrRange,
        request_latency: Optional[float] = None,
        input_token_count: Optional[float] = None,
        tps: Optional[float] = None,
        ttft: Optional[float] = None,
        model_release_date: Optional[date] = None,
        provider_ai_compute_capacity: Optional[float] = None,
        provider_active_model_count: Optional[int] = None,
        if_training_electricity_mix_adpe: Optional[float] = None,
        if_training_electricity_mix_pe: Optional[float] = None,
        if_training_electricity_mix_gwp: Optional[float] = None,
        if_training_electricity_mix_wue: Optional[float] = None,
        **kwargs: Any
) -> Impacts:
    """
    Compute the impacts of an LLM generation request.

    The training phase is computed only when the model release date, the provider AI compute capacity and the
    provider number of active models are all provided. Otherwise, `Impacts.training` is `None`.

    Args:
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_total_parameter_count: Number of total parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption of kgSbeq / kWh (Antimony).
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
        request_latency: Measured request latency in seconds.
        input_token_count: Number of input tokens (optional, used to estimate the pre-fill latency).
        tps: Number of tokens generated per second by the model.
        ttft: Time-to-first-token latency in seconds.
        model_release_date: Release date of the model (optional, used for the training phase).
        provider_ai_compute_capacity: AI compute capacity of the provider in kW (optional, used for the training
            phase).
        provider_active_model_count: Number of models of the provider actively serving requests (optional, used for
            the training phase).
        if_training_electricity_mix_adpe: ADPe impact factor of the electricity mix of the training location in
            kgSbeq / kWh (defaults to the inference electricity mix).
        if_training_electricity_mix_pe: PE impact factor of the electricity mix of the training location in
            MJ / kWh (defaults to the inference electricity mix).
        if_training_electricity_mix_gwp: GWP impact factor of the electricity mix of the training location in
            kgCO2eq / kWh (defaults to the inference electricity mix).
        if_training_electricity_mix_wue: WCF impact factor of the electricity mix of the training location in
            L / kWh (defaults to the inference electricity mix).
        **kwargs: Any other optional parameter of the inference or training dag.
    Returns:
        The impacts of an LLM generation request.
    """
    if request_latency is None:
        request_latency = math.inf

    with_training = model_release_date is not None \
        and provider_ai_compute_capacity is not None \
        and provider_active_model_count is not None

    active_params, total_params = _parameter_bounds(model_active_parameter_count, model_total_parameter_count)

    inference_kwargs = {k: v for k, v in kwargs.items() if k in _INFERENCE_DAG_PARAMETERS}
    training_kwargs = {k: v for k, v in kwargs.items() if k not in _INFERENCE_DAG_PARAMETERS}
    training_kwargs.setdefault("gpu_power", GPU_POWER)

    results: dict[str, Union[RangeValue, float, int]] = {}
    fields = _INFERENCE_FIELDS + (_TRAINING_FIELDS if with_training else [])
    for act_param, tot_param in zip(active_params, total_params):
        res = compute_llm_impacts_dag(
            model_active_parameter_count=act_param,
            model_total_parameter_count=tot_param,
            output_token_count=output_token_count,
            input_token_count=input_token_count,
            request_latency=request_latency,
            if_electricity_mix_adpe=if_electricity_mix_adpe,
            if_electricity_mix_pe=if_electricity_mix_pe,
            if_electricity_mix_gwp=if_electricity_mix_gwp,
            if_electricity_mix_wue=if_electricity_mix_wue,
            datacenter_pue=datacenter_pue,
            datacenter_wue=datacenter_wue,
            tps=tps,
            ttft=ttft,
            **inference_kwargs
        )
        if with_training:
            res = compute_llm_training_impacts_dag(
                inference_results=res,
                model_release_date=cast(date, model_release_date),
                provider_ai_compute_capacity=cast(float, provider_ai_compute_capacity),
                provider_active_model_count=cast(int, provider_active_model_count),
                if_electricity_mix_adpe=_default(if_training_electricity_mix_adpe, if_electricity_mix_adpe),
                if_electricity_mix_pe=_default(if_training_electricity_mix_pe, if_electricity_mix_pe),
                if_electricity_mix_gwp=_default(if_training_electricity_mix_gwp, if_electricity_mix_gwp),
                if_electricity_mix_wue=_default(if_training_electricity_mix_wue, if_electricity_mix_wue),
                **training_kwargs
            )
        _merge_results(results, res, fields)

    energy = Energy(value=results["request_energy"])
    gwp_usage = GWP(value=results["request_usage_gwp"])
    adpe_usage = ADPe(value=results["request_usage_adpe"])
    pe_usage = PE(value=results["request_usage_pe"])
    wcf_usage = WCF(value=results["request_usage_wcf"])
    gwp_embodied = GWP(value=results["request_embodied_gwp"])
    adpe_embodied = ADPe(value=results["request_embodied_adpe"])
    pe_embodied = PE(value=results["request_embodied_pe"])

    training = None
    if with_training:
        training = Training(
            energy=Energy(value=results["request_training_energy"]),
            gwp=GWP(value=results["request_training_gwp"]),
            adpe=ADPe(value=results["request_training_adpe"]),
            pe=PE(value=results["request_training_pe"]),
            wcf=WCF(value=results["request_training_wcf"]),
        )

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
            wcf=wcf_usage
        ),
        embodied=Embodied(
            gwp=gwp_embodied,
            adpe=adpe_embodied,
            pe=pe_embodied
        ),
        training=training
    )


_INFERENCE_DAG_PARAMETERS = set(compute_llm_impacts_dag.__annotations__.keys()) - {"return"}
