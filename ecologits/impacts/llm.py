import math
from typing import Any, Optional, Union, cast

from ecologits.impacts.dag import DAG
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Impacts, Usage
from ecologits.utils.range_value import RangeValue, ValueOrRange

MODEL_QUANTIZATION_BITS = 16

GPU_ENERGY_ALPHA = 1.1665273170451914e-06
GPU_ENERGY_BETA = -0.011205921025579175
GPU_ENERGY_GAMMA = 4.052928146734005e-05

LATENCY_ALPHA = 0.0006785088094353663
LATENCY_BETA = 0.0003119310311688259
LATENCY_GAMMA = 0.019473717579473387

GPU_MEMORY = 80  # GB
GPU_EMBODIED_IMPACT_GWP = 164
GPU_EMBODIED_IMPACT_ADPE = 5.1e-3  # TODO (still A100)
GPU_EMBODIED_IMPACT_PE = 1828      # TODO (still A100)

SERVER_GPUS = 8
SERVER_POWER = 1.2  # kW
SERVER_EMBODIED_IMPACT_GWP = 5700
SERVER_EMBODIED_IMPACT_ADPE = 0.37
SERVER_EMBODIED_IMPACT_PE = 70000

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
def generation_latency(
        model_active_parameter_count: float,
        output_token_count: float,
        batch_size: int,
        latency_alpha: float,
        latency_beta: float,
        latency_gamma: float,
        request_latency: float
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

    Returns:
        The token generation latency in seconds.
    """
    latency_per_token = latency_alpha * model_active_parameter_count + latency_beta * batch_size + latency_gamma
    gpu_latency = output_token_count * latency_per_token
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
def request_energy(
        datacenter_pue: float,
        server_energy: float,
        gpu_required_count: int,
        gpu_energy: ValueOrRange
) -> ValueOrRange:
    """
    Compute the energy consumption of the request.

    Args:
        datacenter_pue: Power Usage Effectiveness of the data center.
        server_energy: Energy consumption of the server in kWh.
        gpu_required_count: Number of required GPUs to load the model.
        gpu_energy: Energy consumption of a single GPU in kWh.

    Returns:
        The energy consumption of the request in kWh.
    """
    return datacenter_pue * (server_energy + gpu_required_count * gpu_energy)


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
        request_energy: ValueOrRange,
        if_electricity_mix_wue: float,
        datacenter_wue: float,
        datacenter_pue: float
) -> ValueOrRange:
    """
    Compute the water usage impact of the request.

    Args:
        request_energy: Energy consumption of the request in kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
    Returns:
        The water usage impact of the request in liters.
    """
    return request_energy * (datacenter_wue + datacenter_pue * if_electricity_mix_wue)


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
def request_embodied_gwp(
        server_gpu_embodied_gwp: float,
        server_lifetime: float,
        generation_latency: ValueOrRange,
        batch_size: int
) -> ValueOrRange:
    """
    Compute the Global Warming Potential (GWP) embodied impact of the request.

    Args:
        server_gpu_embodied_gwp: GWP embodied impact of the server and the GPUs in kgCO2eq.
        server_lifetime: Lifetime duration of the server in seconds.
        generation_latency: Token generation latency in seconds.
        batch_size: Number of requests handled concurrently by the server.

    Returns:
        The GWP embodied impact of the request in kgCO2eq.
    """
    return generation_latency * server_gpu_embodied_gwp / (server_lifetime * batch_size)


@dag.asset
def request_embodied_adpe(
        server_gpu_embodied_adpe: float,
        server_lifetime: float,
        generation_latency: ValueOrRange,
        batch_size: int
) -> ValueOrRange:
    """
    Compute the Abiotic Depletion Potential for Elements (ADPe) embodied impact of the request.

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
        model_quantization_bits: Optional[int] = MODEL_QUANTIZATION_BITS,
        gpu_energy_alpha: Optional[float] = GPU_ENERGY_ALPHA,
        gpu_energy_beta: Optional[float] = GPU_ENERGY_BETA,
        gpu_energy_gamma: Optional[float] = GPU_ENERGY_GAMMA,
        latency_alpha: Optional[float] = LATENCY_ALPHA,
        latency_beta: Optional[float] = LATENCY_BETA,
        latency_gamma: Optional[float] = LATENCY_GAMMA,
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
        batch_size: Optional[float] =  BATCH_SIZE
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
        model_quantization_bits: Number of bits used to represent the model weights.
        gpu_energy_alpha: Alpha coefficient of the "GPU energy" regression.
        gpu_energy_beta: Beta coefficient of the "GPU energy" regression.
        gpu_energy_gamma: Gamma coefficient of the "GPU energy" regression.
        latency_alpha: Alpha coefficient of the "Latency" regression.
        latency_beta: Beta coefficient of the "Latency" regression.
        latency_gamma: Gamma coefficient of the "Latency" regression.
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
        batch_size: The number of requests handled concurrently by the server.
    Returns:
        The environmental impacts dag with all intermediate states.
    """
    results = dag.execute(
        model_active_parameter_count=model_active_parameter_count,
        model_total_parameter_count=model_total_parameter_count,
        model_quantization_bits=model_quantization_bits,
        output_token_count=output_token_count,
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
        batch_size=batch_size
    )
    return results


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
        **kwargs: Any
) -> Impacts:
    """
    Compute the impacts of an LLM generation request.

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
        **kwargs: Any other optional parameter.
    Returns:
        The impacts of an LLM generation request.
    """
    if request_latency is None:
        request_latency = math.inf

    active_params = [model_active_parameter_count]
    total_params = [model_total_parameter_count]

    if isinstance(model_active_parameter_count, RangeValue) or isinstance(model_total_parameter_count, RangeValue):
        if isinstance(model_active_parameter_count, RangeValue):
            active_params = [model_active_parameter_count.min, model_active_parameter_count.max]
        else:
            active_params = [model_active_parameter_count, model_active_parameter_count]
        if isinstance(model_total_parameter_count, RangeValue):
            total_params = [model_total_parameter_count.min, model_total_parameter_count.max]
        else:
            total_params = [model_total_parameter_count, model_total_parameter_count]

    results: dict[str, Union[RangeValue, float, int]] = {}
    fields = ["request_energy", "request_usage_gwp", "request_usage_adpe", "request_usage_pe", "request_usage_wcf",
              "request_embodied_gwp", "request_embodied_adpe", "request_embodied_pe"]
    for act_param, tot_param in zip(active_params, total_params):
        res = compute_llm_impacts_dag(
            model_active_parameter_count=act_param,
            model_total_parameter_count=tot_param,
            output_token_count=output_token_count,
            request_latency=request_latency,
            if_electricity_mix_adpe=if_electricity_mix_adpe,
            if_electricity_mix_pe=if_electricity_mix_pe,
            if_electricity_mix_gwp=if_electricity_mix_gwp,
            if_electricity_mix_wue=if_electricity_mix_wue,
            datacenter_pue=datacenter_pue,
            datacenter_wue=datacenter_wue,
            **kwargs
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
            wcf=wcf_usage
        ),
        embodied=Embodied(
            gwp=gwp_embodied,
            adpe=adpe_embodied,
            pe=pe_embodied
        )
    )
