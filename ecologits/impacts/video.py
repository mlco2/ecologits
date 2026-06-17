import math
from typing import Any, Optional

from ecologits.impacts.dag import DAG
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Impacts, Usage
from ecologits.utils.range_value import ValueOrRange

dag = DAG()

SERVER_ACCELERATORS = 8
HARDWARE_LIFESPAN = 3 * 365 * 24 * 60 * 60


@dag.asset
def generation_latency(
        video_width: int,
        video_height: int,
        video_frames_count: int,
        n: float,
        m: float,
        n1: float,
        m1: float,
        n2: float,
        m2: float,
        g: float,
        non_audio_weight: float,
        request_latency: float,
) -> float:
    """
    Compute the generation latency.

    Args:
        video_width: Width of the frames.
        video_height: Height of the frames
        video_frames_count: Number of frames.
        n: N coefficient of the latency regression.
        m: M coefficient of the latency regression.
        n1: N1 coefficient of the latency regression.
        m1: M1 coefficient of the latency regression.
        n2: N2 coefficient of the latency regression.
        m2: M2 coefficient of the latency regression.
        g: G coefficient of the latency regression.
        non_audio_weight: Proportional difference in generation latency without audio <1 or with audio =1..
        request_latency: Measured request latency in seconds.
    """
    width_height = video_width * video_height
    spaciotemporal_volume = width_height * video_frames_count / 1000
    estimated_latency = (
        n * video_frames_count * width_height ** 2
        + m * spaciotemporal_volume
        + m1 * video_frames_count
        + m2 * width_height ** 2
        + n1 * video_frames_count ** 2
        + n2 * spaciotemporal_volume ** 2
        + g
    ) * non_audio_weight
    if request_latency < estimated_latency:
        return request_latency
    return estimated_latency


@dag.asset
def server_accelerator_energy(
        generation_latency: float,
        server_accelerator_power: ValueOrRange,
) -> ValueOrRange:
    return (generation_latency / 3600) * server_accelerator_power


@dag.asset
def request_energy(
        datacenter_pue: float,
        server_accelerator_energy: ValueOrRange,
) -> ValueOrRange:
    return datacenter_pue * server_accelerator_energy


@dag.asset
def request_usage_gwp(
        request_energy: ValueOrRange,
        if_electricity_mix_gwp: float,
) -> ValueOrRange:
    return request_energy * if_electricity_mix_gwp


@dag.asset
def request_usage_adpe(
        request_energy: ValueOrRange,
        if_electricity_mix_adpe: float,
) -> ValueOrRange:
    return request_energy * if_electricity_mix_adpe


@dag.asset
def request_usage_pe(
        request_energy: ValueOrRange,
        if_electricity_mix_pe: float,
) -> ValueOrRange:
    return request_energy * if_electricity_mix_pe


@dag.asset
def request_usage_wcf(
        server_accelerator_energy: ValueOrRange,
        if_electricity_mix_wue: float,
        datacenter_wue: float,
        datacenter_pue: float,
) -> ValueOrRange:
    return server_accelerator_energy * (datacenter_wue + datacenter_pue * if_electricity_mix_wue)


@dag.asset
def server_accelerator_embodied_gwp(
        server_embodied_gwp: float,
        server_accelerator_count: float,
        accelerator_embodied_gwp: float,
) -> float:
    return server_embodied_gwp + server_accelerator_count * accelerator_embodied_gwp


@dag.asset
def server_accelerator_embodied_adpe(
        server_embodied_adpe: float,
        server_accelerator_count: float,
        accelerator_embodied_adpe: float,
) -> float:
    return server_embodied_adpe + server_accelerator_count * accelerator_embodied_adpe


@dag.asset
def server_accelerator_embodied_pe(
        server_embodied_pe: float,
        server_accelerator_count: float,
        accelerator_embodied_pe: float,
) -> float:
    return server_embodied_pe + server_accelerator_count * accelerator_embodied_pe


@dag.asset
def request_embodied_gwp(
        server_accelerator_embodied_gwp: float,
        server_lifetime: float,
        generation_latency: float,
) -> float:
    return (generation_latency / server_lifetime) * server_accelerator_embodied_gwp


@dag.asset
def request_embodied_adpe(
        server_accelerator_embodied_adpe: float,
        server_lifetime: float,
        generation_latency: float,
) -> float:
    return (generation_latency / server_lifetime) * server_accelerator_embodied_adpe


@dag.asset
def request_embodied_pe(
        server_accelerator_embodied_pe: float,
        server_lifetime: float,
        generation_latency: float,
) -> float:
    return (generation_latency / server_lifetime) * server_accelerator_embodied_pe


def compute_video_impacts_dag(
        video_width: int,
        video_height: int,
        video_frames_count: int,
        request_latency: float,
        server_accelerator_power: float | ValueOrRange,
        if_electricity_mix_adpe: float,
        if_electricity_mix_pe: float,
        if_electricity_mix_gwp: float,
        if_electricity_mix_wue: float,
        datacenter_pue: ValueOrRange,
        datacenter_wue: ValueOrRange,
        accelerator_embodied_gwp: float,
        accelerator_embodied_adpe: float,
        accelerator_embodied_pe: float,
        server_embodied_gwp: float,
        server_embodied_adpe: float,
        server_embodied_pe: float,
        n: float = 0,
        m: float = 0,
        n1: float = 0,
        m1: float = 0,
        n2: float = 0,
        m2: float = 0,
        g: float = 0,
        non_audio_weight: float = 1,
        server_accelerator_count: int = SERVER_ACCELERATORS,
        server_lifetime: float = HARDWARE_LIFESPAN,
) -> dict[str, Any]:
    results = dag.execute(
        video_width=video_width,
        video_height=video_height,
        video_frames_count=video_frames_count,
        request_latency=request_latency,
        server_accelerator_power=server_accelerator_power,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        accelerator_embodied_gwp=accelerator_embodied_gwp,
        accelerator_embodied_adpe=accelerator_embodied_adpe,
        accelerator_embodied_pe=accelerator_embodied_pe,
        server_embodied_gwp=server_embodied_gwp,
        server_embodied_adpe=server_embodied_adpe,
        server_embodied_pe=server_embodied_pe,
        n=n,
        m=m,
        n1=n1,
        m1=m1,
        n2=n2,
        m2=m2,
        g=g,
        non_audio_weight=non_audio_weight,
        server_accelerator_count=server_accelerator_count,
        server_lifetime=server_lifetime,
    )
    return results


def compute_video_impacts(
        video_width: int,
        video_height: int,
        video_frames_count: int,
        server_accelerator_power: float | ValueOrRange,
        if_electricity_mix_adpe: float,
        if_electricity_mix_pe: float,
        if_electricity_mix_gwp: float,
        if_electricity_mix_wue: float,
        datacenter_pue: ValueOrRange,
        datacenter_wue: ValueOrRange,
        accelerator_embodied_gwp: float,
        accelerator_embodied_adpe: float,
        accelerator_embodied_pe: float,
        server_embodied_gwp: float,
        server_embodied_adpe: float,
        server_embodied_pe: float,
        n: float = 0,
        m: float = 0,
        n1: float = 0,
        m1: float = 0,
        n2: float = 0,
        m2: float = 0,
        g: float = 0,
        non_audio_weight: float = 1,
        request_latency: Optional[float] = None,
        **kwargs: Any,
) -> Impacts:
    """
    Compute the environmental impacts of a video generation request.

    Args:
        video_width: Width of the frames.
        video_height: Height of the frames.
        video_frames_count: Number of frames.
        server_accelerator_power: Power consumption of the server (with accelerators) in W.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh.
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        accelerator_embodied_gwp: GWP embodied impact of a single accelerator in kgCO2eq.
        accelerator_embodied_adpe: ADPe embodied impact of a single accelerator in kgSbeq.
        accelerator_embodied_pe: PE embodied impact of a single accelerator in MJ.
        server_embodied_gwp: GWP embodied impact of the server in kgCO2eq.
        server_embodied_adpe: ADPe embodied impact of the server in kgSbeq.
        server_embodied_pe: PE embodied impact of the server in MJ.
        n: N coefficient of the latency regression.
        m: M coefficient of the latency regression.
        n1: N1 coefficient of the latency regression.
        m1: M1 coefficient of the latency regression.
        n2: N2 coefficient of the latency regression.
        m2: M2 coefficient of the latency regression.
        g: Constant coefficient of the latency regression.
        non_audio_weight: Latency scaling factor when generating without audio.
        request_latency: Measured request latency in seconds (optional).

    Returns:
        The impacts of the video generation request.
    """
    if request_latency is None:
        request_latency = math.inf

    results = compute_video_impacts_dag(
        video_width=video_width,
        video_height=video_height,
        video_frames_count=video_frames_count,
        request_latency=request_latency,
        server_accelerator_power=server_accelerator_power,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        accelerator_embodied_gwp=accelerator_embodied_gwp,
        accelerator_embodied_adpe=accelerator_embodied_adpe,
        accelerator_embodied_pe=accelerator_embodied_pe,
        server_embodied_gwp=server_embodied_gwp,
        server_embodied_adpe=server_embodied_adpe,
        server_embodied_pe=server_embodied_pe,
        n=n,
        m=m,
        n1=n1,
        m1=m1,
        n2=n2,
        m2=m2,
        g=g,
        non_audio_weight=non_audio_weight,
        **kwargs
    )

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
