from operator import ge, gt

import numpy as np
import pytest

from ecologits.impacts.llm import compute_llm_impacts
from ecologits.impacts.modeling import Impacts, Energy, GWP, ADPe, PE, WCF, Usage, Embodied


@pytest.mark.parametrize(
    ['model_active_parameter_count', 'model_total_parameter_count', 'output_token_count', 'request_latency',
     'if_electricity_mix_adpe', 'if_electricity_mix_pe', 'if_electricity_mix_gwp', 'if_electricity_mix_wue',
     'datacenter_pue', 'datacenter_wue'],
    [
        (7.3, 7.3, 200, 5, 0.0000000737708, 9.988, 0.590478, 5.04, 1.26, 0.37),  # Mistral 7B with World mix
        (12.9, 46.7, 200, 10, 0.0000000737708, 9.988, 0.590478, 5.04, 1.26, 0.37)  # Mixtral 8x7B with World mix
    ]
)
def test_compute_llm_impacts(model_active_parameter_count: float,
                             model_total_parameter_count: float,
                             output_token_count: int,
                             request_latency: float,
                             if_electricity_mix_adpe: float,
                             if_electricity_mix_pe: float,
                             if_electricity_mix_gwp: float,
                             if_electricity_mix_wue: float,
                             datacenter_pue: float,
                             datacenter_wue: float) -> None:
    impacts = compute_llm_impacts(
        model_active_parameter_count=model_active_parameter_count,
        model_total_parameter_count=model_total_parameter_count,
        output_token_count=output_token_count,
        request_latency=request_latency,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue
    )
    assert impacts.energy.value > 0
    assert impacts.gwp.value > 0
    assert impacts.adpe.value > 0
    assert impacts.pe.value > 0
    assert impacts.wcf.value > 0
    assert impacts.usage.energy.value > 0
    assert impacts.usage.gwp.value > 0
    assert impacts.usage.adpe.value > 0
    assert impacts.usage.pe.value > 0
    assert impacts.usage.wcf.value > 0
    assert impacts.embodied.gwp.value > 0
    assert impacts.embodied.adpe.value > 0
    assert impacts.embodied.pe.value > 0


def compare_impacts(impacts: Impacts, prev_impacts: Impacts, op=gt):
    assert op(impacts.energy, prev_impacts.energy)
    assert op(impacts.gwp, prev_impacts.gwp)
    assert op(impacts.adpe, prev_impacts.adpe)
    assert op(impacts.pe, prev_impacts.pe)
    assert op(impacts.wcf, prev_impacts.wcf)
    assert op(impacts.usage.energy, prev_impacts.usage.energy)
    assert op(impacts.usage.gwp, prev_impacts.usage.gwp)
    assert op(impacts.usage.adpe, prev_impacts.usage.adpe)
    assert op(impacts.usage.pe, prev_impacts.usage.pe)
    assert op(impacts.embodied.gwp, prev_impacts.embodied.gwp)
    assert op(impacts.embodied.adpe, prev_impacts.embodied.adpe)
    assert op(impacts.embodied.pe, prev_impacts.embodied.pe)


@pytest.mark.parametrize(
    ['if_electricity_mix_adpe', 'if_electricity_mix_pe', 'if_electricity_mix_gwp', 'if_electricity_mix_wue',
     'datacenter_pue', 'datacenter_wue'],
    [
        (0.0000000737708, 9.988, 0.590478, 5.04, 1.26, 0.37),  # Mistral 7B with World mix
    ]
)
def test_compute_llm_impacts_monotonicity_on_parameters(if_electricity_mix_adpe: float,
                                                        if_electricity_mix_pe: float,
                                                        if_electricity_mix_gwp: float,
                                                        if_electricity_mix_wue: float,
                                                        datacenter_pue: float,
                                                        datacenter_wue: float):
    zero_impacts = Impacts(
        energy=Energy(value=0),
        gwp=GWP(value=0),
        adpe=ADPe(value=0),
        pe=PE(value=0),
        wcf=WCF(value=0),
        usage=Usage(
            energy=Energy(value=0),
            gwp=GWP(value=0),
            adpe=ADPe(value=0),
            pe=PE(value=0),
            wcf=WCF(value=0)
        ),
        embodied=Embodied(
            gwp=GWP(value=0),
            adpe=ADPe(value=0),
            pe=PE(value=0),
        )
    )
    prev_impacts = zero_impacts.model_copy(deep=True)

    for total_parameters in np.logspace(-4, 4, num=20):
        impacts = compute_llm_impacts(
            model_active_parameter_count=total_parameters,
            model_total_parameter_count=total_parameters,
            output_token_count=100,
            if_electricity_mix_adpe=if_electricity_mix_adpe,
            if_electricity_mix_pe=if_electricity_mix_pe,
            if_electricity_mix_gwp=if_electricity_mix_gwp,
            if_electricity_mix_wue=if_electricity_mix_wue,
            datacenter_pue=datacenter_pue,
            datacenter_wue=datacenter_wue
        )

        compare_impacts(impacts, prev_impacts, op=gt)
        prev_impacts = impacts.model_copy(deep=True)

        prev_impacts_moe = zero_impacts.model_copy(deep=True)
        for active_parameters in np.linspace(start=total_parameters / 10, stop=total_parameters, num=10):
            impacts_moe = compute_llm_impacts(
                model_active_parameter_count=active_parameters,
                model_total_parameter_count=total_parameters,
                output_token_count=100,
                if_electricity_mix_adpe=if_electricity_mix_adpe,
                if_electricity_mix_pe=if_electricity_mix_pe,
                if_electricity_mix_gwp=if_electricity_mix_gwp,
                if_electricity_mix_wue=if_electricity_mix_wue,
                datacenter_pue=datacenter_pue,
                datacenter_wue=datacenter_wue
            )

            compare_impacts(impacts_moe, prev_impacts_moe, op=gt)
            compare_impacts(impacts, impacts_moe, op=ge)
            prev_impacts_moe = impacts_moe.model_copy(deep=True)


def test_compute_llm_impacts_with_throughput_and_latency():
    other_params = dict(
        model_active_parameter_count=20,
        model_total_parameter_count=20,
        output_token_count=100,
        if_electricity_mix_adpe=0.0000000737708,
        if_electricity_mix_pe=9.988,
        if_electricity_mix_gwp=0.590478,
        if_electricity_mix_wue=5.04,
        datacenter_pue=1.26,
        datacenter_wue=0.37,
    )

    impacts = compute_llm_impacts(**other_params)           # Biggest estimated latency by default
    impacts_with_request_latency = compute_llm_impacts(     # Forced low request latency
        request_latency=1,
        **other_params
    )
    impacts_fast_tps_low_lat = compute_llm_impacts(         # Fast generation, low ttft
        tps=1000,
        ttft=0.1,
        **other_params
    )
    impacts_fast_tps_high_lat = compute_llm_impacts(        # Fast generation, high ttft
        tps=1000,
        ttft=100,
        **other_params
    )
    assert impacts_fast_tps_high_lat.energy.value \
           > impacts.energy.value \
           > impacts_with_request_latency.energy.value \
           > impacts_fast_tps_low_lat.energy.value
