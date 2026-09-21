import pytest

from ecologits.impacts.llm import (
    DATACENTER_BUILDING_EMBODIED_GWP_FACTOR,
    NETWORK_EMBODIED_IMPACT_GWP,
    NETWORK_POWER,
    PREFILL_LATENCY_INTERCEPT,
    compute_llm_impacts,
    compute_llm_impacts_dag,
)

WORLD_MIX = {
    "if_electricity_mix_adpe": 0.0000000737708,
    "if_electricity_mix_pe": 9.988,
    "if_electricity_mix_gwp": 0.590478,
    "if_electricity_mix_wue": 5.04,
}


def _dag(**kwargs):
    params = {
        "model_active_parameter_count": 76,
        "model_total_parameter_count": 76,
        "output_token_count": 1000,
        "request_latency": 1e9,
        "datacenter_pue": 1.2,
        "datacenter_wue": 0.569,
        "tps": 40,
        "ttft": 0.62,
        **WORLD_MIX,
    }
    params.update(kwargs)
    return compute_llm_impacts_dag(**params)


def test_network_constants():
    assert NETWORK_POWER == pytest.approx(0.16108, rel=1e-4)
    assert NETWORK_EMBODIED_IMPACT_GWP == pytest.approx(660.1, rel=0.001)


def test_prefill_latency_without_input_tokens_uses_ttft():
    results = _dag(input_token_count=None)
    assert results["prefill_latency"] == 0.62


def test_prefill_latency_without_input_tokens_and_ttft():
    results = _dag(input_token_count=None, ttft=None)
    assert results["prefill_latency"] == 0


def test_prefill_latency_scales_with_input_tokens():
    small = _dag(input_token_count=100)
    big = _dag(input_token_count=10_000)
    assert small["prefill_latency"] > PREFILL_LATENCY_INTERCEPT
    assert big["prefill_latency"] > small["prefill_latency"]
    # Coefficient of Mistral 7B scaled by the TTFT ratio: 0.062ms * (0.62 / 0.44) per token
    assert big["prefill_latency"] == pytest.approx(0.062e-3 * 0.62 / 0.44 * 10_000 + PREFILL_LATENCY_INTERCEPT)
    assert big["generation_latency"] > small["generation_latency"]


def test_prefill_latency_scales_with_parameters_without_ttft():
    small = _dag(input_token_count=1000, ttft=None, model_active_parameter_count=7.3)
    big = _dag(input_token_count=1000, ttft=None, model_active_parameter_count=73)
    assert big["prefill_latency"] == pytest.approx(10 * (small["prefill_latency"] - PREFILL_LATENCY_INTERCEPT)
                                                   + PREFILL_LATENCY_INTERCEPT)


def test_network_energy_and_embodied():
    results = _dag(input_token_count=1000)
    assert results["network_energy"] > 0
    assert results["network_energy"] == pytest.approx(results["server_energy"] * NETWORK_POWER / 1.2)
    assert results["request_it_energy"] == pytest.approx(
        results["server_energy"] + results["gpu_required_count"] * results["gpu_energy"] + results["network_energy"]
    )
    assert results["request_network_embodied_gwp"] > 0
    assert results["request_datacenter_embodied_gwp"] == pytest.approx(
        results["request_it_energy"] * DATACENTER_BUILDING_EMBODIED_GWP_FACTOR
    )
    assert results["request_embodied_gwp"] == pytest.approx(
        results["request_hardware_embodied_gwp"]
        + results["request_network_embodied_gwp"]
        + results["request_datacenter_embodied_gwp"]
    )


def test_network_and_building_can_be_disabled():
    results = _dag(network_power=0, network_embodied_gwp=0, datacenter_building_embodied_gwp_factor=0)
    assert results["network_energy"] == 0
    assert results["request_embodied_gwp"] == results["request_hardware_embodied_gwp"]


def test_compute_llm_impacts_with_input_tokens():
    params = {
        "model_active_parameter_count": 76,
        "model_total_parameter_count": 76,
        "output_token_count": 1000,
        "request_latency": 100,
        "datacenter_pue": 1.2,
        "datacenter_wue": 0.569,
        "tps": 40,
        "ttft": 0.62,
        **WORLD_MIX,
    }
    short = compute_llm_impacts(input_token_count=100, **params)
    long = compute_llm_impacts(input_token_count=100_000, **params)
    unknown = compute_llm_impacts(**params)
    assert long.energy.value > short.energy.value
    assert long.embodied.gwp.value > short.embodied.gwp.value
    assert unknown.training is None
    assert long.gwp == long.usage.gwp + long.embodied.gwp
