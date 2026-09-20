from datetime import date

import pytest

from ecologits.impacts.llm import compute_llm_impacts, compute_llm_impacts_dag
from ecologits.impacts.modeling import Training
from ecologits.impacts.training import compute_llm_training_impacts_dag
from ecologits.utils.range_value import RangeValue

WORLD_MIX = {
    "if_electricity_mix_adpe": 0.0000000737708,
    "if_electricity_mix_pe": 9.988,
    "if_electricity_mix_gwp": 0.590478,
    "if_electricity_mix_wue": 5.04,
}
TRAINING_KWARGS = {
    "model_release_date": date(2025, 4, 14),
    "provider_ai_compute_capacity": 2_000_000,
    "provider_active_model_count": 15,
}


def _compute(output_token_count=1000, **kwargs):
    params = {
        "model_active_parameter_count": 76,
        "model_total_parameter_count": 76,
        "output_token_count": output_token_count,
        "request_latency": 100,
        "datacenter_pue": 1.2,
        "datacenter_wue": 0.569,
        **WORLD_MIX,
        **TRAINING_KWARGS,
    }
    params.update(kwargs)
    return compute_llm_impacts(**params)


def test_training_phase_is_reported():
    impacts = _compute()
    assert isinstance(impacts.training, Training)
    assert impacts.training.energy.value > 0
    assert impacts.training.gwp.value > 0
    assert impacts.training.adpe.value > 0
    assert impacts.training.pe.value > 0
    assert impacts.training.wcf.value > 0


def test_training_phase_not_included_in_totals():
    impacts = _compute()
    assert impacts.gwp == impacts.usage.gwp + impacts.embodied.gwp
    assert impacts.energy == impacts.usage.energy


@pytest.mark.parametrize("missing", ["model_release_date", "provider_ai_compute_capacity",
                                     "provider_active_model_count"])
def test_training_phase_requires_all_inputs(missing):
    impacts = _compute(**{missing: None})
    assert impacts.training is None


def test_training_phase_is_proportional_to_output_tokens():
    small = _compute(output_token_count=100)
    big = _compute(output_token_count=1000)
    assert big.training.energy.value == pytest.approx(10 * small.training.energy.value)
    assert big.training.gwp.value > small.training.gwp.value


def test_training_phase_with_zero_output_tokens():
    impacts = _compute(output_token_count=0)
    assert impacts.training.energy.value == 0
    assert impacts.training.gwp.value == 0


def test_training_phase_increases_with_model_size_and_recency():
    base = _compute()
    bigger = _compute(model_active_parameter_count=400, model_total_parameter_count=400)
    newer = _compute(model_release_date=date(2026, 4, 14))
    assert bigger.training.energy.value > base.training.energy.value
    assert newer.training.energy.value > base.training.energy.value


def test_training_phase_decreases_with_provider_capacity():
    base = _compute()
    more_capacity = _compute(provider_ai_compute_capacity=4_000_000)
    more_models = _compute(provider_active_model_count=30)
    assert more_capacity.training.energy.value == pytest.approx(base.training.energy.value / 2)
    assert more_models.training.energy.value == pytest.approx(base.training.energy.value * 2)


def test_training_phase_uses_training_electricity_mix():
    base = _compute()
    cleaner = _compute(if_training_electricity_mix_gwp=0.05)
    assert cleaner.training.gwp.value < base.training.gwp.value
    assert cleaner.training.energy.value == base.training.energy.value
    assert cleaner.usage.gwp.value == base.usage.gwp.value


def test_training_phase_with_range_values():
    impacts = _compute(
        model_active_parameter_count=RangeValue(min=40, max=112),
        model_total_parameter_count=RangeValue(min=40, max=112),
        datacenter_pue=RangeValue(min=1.09, max=1.14),
    )
    assert isinstance(impacts.training.energy.value, RangeValue)
    assert isinstance(impacts.training.gwp.value, RangeValue)
    assert isinstance(impacts.training.wcf.value, RangeValue)
    assert impacts.training.energy.value.min < impacts.training.energy.value.max


def test_training_dag_intermediate_values():
    inference = compute_llm_impacts_dag(
        model_active_parameter_count=76,
        model_total_parameter_count=76,
        output_token_count=1000,
        request_latency=100,
        datacenter_pue=1.2,
        datacenter_wue=0.569,
        **WORLD_MIX,
    )
    results = compute_llm_training_impacts_dag(
        inference_results=inference,
        gpu_power=0.7,
        **WORLD_MIX,
        **TRAINING_KWARGS,
    )
    # Regression on Epoch AI data: 10^(0.0006 * 1930 + 17.151) * (76e9)^0.541
    assert results["training_flops"] == pytest.approx(1.566e24, rel=0.01)
    assert results["infrastructure_overhead_ratio"] > 1
    assert results["rnd_it_energy"] == pytest.approx(4 * results["training_it_energy"])
    assert results["training_hdd_count"] >= 1
    assert results["model_lifetime_output_token_count"] > 0
    assert results["request_training_energy"] == pytest.approx(results["request_training_it_energy"] * 1.2)


def test_training_dag_override_intermediate_value():
    base = _compute()
    overridden = _compute(model_lifetime_output_token_count=1e12)
    assert overridden.training.energy.value != base.training.energy.value
