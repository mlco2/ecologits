"""Regression tests for WCF PUE double-counting bug (#230)."""
import pytest
from ecologits.impacts.llm import compute_llm_impacts


def _base_kwargs(**overrides):
    defaults = dict(
        model_active_parameter_count=7.0,
        model_total_parameter_count=7.0,
        output_token_count=100,
        if_electricity_mix_adpe=1e-8,
        if_electricity_mix_pe=9.0,
        if_electricity_mix_gwp=0.4,
        if_electricity_mix_wue=0.5,
        datacenter_pue=1.2,
        datacenter_wue=1.8,
        request_latency=2.0,
    )
    defaults.update(overrides)
    return defaults


def test_wcf_does_not_scale_with_pue_squared():
    """WCF must not grow with PUE^2; doubling PUE should not quadruple WCF."""
    impacts_low = compute_llm_impacts(**_base_kwargs(datacenter_pue=1.0))
    impacts_high = compute_llm_impacts(**_base_kwargs(datacenter_pue=2.0))

    wcf_low = impacts_low.wcf.value
    wcf_high = impacts_high.wcf.value

    # If PUE were double-counted the ratio would be ~4x; correct formula keeps it sub-linear.
    ratio = wcf_high / wcf_low if isinstance(wcf_high, (int, float)) else wcf_high.mean / wcf_low.mean
    assert ratio < 4.0, f"WCF ratio {ratio:.2f} suggests PUE is being squared"


def test_wcf_zero_wue_equals_elec_mix_contribution_only():
    """With datacenter_wue=0, WCF should equal request_energy * if_electricity_mix_wue."""
    impacts = compute_llm_impacts(**_base_kwargs(datacenter_wue=0.0, if_electricity_mix_wue=1.0))
    wcf = impacts.wcf.value
    energy = impacts.energy.value
    wcf_val = wcf.mean if hasattr(wcf, "mean") else wcf
    energy_val = energy.mean if hasattr(energy, "mean") else energy
    assert abs(wcf_val - energy_val) < 1e-9, "WCF with zero datacenter WUE should equal energy * mix WUE"


def test_wcf_positive():
    """WCF should always be positive."""
    impacts = compute_llm_impacts(**_base_kwargs())
    wcf = impacts.wcf.value
    val = wcf.min if hasattr(wcf, "min") else wcf
    assert val > 0
