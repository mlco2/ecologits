import pytest

from ecologits.impacts.diffusion import compute_diffusion_impacts
from ecologits.tracers.utils import diffusion_impacts
from ecologits.utils.range_value import RangeValue

_SDXL_PARAMS = dict(
    model_total_parameter_count=2.6,
    flops_per_step=6.7e12,
    n_steps=30,
    guidance_scale=7.5,
    cfg_double_pass=True,
    denoising_strength=1.0,
    n_frames=1,
    default_frames=1,
    if_electricity_mix_gwp=0.29,
    if_electricity_mix_adpe=7.37e-7,
    if_electricity_mix_pe=3.98,
    if_electricity_mix_wue=1.0,
    datacenter_pue=1.2,
    datacenter_wue=0.5,
)

_ENERGY_PLAUSIBLE_MIN = 0.0005
_ENERGY_PLAUSIBLE_MAX = 0.01


def _energy_mean(value):
    if isinstance(value, RangeValue):
        return (value.min + value.max) / 2
    return value


def test_diffusion_impacts_txt2img_sdxl_energy_is_range_value():
    result = compute_diffusion_impacts(**_SDXL_PARAMS)

    assert isinstance(result.energy.value, RangeValue)


def test_diffusion_impacts_txt2img_sdxl_gwp_positive():
    result = compute_diffusion_impacts(**_SDXL_PARAMS)

    gwp = result.gwp.value
    gwp_min = gwp.min if isinstance(gwp, RangeValue) else gwp
    assert gwp_min > 0


def test_diffusion_impacts_txt2img_sdxl_energy_in_plausible_range():
    result = compute_diffusion_impacts(**_SDXL_PARAMS)

    energy = result.energy.value
    assert isinstance(energy, RangeValue)
    assert _ENERGY_PLAUSIBLE_MIN <= energy.min
    assert energy.max <= _ENERGY_PLAUSIBLE_MAX


def test_diffusion_impacts_img2img_sdxl_has_editing_unvalidated_warning():
    result = diffusion_impacts(
        provider="stabilityai",
        model_name="stable-diffusion-xl",
        task="img2img",
        n_steps=30,
        guidance_scale=7.5,
        denoising_strength=0.5,
    )

    assert result.has_warnings
    assert any(w.code == "modality-editing-unvalidated" for w in result.warnings)


def test_diffusion_impacts_img2img_sdxl_energy_approximately_half_of_txt2img():
    txt2img = diffusion_impacts(
        provider="stabilityai",
        model_name="stable-diffusion-xl",
        task="txt2img",
        n_steps=30,
        guidance_scale=7.5,
        denoising_strength=1.0,
    )
    img2img = diffusion_impacts(
        provider="stabilityai",
        model_name="stable-diffusion-xl",
        task="img2img",
        n_steps=30,
        guidance_scale=7.5,
        denoising_strength=0.5,
    )

    txt2img_mean = _energy_mean(txt2img.energy.value)
    img2img_mean = _energy_mean(img2img.energy.value)
    ratio = img2img_mean / txt2img_mean
    assert ratio == pytest.approx(0.5, rel=0.20)
