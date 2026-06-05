from copy import deepcopy

import pytest

from ecologits.estimations import video as video_estimations
from ecologits.impacts.video import HARDWARE_LIFESPAN
from ecologits.status_messages import (
    ElectricityMixADPeDefaultWarning,
    ElectricityMixPEDefaultWarning,
    ElectricityMixWUEDefaultWarning,
)
from ecologits.tracers.utils import ImpactsOutput


def test_impacts_video_generation_uses_number_of_accelerators_from_hardware_configuration(monkeypatch):
    hardware_name = "dgx_h800"
    hardware = deepcopy(video_estimations._HARDWARE_CONFIGURATIONS[hardware_name])
    hardware["number_of_accelerators"] = 4
    hardware["server_power"] = {
        "p2_5": 0.0,
        "mean": 0.0,
        "p97_5": 0.0,
    }
    hardware["server_embodied"] = {
        "gwp": 0.0,
        "adpe": 0.0,
        "pe": 0.0,
    }
    hardware["accelerator_embodied"] = {
        "gwp": 1.0,
        "adpe": 1.0,
        "pe": 1.0,
    }
    monkeypatch.setitem(video_estimations._HARDWARE_CONFIGURATIONS, hardware_name, hardware)

    impacts = video_estimations.video_impacts(
        model_name="klingai/kling-v1.6",
        resolution="1280x720",
        duration=5,
        with_audio=True,
        datacenter_pue=1.0,
        datacenter_wue=0.0,
    )

    expected_generation_latency = 1.52e-03 * (1280 * 720 * 121 / 1000)
    expected_embodied_value = (expected_generation_latency / HARDWARE_LIFESPAN) * 4

    assert impacts.energy.value.min == 0.0
    assert impacts.energy.value.max == 0.0
    assert impacts.embodied.gwp.value == pytest.approx(expected_embodied_value)
    assert impacts.embodied.adpe.value == pytest.approx(expected_embodied_value)
    assert impacts.embodied.pe.value == pytest.approx(expected_embodied_value)


def test_impacts_video_generation_supports_tpu_backed_models():
    impacts = video_estimations.video_impacts(
        model_name="google/veo-3.0",
        resolution="1280x720",
        duration=5,
        with_audio=True,
        datacenter_pue=1.0,
        datacenter_wue=0.0,
    )

    assert impacts.errors is None
    assert impacts.energy.value.min > 0
    assert impacts.energy.value.max > 0
    assert impacts.embodied.gwp.value > 0


def test_impacts_video_generation_supports_h200_open_models(monkeypatch):
    captured = {}

    def fake_compute_video_impacts(**kwargs):
        captured.update(kwargs)
        return ImpactsOutput()

    monkeypatch.setattr(video_estimations, "compute_video_impacts", fake_compute_video_impacts)

    video_estimations.video_impacts(
        model_name="tencent/hunyuanvideo",
        resolution="1280x720",
        duration=5,
        with_audio=True,
    )

    assert captured["n"] == pytest.approx(1.73320916013e-13)
    assert captured["m"] == pytest.approx(1.84153473264e-04)
    assert captured["n1"] == 0
    assert captured["n2"] == pytest.approx(5.19962748039e-09)
    assert captured["g"] == pytest.approx(1.19158129759e-01)
    assert captured["server_accelerator_count"] == 8


def test_impacts_video_generation_uses_h200_open_model_latency_regression():
    impacts = video_estimations.video_impacts(
        model_name="tencent/hunyuanvideo",
        resolution="1280x720",
        duration=5,
        with_audio=True,
        datacenter_pue=1.0,
        datacenter_wue=0.0,
    )

    assert impacts.energy.value.mean == pytest.approx(0.19306527276816782)


def test_impacts_video_generation_supports_single_h200_ltx_models():
    impacts = video_estimations.video_impacts(
        model_name="lightricks/ltx-2-t2v",
        resolution="1024x576",
        duration=5,
        with_audio=False,
        datacenter_pue=1.0,
        datacenter_wue=0.0,
    )

    assert impacts.errors is None
    assert impacts.energy.value.min > 0
    assert impacts.energy.value.max > 0
    assert impacts.embodied.gwp.value > 0


def test_impacts_video_generation_uses_provider_configuration(monkeypatch):
    captured = {}

    def fake_compute_video_impacts(**kwargs):
        captured.update(kwargs)
        return ImpactsOutput()

    monkeypatch.setattr(video_estimations, "compute_video_impacts", fake_compute_video_impacts)

    video_estimations.video_impacts(
        model_name="runway/gen-4.5",
        resolution="1280x720",
        duration=5,
        with_audio=True,
    )

    assert captured["datacenter_pue"].min == pytest.approx(1.09)
    assert captured["datacenter_pue"].max == pytest.approx(1.14)
    assert captured["datacenter_wue"].min == pytest.approx(0.13)
    assert captured["datacenter_wue"].max == pytest.approx(0.999)


def test_impacts_video_generation_uses_datacenter_location_when_provided():
    impacts = video_estimations.video_impacts(
        model_name="runway/gen-4.5",
        resolution="1280x720",
        duration=5,
        with_audio=True,
        datacenter_location="UNKNOWN-ZONE",
    )

    assert impacts.errors[0].message == "Could not find electricity mix for `UNKNOWN-ZONE` zone."


def test_impacts_video_generation_adds_electricity_mix_warnings():
    impacts = video_estimations.video_impacts(
        model_name="runway/gen-4.5",
        resolution="1280x720",
        duration=5,
        with_audio=True,
        datacenter_location="ABW",
    )

    assert impacts.has_warnings
    assert any(isinstance(w, ElectricityMixADPeDefaultWarning) for w in impacts.warnings)
    assert any(isinstance(w, ElectricityMixPEDefaultWarning) for w in impacts.warnings)
    assert any(isinstance(w, ElectricityMixWUEDefaultWarning) for w in impacts.warnings)


def test_video_models_reference_provider_configurations():
    provider_configurations = video_estimations._PROVIDER_CONFIGURATIONS

    for model_info in video_estimations._MODELS_INFO.values():
        assert model_info["provider"] in provider_configurations
