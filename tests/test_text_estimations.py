import math

import pytest

from ecologits.estimations import text_impacts
from ecologits.model_repository import models
from ecologits.status_messages import (
    ElectricityMixADPeDefaultWarning,
    ElectricityMixPEDefaultWarning,
    ElectricityMixWUEDefaultWarning,
    ModelNotRegisteredError,
    ZoneNotRegisteredError,
)
from ecologits.tracers.utils import ImpactsOutput, llm_impacts
from ecologits.utils.range_value import RangeValue


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("openai", "gpt-4.1-nano"),
        ("mistralai", "mistral-small-latest"),
        ("anthropic", "claude-haiku-4-5"),
        ("google_genai", "gemini-2.5-flash"),
        ("cohere", "command-r-08-2024"),
        ("huggingface_hub", "mistralai/Mistral-7B-v0.1"),
        ("huggingface_hub", "mistralai/Mixtral-8x7B-v0.1"),
    ],
)
def test_text_impacts_matches_tracer_estimation(provider, model_name):
    namespaces = {"google_genai": "google"}
    model_id = model_name if provider == "huggingface_hub" else f"{namespaces.get(provider, provider)}/{model_name}"
    impacts = text_impacts(model_id, provider, 100, datacenter_location=None)
    expected = llm_impacts(provider, model_name, 100, request_latency=math.inf)

    assert isinstance(impacts, ImpactsOutput)
    assert not impacts.has_errors
    assert impacts == expected
    for field in ("energy", "gwp", "adpe", "pe", "wcf"):
        value = getattr(impacts, field).value
        minimum = value.min if isinstance(value, RangeValue) else value
        assert math.isfinite(minimum)
        assert minimum > 0


def test_text_impacts_uses_world_mix_by_default():
    default = text_impacts("openai/gpt-4.1-nano", "openai", 100)
    world = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_location="WOR")
    provider = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_location=None)
    usa = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_location="USA")

    assert default == world
    assert provider == usa
    assert default.energy == provider.energy
    assert default.usage.gwp != provider.usage.gwp


@pytest.mark.parametrize("pue", [2.4, RangeValue(min=2.4, max=3.6)])
def test_text_impacts_datacenter_overrides(pue):
    baseline = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_pue=1.2)
    overridden = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_pue=pue, datacenter_wue=0)

    assert overridden.energy.value.min == pytest.approx(baseline.energy.value.min * 2)
    max_multiplier = 3 if isinstance(pue, RangeValue) else 2
    assert overridden.energy.value.max == pytest.approx(baseline.energy.value.max * max_multiplier)
    assert overridden.embodied == baseline.embodied


def test_text_impacts_zero_wue_and_range_wue():
    dry = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_wue=0)
    wet = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_wue=RangeValue(min=1, max=2))

    assert wet.energy == dry.energy
    assert wet.gwp == dry.gwp
    assert wet.wcf.value.min > dry.wcf.value.min
    assert wet.wcf.value.max > dry.wcf.value.max


@pytest.mark.parametrize("tokens", [0, 200])
def test_text_impacts_scales_with_output_tokens(tokens):
    kwargs = {"model_name": "mistralai/Mistral-7B-v0.1", "provider": "huggingface_hub"}
    baseline = text_impacts(**kwargs, output_tokens=100)
    impacts = text_impacts(**kwargs, output_tokens=tokens)

    for field in ("energy", "gwp", "adpe", "pe", "wcf"):
        actual = getattr(impacts, field).value
        expected = getattr(baseline, field).value * tokens / 100
        if isinstance(expected, RangeValue):
            assert actual.min == pytest.approx(expected.min)
            assert actual.max == pytest.approx(expected.max)
        else:
            assert actual == pytest.approx(expected)


def test_text_impacts_uses_registered_throughput(monkeypatch):
    from ecologits.model_repository import Deployment

    model = models.find_model("huggingface_hub", "mistralai/Mistral-7B-v0.1")
    monkeypatch.setattr(model, "deployment", Deployment(tps=1000, ttft=0.1))
    fast = text_impacts(model.name, "huggingface_hub", 100)
    monkeypatch.setattr(model, "deployment", Deployment(tps=10, ttft=1))
    slow = text_impacts(model.name, "huggingface_hub", 100)

    assert slow.energy.value > fast.energy.value
    assert slow.embodied.gwp.value > fast.embodied.gwp.value


@pytest.mark.parametrize("tokens", [-1, 1.5, True, "100", None])
def test_text_impacts_rejects_invalid_token_counts(tokens):
    with pytest.raises(ValueError, match="output_tokens must be a non-negative integer"):
        text_impacts("openai/gpt-4.1-nano", "openai", tokens)


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("openai", "openai/unknown-model"),
        ("azure", "openai/gpt-4.1-nano"),
        ("aws", "openai/gpt-4.1-nano"),
        ("openai", "anthropic/gpt-4.1-nano"),
        ("openai", "gpt-4.1-nano"),
    ],
)
def test_text_impacts_unknown_model_or_provider(provider, model_name):
    impacts = text_impacts(model_name, provider, 100)

    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ModelNotRegisteredError)
    assert model_name in impacts.errors[0].message
    assert impacts.energy is None


def test_text_impacts_unknown_zone():
    impacts = text_impacts("openai/gpt-4.1-nano", "openai", 100, datacenter_location="UNKNOWN")

    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ZoneNotRegisteredError)
    assert impacts.energy is None


def test_text_impacts_preserves_model_and_electricity_warnings():
    model = models.find_model("openai", "gpt-4.1-nano")
    impacts = text_impacts(f"openai/{model.name}", "openai", 100, datacenter_location="ABW")

    assert not impacts.has_errors
    assert all(warning in impacts.warnings for warning in model.warnings)
    for warning_type in (ElectricityMixADPeDefaultWarning, ElectricityMixPEDefaultWarning,
                         ElectricityMixWUEDefaultWarning):
        assert any(isinstance(warning, warning_type) for warning in impacts.warnings)
