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
from ecologits.tracers.utils import ImpactsOutput, _llm_impacts
from ecologits.utils.range_value import RangeValue


@pytest.mark.parametrize(
    ("provider", "internal_provider", "model_name"),
    [
        ("openai", "openai", "gpt-6-luna"),
        ("mistralai", "mistralai", "mistral-small-latest"),
        ("anthropic", "anthropic", "claude-haiku-4-5"),
        ("google", "google_genai", "gemini-2.5-flash"),
        ("cohere", "cohere", "command-r-08-2024"),
    ],
)
def test_text_impacts_matches_tracer_estimation(provider, internal_provider, model_name):
    impacts = text_impacts(f"{provider}/{model_name}", 1000, 100, provider=provider, datacenter_location="WOR")
    expected = _llm_impacts(internal_provider, model_name, 100, request_latency=math.inf, electricity_mix_zone="WOR")

    assert isinstance(impacts, ImpactsOutput)
    assert not impacts.has_errors
    assert impacts == expected
    for field in ("energy", "gwp", "adpe", "pe", "wcf"):
        value = getattr(impacts, field).value
        minimum = value.min if isinstance(value, RangeValue) else value
        assert math.isfinite(minimum)
        assert minimum > 0


@pytest.mark.parametrize(
    ("model_name", "provider", "location"),
    [
        ("openai/gpt-6-luna", "openai", "USA"),
        ("mistralai/mistral-small-latest", "mistralai", "SWE"),
    ],
)
def test_text_impacts_uses_provider_location_by_default(model_name, provider, location):
    default = text_impacts(model_name, 1000, 100, provider=provider)
    world = text_impacts(model_name, 1000, 100, provider=provider, datacenter_location="WOR")
    unspecified = text_impacts(model_name, 1000, 100, provider=provider, datacenter_location=None)
    explicit = text_impacts(model_name, 1000, 100, provider=provider, datacenter_location=location)

    assert not default.has_errors
    assert default == unspecified == explicit == text_impacts(model_name, 1000, 100)
    assert default.energy == world.energy
    assert default.usage.gwp != world.usage.gwp


@pytest.mark.parametrize(
    ("model_name", "provider", "internal_provider", "registered_name"),
    [
        ("openai/gpt-6-luna", "openai", "openai", "gpt-6-luna"),
        ("anthropic/claude-haiku-4-5", "anthropic", "anthropic", "claude-haiku-4-5"),
        ("cohere/command-r-08-2024", "cohere", "cohere", "command-r-08-2024"),
        ("google/gemini-2.5-flash", "google", "google_genai", "gemini-2.5-flash"),
        ("mistralai/mistral-small-latest", "mistralai", "mistralai", "mistral-small-latest"),
    ],
)
def test_default_provider_is_developer_api(model_name, provider, internal_provider, registered_name):
    default = text_impacts(model_name, 1000, 100)
    expected = _llm_impacts(internal_provider, registered_name, 100, request_latency=math.inf)

    assert not default.has_errors
    assert default == text_impacts(model_name, 1000, 100, provider=provider) == expected


@pytest.mark.parametrize(
    ("developer_id", "api_name"),
    [
        ("deepseek/deepseek-v4.1-flash-1017", "deepseek-v4-1-flash-1017"),
        ("zai/glm-5.2", "zai-glm-5-2"),
    ],
)
def test_third_party_model_on_mistral(developer_id, api_name):
    impacts = text_impacts(developer_id, 1000, 100, provider="mistralai")
    expected = _llm_impacts("mistralai", api_name, 100, request_latency=math.inf)
    assert not impacts.has_errors
    assert impacts == expected
    assert text_impacts(developer_id, 1000, 100).has_errors


def test_model_without_default_requires_provider():
    name = "zai/glm-5.2"
    error = text_impacts(name, 1000, 100).errors[0]
    assert isinstance(error, ModelNotRegisteredError)
    assert "provider" in error.message
    assert not text_impacts(name, 1000, 100, provider="mistralai").has_errors


@pytest.mark.parametrize("pue", [2.4, RangeValue(min=2.4, max=3.6)])
def test_text_impacts_datacenter_overrides(pue):
    baseline = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_pue=1.2)
    overridden = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_pue=pue, datacenter_wue=0)

    assert overridden.energy.value.min == pytest.approx(baseline.energy.value.min * 2)
    max_multiplier = 3 if isinstance(pue, RangeValue) else 2
    assert overridden.energy.value.max == pytest.approx(baseline.energy.value.max * max_multiplier)
    assert overridden.embodied == baseline.embodied


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("datacenter_pue", 0.99),
        ("datacenter_wue", -1),
        ("datacenter_pue", True),
        ("datacenter_wue", math.nan),
        ("datacenter_pue", math.inf),
        ("datacenter_pue", RangeValue(min=0.9, max=1.2)),
        ("datacenter_wue", RangeValue(min=0, max=math.inf)),
    ],
)
def test_text_impacts_rejects_invalid_datacenter_override(name, value):
    with pytest.raises(ValueError, match=name):
        text_impacts("openai/gpt-6-luna", 1000, 100, **{name: value})


def test_text_impacts_accepts_minimum_datacenter_overrides():
    impacts = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_pue=1, datacenter_wue=0)
    assert not impacts.has_errors


def test_text_impacts_zero_wue_and_range_wue():
    dry = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_wue=0)
    wet = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_wue=RangeValue(min=1, max=2))

    assert wet.energy == dry.energy
    assert wet.gwp == dry.gwp
    assert wet.wcf.value.min > dry.wcf.value.min
    assert wet.wcf.value.max > dry.wcf.value.max


@pytest.mark.parametrize("tokens", [0, 200])
def test_text_impacts_scales_with_output_tokens(tokens, monkeypatch):
    # Without registered throughput, the latency regression scales linearly with output tokens.
    monkeypatch.setattr(models.find_model("openai", "gpt-6-luna"), "deployment", None)
    kwargs = {"model_name": "openai/gpt-6-luna", "input_tokens": 1000}
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

    model = models.find_model("openai", "gpt-6-luna")
    monkeypatch.setattr(model, "deployment", Deployment(tps=1000, ttft=0.1))
    fast = text_impacts("openai/gpt-6-luna", 1000, 100)
    monkeypatch.setattr(model, "deployment", Deployment(tps=10, ttft=1))
    slow = text_impacts("openai/gpt-6-luna", 1000, 100)

    assert slow.energy.value > fast.energy.value
    assert slow.embodied.gwp.value > fast.embodied.gwp.value


@pytest.mark.parametrize("name", ["input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"])
@pytest.mark.parametrize("tokens", [-1, 1.5, True, False, "100"])
def test_text_impacts_rejects_invalid_token_counts(name, tokens):
    kwargs = {"input_tokens": 1000, "output_tokens": 100, name: tokens}
    with pytest.raises(ValueError, match=f"{name} must be a non-negative integer"):
        text_impacts("openai/gpt-6-luna", **kwargs)


@pytest.mark.parametrize("name", ["input_tokens", "output_tokens"])
def test_text_impacts_rejects_missing_required_token_counts(name):
    kwargs = {"input_tokens": 1000, "output_tokens": 100}
    del kwargs[name]
    with pytest.raises(TypeError, match=name):
        text_impacts("openai/gpt-6-luna", **kwargs)

    kwargs[name] = None
    with pytest.raises(ValueError, match=f"{name} must be a non-negative integer"):
        text_impacts("openai/gpt-6-luna", **kwargs)


def test_text_impacts_cache_counts_require_keywords():
    with pytest.raises(TypeError):
        text_impacts("openai/gpt-6-luna", 1000, 100, 800, 150)


@pytest.mark.parametrize(
    ("input_tokens", "cache_read_tokens", "cache_write_tokens"),
    [
        (0, None, None),
        (0, 0, 0),
        (1000, None, 0),
        (1000, 0, None),
        (1000, 1000, None),
        (1000, None, 1000),
        (1000, 800, 150),
        (1000, 800, 200),
    ],
)
def test_text_impacts_accepts_token_counts_without_changing_estimates(
    input_tokens, cache_read_tokens, cache_write_tokens,
):
    baseline = text_impacts("openai/gpt-6-luna", 0, 100)
    impacts = text_impacts(
        "openai/gpt-6-luna", input_tokens, 100,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    assert not impacts.has_errors
    assert impacts == baseline


@pytest.mark.parametrize(
    ("input_tokens", "cache_read_tokens", "cache_write_tokens"),
    [(0, 1, None), (0, None, 1), (1000, 1001, None), (1000, None, 1001), (1000, 800, 201)],
)
def test_text_impacts_rejects_cache_counts_exceeding_input(input_tokens, cache_read_tokens, cache_write_tokens):
    with pytest.raises(ValueError, match="must not exceed input_tokens"):
        text_impacts(
            "openai/gpt-6-luna", input_tokens, 100,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )


@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("openai", "openai/unknown-model"),
        ("azure", "openai/gpt-6-luna"),
        ("aws", "openai/gpt-6-luna"),
        ("openai", "anthropic/gpt-6-luna"),
        ("openai", "gpt-6-luna"),
        ("google_genai", "google/gemini-2.5-flash"),
        ("huggingface_hub", "mistralai/Mistral-7B-v0.1"),
        ("huggingface", "mistralai/Mistral-7B-v0.1"),
    ],
)
def test_text_impacts_unknown_model_or_provider(provider, model_name):
    impacts = text_impacts(model_name, 1000, 100, provider=provider)

    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ModelNotRegisteredError)
    assert model_name in impacts.errors[0].message
    assert impacts.energy is None


def test_text_impacts_unknown_zone():
    impacts = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_location="UNKNOWN")

    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ZoneNotRegisteredError)
    assert impacts.energy is None


def test_text_impacts_preserves_model_and_electricity_warnings():
    model = models.find_model("openai", "gpt-6-luna")
    impacts = text_impacts(f"openai/{model.name}", 1000, 100, datacenter_location="ABW")

    assert not impacts.has_errors
    assert all(warning in impacts.warnings for warning in model.warnings)
    for warning_type in (ElectricityMixADPeDefaultWarning, ElectricityMixPEDefaultWarning,
                         ElectricityMixWUEDefaultWarning):
        assert any(isinstance(warning, warning_type) for warning in impacts.warnings)


def test_text_impacts_request_latency_caps_estimated_latency():
    estimated = text_impacts("openai/gpt-6-luna", 1000, 100)
    fast = text_impacts("openai/gpt-6-luna", 1000, 100, request_latency=0.001)
    slow = text_impacts("openai/gpt-6-luna", 1000, 100, request_latency=3600)

    assert fast == _llm_impacts("openai", "gpt-6-luna", 100, request_latency=0.001)
    assert fast.energy.value.max < estimated.energy.value.min
    assert slow == estimated


@pytest.mark.parametrize("latency", [0, -1, math.nan, math.inf, True, "1"])
def test_text_impacts_rejects_invalid_request_latency(latency):
    with pytest.raises(ValueError, match="request_latency"):
        text_impacts("openai/gpt-6-luna", 1000, 100, request_latency=latency)
