import warnings

import pytest

from ecologits.estimations import text_impacts
from ecologits.tracers.utils import _llm_impacts, llm_impacts
from ecologits.utils.range_value import RangeValue


@pytest.mark.parametrize(
    ("model_name", "request_latency", "electricity_mix_zone", "datacenter_pue", "datacenter_wue"),
    [
        ("gpt-6-luna", None, None, None, None),
        ("gpt-6-luna", 0.01, None, None, None),
        ("gpt-6-luna", 100, "WOR", 1.2, 0),
        ("gpt-6-luna", 100, "FRA", RangeValue(min=1.1, max=1.3), RangeValue(min=0.1, max=0.3)),
        ("gpt-6-luna", 100, "ABW", None, None),
        ("unknown-model", 100, None, None, None),
        ("gpt-6-luna", 100, "UNKNOWN", None, None),
    ],
)
def test_deprecated_llm_impacts_preserves_results(
    model_name, request_latency, electricity_mix_zone, datacenter_pue, datacenter_wue,
):
    expected = _llm_impacts(
        provider="openai",
        model_name=model_name,
        output_token_count=100,
        request_latency=request_latency,
        electricity_mix_zone=electricity_mix_zone,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
    )

    with pytest.warns(DeprecationWarning, match=r"llm_impacts is deprecated; use ecologits\.estimations\.text_impacts"):
        actual = llm_impacts(
            "openai", model_name, 100, request_latency, electricity_mix_zone,
            datacenter_pue=datacenter_pue,
            datacenter_wue=datacenter_wue,
        )

    assert actual == expected


def test_deprecated_llm_impacts_warns_at_caller():
    with pytest.warns(DeprecationWarning) as caught:
        llm_impacts(provider="openai", model_name="gpt-6-luna", output_token_count=100, request_latency=None)

    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_deprecated_llm_impacts_preserves_measured_latency():
    with pytest.warns(DeprecationWarning):
        short = llm_impacts("openai", "gpt-6-luna", 100, 0.001)
        long = llm_impacts("openai", "gpt-6-luna", 100, 100)

    assert long.energy.value.min > short.energy.value.min
    assert long.embodied.gwp.value.min > short.embodied.gwp.value.min


def test_internal_and_estimation_calls_do_not_emit_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        traced = _llm_impacts("openai", "gpt-6-luna", 100, request_latency=None, electricity_mix_zone="USA")
        estimated = text_impacts("openai/gpt-6-luna", 1000, 100, datacenter_location="USA")

    assert not traced.has_errors
    assert estimated == traced
