"""Tests for Impacts output options: to_dict, to_json, summary (#118)."""
import json
import pytest

from ecologits.impacts.llm import compute_llm_impacts
from ecologits.utils.range_value import RangeValue


def _make_impacts(**overrides):
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
    return compute_llm_impacts(**defaults)


class TestToDict:
    def test_returns_dict(self):
        impacts = _make_impacts()
        d = impacts.to_dict()
        assert isinstance(d, dict)

    def test_has_top_level_keys(self):
        d = _make_impacts().to_dict()
        assert set(d.keys()) == {"energy", "gwp", "adpe", "pe", "wcf", "usage", "embodied"}

    def test_energy_has_value_and_unit(self):
        d = _make_impacts().to_dict()
        assert "value" in d["energy"]
        assert d["energy"]["unit"] == "kWh"

    def test_usage_has_all_impact_types(self):
        d = _make_impacts().to_dict()
        assert set(d["usage"].keys()) == {"energy", "gwp", "adpe", "pe", "wcf"}

    def test_embodied_has_gwp_adpe_pe(self):
        d = _make_impacts().to_dict()
        assert set(d["embodied"].keys()) == {"gwp", "adpe", "pe"}

    def test_range_value_serialized_as_dict(self):
        impacts = _make_impacts(
            model_active_parameter_count=RangeValue(min=3.0, max=7.0),
            model_total_parameter_count=RangeValue(min=3.0, max=7.0),
        )
        d = impacts.to_dict()
        energy_val = d["energy"]["value"]
        assert isinstance(energy_val, dict)
        assert "min" in energy_val and "max" in energy_val and "mean" in energy_val


class TestToJson:
    def test_returns_string(self):
        assert isinstance(_make_impacts().to_json(), str)

    def test_is_valid_json(self):
        result = json.loads(_make_impacts().to_json())
        assert "energy" in result

    def test_indent_parameter(self):
        compact = _make_impacts().to_json(indent=None)
        indented = _make_impacts().to_json(indent=4)
        assert len(indented) > len(compact)

    def test_round_trips_through_json(self):
        impacts = _make_impacts()
        d_original = impacts.to_dict()
        d_round = json.loads(impacts.to_json())
        assert d_original == d_round


class TestSummary:
    def test_returns_string(self):
        assert isinstance(_make_impacts().summary(), str)

    def test_contains_energy_label(self):
        assert "Energy" in _make_impacts().summary()

    def test_contains_gwp_label(self):
        assert "GWP" in _make_impacts().summary()

    def test_contains_wcf_label(self):
        assert "WCF" in _make_impacts().summary()

    def test_contains_unit(self):
        assert "kWh" in _make_impacts().summary()

    def test_range_shows_range(self):
        impacts = _make_impacts(
            model_active_parameter_count=RangeValue(min=3.0, max=7.0),
            model_total_parameter_count=RangeValue(min=3.0, max=7.0),
        )
        assert "range:" in impacts.summary()


class TestRepr:
    def test_impacts_repr_contains_energy(self):
        assert "energy" in repr(_make_impacts()).lower()

    def test_impacts_repr_contains_gwp(self):
        assert "gwp" in repr(_make_impacts()).lower()
