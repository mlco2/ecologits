"""Tests for BaseImpact and Impacts model methods."""
import json
import pytest

from ecologits.impacts.modeling import Energy, GWP, ADPe, PE, WCF, Impacts, Usage, Embodied
from ecologits.utils.range_value import RangeValue


def _make_energy(value=0.001):
    return Energy(value=value)


def _make_gwp(value=0.0004):
    return GWP(value=value)


def _make_impacts_obj():
    e = Energy(value=0.001)
    gwp = GWP(value=0.0004)
    adpe = ADPe(value=1e-10)
    pe = PE(value=0.009)
    wcf = WCF(value=0.0018)
    usage = Usage(energy=e, gwp=gwp, adpe=adpe, pe=pe, wcf=wcf)
    embodied = Embodied(gwp=GWP(value=0.0001), adpe=ADPe(value=1e-11), pe=PE(value=0.001))
    return Impacts(
        energy=e,
        gwp=gwp + GWP(value=0.0001),
        adpe=adpe + ADPe(value=1e-11),
        pe=pe + PE(value=0.001),
        wcf=wcf,
        usage=usage,
        embodied=embodied,
    )


class TestBaseImpactToDict:
    def test_returns_dict(self):
        assert isinstance(_make_energy().to_dict(), dict)

    def test_has_all_keys(self):
        d = _make_energy().to_dict()
        assert set(d.keys()) == {"type", "name", "value", "unit"}

    def test_type_is_energy(self):
        assert _make_energy().to_dict()["type"] == "energy"

    def test_unit_is_kwh(self):
        assert _make_energy().to_dict()["unit"] == "kWh"

    def test_value_is_scalar(self):
        d = _make_energy(value=0.005).to_dict()
        assert d["value"] == 0.005

    def test_range_value_serialized(self):
        e = Energy(value=RangeValue(min=0.001, max=0.002))
        d = e.to_dict()
        assert isinstance(d["value"], dict)
        assert "min" in d["value"] and "max" in d["value"]

    def test_repr_contains_value_and_unit(self):
        r = repr(_make_energy(value=0.005))
        assert "0.005" in r
        assert "kWh" in r


class TestImpactsModelMethods:
    def test_to_dict_is_dict(self):
        assert isinstance(_make_impacts_obj().to_dict(), dict)

    def test_to_json_round_trip(self):
        imp = _make_impacts_obj()
        d = json.loads(imp.to_json())
        assert d["energy"]["unit"] == "kWh"

    def test_summary_is_str(self):
        assert isinstance(_make_impacts_obj().summary(), str)

    def test_repr_is_str(self):
        assert isinstance(repr(_make_impacts_obj()), str)

    def test_impact_addition(self):
        e1 = Energy(value=0.001)
        e2 = Energy(value=0.002)
        result = e1 + e2
        assert result.value == 0.003

    def test_impact_equality(self):
        assert Energy(value=0.001) == Energy(value=0.001)

    def test_impact_type_mismatch_add_raises(self):
        from ecologits.exceptions import ModelingError
        with pytest.raises(ModelingError):
            Energy(value=0.001) + GWP(value=0.001)
