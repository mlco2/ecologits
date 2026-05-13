import operator

import pytest

from ecologits.utils.range_value import RangeValue


def test_range_formats_ok():
    range = RangeValue(min=0.00000006, max=0.00008)
    expected = f"{range.mean:.2f} [{range.min:.2f} - {range.max:.2f}]"
    assert f"{range:.2f}" == expected


@pytest.mark.parametrize("val_1,val_2,op,result", [
    (RangeValue(min=1, max=2), RangeValue(min=1, max=2), operator.eq, True),
    (RangeValue(min=1, max=2), RangeValue(min=1, max=3), operator.eq, False),
    (RangeValue(min=1, max=2), RangeValue(min=2, max=3), operator.eq, False),
    (RangeValue(min=1, max=2), RangeValue(min=1, max=3), operator.ne, True),
    (RangeValue(min=1, max=2), RangeValue(min=2, max=3), operator.ne, True),
    (RangeValue(min=1, max=2), RangeValue(min=1, max=2), operator.ne, False),
    (RangeValue(min=0, max=1), RangeValue(min=2, max=3), operator.le, True),
    (RangeValue(min=0, max=1), RangeValue(min=2, max=3), operator.lt, True),
    (RangeValue(min=2, max=3), RangeValue(min=0, max=1), operator.ge, True),
    (RangeValue(min=2, max=3), RangeValue(min=0, max=1), operator.gt, True),
    (RangeValue(min=0, max=1), RangeValue(min=1, max=2), operator.le, True),
    (RangeValue(min=0, max=1), RangeValue(min=1, max=2), operator.lt, False),
    (RangeValue(min=1, max=2), RangeValue(min=0, max=1), operator.ge, True),
    (RangeValue(min=1, max=2), RangeValue(min=0, max=1), operator.gt, False),
    (RangeValue(min=1.5, max=2), RangeValue(min=1, max=2), operator.ge, True),
    (RangeValue(min=1.5, max=2), RangeValue(min=1, max=2), operator.gt, False),
    (RangeValue(min=1, max=2), RangeValue(min=1.5, max=2), operator.le, True),
    (RangeValue(min=1, max=2), RangeValue(min=1.5, max=2), operator.lt, False),
    (RangeValue(min=0, max=1.5), RangeValue(min=1, max=2), operator.le, True),
    (RangeValue(min=0, max=1.5), RangeValue(min=1, max=2), operator.lt, False),
    (RangeValue(min=1, max=2), RangeValue(min=0, max=1.5), operator.ge, True),
    (RangeValue(min=1, max=2), RangeValue(min=0, max=1.5), operator.gt, False),
    (RangeValue(min=1, max=1), 1, operator.eq, True),
    (RangeValue(min=1, max=2), 1, operator.eq, False),
    (RangeValue(min=1, max=2), 2, operator.eq, False),
    (RangeValue(min=1, max=2), 1, operator.ne, True),
    (RangeValue(min=2, max=3), 1, operator.ne, True),
    (RangeValue(min=1, max=1), 1, operator.ne, False),
    (RangeValue(min=1, max=2), 1.5, operator.ge, False),
    (RangeValue(min=1, max=2), 0.5, operator.ge, True),
    (1.5, RangeValue(min=1, max=2), operator.le, False),
    (0.5, RangeValue(min=1, max=2), operator.le, True),
    (RangeValue(min=1, max=2), 1, operator.ge, True),
    (RangeValue(min=1, max=2), 0, operator.ge, True),
    (1, RangeValue(min=1, max=2), operator.le, True),
    (0, RangeValue(min=1, max=2), operator.le, True),
    (RangeValue(min=-2, max=-1), -1.5, operator.ge, False),
    (RangeValue(min=-2, max=-1), -2.5, operator.ge, True),
    (-1.5, RangeValue(min=-2, max=-1), operator.le, False),
    (-2.5, RangeValue(min=-2, max=-1), operator.le, True),
])
def test_value_range_compare(val_1, val_2, op, result):
    assert op(val_1, val_2) == result


@pytest.mark.parametrize("val_1,val_2,op,exp_result", [
    (RangeValue(min=1, max=2), RangeValue(min=1, max=2), operator.add, RangeValue(min=2, max=4)),
    (RangeValue(min=1, max=2), 1, operator.add, RangeValue(min=2, max=3)),
    (RangeValue(min=1, max=2), 2, operator.mul, RangeValue(min=2, max=4)),
    (RangeValue(min=2, max=4), 2, operator.truediv, RangeValue(min=1, max=2)),
])
def test_value_range_transformation(val_1, val_2, op, exp_result):
    result = op(val_1, val_2)
    assert result.min == exp_result.min and result.max == exp_result.max


class TestRangeValueNewMethods:
    def test_sub_range_range(self):
        r = RangeValue(min=3, max=5) - RangeValue(min=1, max=2)
        assert r.min == 1 and r.max == 4

    def test_sub_range_scalar(self):
        r = RangeValue(min=3, max=5) - 1
        assert r.min == 2 and r.max == 4

    def test_neg(self):
        r = -RangeValue(min=1, max=3)
        assert r.min == -3 and r.max == -1

    def test_abs_positive(self):
        r = abs(RangeValue(min=2, max=5))
        assert r.min == 2 and r.max == 5

    def test_abs_negative(self):
        r = abs(RangeValue(min=-5, max=-2))
        assert r.min == 2 and r.max == 5

    def test_to_dict_keys(self):
        d = RangeValue(min=1.0, max=3.0).to_dict()
        assert set(d.keys()) == {"min", "max", "mean"}

    def test_to_dict_mean(self):
        d = RangeValue(min=1.0, max=3.0).to_dict()
        assert d["mean"] == 2.0

    def test_repr_contains_min_max_mean(self):
        r = repr(RangeValue(min=1, max=3))
        assert "min=1" in r and "max=3" in r and "mean=2.0" in r
