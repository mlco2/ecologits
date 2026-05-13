from typing import Any, Union

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class RangeValue(BaseModel):
    """
    RangeValue data model to represent intervals.

    Attributes:
        min: Lower bound of the interval.
        max: Upper bound of the interval.
    """
    min: Union[float, int]
    max: Union[float, int]

    @property
    def mean(self) -> float:
        return (self.min + self.max) / 2

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("min value must be lower than max value")
        return self

    def __add__(self, other: Union["RangeValue", int, float]) -> "RangeValue":
        if isinstance(other, RangeValue):
            return RangeValue(
                min=self.min + other.min,
                max=self.max + other.max,
            )
        else:
            return RangeValue(
                min=self.min + other,
                max=self.max + other
            )

    def __mul__(self, other: Union["RangeValue", int, float]) -> "RangeValue":
        if isinstance(other, RangeValue):
            return RangeValue(
                min=self.min * other.min,
                max=self.max * other.max,
            )
        else:
            return RangeValue(
                min=self.min * other,
                max=self.max * other
            )

    def __truediv__(self, other: Union[int, float]) -> "RangeValue":
        return RangeValue(
            min=self.min / other,
            max=self.max / other
        )

    def __rsub__(self, other: "Union[int, float]") -> "RangeValue":
        """Support ``scalar - RangeValue`` — note: subtraction is not commutative."""
        return RangeValue(min=other - self.max, max=other - self.min)

    __radd__ = __add__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RangeValue):
            return self.min == other.min and self.max == other.max
        else:
            return self.min == other and self.max == other

    def __le__(self, other: Any) -> bool:
        if isinstance(other, RangeValue):
            return self.max <= other.max
        else:
            return self.max <= other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, RangeValue):
            return self.max < other.min
        else:
            return self.max < other

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, RangeValue):
            return self.min >= other.min
        else:
            return self.min >= other

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, RangeValue):
            return self.min > other.max
        else:
            return self.min > other

    def __format__(self, format_spec: str) -> str:
        return f"{format(self.mean, format_spec)} [{format(self.min, format_spec)} - {format(self.max, format_spec)}]"

    def __repr__(self) -> str:
        return f"RangeValue(min={self.min}, max={self.max}, mean={self.mean})"

    def __sub__(self, other: "Union[RangeValue, int, float]") -> "RangeValue":
        if isinstance(other, RangeValue):
            return RangeValue(min=self.min - other.max, max=self.max - other.min)
        return RangeValue(min=self.min - other, max=self.max - other)

    def __neg__(self) -> "RangeValue":
        return RangeValue(min=-self.max, max=-self.min)

    def __abs__(self) -> "RangeValue":
        return RangeValue(min=min(abs(self.min), abs(self.max)), max=max(abs(self.min), abs(self.max)))

    def to_dict(self) -> dict:
        """Return a plain dictionary representation."""
        return {"min": self.min, "max": self.max, "mean": self.mean}

    __rsub__ = __sub__


ValueOrRange = Union[int, float, RangeValue]
