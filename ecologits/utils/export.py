"""Export utilities for EcoLogits impact data (#118)."""
from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecologits.impacts.modeling import Impacts
    from ecologits.utils.range_value import RangeValue


def _scalar(value: "float | int | RangeValue") -> float:
    """Return the mean if value is a RangeValue, otherwise return the value itself."""
    return value.mean if hasattr(value, "mean") else float(value)


def impacts_to_csv(impacts: "Impacts", include_phases: bool = True) -> str:
    """
    Serialize an :class:`~ecologits.impacts.modeling.Impacts` object to CSV.

    Args:
        impacts: The impacts object to serialize.
        include_phases: When ``True`` (default), include usage and embodied phase rows.

    Returns:
        A CSV string with columns ``phase``, ``metric``, ``value``, ``unit``.

    Example::

        from ecologits.utils.export import impacts_to_csv
        csv_str = impacts_to_csv(response.impacts)
        with open("impacts.csv", "w") as f:
            f.write(csv_str)
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["phase", "metric", "value", "unit"])

    writer.writerow(["total", "energy", _scalar(impacts.energy.value), impacts.energy.unit])
    writer.writerow(["total", "gwp", _scalar(impacts.gwp.value), impacts.gwp.unit])
    writer.writerow(["total", "adpe", _scalar(impacts.adpe.value), impacts.adpe.unit])
    writer.writerow(["total", "pe", _scalar(impacts.pe.value), impacts.pe.unit])
    writer.writerow(["total", "wcf", _scalar(impacts.wcf.value), impacts.wcf.unit])

    if include_phases:
        writer.writerow(["usage", "energy", _scalar(impacts.usage.energy.value), impacts.usage.energy.unit])
        writer.writerow(["usage", "gwp", _scalar(impacts.usage.gwp.value), impacts.usage.gwp.unit])
        writer.writerow(["usage", "adpe", _scalar(impacts.usage.adpe.value), impacts.usage.adpe.unit])
        writer.writerow(["usage", "pe", _scalar(impacts.usage.pe.value), impacts.usage.pe.unit])
        writer.writerow(["usage", "wcf", _scalar(impacts.usage.wcf.value), impacts.usage.wcf.unit])

        writer.writerow(["embodied", "gwp", _scalar(impacts.embodied.gwp.value), impacts.embodied.gwp.unit])
        writer.writerow(["embodied", "adpe", _scalar(impacts.embodied.adpe.value), impacts.embodied.adpe.unit])
        writer.writerow(["embodied", "pe", _scalar(impacts.embodied.pe.value), impacts.embodied.pe.unit])

    return buf.getvalue()
