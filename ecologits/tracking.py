from __future__ import annotations

import operator
from contextvars import ContextVar, Token
from functools import reduce
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from ecologits.impacts.modeling import Embodied, Usage
from ecologits.tracers.utils import ImpactsOutput

if TYPE_CHECKING:
    from ecologits.impacts.modeling import BaseImpact

_Impact = TypeVar("_Impact", bound="BaseImpact")

_active_trackers: ContextVar[tuple[ImpactsTracker, ...]] = ContextVar("ecologits_active_trackers", default=())


class ImpactsTracker:
    """
    Context manager that collects the impacts of the LLM calls made inside it.

    Use it through `EcoLogits.track_impacts()`. Supports both `with` and `async with`.

    Attributes:
        providers: Providers whose calls are tracked.
        calls: Impacts of each tracked call, in completion order.
    """

    def __init__(self, providers: list[str]) -> None:
        self.providers = list(providers)
        self.calls: list[ImpactsOutput] = []
        self._tokens: list[Token[tuple[ImpactsTracker, ...]]] = []

    @property
    def impacts(self) -> ImpactsOutput:
        """Total impacts of the tracked calls."""
        return _sum_impacts(self.calls)

    def __enter__(self) -> ImpactsTracker:
        self._tokens.append(_active_trackers.set((*_active_trackers.get(), self)))
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        _active_trackers.reset(self._tokens.pop())

    async def __aenter__(self) -> ImpactsTracker:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def get_active_trackers(provider: str) -> tuple[ImpactsTracker, ...]:
    """
    Get the trackers open in the current context that track a provider.

    Args:
        provider: Name of the provider.

    Returns:
        The open trackers that track the provider.
    """
    return tuple(tracker for tracker in _active_trackers.get() if provider in tracker.providers)


def is_tracking_enabled(provider: str) -> bool:
    """
    Check if a call to a provider should be tracked.

    Args:
        provider: Name of the provider.

    Returns:
        True if the provider was enabled with `EcoLogits.init()` or is tracked by a tracker open in the current context.
    """
    from ecologits._ecologits import EcoLogits

    return provider in EcoLogits.config.providers or len(get_active_trackers(provider)) > 0


def record_impacts(trackers: tuple[ImpactsTracker, ...], impacts: ImpactsOutput) -> None:
    """
    Add the impacts of a call to the given trackers.

    Args:
        trackers: Trackers that were open when the call started.
        impacts: Impacts of the call.
    """
    for tracker in trackers:
        tracker.calls.append(impacts)


def _sum(values: list[_Impact]) -> _Impact:
    return reduce(operator.add, values)


def _sum_present(values: list[Optional[_Impact]]) -> Optional[_Impact]:
    present = [v for v in values if v is not None]
    return _sum(present) if present else None


def _sum_impacts(calls: list[ImpactsOutput]) -> ImpactsOutput:
    total = ImpactsOutput(
        energy=_sum_present([c.energy for c in calls]),
        gwp=_sum_present([c.gwp for c in calls]),
        adpe=_sum_present([c.adpe for c in calls]),
        pe=_sum_present([c.pe for c in calls]),
        wcf=_sum_present([c.wcf for c in calls]),
    )

    usages = [c.usage for c in calls if c.usage is not None]
    if usages:
        total.usage = Usage(
            energy=_sum([u.energy for u in usages]),
            gwp=_sum([u.gwp for u in usages]),
            adpe=_sum([u.adpe for u in usages]),
            pe=_sum([u.pe for u in usages]),
            wcf=_sum([u.wcf for u in usages]),
        )

    embodieds = [c.embodied for c in calls if c.embodied is not None]
    if embodieds:
        total.embodied = Embodied(
            gwp=_sum([e.gwp for e in embodieds]),
            adpe=_sum([e.adpe for e in embodieds]),
            pe=_sum([e.pe for e in embodieds]),
        )

    for call in calls:
        for warning in call.warnings or []:
            total.add_warning(warning)
        for error in call.errors or []:
            total.add_errors(error)

    return total
