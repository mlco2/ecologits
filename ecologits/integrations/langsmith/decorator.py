"""
`environment_traceable` decorator - wraps LangSmith's `@traceable` and posts
ecologits environmental impacts as feedback on every captured LLM child run,
plus codecarbon runtime emissions on every run in the tree.
"""

# Impacts are posted as LangSmith Feedback scores (not metadata) because LangSmith
# dashboard charts only support feedback_score_avg + feedback_key for custom
# numeric Y-axis metrics; run.metadata fields cannot be plotted.
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from ecologits._ecologits import EcoLogits
from ecologits.integrations.langsmith.constants import DEFAULT_RUN_TYPE
from ecologits.integrations.langsmith.impacts import post_impacts_as_feedback
from ecologits.integrations.langsmith.runtime import (
    build_default_tracker,
    post_runtime_as_feedback,
)
from ecologits.log import logger


def _resolve_zone(explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    return getattr(EcoLogits.config, "electricity_mix_zone", None)


def _run_tracker(tracker: Any) -> tuple[Any, Any]:
    """
    Start the tracker, run start_task, and return (tracker, task_name).

    Using start_task/stop_task rather than start/stop gives us a full
    EmissionsData object (including per-component energy fields) instead of
    just the kgCO2 float returned by stop().
    """
    task_name = "langsmith-trace"
    tracker.start()
    tracker.start_task(task_name)
    return tracker, task_name


def _stop_tracker(tracker: Any, task_name: str | None) -> Any:
    """
    Stop the task and the tracker; return the EmissionsData delta or None.
    """
    emissions_data = None
    try:
        emissions_data = tracker.stop_task(task_name)
    except Exception as exc:
        logger.warning(f"ecologits: codecarbon stop_task failed: {exc}")
    try:
        tracker.stop()
    except Exception as exc:
        logger.warning(f"ecologits: codecarbon stop failed: {exc}")
    return emissions_data


def _make_tracker_factory(
    enable_runtime_emissions: bool,
    runtime_tracker: Any,
) -> Callable[[str | None], Any] | None:
    if not enable_runtime_emissions:
        return None
    if runtime_tracker is not None:
        return lambda _zone: runtime_tracker
    return build_default_tracker


def _wrap_sync(
    func: Callable,
    zone: str | None,
    client: Any,
    tracker_factory: Callable | None,
) -> Callable:
    @functools.wraps(func)
    def sync_body(*args: Any, **kwargs: Any) -> Any:
        tracker = None
        task_name = None
        if tracker_factory is not None:
            try:
                tracker, task_name = _run_tracker(tracker_factory(zone))
            except Exception as exc:
                logger.warning(f"ecologits: codecarbon tracker start failed: {exc}")
                tracker = None
        try:
            result = func(*args, **kwargs)
        finally:
            emissions_data = _stop_tracker(tracker, task_name) if tracker else None
        run_tree = get_current_run_tree()
        post_impacts_as_feedback(run_tree, zone, client)
        if emissions_data is not None:
            post_runtime_as_feedback(run_tree, emissions_data, client)
        return result

    return sync_body


def _wrap_async(
    func: Callable,
    zone: str | None,
    client: Any,
    tracker_factory: Callable | None,
) -> Callable:
    @functools.wraps(func)
    async def async_body(*args: Any, **kwargs: Any) -> Any:
        tracker = None
        task_name = None
        if tracker_factory is not None:
            try:
                tracker, task_name = _run_tracker(tracker_factory(zone))
            except Exception as exc:
                logger.warning(f"ecologits: codecarbon tracker start failed: {exc}")
                tracker = None
        try:
            result = await func(*args, **kwargs)
        finally:
            emissions_data = _stop_tracker(tracker, task_name) if tracker else None
        run_tree = get_current_run_tree()
        post_impacts_as_feedback(run_tree, zone, client)
        if emissions_data is not None:
            post_runtime_as_feedback(run_tree, emissions_data, client)
        return result

    return async_body


def environment_traceable(
    run_type: str = DEFAULT_RUN_TYPE,
    *,
    electricity_mix_zone: str | None = None,
    enable_runtime_emissions: bool = True,
    runtime_tracker: Any = None,
    **traceable_kwargs: Any,
) -> Callable:
    """
    Wrap a function with LangSmith's @traceable and post ecologits impacts.

    All `traceable_kwargs` are forwarded unchanged to `langsmith.traceable`.
    After the wrapped function returns:
    - Every captured LLM child run gets one Feedback score per ecologits impact
      field / variant (energy_wh, gwp_gco2eq, adpe_mgsbeq, pe_kj, wcf_ml x
      min/max/avg).
    - Every run in the tree (LLM + non-LLM, including root) gets one Feedback
      score per codecarbon runtime field (runtime_energy_wh, runtime_gwp_gco2eq,
      runtime_water_ml, runtime_cpu/gpu/ram_energy_wh, runtime_duration_s),
      allocated by exclusive (self) duration share.

    Args:
        enable_runtime_emissions: Set False to skip codecarbon tracking entirely.
        runtime_tracker: Optional pre-configured OfflineEmissionsTracker.
            When provided, its start/stop lifecycle is still managed by this
            decorator. Defaults to a silent OfflineEmissionsTracker built from
            `electricity_mix_zone`.
    """
    client = traceable_kwargs.get("client")

    def decorator(func: Callable) -> Callable:
        zone = _resolve_zone(electricity_mix_zone)
        tracker_factory = _make_tracker_factory(enable_runtime_emissions, runtime_tracker)
        body = (
            _wrap_async(func, zone, client, tracker_factory)
            if asyncio.iscoroutinefunction(func)
            else _wrap_sync(func, zone, client, tracker_factory)
        )
        return traceable(run_type, **traceable_kwargs)(body)

    return decorator
