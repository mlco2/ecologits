"""
codecarbon-based runtime emissions tracking for the LangSmith integration.

Wraps a traced function with an `OfflineEmissionsTracker`, then allocates total
measured emissions to every run in the tree proportionally by exclusive (self)
duration, posting each allocation as `runtime_*` LangSmith Feedback scores.

Allocation by self-time (= run duration minus sum of direct-child durations)
ensures the allocations across all runs sum to the total measured value and
avoids double-counting parent/child overlaps.

Caveats:
- codecarbon measures whole-machine power during the traced interval; for LLM
  runs the "runtime emissions" reflect the client-side baseline while waiting
  on the remote API, not per-call server-side work.
- Async parallel children inflate wall-clock totals, which distorts shares.
"""

from __future__ import annotations

from typing import Any

from ecologits.integrations.langsmith.constants import (
    CODECARBON_FIELD_SCALE,
    CODECARBON_FIELD_TO_KEY,
    CODECARBON_PROJECT_NAME,
    DEFAULT_MEASURE_POWER_SECS,
    DEFAULT_TRACKING_MODE,
    LANGSMITH_FEEDBACK_SOURCE_TYPE,
    SCORE_ROUND_DIGITS,
    SCI_KEY_CARBON_INTENSITY,
)
from ecologits.integrations.langsmith.extractors import (
    compute_latency_seconds,
    iter_all_runs,
)
from ecologits.integrations.langsmith.payload_builder import (
    FEEDBACK_SOURCE_INFO,
    post_payload_safely,
    resolve_client,
)
from ecologits.log import logger

try:
    from codecarbon import OfflineEmissionsTracker

    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


def build_default_tracker(electricity_mix_zone: str | None) -> Any:
    """
    Build a silent OfflineEmissionsTracker scoped to a single traced call.

    Raises ImportError if codecarbon is not installed.
    """
    if not _CODECARBON_AVAILABLE:
        raise ImportError(
            "codecarbon is required for runtime emissions tracking. Install with: pip install ecologits[langsmith]"
        )
    kwargs: dict[str, Any] = {
        "project_name": CODECARBON_PROJECT_NAME,
        "measure_power_secs": DEFAULT_MEASURE_POWER_SECS,
        "tracking_mode": DEFAULT_TRACKING_MODE,
        "save_to_file": False,
        "save_to_api": False,
        "save_to_logger": False,
        "save_to_prometheus": False,
        "allow_multiple_runs": True,
        "log_level": "error",
    }
    if electricity_mix_zone is not None:
        kwargs["country_iso_code"] = electricity_mix_zone
    return OfflineEmissionsTracker(**kwargs)


def compute_self_duration_seconds(run: Any) -> float:
    """
    Return the exclusive (self) duration of a run in seconds.

    Self-time = run duration minus the sum of its direct children's durations.
    Clamped to 0 to guard against clock skew or imprecise timestamps.
    """
    total = compute_latency_seconds(run) or 0.0
    children_total = sum(compute_latency_seconds(child) or 0.0 for child in (getattr(run, "child_runs", None) or []))
    return max(0.0, total - children_total)


def compute_run_share(self_duration: float, total_duration: float) -> float:
    """
    Return the fraction of total duration attributable to one run's self-time.
    """
    if total_duration <= 0:
        return 0.0
    return self_duration / total_duration


def build_runtime_feedback_payloads(
    emissions_data: Any,
    share: float,
    run_id: str,
) -> list[dict]:
    """
    Flatten a scaled, share-weighted EmissionsData snapshot into feedback payloads.
    """
    payloads: list[dict] = []
    for field, key in CODECARBON_FIELD_TO_KEY.items():
        raw = getattr(emissions_data, field, None)
        if raw is None:
            continue
        score = round(float(raw) * share * CODECARBON_FIELD_SCALE[field], SCORE_ROUND_DIGITS)
        payloads.append(
            {
                "run_id": run_id,
                "key": key,
                "score": score,
                "feedback_source_type": LANGSMITH_FEEDBACK_SOURCE_TYPE,
                "source_info": FEEDBACK_SOURCE_INFO,
            }
        )
    return payloads


def build_sci_intensity_payload(emissions_data: Any, run_id: str) -> dict | None:
    """
    Return a sci_I_gco2eq_per_kwh payload for the root run, or None.

    I = emissions (kgCO2eq) / energy_consumed (kWh) × 1e3 → gCO2eq/kWh.
    Posted once on the root run only; intensity is regional and not
    allocable by self-time.
    """
    emissions = getattr(emissions_data, "emissions", None)
    energy = getattr(emissions_data, "energy_consumed", None)
    if emissions is None or energy is None:
        return None
    try:
        e, en = float(emissions), float(energy)
    except (TypeError, ValueError):
        return None
    if en <= 0:
        return None
    score = round(e / en * 1e3, SCORE_ROUND_DIGITS)
    return {
        "run_id": run_id,
        "key": SCI_KEY_CARBON_INTENSITY,
        "score": score,
        "feedback_source_type": LANGSMITH_FEEDBACK_SOURCE_TYPE,
        "source_info": FEEDBACK_SOURCE_INFO,
    }


def post_runtime_as_feedback(
    run_tree: Any,
    emissions_data: Any,
    client: Any = None,
) -> None:
    """
    Walk every run in the tree (root included), allocate runtime emissions by
    self-duration share, and post each as LangSmith Feedback.

    Also posts sci_I_gco2eq_per_kwh (regional grid intensity) on the root run.
    No-op when `run_tree` or `emissions_data` is None.
    """
    if run_tree is None or emissions_data is None:
        return
    total_duration = compute_latency_seconds(run_tree) or 0.0
    if total_duration <= 0:
        logger.warning("ecologits: zero root duration, skipping runtime feedback allocation")
        return
    resolved = resolve_client(client)
    for run in [run_tree, *iter_all_runs(run_tree)]:
        run_id = str(getattr(run, "id", ""))
        if not run_id:
            continue
        self_dur = compute_self_duration_seconds(run)
        share = compute_run_share(self_dur, total_duration)
        for payload in build_runtime_feedback_payloads(emissions_data, share, run_id):
            post_payload_safely(resolved, payload)

    # sci_I posted once on root only.
    root_id = str(getattr(run_tree, "id", ""))
    if root_id:
        intensity_payload = build_sci_intensity_payload(emissions_data, root_id)
        if intensity_payload:
            post_payload_safely(resolved, intensity_payload)
