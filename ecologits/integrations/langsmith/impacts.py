"""
Compute ecologits impacts from LangSmith LLM child runs and post them as
LangSmith Feedback scores on each child run, plus SCI for AI Consumer-Boundary
rates (sci_per_token_mgco2eq, sci_per_workflow_gco2eq) on child and root runs.
"""

# Impacts are posted as LangSmith Feedback scores (not metadata) because LangSmith
# dashboard charts only support feedback_score_avg + feedback_key for custom
# numeric Y-axis metrics; run.metadata fields cannot be plotted.
from __future__ import annotations

from typing import Any

from ecologits.integrations.langsmith.constants import (
    FEEDBACK_KEY_BASE,
    IMPACT_FIELDS,
    SCORE_SCALE,
)
from ecologits.integrations.langsmith.extractors import (
    compute_latency_seconds,
    extract_output_tokens,
    extract_provider_and_model,
    iter_llm_child_runs,
)
from ecologits.integrations.langsmith.payload_builder import (
    FEEDBACK_SOURCE_INFO,
    build_payloads_for_field,
    build_sci_per_token_payloads,
    build_sci_per_workflow_payloads,
    post_payload_safely,
    resolve_client,
)
from ecologits.log import logger
from ecologits.tracers.utils import ImpactsOutput, llm_impacts
from ecologits.utils.range_value import RangeValue


def compute_call_impact(child_run: Any, electricity_mix_zone: str | None) -> ImpactsOutput | None:
    """
    Build an ImpactsOutput for one LangSmith llm child run, or None if inputs are insufficient.
    """
    provider, model_name = extract_provider_and_model(child_run)
    output_tokens = extract_output_tokens(child_run)
    latency = compute_latency_seconds(child_run)
    if not provider or not model_name or output_tokens is None or latency is None:
        return None
    if output_tokens <= 0 or latency <= 0:
        return None
    return llm_impacts(
        provider=provider,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=latency,
        electricity_mix_zone=electricity_mix_zone,
    )


def build_feedback_payloads(impact: ImpactsOutput, run_id: str) -> list[dict]:
    """
    Flatten an ImpactsOutput into one Client.create_feedback payload per scored variant.
    """
    payloads: list[dict] = []
    for field in IMPACT_FIELDS:
        payloads.extend(
            build_payloads_for_field(
                impact=getattr(impact, field, None),
                base_key=FEEDBACK_KEY_BASE[field],
                scale=SCORE_SCALE[field],
                run_id=run_id,
            )
        )
    return payloads


def _gwp_bounds(impact: ImpactsOutput) -> tuple[float | None, float | None]:
    """Extract raw min/max kgCO2eq from an ImpactsOutput gwp field."""
    gwp = getattr(impact, "gwp", None)
    if gwp is None or gwp.value is None:
        return None, None
    value = gwp.value
    if isinstance(value, RangeValue):
        return float(value.min), float(value.max)
    return float(value), float(value)


def post_impacts_as_feedback(
    run_tree: Any,
    electricity_mix_zone: str | None,
    client: Any = None,
) -> None:
    """
    Walk LLM child runs of `run_tree`, compute impacts, post each as feedback.

    Additionally emits:
    - sci_per_token_mgco2eq_{min,max,avg} on each LLM child run.
    - sci_per_workflow_gco2eq_{min,max,avg} on the root run (sum of child GWP).

    No-op when `run_tree` is None. Feedback errors are logged and never raised.
    """
    if run_tree is None:
        return
    resolved = resolve_client(client)
    root_gwp_min: float = 0.0
    root_gwp_max: float = 0.0
    has_root_sci = False

    for child in iter_llm_child_runs(run_tree):
        impact = compute_call_impact(child, electricity_mix_zone)
        if impact is None or impact.has_errors:
            continue
        run_id = str(getattr(child, "id", ""))
        if not run_id:
            continue

        for payload in build_feedback_payloads(impact, run_id):
            post_payload_safely(resolved, payload)

        output_tokens = extract_output_tokens(child) or 0
        for payload in build_sci_per_token_payloads(getattr(impact, "gwp", None), output_tokens, run_id):
            post_payload_safely(resolved, payload)

        lo, hi = _gwp_bounds(impact)
        if lo is not None and hi is not None:
            root_gwp_min += lo
            root_gwp_max += hi
            has_root_sci = True

    if has_root_sci:
        root_id = str(getattr(run_tree, "id", ""))
        if root_id:
            for payload in build_sci_per_workflow_payloads(root_gwp_min, root_gwp_max, root_id):
                post_payload_safely(resolved, payload)
        else:
            logger.warning("ecologits: root run has no id, skipping sci_per_workflow feedback")
