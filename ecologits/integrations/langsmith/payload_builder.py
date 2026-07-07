"""
Shared payload-building utilities for the LangSmith integration.

Centralises the scale/round/source-info pipeline used by both the ecologits
impact emitter (impacts.py) and the codecarbon runtime emitter (runtime.py),
and provides the SCI for AI (ISO/IEC 21031:2024) rate helpers.
"""
from __future__ import annotations

from typing import Any

from langsmith import Client

import ecologits
from ecologits.integrations.langsmith.constants import (
    FEEDBACK_SOURCE_KEY_ORIGIN,
    FEEDBACK_SOURCE_KEY_VERSION,
    FEEDBACK_SOURCE_VALUE_ORIGIN,
    LANGSMITH_FEEDBACK_SOURCE_TYPE,
    SCORE_ROUND_DIGITS,
    SCI_FEEDBACK_METADATA,
    SCI_KEY_PER_TOKEN,
    SCI_KEY_PER_WORKFLOW,
    SCI_SCALE_PER_TOKEN,
    SCI_SCALE_PER_WORKFLOW,
    VARIANT_AVG,
    VARIANT_MAX,
    VARIANT_MIN,
)
from ecologits.log import logger
from ecologits.utils.range_value import RangeValue

FEEDBACK_SOURCE_INFO: dict[str, str] = {
    FEEDBACK_SOURCE_KEY_ORIGIN: FEEDBACK_SOURCE_VALUE_ORIGIN,
    FEEDBACK_SOURCE_KEY_VERSION: ecologits.__version__,
    **SCI_FEEDBACK_METADATA,
}


def resolve_client(client: Any) -> Client:
    if client is not None:
        return client
    return Client()


def post_payload_safely(client: Any, payload: dict) -> None:
    try:
        client.create_feedback(**payload)
    except Exception as exc:
        logger.warning(f"ecologits: failed to post langsmith feedback {payload.get('key')!r}: {exc}")


def scaled_variants(value: Any, scale: float) -> dict[str, float]:
    """Return {min?, max?, avg} from a BaseImpact value, scaled to display units."""
    if value is None:
        return {}
    raw = getattr(value, "value", value)
    if raw is None:
        return {}
    if isinstance(raw, RangeValue):
        return {
            VARIANT_MIN: float(raw.min) * scale,
            VARIANT_MAX: float(raw.max) * scale,
            VARIANT_AVG: float(raw.mean) * scale,
        }
    return {VARIANT_AVG: float(raw) * scale}


def build_payloads_for_field(
    impact: Any,
    base_key: str,
    scale: float,
    run_id: str,
    source_info: dict[str, str] | None = None,
) -> list[dict]:
    """Build one feedback payload per variant of a single impact field."""
    if source_info is None:
        source_info = FEEDBACK_SOURCE_INFO
    payloads: list[dict] = []
    for variant, score in scaled_variants(impact, scale).items():
        payloads.append(
            {
                "run_id": run_id,
                "key": f"{base_key}_{variant}",
                "score": round(score, SCORE_ROUND_DIGITS),
                "feedback_source_type": LANGSMITH_FEEDBACK_SOURCE_TYPE,
                "source_info": source_info,
            }
        )
    return payloads


def _sci_rate_payload(run_id: str, key: str, score: float) -> dict:
    return {
        "run_id": run_id,
        "key": key,
        "score": round(score, SCORE_ROUND_DIGITS),
        "feedback_source_type": LANGSMITH_FEEDBACK_SOURCE_TYPE,
        "source_info": FEEDBACK_SOURCE_INFO,
    }


def build_sci_per_token_payloads(gwp_impact: Any, output_tokens: int, run_id: str) -> list[dict]:
    """
    Emit sci_per_token_mgco2eq_{min,max,avg} for one LLM run.

    R = output_tokens (LLM Consumer functional unit per ISO/IEC 21031:2024).
    Returns empty list when tokens <= 0 or gwp has no value.
    """
    if output_tokens <= 0:
        return []
    variants = scaled_variants(gwp_impact, SCI_SCALE_PER_TOKEN)
    if not variants:
        return []
    payloads: list[dict] = []
    for variant, gwp_scaled in variants.items():
        score = gwp_scaled / output_tokens
        payloads.append(_sci_rate_payload(run_id, f"{SCI_KEY_PER_TOKEN}_{variant}", score))
    return payloads


def build_sci_per_workflow_payloads(
    total_gwp_min: float | None,
    total_gwp_max: float | None,
    run_id: str,
) -> list[dict]:
    """
    Emit sci_per_workflow_gco2eq_{min,max,avg} on the root run.

    R = 1 workflow execution (Agentic AI Consumer functional unit).
    Scale: kgCO2eq → gCO2eq (×1e3).
    """
    bounds: list[tuple[str, float]] = []
    if total_gwp_min is not None:
        bounds.append((VARIANT_MIN, total_gwp_min))
    if total_gwp_max is not None:
        bounds.append((VARIANT_MAX, total_gwp_max))
    if not bounds:
        return []

    valid = [v for _, v in bounds]
    avg = sum(valid) / len(valid)
    payloads = [_sci_rate_payload(run_id, f"{SCI_KEY_PER_WORKFLOW}_{VARIANT_AVG}", avg * SCI_SCALE_PER_WORKFLOW)]
    for variant, val in bounds:
        payloads.append(_sci_rate_payload(run_id, f"{SCI_KEY_PER_WORKFLOW}_{variant}", val * SCI_SCALE_PER_WORKFLOW))
    return payloads
