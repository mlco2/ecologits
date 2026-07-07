"""
Extract ecologits inputs (provider, model, output tokens, latency) from
LangSmith RunTree child runs captured during a traced function call.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ecologits.integrations.langsmith.constants import (
    DEPLOYMENT_TO_MODEL,
    KEY_AZURE_DEPLOYMENT,
    KEY_COMPLETION_TOKENS,
    KEY_EXTRA,
    KEY_INVOCATION_PARAMS,
    KEY_LLM_OUTPUT,
    KEY_LS_MODEL_NAME,
    KEY_LS_PROVIDER,
    KEY_METADATA,
    KEY_MODEL,
    KEY_MODEL_NAME,
    KEY_OUTPUT_TOKENS,
    KEY_RUN_TYPE,
    KEY_SERIALIZED,
    KEY_SERIALIZED_ID,
    KEY_TOKEN_USAGE,
    KEY_USAGE,
    KEY_USAGE_METADATA,
    LLM_RUN_TYPE,
    LS_PROVIDER_TO_ECOLOGITS,
    SERIALIZED_MODULE_TO_PROVIDER,
    TYPE_TO_PROVIDER,
)


def _as_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if obj is None:
        return {}
    return {}


def _run_metadata(run: Any) -> dict:
    extra = getattr(run, KEY_EXTRA, None) or {}
    return _as_dict(extra.get(KEY_METADATA))


def _run_invocation_params(run: Any) -> dict:
    extra = getattr(run, KEY_EXTRA, None) or {}
    return _as_dict(extra.get(KEY_INVOCATION_PARAMS))


def _run_outputs(run: Any) -> dict:
    return _as_dict(getattr(run, "outputs", None))


def _run_serialized(run: Any) -> dict:
    return _as_dict(getattr(run, KEY_SERIALIZED, None))


def _normalise_model_name(raw: str | None) -> str | None:
    if not raw:
        return None
    return DEPLOYMENT_TO_MODEL.get(raw.lower(), raw)


def _provider_from_metadata(metadata: dict) -> str | None:
    ls_provider = metadata.get(KEY_LS_PROVIDER)
    if not ls_provider:
        return None
    return LS_PROVIDER_TO_ECOLOGITS.get(str(ls_provider).lower())


def _provider_from_invocation_params(invocation_params: dict) -> str | None:
    run_type = str(invocation_params.get(KEY_RUN_TYPE) or "").lower()
    if not run_type:
        return None
    return TYPE_TO_PROVIDER.get(run_type)


def _provider_from_serialized(serialized: dict) -> str | None:
    fragments = serialized.get(KEY_SERIALIZED_ID) or []
    if not isinstance(fragments, list):
        return None
    for fragment in fragments:
        provider = SERIALIZED_MODULE_TO_PROVIDER.get(str(fragment).lower())
        if provider:
            return provider
    return None


def _model_name_from_sources(metadata: dict, invocation_params: dict) -> str | None:
    raw = (
        metadata.get(KEY_LS_MODEL_NAME)
        or invocation_params.get(KEY_LS_MODEL_NAME)
        or invocation_params.get(KEY_MODEL_NAME)
        or invocation_params.get(KEY_MODEL)
        or invocation_params.get(KEY_AZURE_DEPLOYMENT)
    )
    return _normalise_model_name(raw)


def extract_provider_and_model(run: Any) -> tuple[str | None, str | None]:
    """
    Return the ecologits provider key and the normalised model name.
    """
    metadata = _run_metadata(run)
    invocation_params = _run_invocation_params(run)
    provider = (
        _provider_from_metadata(metadata)
        or _provider_from_invocation_params(invocation_params)
        or _provider_from_serialized(_run_serialized(run))
    )
    model_name = _model_name_from_sources(metadata, invocation_params)
    return provider, model_name


def _usage_metadata_from_run(run: Any) -> dict:
    metadata = _run_metadata(run)
    return _as_dict(_run_outputs(run).get(KEY_USAGE_METADATA) or metadata.get(KEY_USAGE_METADATA))


def _legacy_usage_from_outputs(run: Any) -> dict:
    # Fallback for pre-usage_metadata LangChain run shapes:
    # - outputs["usage"] dict (LangChain <0.2 direct token counts)
    # - outputs["llm_output"]["token_usage"] (oldest LangChain LLM wrapper shape)
    # Safe to remove once all traced runs come from LangChain >=0.2 / wrap_openai.
    outputs = _run_outputs(run)
    legacy = outputs.get(KEY_USAGE) or _as_dict(outputs.get(KEY_LLM_OUTPUT)).get(KEY_TOKEN_USAGE)
    return _as_dict(legacy)


def extract_output_tokens(run: Any) -> int | None:
    """
    Return the number of generated/output tokens for an LLM child run, or None.
    """
    # TODO: ecologits llm_impacts() expects only generated/output tokens
    # (completion_tokens), not prompt+completion. When ecologits exposes a full
    # token-type API (separate prompt / cached / reasoning / completion buckets),
    # extend this extractor + the llm_impacts call to pass each bucket explicitly.
    usage_metadata = _usage_metadata_from_run(run)
    if usage_metadata.get(KEY_OUTPUT_TOKENS) is not None:
        return int(usage_metadata[KEY_OUTPUT_TOKENS])
    legacy = _legacy_usage_from_outputs(run)
    if legacy.get(KEY_COMPLETION_TOKENS) is not None:
        return int(legacy[KEY_COMPLETION_TOKENS])
    if legacy.get(KEY_OUTPUT_TOKENS) is not None:
        return int(legacy[KEY_OUTPUT_TOKENS])
    return None


def compute_latency_seconds(run: Any) -> float | None:
    """
    Return clock latency in seconds from a run's start/end timestamps.
    """
    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)
    if start is None or end is None:
        return None
    try:
        return (end - start).total_seconds()
    except (TypeError, AttributeError):
        return None


def iter_all_runs(run_tree: Any) -> Iterator[Any]:
    """
    Yield every descendant run in depth-first order, regardless of run_type.
    """
    if run_tree is None:
        return
    stack: list[Any] = list(reversed(getattr(run_tree, "child_runs", None) or []))
    while stack:
        child = stack.pop()
        yield child
        stack.extend(reversed(getattr(child, "child_runs", None) or []))


def iter_llm_child_runs(run_tree: Any) -> Iterator[Any]:
    """
    Yield every descendant run with `run_type == LLM_RUN_TYPE` in depth-first order.
    """
    for run in iter_all_runs(run_tree):
        if getattr(run, "run_type", None) == LLM_RUN_TYPE:
            yield run
