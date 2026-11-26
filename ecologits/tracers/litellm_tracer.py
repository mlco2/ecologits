import re
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Optional, Union

import litellm
from litellm import AsyncCompletions, Completions
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper
from rapidfuzz import fuzz, process
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.model_repository import Providers, models
from ecologits.status_messages import ModelNameFuzzyMatchWarning
from ecologits.tracers.utils import AsyncImpactsStream, ImpactsOutput, ImpactsStream, _llm_impacts
from ecologits.tracking import ImpactsTracker, get_active_trackers, is_tracking_enabled, record_impacts

PROVIDER = "litellm"


class ChatCompletion(ModelResponse):
    """
    Wrapper of `litellm.types.utils.ModelResponse` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class ChatCompletionChunk(ModelResponse):
    """
    Wrapper of `litellm.types.utils.ModelResponse` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


_PROVIDERS = [p.value for p in Providers]
_model_choices = [f"{m.provider.value}/{m.name}" for m in models.list_models()]
# Bedrock region and vendor prefixes (`us-gov.`, `anthropic.`); model names have a digit before any dot.
_HOST_PREFIX = re.compile(r"^(?:[a-z-]+\.)+")


def _host_variants(name: str) -> list[str]:
    """Names without host details, e.g. `us.anthropic.claude-opus-4-6-v1` -> `claude-opus-4-6`."""
    name = _HOST_PREFIX.sub("", name)
    return [name, name.replace("@", "-"), name.split("@")[0], re.sub(r"(-v\d+)?(-\d+)?:\d+$|-v\d+$", "", name)]


def _numbers(name: str) -> list[str]:
    return sorted(re.findall(r"\d+", name))


def _exact(name: str) -> Optional[tuple[str, str]]:
    found = [(p, name) for p in _PROVIDERS if models.find_model(p, name) is not None]
    return found[0] if len(found) == 1 else None


def _match(model_name: Optional[str]) -> Optional[tuple[str, str, bool]]:
    """Return (provider, model_name, is_guess) for a LiteLLM model name, or None."""
    if not model_name:
        return None
    # Hugging Face ids keep their organisation; `openai/` also prefixes other OpenAI-compatible hosts.
    names = [model_name] + ([rest] if (rest := model_name.partition("/")[2]) else [])
    for name in names:
        for candidate in _host_variants(name):
            if (match := _exact(candidate)) is not None:
                return (*match, False)
    for name in names:
        if (match := _exact(f"{name}-latest")) is not None:
            return (*match, True)
    tail = model_name.rsplit("/", 1)[-1]
    if not _numbers(tail):
        return None
    fuzzy = process.extractOne(
        query=model_name, choices=_model_choices, scorer=fuzz.token_sort_ratio, score_cutoff=51
    )
    if fuzzy is not None:
        provider, name = fuzzy[0].split("/", 1)
        target = _numbers(name.rsplit("/", 1)[-1])
        # A different version, size or date is a different model, so the numbers must be equal.
        if target in (_numbers(_host_variants(tail)[0]), _numbers(_host_variants(tail)[-1])):
            return provider, name, True
    return None


def litellm_match_model(model_name: str) -> Optional[tuple[str, str]]:
    """
    Match according provider and model from a litellm model_name.

    Looks up the exact model name (aliases included), then the name without host details
    (Bedrock region and vendor prefixes, version suffixes, Vertex `@` versions), then
    `<name>-latest`. As a last step it uses fuzzy matching, but only when the name has
    numbers and they are the same in both names. Other names return None.

    Args:
        model_name: Name of the model as used in litellm.

    Returns:
        A tuple (provider, model_name) matching a record of the ModelRepository.
    """
    match = _match(model_name)
    return None if match is None else (match[0], match[1])


def _litellm_impacts(
    model_name: str,
    output_token_count: int,
    request_latency: float
) -> tuple[tuple[str, str], ImpactsOutput]:
    provider, name, fuzzy = _match(model_name) or (PROVIDER, model_name, False)
    impacts = _llm_impacts(
        provider=provider,
        model_name=name,
        output_token_count=output_token_count,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
    if fuzzy and impacts is not None:
        impacts.add_warning(ModelNameFuzzyMatchWarning())
    return (provider, name), impacts


def litellm_chat_wrapper(
    wrapped: Callable,
    instance: Completions,
    args: Any,
    kwargs: Any
) -> Union[ChatCompletion, CustomStreamWrapper]:
    """
    Function that wraps a LiteLLM answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletion` or `CustomStreamWrapper` with impacts
    """
    if not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    if kwargs.get("stream", False):
        timer_start = time.perf_counter()
        stream = wrapped(*args, **kwargs)
        return ImpactsStream(stream, litellm_chat_wrapper_stream(stream, timer_start, trackers))
    else:
        return litellm_chat_wrapper_non_stream(wrapped, instance, args, kwargs, trackers)


def litellm_chat_wrapper_stream(
    stream: CustomStreamWrapper,
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...] = ()
) -> Iterator[Any]:
    token_count = 0
    impacts: ImpactsOutput | None = None
    for i, chunk in enumerate(stream):
        if i > 0 and chunk.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start

        model_match, impacts = _litellm_impacts(chunk.model, token_count, request_latency)
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and hasattr(chunk, "usage") \
                    and chunk.usage is not None:
                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=model_match[0],
                    model=model_match[1],
                    endpoint="/chat/completions"
                )

            yield ChatCompletionChunk.model_construct(**dict(chunk), impacts=impacts)
        else:
            yield chunk
    if impacts is not None:
        record_impacts(trackers, impacts)


def litellm_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = ()
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_match, impacts = _litellm_impacts(response.model, response.usage.completion_tokens, request_latency)
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=model_match[0],
                model=model_match[1],
                endpoint="/chat/completions"
            )

        record_impacts(trackers, impacts)
        return ChatCompletion.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def litellm_async_chat_wrapper(
        wrapped: Callable,
        instance: AsyncCompletions,
        args: Any,
        kwargs: Any
) -> Union[ChatCompletion, CustomStreamWrapper]:
    """
    Function that wraps a LiteLLM answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletion` or `CustomStreamWrapper` with impacts
    """
    if not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    if kwargs.get("stream", False):
        timer_start = time.perf_counter()
        stream = await wrapped(*args, **kwargs)
        return AsyncImpactsStream(stream, litellm_async_chat_wrapper_stream(stream, timer_start, trackers))
    else:
        return await litellm_async_chat_wrapper_base(wrapped, instance, args, kwargs, trackers)


async def litellm_async_chat_wrapper_base(
    wrapped: Callable,
    instance: AsyncCompletions,      # noqa: ARG001
    args: Any,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = ()
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_match, impacts = _litellm_impacts(response.model, response.usage.completion_tokens, request_latency)
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=model_match[0],
                model=model_match[1],
                endpoint="/chat/completions"
            )

        record_impacts(trackers, impacts)
        return ChatCompletion.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def litellm_async_chat_wrapper_stream(
    stream: CustomStreamWrapper,
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...] = ()
) -> AsyncIterator[Any]:
    i = 0
    token_count = 0
    impacts: ImpactsOutput | None = None
    async for chunk in stream:
        if i > 0 and chunk.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_match, impacts = _litellm_impacts(chunk.model, token_count, request_latency)
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and hasattr(chunk, "usage") \
                    and chunk.usage is not None:
                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=model_match[0],
                    model=model_match[1],
                    endpoint="/chat/completions"
                )

            yield ChatCompletionChunk.model_construct(**dict(chunk), impacts=impacts)
        else:
            yield chunk
        i += 1
    if impacts is not None:
        record_impacts(trackers, impacts)


class LiteLLMInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all LiteLLM calls
    """

    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": litellm,
                "name": "completion",
                "wrapper": litellm_chat_wrapper,
            },
            {
                "module": litellm,
                "name": "acompletion",
                "wrapper": litellm_async_chat_wrapper,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
