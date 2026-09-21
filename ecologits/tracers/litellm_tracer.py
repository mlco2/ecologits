import time
from typing import Any, Callable, Optional, Union

import litellm
from litellm import AsyncCompletions, Completions
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper
from rapidfuzz import fuzz, process
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.model_repository import models
from ecologits.tracers.utils import ImpactsOutput, llm_impacts


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


_model_choices = [f"{m.provider.value}/{m.name}" for m in models.list_models()]


def litellm_match_model(model_name: str) -> Optional[tuple[str, str]]:
    """
    Match according provider and model from a litellm model_name.

    Args:
        model_name: Name of the model as used in litellm.

    Returns:
        A tuple (provider, model_name) matching a record of the ModelRepository.
    """
    candidate = process.extractOne(
        query=model_name,
        choices=_model_choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=51
    )
    if candidate is not None:
        provider, model_name = candidate[0].split("/", 1)
        return provider, model_name
    return None


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

    if kwargs.get("stream", False):
        return litellm_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return litellm_chat_wrapper_non_stream(wrapped, instance, args, kwargs)


def litellm_chat_wrapper_stream(  # type: ignore[misc]
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> CustomStreamWrapper:
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    token_count = 0
    for i, chunk in enumerate(stream):
        if i > 0 and chunk.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start

        model_match = litellm_match_model(chunk.model)
        if model_match is not None:
            impacts = llm_impacts(
                provider=model_match[0],
                model_name=model_match[1],
                output_token_count=token_count,
                input_token_count=chunk.usage.prompt_tokens if getattr(chunk, "usage", None) else None,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
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

                yield ChatCompletionChunk(**chunk.model_dump(), impacts=impacts)
            else:
                yield chunk
        else:
            yield chunk


def litellm_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_match = litellm_match_model(response.model)
    if model_match is None:
        return response
    impacts = llm_impacts(
        provider=model_match[0],
        model_name=model_match[1],
        output_token_count=response.usage.completion_tokens,
        input_token_count=response.usage.prompt_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
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

        return ChatCompletion(**response.model_dump(), impacts=impacts)
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
    if kwargs.get("stream", False):
        return litellm_async_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return await litellm_async_chat_wrapper_base(wrapped, instance, args, kwargs)


async def litellm_async_chat_wrapper_base(
    wrapped: Callable,
    instance: AsyncCompletions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_match = litellm_match_model(response.model)
    if model_match is None:
        return response
    impacts = llm_impacts(
        provider=model_match[0],
        model_name=model_match[1],
        output_token_count=response.usage.completion_tokens,
        input_token_count=response.usage.prompt_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
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

        return ChatCompletion(**response.model_dump(), impacts=impacts)
    else:
        return response


async def litellm_async_chat_wrapper_stream(  # type: ignore[misc]
    wrapped: Callable,
    instance: AsyncCompletions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> CustomStreamWrapper:
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    i = 0
    token_count = 0
    async for chunk in stream:
        if i > 0 and chunk.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_match = litellm_match_model(chunk.model)
        if model_match is not None:
            impacts = llm_impacts(
                provider=model_match[0],
                model_name=model_match[1],
                output_token_count=token_count,
                input_token_count=chunk.usage.prompt_tokens if getattr(chunk, "usage", None) else None,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
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

                yield ChatCompletionChunk(**chunk.model_dump(), impacts=impacts)
            else:
                yield chunk
        else:
            yield chunk
        i += 1


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
