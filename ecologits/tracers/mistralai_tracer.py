import time
from collections.abc import AsyncGenerator, Iterable
from typing import Any, Callable

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse as _ChatCompletionResponse
from mistralai.models import CompletionChunk as _CompletionChunk
from mistralai.models import CompletionEvent
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, llm_impacts

PROVIDER = "mistralai"


class ChatCompletionResponse(_ChatCompletionResponse):
    """
    Wrapper of `mistralai.models.ChatCompletionResponse` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class CompletionChunk(_CompletionChunk):
    """
    Wrapper of `mistralai.models.CompletionChunk` with `ImpactsOutput`
    """
    impacts: ImpactsOutput



def mistralai_chat_wrapper(
    wrapped: Callable, instance: Mistral, args: Any, kwargs: Any  # noqa: ARG001
) -> ChatCompletionResponse:
    """
    Function that wraps a MistralAI answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletionResponse` with impacts
    """
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=response.model,
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
                provider=PROVIDER,
                model=response.model,
                endpoint="/chat/completions"
            )

        return ChatCompletionResponse(**response.model_dump(), impacts=impacts)
    else:
        return response


def mistralai_chat_wrapper_stream(
    wrapped: Callable, instance: Mistral, args: Any, kwargs: Any  # noqa: ARG001
) -> Iterable[CompletionEvent]:
    """
    Function that wraps a MistralAI answer with computed impacts in streaming mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `Iterable[CompletionEvent]` with impacts
    """
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    token_count = 0
    for i, chunk in enumerate(stream):
        if i > 0 and chunk.data.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.data.model
        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=token_count,
            input_token_count=chunk.data.usage.prompt_tokens if chunk.data.usage is not None else None,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                and chunk.data.choices[0].finish_reason is not None:

                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=chunk.data.usage.prompt_tokens,
                    output_tokens=chunk.data.usage.completion_tokens,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/chat/completions"
                )

            chunk.data = CompletionChunk(**chunk.data.model_dump(), impacts=impacts)
            yield chunk
        else:
            yield chunk


async def mistralai_async_chat_wrapper(
    wrapped: Callable,
    instance: Mistral,  # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> ChatCompletionResponse:
    """
    Function that wraps a MistralAI answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletionResponse` with impacts
    """
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=response.model,
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
                provider=PROVIDER,
                model=response.model,
                endpoint="/chat/completions"
            )

        return ChatCompletionResponse(**response.model_dump(), impacts=impacts)
    else:
        return response

async def _generator(
    stream: AsyncGenerator[CompletionEvent, None],
    timer_start: float
) -> AsyncGenerator[CompletionEvent, None]:
    token_count = 0
    async for chunk in stream:
        if chunk.data.usage is not None:
            token_count = chunk.data.usage.completion_tokens
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.data.model
        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=token_count,
            input_token_count=chunk.data.usage.prompt_tokens if chunk.data.usage is not None else None,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                and chunk.data.choices[0].finish_reason is not None:

                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=chunk.data.usage.prompt_tokens,
                    output_tokens=chunk.data.usage.completion_tokens,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/chat/completions"
                )

            chunk.data = CompletionChunk(**chunk.data.model_dump(), impacts=impacts)
            yield chunk
        else:
            yield chunk


async def mistralai_async_chat_wrapper_stream(
    wrapped: Callable,
    instance: Mistral,  # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> AsyncGenerator[CompletionEvent, None]:
    """
    Function that wraps a MistralAI answer with computed impacts in streaming and async mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `AsyncGenerator[CompletionEvent, None]` with impacts
    """
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    return _generator(stream, timer_start)


class MistralAIInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all MistralAI calls
    """
    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "mistralai.chat",
                "name": "Chat.complete",
                "wrapper": mistralai_chat_wrapper,
            },
            {
                "module": "mistralai.chat",
                "name": "Chat.complete_async",
                "wrapper": mistralai_async_chat_wrapper,
            },
            {
                "module": "mistralai.chat",
                "name": "Chat.stream",
                "wrapper": mistralai_chat_wrapper_stream,
            },
            {
                "module": "mistralai.chat",
                "name": "Chat.stream_async",
                "wrapper": mistralai_async_chat_wrapper_stream,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
