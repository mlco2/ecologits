import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Optional

from cohere import AsyncClient, Client
from cohere.types.non_streamed_chat_response import NonStreamedChatResponse as _NonStreamedChatResponse
from cohere.types.streamed_chat_response import StreamedChatResponse
from cohere.types.streamed_chat_response import StreamEndStreamedChatResponse as _StreamEndStreamedChatResponse
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, llm_impacts

PROVIDER = "cohere"


class NonStreamedChatResponse(_NonStreamedChatResponse):
    """
    Wrapper of `cohere.types.non_streamed_chat_response.NonStreamedChatResponse` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None

    class Config:
        arbitrary_types_allowed = True


class StreamEndStreamedChatResponse(_StreamEndStreamedChatResponse):
    """
    Wrapper of `cohere.types.streamed_chat_response.StreamEndStreamedChatResponse` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None

    class Config:
        arbitrary_types_allowed = True


def cohere_chat_wrapper(
    wrapped: Callable, instance: Client, args: Any, kwargs: Any  # noqa: ARG001
) -> NonStreamedChatResponse:
    """
    Function that wraps a Cohere answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `NonStreamedChatResponse` with impacts
    """
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    output_tokens = response.meta.tokens.output_tokens
    model_name = kwargs.get("model", "command-r")
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        input_token_count=response.meta.tokens.input_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=response.meta.tokens.input_tokens,
                output_tokens=response.meta.tokens.output_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=PROVIDER,
                model=model_name,
                endpoint="/chat"
            )

        return NonStreamedChatResponse(**response.dict(), impacts=impacts)
    else:
        return response


async def cohere_async_chat_wrapper(
    wrapped: Callable, instance: AsyncClient, args: Any, kwargs: Any    # noqa: ARG001
) -> NonStreamedChatResponse:
    """
    Function that wraps a Cohere answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `NonStreamedChatResponse` with impacts
    """
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    output_tokens = response.meta.tokens.output_tokens
    model_name = kwargs.get("model", "command-r")
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        input_token_count=response.meta.tokens.input_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=response.meta.tokens.input_tokens,
                output_tokens=response.meta.tokens.output_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=PROVIDER,
                model=model_name,
                endpoint="/chat"
            )

        return NonStreamedChatResponse(**response.dict(), impacts=impacts)
    else:
        return response


def cohere_stream_chat_wrapper(
    wrapped: Callable, instance: Client, args: Any, kwargs: Any # noqa: ARG001
) -> Iterator[StreamedChatResponse]:
    """
    Function that wraps a Cohere answer with computed impacts in streaming mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `Iterator[StreamedChatResponse]` with impacts
    """

    model_name = kwargs.get("model", "command-r")
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    for event in stream:
        if event.event_type == "stream-end":
            request_latency = time.perf_counter() - timer_start
            output_tokens = event.response.meta.tokens.output_tokens
            impacts = llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_tokens,
                input_token_count=event.response.meta.tokens.input_tokens,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )

            if impacts is not None:
                if EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=event.response.meta.tokens.input_tokens,
                        output_tokens=event.response.meta.tokens.output_tokens,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint="/chat"
                    )

                yield StreamEndStreamedChatResponse(**event.dict(), impacts=impacts)
            else:
                yield event
        else:
            yield event


async def cohere_async_stream_chat_wrapper(
    wrapped: Callable, instance: AsyncClient, args: Any, kwargs: Any # noqa: ARG001
) -> AsyncIterator[StreamedChatResponse]:
    """
    Function that wraps a Cohere answer with computed impacts in streaming and async mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `AsyncIterator[StreamedChatResponse]` with impacts
    """

    model_name = kwargs.get("model", "command-r")
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    async for event in stream:
        if event.event_type == "stream-end":
            request_latency = time.perf_counter() - timer_start
            output_tokens = event.response.meta.tokens.output_tokens
            impacts = llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_tokens,
                input_token_count=event.response.meta.tokens.input_tokens,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
            if impacts is not None:
                if EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=event.response.meta.tokens.input_tokens,
                        output_tokens=event.response.meta.tokens.output_tokens,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint="/chat"
                    )

                yield StreamEndStreamedChatResponse(**event.dict(), impacts=impacts)
            else:
                yield event
        else:
            yield event


class CohereInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all Cohere calls
    """

    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "cohere.base_client",
                "name": "BaseCohere.chat",
                "wrapper": cohere_chat_wrapper,
            },
            {
                "module": "cohere.base_client",
                "name": "AsyncBaseCohere.chat",
                "wrapper": cohere_async_chat_wrapper,
            },
            {
                "module": "cohere.base_client",
                "name": "BaseCohere.chat_stream",
                "wrapper": cohere_stream_chat_wrapper,
            },
            {
                "module": "cohere.base_client",
                "name": "AsyncBaseCohere.chat_stream",
                "wrapper": cohere_async_stream_chat_wrapper,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )

