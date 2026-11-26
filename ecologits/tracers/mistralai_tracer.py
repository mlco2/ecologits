import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from importlib.util import find_spec
from types import TracebackType
from typing import Any, Callable, Optional

from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, _llm_impacts
from ecologits.tracking import ImpactsTracker, get_active_trackers, is_tracking_enabled, record_impacts

if find_spec("mistralai.chat") is not None:
    from mistralai.chat import Chat
    from mistralai.models import ChatCompletionResponse as _ChatCompletionResponse
    from mistralai.models import CompletionChunk as _CompletionChunk
    from mistralai.models import CompletionEvent
else:
    from mistralai.client.chat import Chat
    from mistralai.client.models import ChatCompletionResponse as _ChatCompletionResponse
    from mistralai.client.models import CompletionChunk as _CompletionChunk
    from mistralai.client.models import CompletionEvent

PROVIDER = "mistralai"


class ChatCompletionResponse(_ChatCompletionResponse):
    """
    Wrapper of `mistralai.client.models.ChatCompletionResponse` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class CompletionChunk(_CompletionChunk):
    """
    Wrapper of `mistralai.client.models.CompletionChunk` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class EventStream:
    """
    Wrapper of the SDK's `EventStream` that adds impacts to each chunk while keeping its interface (context manager,
    `close()`, `response`)
    """

    def __init__(self, stream: Any, timer_start: float, trackers: tuple[ImpactsTracker, ...]) -> None:
        self._stream = stream
        self._events = _stream_generator(stream, timer_start, trackers)

    def __iter__(self) -> "EventStream":
        return self

    def __next__(self) -> CompletionEvent:
        return next(self._events)

    def __enter__(self) -> "EventStream":
        self._stream.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        self._stream.__exit__(exc_type, exc, exc_tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class EventStreamAsync:
    """
    Wrapper of the SDK's `EventStreamAsync` that adds impacts to each chunk while keeping its interface (async context
    manager, `close()`, `response`)
    """

    def __init__(self, stream: Any, timer_start: float, trackers: tuple[ImpactsTracker, ...]) -> None:
        self._stream = stream
        self._events = _generator(stream, timer_start, trackers)

    def __aiter__(self) -> "EventStreamAsync":
        return self

    async def __anext__(self) -> CompletionEvent:
        return await self._events.__anext__()

    async def __aenter__(self) -> "EventStreamAsync":
        await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        await self._stream.__aexit__(exc_type, exc, exc_tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def mistralai_chat_wrapper(
    wrapped: Callable, instance: Chat, args: Any, kwargs: Any  # noqa: ARG001
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
    if not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=response.model,
        output_token_count=response.usage.completion_tokens,
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

        record_impacts(trackers, impacts)
        return ChatCompletionResponse.model_construct(**dict(response), impacts=impacts)
    else:
        return response


def mistralai_chat_wrapper_stream(
    wrapped: Callable, instance: Chat, args: Any, kwargs: Any  # noqa: ARG001
) -> EventStream:
    """
    Function that wraps a MistralAI answer with computed impacts in streaming mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `EventStream` with impacts
    """
    if not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    return EventStream(stream, timer_start, trackers)


def _stream_generator(
    stream: Iterator[CompletionEvent],
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...]
) -> Iterator[CompletionEvent]:
    token_count = 0
    impacts: ImpactsOutput | None = None
    for i, chunk in enumerate(stream):
        if i > 0 and chunk.data.choices[0].finish_reason is None:
            token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.data.model
        impacts = _llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=token_count,
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

            chunk.data = CompletionChunk.model_construct(**dict(chunk.data), impacts=impacts)
            yield chunk
        else:
            yield chunk
    if impacts is not None:
        record_impacts(trackers, impacts)


async def mistralai_async_chat_wrapper(
    wrapped: Callable,
    instance: Chat,  # noqa: ARG001
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
    if not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=response.model,
        output_token_count=response.usage.completion_tokens,
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

        record_impacts(trackers, impacts)
        return ChatCompletionResponse.model_construct(**dict(response), impacts=impacts)
    else:
        return response

async def _generator(
    stream: AsyncIterator[CompletionEvent],
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...] = ()
) -> AsyncGenerator[CompletionEvent, None]:
    token_count = 0
    impacts: ImpactsOutput | None = None
    async for chunk in stream:
        if chunk.data.usage is not None:
            token_count = chunk.data.usage.completion_tokens
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.data.model
        impacts = _llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=token_count,
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

            chunk.data = CompletionChunk.model_construct(**dict(chunk.data), impacts=impacts)
            yield chunk
        else:
            yield chunk
    if impacts is not None:
        record_impacts(trackers, impacts)


async def mistralai_async_chat_wrapper_stream(
    wrapped: Callable,
    instance: Chat,  # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> EventStreamAsync:
    """
    Function that wraps a MistralAI answer with computed impacts in streaming and async mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `EventStreamAsync` with impacts
    """
    if not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    return EventStreamAsync(stream, timer_start, trackers)


class MistralAIInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all MistralAI calls
    """
    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": Chat.__module__,
                "name": "Chat.complete",
                "wrapper": mistralai_chat_wrapper,
            },
            {
                "module": Chat.__module__,
                "name": "Chat.complete_async",
                "wrapper": mistralai_async_chat_wrapper,
            },
            {
                "module": Chat.__module__,
                "name": "Chat.stream",
                "wrapper": mistralai_chat_wrapper_stream,
            },
            {
                "module": Chat.__module__,
                "name": "Chat.stream_async",
                "wrapper": mistralai_async_chat_wrapper_stream,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
