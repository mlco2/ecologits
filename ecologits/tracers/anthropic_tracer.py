import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Iterable, Iterator
from types import TracebackType
from typing import Any, Callable, Generic, Optional, TypeVar

from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.streaming import AsyncMessageStream as _AsyncMessageStream
from anthropic.lib.streaming import MessageStream as _MessageStream
from anthropic.types import Message as _Message
from typing_extensions import override
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import (
    AsyncImpactsStream,
    ImpactsOutput,
    ImpactsStream,
    _llm_impacts,
    is_raw_response_request,
)
from ecologits.tracking import ImpactsTracker, get_active_trackers, is_tracking_enabled, record_impacts

PROVIDER = "anthropic"

# Older SDKs have no `output_format` attribute; an explicit `None` must still be passed to `MessageStream`.
_MISSING: Any = object()

MessageStreamT = TypeVar("MessageStreamT", bound=_MessageStream)
AsyncMessageStreamT = TypeVar("AsyncMessageStreamT", bound=_AsyncMessageStream)


class Message(_Message):
    """
    Wrapper of `anthropic.types.Message` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class MessageStream(_MessageStream):
    """
    Wrapper of `anthropic.lib.streaming.MessageStream` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None
    impacts_trackers: tuple[ImpactsTracker, ...] = ()

    @override
    def __stream__(self) -> Iterator[Any]:  # type: ignore[misc]
        for event in _stream_events(super().__stream__(), time.perf_counter(), self.impacts_trackers):
            if event.type == "message_stop":
                self.impacts = event.impacts
            yield event


class AsyncMessageStream(_AsyncMessageStream):
    """
    Wrapper of `anthropic.lib.streaming.AsyncMessageStream` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None
    impacts_trackers: tuple[ImpactsTracker, ...] = ()

    @override
    async def __stream__(self) -> AsyncIterator[Any]:  # type: ignore[misc]
        async for event in _async_stream_events(super().__stream__(), time.perf_counter(), self.impacts_trackers):
            if event.type == "message_stop":
                self.impacts = event.impacts
            yield event


class MessageStreamManager(Generic[MessageStreamT]):
    """
    Re-writing of Anthropic's `MessageStreamManager` with wrapped `MessageStream`
    """

    def __init__(
        self, api_request: Callable[[], Any], output_format: Any, trackers: tuple[ImpactsTracker, ...] = ()
    ) -> None:
        self.__api_request = api_request
        self.__output_format = output_format
        self.__trackers = trackers

    def __enter__(self) -> MessageStream:
        raw_stream = self.__api_request()
        stream_options = {"output_format": self.__output_format} if self.__output_format is not _MISSING else {}
        self.__stream = MessageStream(raw_stream, **stream_options)
        self.__stream.impacts_trackers = self.__trackers
        return self.__stream

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        if self.__stream is not None:
            self.__stream.close()


class AsyncMessageStreamManager(Generic[AsyncMessageStreamT]):
    """
    Re-writing of Anthropic's `AsyncMessageStreamManager` with wrapped `AsyncMessageStream`
    """
    def __init__(
        self, api_request: Awaitable[Any], output_format: Any, trackers: tuple[ImpactsTracker, ...] = ()
    ) -> None:
        self.__api_request = api_request
        self.__output_format = output_format
        self.__trackers = trackers

    async def __aenter__(self) -> AsyncMessageStream:
        raw_stream = await self.__api_request
        stream_options = {"output_format": self.__output_format} if self.__output_format is not _MISSING else {}
        self.__stream = AsyncMessageStream(raw_stream, **stream_options)
        self.__stream.impacts_trackers = self.__trackers
        return self.__stream

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        if self.__stream is not None:
            await self.__stream.close()


def _stream_impacts(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...],
) -> ImpactsOutput:
    request_latency = time.perf_counter() - timer_start
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
    record_impacts(trackers, impacts)
    if impacts is not None and EcoLogits.config.opentelemetry:
        EcoLogits.config.opentelemetry.record_request(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_latency=request_latency,
            impacts=impacts,
            provider=PROVIDER,
            model=model_name,
            endpoint="/messages"
        )
    return impacts


def _stream_events(
    stream: Iterable[Any],
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...],
) -> Iterator[Any]:
    model_name, input_tokens, output_tokens = "", 0, 0
    for event in stream:
        if event.type == "message_start":
            model_name = event.message.model
            input_tokens = event.message.usage.input_tokens
            output_tokens += event.message.usage.output_tokens
        elif event.type == "message_delta":
            output_tokens += event.usage.output_tokens
        elif event.type == "message_stop":
            event.impacts = _stream_impacts(model_name, input_tokens, output_tokens, timer_start, trackers)
        yield event


async def _async_stream_events(
    stream: AsyncIterable[Any],
    timer_start: float,
    trackers: tuple[ImpactsTracker, ...],
) -> AsyncIterator[Any]:
    model_name, input_tokens, output_tokens = "", 0, 0
    async for event in stream:
        if event.type == "message_start":
            model_name = event.message.model
            input_tokens = event.message.usage.input_tokens
            output_tokens += event.message.usage.output_tokens
        elif event.type == "message_delta":
            output_tokens += event.usage.output_tokens
        elif event.type == "message_stop":
            event.impacts = _stream_impacts(model_name, input_tokens, output_tokens, timer_start, trackers)
        yield event


def anthropic_chat_wrapper(
    wrapped: Callable, instance: Anthropic, args: Any, kwargs: Any  # noqa: ARG001
) -> Message:
    """
    Function that wraps an Anthropic answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `Message` with impacts
    """
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)

    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    if kwargs.get("stream", False):
        return ImpactsStream(response, _stream_events(response, timer_start, trackers))
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=response.usage.output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=PROVIDER,
                model=model_name,
                endpoint="/messages"
            )

        record_impacts(trackers, impacts)
        return Message.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def anthropic_async_chat_wrapper(
    wrapped: Callable, instance: AsyncAnthropic, args: Any, kwargs: Any  # noqa: ARG001
) -> Message:
    """
    Function that wraps an Anthropic answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `Message` with impacts
    """
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)

    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    if kwargs.get("stream", False):
        return AsyncImpactsStream(response, _async_stream_events(response, timer_start, trackers))
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=response.usage.output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
    )
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=PROVIDER,
                model=model_name,
                endpoint="/messages"
            )

        record_impacts(trackers, impacts)
        return Message.model_construct(**dict(response), impacts=impacts)
    else:
        return response


def anthropic_stream_chat_wrapper(
    wrapped: Callable, instance: Anthropic, args: Any, kwargs: Any  # noqa: ARG001
) -> MessageStreamManager:
    """
    Function that wraps an Anthropic answer with computed impacts in streaming mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `MessageStreamManager` with impacts
    """
    response = wrapped(*args, **kwargs)
    if not is_tracking_enabled(PROVIDER):
        return response
    api_request = response._MessageStreamManager__api_request  # noqa: SLF001
    output_format = getattr(response, "_MessageStreamManager__output_format", _MISSING)
    return MessageStreamManager(api_request, output_format, get_active_trackers(PROVIDER))


def anthropic_async_stream_chat_wrapper(
    wrapped: Callable, instance: AsyncAnthropic, args: Any, kwargs: Any  # noqa: ARG001
) -> AsyncMessageStreamManager:
    """
    Function that wraps an Anthropic answer with computed impacts in streaming and async mode

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `AsyncMessageStreamManager` with impacts
    """
    response = wrapped(*args, **kwargs)
    if not is_tracking_enabled(PROVIDER):
        return response
    api_request = response._AsyncMessageStreamManager__api_request  # noqa: SLF001
    output_format = getattr(response, "_AsyncMessageStreamManager__output_format", _MISSING)
    return AsyncMessageStreamManager(api_request, output_format, get_active_trackers(PROVIDER))


class AnthropicInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all Anthropic calls
    """
    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "anthropic.resources",
                "name": "Messages.create",
                "wrapper": anthropic_chat_wrapper,
            },
            {
                "module": "anthropic.resources",
                "name": "AsyncMessages.create",
                "wrapper": anthropic_async_chat_wrapper,
            },
            {
                "module": "anthropic.resources",
                "name": "Messages.stream",
                "wrapper": anthropic_stream_chat_wrapper,
            },
            {
                "module": "anthropic.resources",
                "name": "AsyncMessages.stream",
                "wrapper": anthropic_async_stream_chat_wrapper,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
