import time
from collections.abc import AsyncIterator, Awaitable, Iterator
from types import TracebackType
from typing import Any, Callable, Generic, Optional, TypeVar

from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.streaming import AsyncMessageStream as _AsyncMessageStream
from anthropic.lib.streaming import MessageStream as _MessageStream
from anthropic.types import Message as _Message
from anthropic.types.message_delta_event import MessageDeltaEvent
from anthropic.types.message_start_event import MessageStartEvent
from typing_extensions import override
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, llm_impacts

PROVIDER = "anthropic"

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

    @override
    def __stream_text__(self) -> Iterator[str]:  # type: ignore[misc]
        timer_start = time.perf_counter()
        output_tokens = 0
        model_name = None
        input_tokens = 0
        for chunk in self:
            if type(chunk) is MessageStartEvent:
                message = chunk.message
                model_name = message.model
                input_tokens = message.usage.input_tokens
                output_tokens += message.usage.output_tokens
            elif type(chunk) is MessageDeltaEvent:
                output_tokens += chunk.usage.output_tokens
            elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text
        request_latency = time.perf_counter() - timer_start
        if model_name is not None:
            impacts = llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_tokens,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
            self.impacts = impacts

            if impacts is not None \
                and EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint="/messages"
                    )



class AsyncMessageStream(_AsyncMessageStream):
    """
    Wrapper of `anthropic.lib.streaming.AsyncMessageStream` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None

    @override
    async def __stream_text__(self) -> AsyncIterator[str]:  # type: ignore[misc]
        timer_start = time.perf_counter()
        output_tokens = 0
        model_name = None
        input_tokens = 0
        async for chunk in self:
            if type(chunk) is MessageStartEvent:
                message = chunk.message
                model_name = message.model
                input_tokens = message.usage.input_tokens
                output_tokens += message.usage.output_tokens
            elif type(chunk) is MessageDeltaEvent:
                output_tokens += chunk.usage.output_tokens
            elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text
        request_latency = time.perf_counter() - timer_start
        if model_name is not None:
            impacts = llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_tokens,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
            self.impacts = impacts

            if impacts is not None \
                and EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint="/messages"
                    )



class MessageStreamManager(Generic[MessageStreamT]):
    """
    Re-writing of Anthropic's `MessageStreamManager` with wrapped `MessageStream`
    """

    def __init__(self, api_request: Callable[[], Any], output_format: Any) -> None:
        self.__api_request = api_request
        self.__output_format = output_format

    def __enter__(self) -> MessageStream:
        raw_stream = self.__api_request()
        self.__stream = MessageStream(raw_stream, output_format=self.__output_format)
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
    def __init__(self, api_request: Awaitable[Any], output_format: Any) -> None:
        self.__api_request = api_request
        self.__output_format = output_format

    async def __aenter__(self) -> AsyncMessageStream:
        raw_stream = await self.__api_request
        self.__stream = AsyncMessageStream(raw_stream, output_format=self.__output_format)
        return self.__stream

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        if self.__stream is not None:
            await self.__stream.close()


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

    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = llm_impacts(
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

        return Message(**response.model_dump(), impacts=impacts)
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

    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = llm_impacts(
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

        return Message(**response.model_dump(), impacts=impacts)
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
    return MessageStreamManager(  # noqa: SLF001
        response._MessageStreamManager__api_request,
        response._MessageStreamManager__output_format,
    )


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
    return AsyncMessageStreamManager(  # noqa: SLF001
        response._AsyncMessageStreamManager__api_request,
        response._AsyncMessageStreamManager__output_format,
    )


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
