import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Optional

from openai import AsyncStream, Stream
from openai.resources.chat import AsyncCompletions, Completions
from openai.resources.responses.responses import AsyncResponses, Responses
from openai.types.chat import ChatCompletion as _ChatCompletion
from openai.types.chat import ChatCompletionChunk as _ChatCompletionChunk
from openai.types.chat import ParsedChatCompletion as _ParsedChatCompletion
from openai.types.responses import ParsedResponse as _ParsedResponse
from openai.types.responses import Response as _Response
from openai.types.responses import ResponseStreamEvent
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.model_repository import models
from ecologits.tracers.utils import (
    AsyncImpactsStream,
    ImpactsOutput,
    ImpactsStream,
    _llm_impacts,
    is_raw_response_request,
)
from ecologits.tracking import ImpactsTracker, get_active_trackers, is_tracking_enabled, record_impacts

PROVIDER = "openai"


def _registered_model_name(response_model: str, requested_model: Optional[str]) -> str:
    """Use the requested model when only it is registered, e.g. Azure returns a dated id for a deployment."""
    if (
        requested_model is not None
        and models.find_model(PROVIDER, response_model) is None
        and models.find_model(PROVIDER, requested_model) is not None
    ):
        return requested_model
    return response_model


class ChatCompletion(_ChatCompletion):
    """
    Wrapper of `openai.types.chat.ChatCompletion` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class ChatCompletionChunk(_ChatCompletionChunk):
    """
    Wrapper of `openai.types.chat.ChatCompletionChunk` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class Response(_Response):
    """
    Wrapper of `openai.types.responses.Response` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class ParsedResponse(_ParsedResponse):
    """
    Wrapper of `openai.types.responses.ParsedResponse` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


class ParsedChatCompletion(_ParsedChatCompletion):
    """
    Wrapper of `openai.types.chat.ParsedChatCompletion` with `ImpactsOutput`
    """
    impacts: ImpactsOutput


def openai_chat_wrapper(
    wrapped: Callable,
    instance: Completions,
    args: Any,
    kwargs: Any
) -> ChatCompletion | _ChatCompletion | Stream[ChatCompletionChunk] | Stream[_ChatCompletionChunk]:
    """
    Function that wraps an OpenAI answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletion` or `Stream[ChatCompletionChunk]` with impacts
    """
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    if kwargs.get("stream", False):
        timer_start = time.perf_counter()
        stream = wrapped(*args, **kwargs)
        return ImpactsStream(stream, openai_chat_wrapper_stream(stream, timer_start, kwargs, trackers))
    else:
        return openai_chat_wrapper_non_stream(wrapped, instance, args, kwargs, trackers)


def openai_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
    cls: type[_ChatCompletion] = ChatCompletion,
    endpoint: str = "/chat/completions",
) -> ChatCompletion | _ChatCompletion:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = _registered_model_name(response.model, kwargs.get("model"))
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
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
                model=model_name,
                endpoint=endpoint
            )

        record_impacts(trackers, impacts)
        return cls.model_construct(**dict(response), impacts=impacts)
    else:
        return response


def openai_chat_wrapper_stream(
    stream: Stream[_ChatCompletionChunk],
    timer_start: float,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
) -> Iterator[ChatCompletionChunk | _ChatCompletionChunk]:
    output_token_count = 0
    impacts: Optional[ImpactsOutput] = None
    for i, chunk in enumerate(stream):
        # azure openai has an empty first chunk so we skip it
        if i == 0 and chunk.model == "":
            continue
        if i > 0 and chunk.choices and chunk.choices[0].finish_reason is None:
            output_token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = _registered_model_name(chunk.model, kwargs.get("model"))
        impacts = _llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=output_token_count,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and chunk.choices[0].finish_reason is not None:
                import tiktoken

                # Compute input tokens
                encoder = tiktoken.get_encoding("cl100k_base")
                input_token_count =  len(encoder.encode(kwargs["messages"][0]["content"]))

                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=input_token_count,
                    output_tokens=output_token_count,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/chat/completions"
                )

            yield ChatCompletionChunk.model_construct(**dict(chunk), impacts=impacts)
        else:
            yield chunk
    if impacts is not None:
        record_impacts(trackers, impacts)


async def openai_async_chat_wrapper(
    wrapped: Callable,
    instance: AsyncCompletions,
    args: Any,
    kwargs: Any,
) -> ChatCompletion | _ChatCompletion | AsyncStream[ChatCompletionChunk]:
    """
    Function that wraps an OpenAI answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletion` or `AsyncStream[ChatCompletionChunk]` with impacts
    """
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    if kwargs.get("stream", False):
        timer_start = time.perf_counter()
        stream = await wrapped(*args, **kwargs)
        return AsyncImpactsStream(stream, openai_async_chat_wrapper_stream(stream, timer_start, kwargs, trackers))
    else:
        return await openai_async_chat_wrapper_base(wrapped, instance, args, kwargs, trackers)


async def openai_async_chat_wrapper_base(
    wrapped: Callable,
    instance: AsyncCompletions,     # noqa: ARG001
    args: Any,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
    cls: type[_ChatCompletion] = ChatCompletion,
    endpoint: str = "/chat/completions",
) -> ChatCompletion | _ChatCompletion:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = _registered_model_name(response.model, kwargs.get("model"))
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
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
                model=model_name,
                endpoint=endpoint
            )

        record_impacts(trackers, impacts)
        return cls.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def openai_async_chat_wrapper_stream(
    stream: AsyncStream[_ChatCompletionChunk],
    timer_start: float,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
) -> AsyncIterator[ChatCompletionChunk | _ChatCompletionChunk]:
    i = 0
    output_token_count = 0
    impacts: Optional[ImpactsOutput] = None
    async for chunk in stream:
        if i == 0 and chunk.model == "":
            continue
        if i > 0 and chunk.choices and chunk.choices[0].finish_reason is None:
            output_token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = _registered_model_name(chunk.model, kwargs.get("model"))
        impacts = _llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=output_token_count,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and chunk.choices[0].finish_reason is not None:
                import tiktoken

                # Compute input tokens
                encoder = tiktoken.get_encoding("cl100k_base")
                input_token_count =  len(encoder.encode(kwargs["messages"][0]["content"]))

                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=input_token_count,
                    output_tokens=output_token_count,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/chat/completions"
                )

            yield ChatCompletionChunk.model_construct(**dict(chunk), impacts=impacts)
        else:
            yield chunk
        i += 1
    if impacts is not None:
        record_impacts(trackers, impacts)


def openai_responses_wrapper(
    wrapped: Callable,
    instance: Responses,
    args: Any,
    kwargs: Any
) -> Response | _Response | Stream[ResponseStreamEvent]:
    """
    Function that wraps an OpenAI Responses API answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `Response` or `Stream[ResponseStreamEvent]` with impacts
    """
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    if kwargs.get("stream", False):
        timer_start = time.perf_counter()
        stream = wrapped(*args, **kwargs)
        return ImpactsStream(stream, openai_responses_wrapper_stream(stream, timer_start, kwargs, trackers))
    else:
        return openai_responses_wrapper_non_stream(wrapped, instance, args, kwargs, trackers)


def openai_responses_wrapper_non_stream(
    wrapped: Callable,
    instance: Responses,      # noqa: ARG001
    args: Any,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
) -> Response | _Response:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = _registered_model_name(response.model, kwargs.get("model"))
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
                endpoint="/responses"
            )

        record_impacts(trackers, impacts)
        return Response.model_construct(**dict(response), impacts=impacts)
    else:
        return response


def openai_responses_wrapper_stream(
    stream: Stream[ResponseStreamEvent],
    timer_start: float,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
) -> Iterator[ResponseStreamEvent]:
    for event in stream:
        if event.type == "response.completed":
            request_latency = time.perf_counter() - timer_start
            model_name = _registered_model_name(event.response.model, kwargs.get("model"))
            output_token_count = event.response.usage.output_tokens
            impacts = _llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_token_count,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
            if impacts is not None:
                if EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=output_token_count,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint="/responses"
                    )
                event.response = Response.model_construct(**dict(event.response), impacts=impacts)
                event.impacts = impacts
                record_impacts(trackers, impacts)
        yield event


async def openai_async_responses_wrapper(
    wrapped: Callable,
    instance: AsyncResponses,
    args: Any,
    kwargs: Any,
) -> Response | _Response | AsyncStream[ResponseStreamEvent]:
    """
    Function that wraps an OpenAI Responses API answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `Response` or `AsyncStream[ResponseStreamEvent]` with impacts
    """
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    if kwargs.get("stream", False):
        timer_start = time.perf_counter()
        stream = await wrapped(*args, **kwargs)
        return AsyncImpactsStream(stream, openai_async_responses_wrapper_stream(stream, timer_start, kwargs, trackers))
    else:
        return await openai_async_responses_wrapper_base(wrapped, instance, args, kwargs, trackers)


async def openai_async_responses_wrapper_base(
    wrapped: Callable,
    instance: AsyncResponses,     # noqa: ARG001
    args: Any,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
) -> Response | _Response:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = _registered_model_name(response.model, kwargs.get("model"))
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
                endpoint="/responses"
            )
        record_impacts(trackers, impacts)
        return Response.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def openai_async_responses_wrapper_stream(
    stream: AsyncStream[ResponseStreamEvent],
    timer_start: float,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...] = (),
) -> AsyncIterator[ResponseStreamEvent]:
    async for event in stream:
        if event.type == "response.completed":
            request_latency = time.perf_counter() - timer_start
            model_name = _registered_model_name(event.response.model, kwargs.get("model"))
            output_token_count = event.response.usage.output_tokens
            impacts = _llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_token_count,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone
            )
            if impacts is not None:
                if EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=output_token_count,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint="/responses"
                    )
                event.response = Response.model_construct(**dict(event.response), impacts=impacts)
                event.impacts = impacts
                record_impacts(trackers, impacts)
        yield event


def openai_responses_parse_wrapper(
    wrapped: Callable,
    instance: Responses,  # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> ParsedResponse | _ParsedResponse:
    if not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = _registered_model_name(response.model, kwargs.get("model"))
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=response.usage.output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone,
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
                endpoint="/responses/parse",
            )
        record_impacts(trackers, impacts)
        return ParsedResponse.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def openai_async_responses_parse_wrapper(
    wrapped: Callable,
    instance: AsyncResponses,  # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> ParsedResponse | _ParsedResponse:
    if not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = _registered_model_name(response.model, kwargs.get("model"))
    impacts = _llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=response.usage.output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone,
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
                endpoint="/responses/parse",
            )

        record_impacts(trackers, impacts)
        return ParsedResponse.model_construct(**dict(response), impacts=impacts)
    else:
        return response


def openai_chat_parse_wrapper(
    wrapped: Callable,
    instance: Completions,
    args: Any,
    kwargs: Any,
) -> _ChatCompletion:
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    return openai_chat_wrapper_non_stream(
        wrapped, instance, args, kwargs, get_active_trackers(PROVIDER),
        cls=ParsedChatCompletion, endpoint="/chat/completions/parse",
    )


async def openai_async_chat_parse_wrapper(
    wrapped: Callable,
    instance: AsyncCompletions,
    args: Any,
    kwargs: Any,
) -> _ChatCompletion:
    if is_raw_response_request(kwargs) or not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    return await openai_async_chat_wrapper_base(
        wrapped, instance, args, kwargs, get_active_trackers(PROVIDER),
        cls=ParsedChatCompletion, endpoint="/chat/completions/parse",
    )


class OpenAIInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all OpenAI calls
    """
    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "openai.resources.chat.completions",
                "name": "Completions.create",
                "wrapper": openai_chat_wrapper,
            },
            {
                "module": "openai.resources.chat.completions",
                "name": "AsyncCompletions.create",
                "wrapper": openai_async_chat_wrapper,
            },
            {
                "module": "openai.resources.responses.responses",
                "name": "Responses.create",
                "wrapper": openai_responses_wrapper,
            },
            {
                "module": "openai.resources.responses.responses",
                "name": "AsyncResponses.create",
                "wrapper": openai_async_responses_wrapper,
            },
            {
                "module": "openai.resources.responses.responses",
                "name": "Responses.parse",
                "wrapper": openai_responses_parse_wrapper,
            },
            {
                "module": "openai.resources.responses.responses",
                "name": "AsyncResponses.parse",
                "wrapper": openai_async_responses_parse_wrapper
            }
        ]
        # Older openai 1.x releases only have `client.beta.chat.completions.parse`.
        if hasattr(Completions, "parse"):
            self.wrapped_methods += [
                {
                    "module": "openai.resources.chat.completions",
                    "name": "Completions.parse",
                    "wrapper": openai_chat_parse_wrapper,
                },
                {
                    "module": "openai.resources.chat.completions",
                    "name": "AsyncCompletions.parse",
                    "wrapper": openai_async_chat_parse_wrapper,
                },
            ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
