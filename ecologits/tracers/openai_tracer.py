import time
from typing import Any, Callable, Union

from openai import AsyncStream, Stream
from openai.resources.chat import AsyncCompletions, Completions
from openai.resources.responses.responses import AsyncResponses, Responses
from openai.types.chat import ChatCompletion as _ChatCompletion
from openai.types.chat import ChatCompletionChunk as _ChatCompletionChunk
from openai.types.responses import Response as _Response
from openai.types.responses import ResponseStreamEvent
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, llm_impacts

PROVIDER = "openai"


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


def openai_chat_wrapper(
    wrapped: Callable,
    instance: Completions,
    args: Any,
    kwargs: Any
) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
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
    if kwargs.get("stream", False):
        return openai_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return openai_chat_wrapper_non_stream(wrapped, instance, args, kwargs)


def openai_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = llm_impacts(
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
                endpoint="/chat/completions"
            )

        return ChatCompletion(**response.model_dump(), impacts=impacts)
    else:
        return response


def openai_chat_wrapper_stream(  # type: ignore[misc]
    wrapped: Callable,
    instance: Completions,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> Stream[ChatCompletionChunk]:
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    output_token_count = 0
    for i, chunk in enumerate(stream):
        # azure openai has an empty first chunk so we skip it
        if i == 0 and chunk.model == "":
            continue
        if i > 0 and chunk.choices and chunk.choices[0].finish_reason is None:
            output_token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.model
        impacts = llm_impacts(
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

            yield ChatCompletionChunk(**chunk.model_dump(), impacts=impacts)
        else:
            yield chunk


async def openai_async_chat_wrapper(
    wrapped: Callable,
    instance: AsyncCompletions,
    args: Any,
    kwargs: Any,
) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
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
    if kwargs.get("stream", False):
        return openai_async_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return await openai_async_chat_wrapper_base(wrapped, instance, args, kwargs)


async def openai_async_chat_wrapper_base(
    wrapped: Callable,
    instance: AsyncCompletions,     # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> ChatCompletion:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = response.model
    impacts = llm_impacts(
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
                endpoint="/chat/completions"
            )

        return ChatCompletion(**response.model_dump(), impacts=impacts)
    else:
        return response


async def openai_async_chat_wrapper_stream(  # type: ignore[misc]
    wrapped: Callable,
    instance: AsyncCompletions,     # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> AsyncStream[ChatCompletionChunk]:
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    i = 0
    output_token_count = 0
    async for chunk in stream:
        if i == 0 and chunk.model == "":
            continue
        if i > 0 and chunk.choices and chunk.choices[0].finish_reason is None:
            output_token_count += 1
        request_latency = time.perf_counter() - timer_start
        model_name = chunk.model
        impacts = llm_impacts(
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

            yield ChatCompletionChunk(**chunk.model_dump(), impacts=impacts)
        else:
            yield chunk
        i += 1


def openai_responses_wrapper(
    wrapped: Callable,
    instance: Responses,
    args: Any,
    kwargs: Any
) -> Union[Response, Stream[ResponseStreamEvent]]:
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
    if kwargs.get("stream", False):
        return openai_responses_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return openai_responses_wrapper_non_stream(wrapped, instance, args, kwargs)


def openai_responses_wrapper_non_stream(
    wrapped: Callable,
    instance: Responses,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> Response:
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
                endpoint="/responses"
            )

        return Response(**response.model_dump(), impacts=impacts)
    else:
        return response


def openai_responses_wrapper_stream(  # type: ignore[misc]
    wrapped: Callable,
    instance: Responses,      # noqa: ARG001
    args: Any,
    kwargs: Any
) -> Stream[ResponseStreamEvent]:
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    model_name = kwargs["model"]
    output_token_count = 0
    for event in stream:
        if event.type == "response.output_text.delta":
            output_token_count += 1
        request_latency = time.perf_counter() - timer_start
        if event.type == "response.completed":
            model_name = event.response.model
            output_token_count = event.response.usage.output_tokens
        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=output_token_count,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and event.type == "response.completed":
                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=event.response.usage.input_tokens,
                    output_tokens=output_token_count,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/responses"
                )
            if event.type == "response.completed":
                event.response = Response(**event.response.model_dump(), impacts=impacts)
            event.impacts = impacts
            yield event
        else:
            yield event


async def openai_async_responses_wrapper(
    wrapped: Callable,
    instance: AsyncResponses,
    args: Any,
    kwargs: Any,
) -> Union[Response, AsyncStream[ResponseStreamEvent]]:
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
    if kwargs.get("stream", False):
        return openai_async_responses_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return await openai_async_responses_wrapper_base(wrapped, instance, args, kwargs)


async def openai_async_responses_wrapper_base(
    wrapped: Callable,
    instance: AsyncResponses,     # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> Response:
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
                endpoint="/responses"
            )

        return Response(**response.model_dump(), impacts=impacts)
    else:
        return response


async def openai_async_responses_wrapper_stream(  # type: ignore[misc]
    wrapped: Callable,
    instance: AsyncResponses,     # noqa: ARG001
    args: Any,
    kwargs: Any,
) -> AsyncStream[ResponseStreamEvent]:
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    model_name = kwargs["model"]
    output_token_count = 0
    async for event in stream:
        if event.type == "response.output_text.delta":
            output_token_count += 1
        request_latency = time.perf_counter() - timer_start
        if event.type == "response.completed":
            model_name = event.response.model
            output_token_count = event.response.usage.output_tokens
        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=output_token_count,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and event.type == "response.completed":
                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=event.response.usage.input_tokens,
                    output_tokens=output_token_count,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/responses"
                )
            if event.type == "response.completed":
                event.response = Response(**event.response.model_dump(), impacts=impacts)
            event.impacts = impacts
            yield event
        else:
            yield event


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
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
