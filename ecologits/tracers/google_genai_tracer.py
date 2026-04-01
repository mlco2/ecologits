import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Optional

from google.genai.models import Models
from google.genai.types import GenerateContentResponse as _GenerateContentResponse
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, llm_impacts

PROVIDER = "google_genai"


class GenerateContentResponse(_GenerateContentResponse):
    """
    Wrapper of `google.genai.types.GenerateContentResponse` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None


def google_genai_content_wrapper(
    wrapped: Callable,
    instance: Models,   # noqa: ARG001
    args: Any,
    kwargs: Any
) -> GenerateContentResponse:
    """
    Function that wraps Google GenAI answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `GenerateContentResponse` with impacts
    """
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = kwargs["model"]
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count + (response.usage_metadata.thoughts_token_count or 0)
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone,
    )
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=PROVIDER,
                model=model_name,
                endpoint=f"/v1beta/models/{model_name}:generateContent"
            )

        return GenerateContentResponse(**response.model_dump(), impacts=impacts)
    else:
        return response


def google_genai_content_stream_wrapper(
    wrapped: Callable,
    instance: Models,   # noqa: ARG001
    args: Any,
    kwargs: Any
) -> Iterator[GenerateContentResponse]:
    """
    Function that wraps Google GenAI answer with computed impacts in streaming mode.

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Yields:
        A wrapped `GenerateContentResponse` with impacts
    """
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    for chunk in stream:
        if chunk.candidates[0].finish_reason is None:
            yield GenerateContentResponse(**chunk.model_dump(), impacts=None)

        else:
            request_latency = time.perf_counter() - timer_start
            model_name = kwargs["model"]
            input_tokens = chunk.usage_metadata.prompt_token_count
            output_tokens = chunk.usage_metadata.candidates_token_count + (chunk.usage_metadata.thoughts_token_count or 0)
            impacts = llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_tokens,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone,
            )
            if impacts is not None:
                if EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint=f"/v1beta/models/{model_name}:generateContent"
                    )

                yield GenerateContentResponse(**chunk.model_dump(), impacts=impacts)
            else:
                yield GenerateContentResponse(**chunk.model_dump(), impacts=None)


async def google_genai_async_content_wrapper(
    wrapped: Callable,
    instance: Models,   # noqa: ARG001
    args: Any,
    kwargs: Any
) -> GenerateContentResponse:
    """
    Function that wraps Google GenAI answer with computed impacts in async mode.

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `GenerateContentResponse` with impacts
    """
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = kwargs["model"]
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count + (response.usage_metadata.thoughts_token_count or 0)
    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone,
    )
    if impacts is not None:
        if EcoLogits.config.opentelemetry:
            EcoLogits.config.opentelemetry.record_request(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_latency=request_latency,
                impacts=impacts,
                provider=PROVIDER,
                model=model_name,
                endpoint=f"/v1beta/models/{model_name}:generateContent"
            )

        return GenerateContentResponse(**response.model_dump(), impacts=impacts)
    else:
        return response


async def _generator(
    stream: AsyncIterator[GenerateContentResponse],
    timer_start: float,
    model_name: str
) -> AsyncIterator[GenerateContentResponse]:
    async for chunk in stream:
        if chunk.candidates[0].finish_reason is None:
            yield GenerateContentResponse(**chunk.model_dump(), impacts=None)

        else:
            request_latency = time.perf_counter() - timer_start
            input_tokens = chunk.usage_metadata.prompt_token_count
            output_tokens = chunk.usage_metadata.candidates_token_count + (chunk.usage_metadata.thoughts_token_count or 0)
            impacts = llm_impacts(
                provider=PROVIDER,
                model_name=model_name,
                output_token_count=output_tokens,
                request_latency=request_latency,
                electricity_mix_zone=EcoLogits.config.electricity_mix_zone,
            )
            if impacts is not None:
                if EcoLogits.config.opentelemetry:
                    EcoLogits.config.opentelemetry.record_request(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        request_latency=request_latency,
                        impacts=impacts,
                        provider=PROVIDER,
                        model=model_name,
                        endpoint=f"/v1beta/models/{model_name}:generateContent"
                    )

                yield GenerateContentResponse(**chunk.model_dump(), impacts=impacts)
            else:
                yield GenerateContentResponse(**chunk.model_dump(), impacts=None)


async def google_genai_async_content_stream_wrapper(
    wrapped: Callable,
    instance: Models,   # noqa: ARG001
    args: Any,
    kwargs: Any
) -> AsyncIterator[GenerateContentResponse]:
    """
    Function that wraps Google GenAI answer with computed impacts in async and streaming mode.

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Yields:
        A wrapped `GenerateContentResponse` with impacts
    """
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    return _generator(stream, timer_start=timer_start, model_name=kwargs["model"])


class GoogleGenaiInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all Google GenAI calls
    """

    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "google.genai.models",
                "name": "Models.generate_content",
                "wrapper": google_genai_content_wrapper,
            },
            {
                "module": "google.genai.models",
                "name": "Models.generate_content_stream",
                "wrapper": google_genai_content_stream_wrapper,
            },
            {
                "module": "google.genai.models",
                "name": "AsyncModels.generate_content",
                "wrapper": google_genai_async_content_wrapper
            },
            {
                "module": "google.genai.models",
                "name": "AsyncModels.generate_content_stream",
                "wrapper": google_genai_async_content_stream_wrapper
            }
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
