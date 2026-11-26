import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Optional

from google.genai.models import AsyncModels, Models
from google.genai.types import GenerateContentResponse as _GenerateContentResponse
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits import EcoLogits
from ecologits.tracers.utils import AsyncImpactsStream, ImpactsOutput, ImpactsStream, _llm_impacts
from ecologits.tracking import ImpactsTracker, get_active_trackers, is_tracking_enabled, record_impacts

PROVIDER = "google_genai"


class GenerateContentResponse(_GenerateContentResponse):
    """
    Wrapper of `google.genai.types.GenerateContentResponse` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None


def _endpoint(instance: Models | AsyncModels, kwargs: dict[str, Any], method: str) -> str:
    config = kwargs.get("config")
    options = config.get("http_options") if isinstance(config, dict) else getattr(config, "http_options", None)
    version = options.get("api_version") if isinstance(options, dict) else getattr(options, "api_version", None)
    if version is None:
        version = instance._api_client.get_read_only_http_options().get("api_version")  # noqa: SLF001
    model = kwargs["model"].removeprefix("models/")
    prefix = f"/{version}" if version else ""
    return f"{prefix}/models/{model}:{method}"


def google_genai_content_wrapper(
    wrapped: Callable,
    instance: Models,
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
    if not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = kwargs["model"]
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = (
        response.usage_metadata.candidates_token_count
        + (response.usage_metadata.thoughts_token_count or 0)
    )
    impacts = _llm_impacts(
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
                endpoint=_endpoint(instance, kwargs, "generateContent")
            )

        record_impacts(trackers, impacts)
        return GenerateContentResponse.model_construct(**dict(response), impacts=impacts)
    else:
        return response


def google_genai_content_stream_wrapper(
    wrapped: Callable,
    instance: Models,
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
    if not is_tracking_enabled(PROVIDER):
        return wrapped(*args, **kwargs)
    stream = wrapped(*args, **kwargs)
    return ImpactsStream(stream, _stream_generator(stream, instance, kwargs, get_active_trackers(PROVIDER)))


def _stream_generator(
    stream: Iterator[GenerateContentResponse],
    instance: Models,
    kwargs: Any,
    trackers: tuple[ImpactsTracker, ...]
) -> Iterator[GenerateContentResponse]:
    # The SDK sends the request on the first iteration.
    timer_start = time.perf_counter()
    for chunk in stream:
        if chunk.candidates[0].finish_reason is None:
            yield GenerateContentResponse.model_construct(**dict(chunk), impacts=None)

        else:
            request_latency = time.perf_counter() - timer_start
            model_name = kwargs["model"]
            input_tokens = chunk.usage_metadata.prompt_token_count
            output_tokens = (
                chunk.usage_metadata.candidates_token_count
                + (chunk.usage_metadata.thoughts_token_count or 0)
            )
            impacts = _llm_impacts(
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
                        endpoint=_endpoint(instance, kwargs, "streamGenerateContent")
                    )

                record_impacts(trackers, impacts)
                yield GenerateContentResponse.model_construct(**dict(chunk), impacts=impacts)
            else:
                yield GenerateContentResponse.model_construct(**dict(chunk), impacts=None)


async def google_genai_async_content_wrapper(
    wrapped: Callable,
    instance: AsyncModels,
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
    if not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    trackers = get_active_trackers(PROVIDER)
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    model_name = kwargs["model"]
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = (
        response.usage_metadata.candidates_token_count
        + (response.usage_metadata.thoughts_token_count or 0)
    )
    impacts = _llm_impacts(
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
                endpoint=_endpoint(instance, kwargs, "generateContent")
            )

        record_impacts(trackers, impacts)
        return GenerateContentResponse.model_construct(**dict(response), impacts=impacts)
    else:
        return response


async def _generator(
    stream: AsyncIterator[GenerateContentResponse],
    instance: AsyncModels,
    kwargs: dict[str, Any],
    trackers: tuple[ImpactsTracker, ...] = ()
) -> AsyncIterator[GenerateContentResponse]:
    model_name = kwargs["model"]
    # The SDK sends the request on the first iteration.
    timer_start = time.perf_counter()
    async for chunk in stream:
        if chunk.candidates[0].finish_reason is None:
            yield GenerateContentResponse.model_construct(**dict(chunk), impacts=None)

        else:
            request_latency = time.perf_counter() - timer_start
            input_tokens = chunk.usage_metadata.prompt_token_count
            output_tokens = (
                chunk.usage_metadata.candidates_token_count
                + (chunk.usage_metadata.thoughts_token_count or 0)
            )
            impacts = _llm_impacts(
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
                        endpoint=_endpoint(instance, kwargs, "streamGenerateContent")
                    )

                record_impacts(trackers, impacts)
                yield GenerateContentResponse.model_construct(**dict(chunk), impacts=impacts)
            else:
                yield GenerateContentResponse.model_construct(**dict(chunk), impacts=None)


async def google_genai_async_content_stream_wrapper(
    wrapped: Callable,
    instance: AsyncModels,
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
    if not is_tracking_enabled(PROVIDER):
        return await wrapped(*args, **kwargs)
    stream = await wrapped(*args, **kwargs)
    return AsyncImpactsStream(stream, _generator(stream, instance, kwargs, get_active_trackers(PROVIDER)))


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
