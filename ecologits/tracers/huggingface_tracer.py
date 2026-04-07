import time
from collections.abc import AsyncIterable, Iterable
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Union

import tiktoken
from huggingface_hub import AsyncInferenceClient, InferenceClient  # type: ignore[import-untyped]
from huggingface_hub import ChatCompletionOutput as _ChatCompletionOutput
from huggingface_hub import ChatCompletionStreamOutput as _ChatCompletionStreamOutput
from wrapt import wrap_function_wrapper  # type: ignore[import-untyped]

from ecologits._ecologits import EcoLogits
from ecologits.tracers.utils import ImpactsOutput, llm_impacts

PROVIDER = "huggingface_hub"
HF_INFERENCE_URL_PREFIXES = (
    "https://api-inference.huggingface.co/models/",
    "https://router.huggingface.co/hf-inference/models/",
)


@dataclass
class ChatCompletionOutput(_ChatCompletionOutput):
    """
    Wrapper of `huggingface_hub.ChatCompletionOutput` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None


@dataclass
class ChatCompletionStreamOutput(_ChatCompletionStreamOutput):
    """
    Wrapper of `huggingface_hub.ChatCompletionStreamOutput` with `ImpactsOutput`
    """
    impacts: Optional[ImpactsOutput] = None


def _resolve_model_name(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        for prefix in HF_INFERENCE_URL_PREFIXES:
            if value.startswith(prefix):
                return value.removeprefix(prefix).removesuffix("/v1/chat/completions")
        return value

    return None


def huggingface_chat_wrapper(
    wrapped: Callable,
    instance: InferenceClient,
    args: Any,
    kwargs: Any
) -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]:
    """
    Function that wraps a HuggingFace Hub answer with computed impacts

    Args:
        wrapped: Callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletionOutput` or `Iterable[ChatCompletionStreamOutput]` with impacts
    """

    if kwargs.get("stream", False):
        return huggingface_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return huggingface_chat_wrapper_non_stream(wrapped, instance, args, kwargs)


def huggingface_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: InferenceClient,
    args: Any,
    kwargs: Any
) -> ChatCompletionOutput:
    timer_start = time.perf_counter()
    response = wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    output_tokens = response.usage["completion_tokens"]
    input_tokens = response.usage["prompt_tokens"]
    model_name = _resolve_model_name(response.model, kwargs.get("model"), instance.model)
    if model_name is None:
        return response

    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
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
                endpoint="/chat/completions"
            )

        return ChatCompletionOutput(**asdict(response), impacts=impacts)
    else:
        return response


def huggingface_chat_wrapper_stream(
    wrapped: Callable,
    instance: InferenceClient,
    args: Any,
    kwargs: Any
) -> Iterable[ChatCompletionStreamOutput]:
    encoder = tiktoken.get_encoding("cl100k_base")
    prompt_text = "".join([m["content"] for m in kwargs["messages"]])
    input_tokens = len(encoder.encode(prompt_text))
    default_model_name = _resolve_model_name(kwargs.get("model"), instance.model)
    timer_start = time.perf_counter()
    stream = wrapped(*args, **kwargs)
    output_tokens = 0
    for chunk in stream:
        output_tokens += 1
        request_latency = time.perf_counter() - timer_start
        model_name = _resolve_model_name(getattr(chunk, "model", None), default_model_name)
        if model_name is None:
            yield chunk
            continue

        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=output_tokens,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and chunk.choices[0]["finish_reason"] is not None:
                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/chat/completions"
                )

            yield ChatCompletionStreamOutput(**asdict(chunk), impacts=impacts)
        else:
            yield chunk


async def huggingface_async_chat_wrapper(
    wrapped: Callable,
    instance: AsyncInferenceClient,
    args: Any,
    kwargs: Any
) -> Union[ChatCompletionOutput, AsyncIterable[ChatCompletionStreamOutput]]:
    """
    Function that wraps a HuggingFace Hub answer with computed impacts in async mode

    Args:
        wrapped: Async callable that returns the LLM response
        instance: Never used - for compatibility with `wrapt`
        args: Arguments of the callable
        kwargs: Keyword arguments of the callable

    Returns:
        A wrapped `ChatCompletionOutput` or `AsyncIterable[ChatCompletionStreamOutput]]` with impacts
    """

    if kwargs.get("stream", False):
        return huggingface_async_chat_wrapper_stream(wrapped, instance, args, kwargs)
    else:
        return await huggingface_async_chat_wrapper_non_stream(wrapped, instance, args, kwargs)


async def huggingface_async_chat_wrapper_non_stream(
    wrapped: Callable,
    instance: AsyncInferenceClient,
    args: Any,
    kwargs: Any
) -> ChatCompletionOutput:
    timer_start = time.perf_counter()
    response = await wrapped(*args, **kwargs)
    request_latency = time.perf_counter() - timer_start
    output_tokens = response.usage["completion_tokens"]
    input_tokens = response.usage["prompt_tokens"]
    model_name = _resolve_model_name(response.model, kwargs.get("model"), instance.model)
    if model_name is None:
        return response

    impacts = llm_impacts(
        provider=PROVIDER,
        model_name=model_name,
        output_token_count=output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=EcoLogits.config.electricity_mix_zone
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
                endpoint="/chat/completions"
            )

        return ChatCompletionOutput(**asdict(response), impacts=impacts)
    else:
        return response


async def huggingface_async_chat_wrapper_stream(
    wrapped: Callable,
    instance: AsyncInferenceClient,
    args: Any,
    kwargs: Any
) -> AsyncIterable[ChatCompletionStreamOutput]:
    encoder = tiktoken.get_encoding("cl100k_base")
    prompt_text = "".join([m["content"] for m in kwargs["messages"]])
    input_tokens = len(encoder.encode(prompt_text))
    default_model_name = _resolve_model_name(kwargs.get("model"), instance.model)
    timer_start = time.perf_counter()
    stream = await wrapped(*args, **kwargs)
    output_tokens = 0
    async for chunk in stream:
        output_tokens += 1
        request_latency = time.perf_counter() - timer_start
        model_name = _resolve_model_name(getattr(chunk, "model", None), default_model_name)
        if model_name is None:
            yield chunk
            continue

        impacts = llm_impacts(
            provider=PROVIDER,
            model_name=model_name,
            output_token_count=output_tokens,
            request_latency=request_latency,
            electricity_mix_zone=EcoLogits.config.electricity_mix_zone
        )
        if impacts is not None:
            if EcoLogits.config.opentelemetry \
                    and chunk.choices[0]["finish_reason"] is not None:
                EcoLogits.config.opentelemetry.record_request(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_latency=request_latency,
                    impacts=impacts,
                    provider=PROVIDER,
                    model=model_name,
                    endpoint="/chat/completions"
                )

            yield ChatCompletionStreamOutput(**asdict(chunk), impacts=impacts)
        else:
            yield chunk


class HuggingfaceInstrumentor:
    """
    Instrumentor initialized by EcoLogits to automatically wrap all HuggingFace Hub calls
    """

    def __init__(self) -> None:
        self.wrapped_methods = [
            {
                "module": "huggingface_hub.inference._client",
                "name": "InferenceClient.chat_completion",
                "wrapper": huggingface_chat_wrapper,
            },
            {
                "module": "huggingface_hub.inference._generated._async_client",
                "name": "AsyncInferenceClient.chat_completion",
                "wrapper": huggingface_async_chat_wrapper,
            },
        ]

    def instrument(self) -> None:
        for wrapper in self.wrapped_methods:
            wrap_function_wrapper(
                wrapper["module"], wrapper["name"], wrapper["wrapper"]
            )
