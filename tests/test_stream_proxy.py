import asyncio
import time
from types import SimpleNamespace

import pytest
from google.genai.types import (
    Candidate,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
)

from ecologits.tracers import cohere_tracer, google_genai_tracer
from ecologits.tracers.utils import AsyncImpactsStream, ImpactsStream


def test_with_needs_an_sdk_context_manager():
    body_ran = []
    stream = ImpactsStream(iter([]), iter([]))

    with pytest.raises(AttributeError), stream:
        body_ran.append(True)

    assert not body_ran


@pytest.mark.asyncio
async def test_async_with_needs_an_sdk_context_manager():
    async def empty():
        return
        yield

    body_ran = []
    stream = AsyncImpactsStream(empty(), empty())

    with pytest.raises(AttributeError):
        async with stream:
            body_ran.append(True)

    assert not body_ran


def test_next_yields_the_items_with_impacts():
    stream = ImpactsStream(iter(["raw"]), iter(["with impacts"]))

    assert next(stream) == "with impacts"


@pytest.mark.asyncio
async def test_anext_yields_the_items_with_impacts():
    async def items(value):
        yield value

    stream = AsyncImpactsStream(items("raw"), items("with impacts"))

    assert await anext(stream) == "with impacts"


def _cohere_stream_end():
    tokens = SimpleNamespace(input_tokens=5, output_tokens=3)
    return SimpleNamespace(event_type="stream-end", response=SimpleNamespace(meta=SimpleNamespace(tokens=tokens)))


def _genai_last_chunk():
    usage = GenerateContentResponseUsageMetadata(prompt_token_count=5, candidates_token_count=3)
    return GenerateContentResponse(candidates=[Candidate(finish_reason=FinishReason.STOP)], usage_metadata=usage)


@pytest.mark.parametrize(("tracer", "wrapper", "last_event"), [
    (cohere_tracer, cohere_tracer.cohere_stream_chat_wrapper, _cohere_stream_end),
    (google_genai_tracer, google_genai_tracer.google_genai_content_stream_wrapper, _genai_last_chunk),
])
def test_lazy_sdk_stream_latency_starts_at_first_iteration(monkeypatch, tracer, wrapper, last_event):
    # These SDKs return a generator that sends the request on the first iteration.
    latencies = []
    monkeypatch.setattr(tracer, "is_tracking_enabled", lambda provider: True)
    monkeypatch.setattr(tracer, "_llm_impacts", lambda **kwargs: latencies.append(kwargs["request_latency"]))

    def sdk_stream(**kwargs):
        yield last_event()

    stream = wrapper(sdk_stream, None, (), {"model": "test-model"})
    time.sleep(0.2)
    list(stream)

    assert latencies and latencies[0] < 0.2


@pytest.mark.asyncio
async def test_lazy_async_sdk_stream_latency_starts_at_first_iteration(monkeypatch):
    latencies = []
    monkeypatch.setattr(google_genai_tracer, "is_tracking_enabled", lambda provider: True)
    monkeypatch.setattr(google_genai_tracer, "_llm_impacts", lambda **kwargs: latencies.append(kwargs["request_latency"]))

    async def sdk_stream(**kwargs):
        async def chunks():
            yield _genai_last_chunk()
        return chunks()

    stream = await google_genai_tracer.google_genai_async_content_stream_wrapper(
        sdk_stream, None, (), {"model": "test-model"}
    )
    await asyncio.sleep(0.2)
    [chunk async for chunk in stream]

    assert latencies and latencies[0] < 0.2
