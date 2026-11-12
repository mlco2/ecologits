import os
import threading

import pytest
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from ecologits import EcoLogits
from ecologits.exceptions import EcoLogitsError

OPENAI_CHAT = "test_openai/test_openai_chat.yaml"
OPENAI_ASYNC_CHAT = "test_openai/test_openai_async_chat.yaml"
OPENAI_STREAM_CHAT = "test_openai/test_openai_stream_chat.yaml"
OPENAI_RESPONSES = "test_openai/test_openai_responses.yaml"
ANTHROPIC_CHAT = "test_anthropic/test_anthropic_chat.yaml"
ANTHROPIC_ASYNC_STREAM_CHAT = "test_anthropic/test_anthropic_async_stream_chat.yaml"


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return os.path.join(os.path.dirname(__file__), "cassettes")


@pytest.fixture
def no_global_init(monkeypatch):
    monkeypatch.setattr(EcoLogits.config, "providers", [])


def openai_chat(client):
    return client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello World!"}]
    )


def anthropic_chat(client):
    return client.messages.create(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    )


def test_empty_tracker():
    with EcoLogits.track_impacts() as tracker:
        pass
    assert tracker.calls == []
    assert tracker.impacts.energy is None
    assert tracker.impacts.gwp is None


def test_unsupported_provider():
    with pytest.raises(EcoLogitsError):
        EcoLogits.track_impacts(providers=["mistralai"])


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_openai_chat(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as tracker:
        response = openai_chat(client)
    assert len(tracker.calls) == 1
    assert tracker.calls[0] == response.impacts
    assert tracker.impacts.energy == response.impacts.energy
    assert tracker.impacts.gwp == response.impacts.gwp


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_calls_outside_block_are_not_tracked(no_global_init):
    client = OpenAI()
    before = openai_chat(client)
    with EcoLogits.track_impacts() as tracker:
        inside = openai_chat(client)
    after = openai_chat(client)

    assert not hasattr(before, "impacts")
    assert not hasattr(after, "impacts")
    assert inside.impacts.energy.value > 0
    assert len(tracker.calls) == 1


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_calls_outside_block_still_tracked_with_init(tracer_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as tracker:
        openai_chat(client)
    outside = openai_chat(client)

    assert outside.impacts.energy.value > 0
    assert len(tracker.calls) == 1


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_impacts_are_summed(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as tracker:
        first = openai_chat(client)
        second = openai_chat(client)

    assert len(tracker.calls) == 2
    assert tracker.impacts.energy == first.impacts.energy + second.impacts.energy
    assert tracker.impacts.gwp == first.impacts.gwp + second.impacts.gwp
    assert tracker.impacts.usage.energy == first.impacts.usage.energy + second.impacts.usage.energy
    assert tracker.impacts.embodied.gwp == first.impacts.embodied.gwp + second.impacts.embodied.gwp


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_nested_blocks(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as outer:
        openai_chat(client)
        with EcoLogits.track_impacts() as inner:
            openai_chat(client)

    assert len(inner.calls) == 1
    assert len(outer.calls) == 2


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_tracker_only_counts_its_providers(no_global_init):
    EcoLogits.track_impacts(providers="openai")
    client = OpenAI()
    with EcoLogits.track_impacts(providers="anthropic") as tracker:
        response = openai_chat(client)

    assert not hasattr(response, "impacts")
    assert tracker.calls == []


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_other_threads_are_not_tracked(no_global_init):
    client = OpenAI()
    results = []
    with EcoLogits.track_impacts() as tracker:
        thread = threading.Thread(target=lambda: results.append(openai_chat(client)))
        thread.start()
        thread.join()
        openai_chat(client)

    assert not hasattr(results[0], "impacts")
    assert len(tracker.calls) == 1


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_ASYNC_CHAT)
@pytest.mark.asyncio
async def test_openai_async_chat(no_global_init):
    client = AsyncOpenAI()
    async with EcoLogits.track_impacts() as tracker:
        response = await openai_chat(client)
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_STREAM_CHAT)
def test_openai_stream_chat(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as tracker:
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Hello World!"}],
            stream=True
        )
        chunks = list(stream)

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == chunks[-1].impacts


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_STREAM_CHAT)
def test_stream_read_after_block_is_still_tracked(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as tracker:
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Hello World!"}],
            stream=True
        )
    chunks = list(stream)

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == chunks[-1].impacts


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_RESPONSES)
def test_openai_responses(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts() as tracker:
        response = client.responses.create(model="gpt-5-nano", input="Hello World!")
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(ANTHROPIC_CHAT)
def test_anthropic_chat(no_global_init):
    client = Anthropic()
    outside = anthropic_chat(client)
    with EcoLogits.track_impacts(providers="anthropic") as tracker:
        response = anthropic_chat(client)

    assert not hasattr(outside, "impacts")
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr
@pytest.mark.default_cassette(ANTHROPIC_ASYNC_STREAM_CHAT)
@pytest.mark.asyncio
async def test_anthropic_async_stream_chat(no_global_init):
    client = AsyncAnthropic()
    async with EcoLogits.track_impacts() as tracker:
        async with client.messages.stream(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello World!"}],
            model="claude-haiku-4-5",
        ) as stream:
            async for _ in stream.text_stream:
                pass

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == stream.impacts
