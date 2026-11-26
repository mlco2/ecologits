import os
import threading

import litellm
import pytest
from anthropic import Anthropic, AsyncAnthropic
from cohere import AsyncClient, Client
from huggingface_hub import InferenceClient
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from ecologits import EcoLogits
from ecologits.exceptions import EcoLogitsError

OPENAI_CHAT = "test_openai/test_openai_chat.yaml"
OPENAI_ASYNC_CHAT = "test_openai/test_openai_async_chat.yaml"
OPENAI_STREAM_CHAT = "test_openai/test_openai_stream_chat.yaml"
OPENAI_RESPONSES = "test_openai/test_openai_responses.yaml"
OPENAI_RESPONSES_PARSE = "test_openai/test_openai_responses_parse.yaml"
OPENAI_ASYNC_RESPONSES_PARSE = "test_openai/test_openai_async_responses_parse.yaml"
ANTHROPIC_CHAT = "test_anthropic/test_anthropic_chat.yaml"
ANTHROPIC_ASYNC_STREAM_CHAT = "test_anthropic/test_anthropic_async_stream_chat.yaml"
MISTRALAI_ASYNC_CHAT = "test_mistralai/test_mistralai_async_chat.yaml"
MISTRALAI_STREAM_CHAT = "test_mistralai/test_mistralai_stream_chat.yaml"
COHERE_STREAM_CHAT = "test_cohere/test_cohere_stream_chat.yaml"
COHERE_ASYNC_STREAM_CHAT = "test_cohere/test_cohere_async_stream_chat.yaml"
GOOGLE_GENAI_CONTENT_STREAM = "test_google_genai/test_google_genai_content_stream.yaml"
HUGGINGFACE_HUB_STREAM_CHAT = "test_huggingface_hub/test_huggingface_hub_stream_chat.yaml"
LITELLM_CHAT = "test_litellm/test_litellm_chat.yaml"
LITELLM_STREAM_CHAT = "test_litellm/test_litellm_stream_chat.yaml"

HF_INFERENCE_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return os.path.join(os.path.dirname(__file__), "cassettes")


@pytest.fixture
def no_global_init(monkeypatch):
    monkeypatch.setattr(EcoLogits.config, "providers", [])


def openai_chat(client):
    return client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    )


class GreetingDetection(BaseModel):
    is_greeting: bool


def openai_responses_parse(client):
    return client.responses.parse(
        model="gpt-6-luna",
        input=[{"role": "user", "content": "Hello World!"}],
        text_format=GreetingDetection,
    )


def anthropic_chat(client):
    return client.messages.create(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-5-5",
    )


def test_empty_tracker():
    with EcoLogits.track_impacts(providers="openai") as tracker:
        pass
    assert tracker.calls == []
    assert tracker.impacts.energy is None
    assert tracker.impacts.gwp is None


def test_unknown_provider():
    with pytest.raises(EcoLogitsError):
        EcoLogits.track_impacts(providers=["unknown"])


def test_no_providers():
    with pytest.raises(EcoLogitsError):
        EcoLogits.track_impacts(providers=[])


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_openai_chat(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts(providers="openai") as tracker:
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
    with EcoLogits.track_impacts(providers="openai") as tracker:
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
    with EcoLogits.track_impacts(providers="openai") as tracker:
        openai_chat(client)
    outside = openai_chat(client)

    assert outside.impacts.energy.value > 0
    assert len(tracker.calls) == 1


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_CHAT)
def test_impacts_are_summed(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts(providers="openai") as tracker:
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
    with EcoLogits.track_impacts(providers="openai") as outer:
        openai_chat(client)
        with EcoLogits.track_impacts(providers="openai") as inner:
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
    with EcoLogits.track_impacts(providers="openai") as tracker:
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
    async with EcoLogits.track_impacts(providers="openai") as tracker:
        response = await openai_chat(client)
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_STREAM_CHAT)
def test_openai_stream_chat(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts(providers="openai") as tracker:
        stream = client.chat.completions.create(
            model="gpt-6-luna",
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
    with EcoLogits.track_impacts(providers="openai") as tracker:
        stream = client.chat.completions.create(
            model="gpt-6-luna",
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
    with EcoLogits.track_impacts(providers="openai") as tracker:
        response = client.responses.create(model="gpt-6-luna", input="Hello World!")
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_RESPONSES_PARSE)
def test_openai_responses_parse(no_global_init):
    client = OpenAI()
    with EcoLogits.track_impacts(providers="openai") as tracker:
        response = openai_responses_parse(client)

    assert isinstance(response.output_parsed, GreetingDetection)
    assert tracker.calls == [response.impacts]


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(OPENAI_RESPONSES_PARSE)
def test_parsed_responses_outside_block_are_not_tracked(no_global_init):
    client = OpenAI()
    before = openai_responses_parse(client)
    with EcoLogits.track_impacts(providers="openai") as tracker:
        inside = openai_responses_parse(client)
    after = openai_responses_parse(client)

    assert not hasattr(before, "impacts")
    assert not hasattr(after, "impacts")
    assert tracker.calls == [inside.impacts]


@pytest.mark.vcr
@pytest.mark.default_cassette(OPENAI_ASYNC_RESPONSES_PARSE)
@pytest.mark.asyncio
async def test_openai_async_responses_parse(no_global_init):
    client = AsyncOpenAI()
    async with EcoLogits.track_impacts(providers="openai") as tracker:
        response = await openai_responses_parse(client)

    assert isinstance(response.output_parsed, GreetingDetection)
    assert tracker.calls == [response.impacts]


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
    async with EcoLogits.track_impacts(providers="anthropic") as tracker:
        async with client.messages.stream(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello World!"}],
            model="claude-haiku-5-5",
        ) as stream:
            async for _ in stream.text_stream:
                pass

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == stream.impacts


@pytest.mark.vcr
@pytest.mark.default_cassette(MISTRALAI_ASYNC_CHAT)
@pytest.mark.asyncio
async def test_mistralai_async_chat(no_global_init, mistralai_client):
    async with EcoLogits.track_impacts(providers="mistralai") as tracker:
        response = await mistralai_client.chat.complete_async(
            messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
        )
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(MISTRALAI_STREAM_CHAT)
def test_mistralai_stream_chat(no_global_init, mistralai_client):
    outside = list(mistralai_client.chat.stream(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    ))
    with EcoLogits.track_impacts(providers="mistralai") as tracker:
        stream = mistralai_client.chat.stream(
            messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
        )
    chunks = list(stream)

    assert not hasattr(outside[-1].data, "impacts")
    assert len(tracker.calls) == 1
    assert tracker.calls[0] == chunks[-1].data.impacts


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(COHERE_STREAM_CHAT)
def test_cohere_stream_chat(no_global_init):
    client = Client()
    outside = list(client.chat_stream(message="Tell me a short story", model="command-r7b-12-2024", max_tokens=100))
    with EcoLogits.track_impacts(providers="cohere") as tracker:
        stream = client.chat_stream(message="Tell me a short story", model="command-r7b-12-2024", max_tokens=100)
    events = list(stream)

    assert not hasattr(outside[-1], "impacts")
    assert len(tracker.calls) == 1
    assert tracker.calls[0] == events[-1].impacts


@pytest.mark.vcr
@pytest.mark.default_cassette(COHERE_ASYNC_STREAM_CHAT)
@pytest.mark.asyncio
async def test_cohere_async_stream_chat(no_global_init):
    client = AsyncClient()
    async with EcoLogits.track_impacts(providers="cohere") as tracker:
        stream = client.chat_stream(message="Tell me a short story", model="command-r7b-12-2024", max_tokens=100)
    events = [event async for event in stream]

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == events[-1].impacts


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(GOOGLE_GENAI_CONTENT_STREAM)
def test_google_genai_content_stream(no_global_init, google_genai_client):
    models = google_genai_client.models
    outside = list(models.generate_content_stream(model="gemini-2.5-flash-lite", contents="Tell me a joke!"))
    with EcoLogits.track_impacts(providers="google_genai") as tracker:
        stream = models.generate_content_stream(model="gemini-2.5-flash-lite", contents="Tell me a joke!")
    chunks = list(stream)

    assert not hasattr(outside[-1], "impacts")
    assert len(tracker.calls) == 1
    assert tracker.calls[0] == chunks[-1].impacts


@pytest.mark.vcr
@pytest.mark.default_cassette(HUGGINGFACE_HUB_STREAM_CHAT)
def test_huggingface_hub_stream_chat(no_global_init):
    client = InferenceClient(model=HF_INFERENCE_MODEL_URL)
    with EcoLogits.track_impacts(providers="huggingface_hub") as tracker:
        stream = client.chat_completion(
            messages=[{"role": "user", "content": "Hello World!"}],
            max_tokens=15,
            stream=True
        )
    chunks = list(stream)

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == chunks[-1].impacts


@pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.default_cassette(LITELLM_CHAT)
def test_litellm_chat(no_global_init):
    outside = litellm.completion(model="mistral/ministral-3b-2512", messages=[{"content": "Hello World!", "role": "user"}])
    with EcoLogits.track_impacts(providers="litellm") as tracker:
        response = litellm.completion(
            model="mistral/ministral-3b-2512",
            messages=[{"content": "Hello World!", "role": "user"}]
        )

    assert not hasattr(outside, "impacts")
    assert len(tracker.calls) == 1
    assert tracker.impacts.energy == response.impacts.energy


@pytest.mark.vcr
@pytest.mark.default_cassette(LITELLM_STREAM_CHAT)
def test_litellm_stream_chat(no_global_init):
    with EcoLogits.track_impacts(providers="litellm") as tracker:
        stream = litellm.completion(
            model="mistral/ministral-3b-2512",
            messages=[{"role": "user", "content": "Hello World!"}],
            stream=True
        )
        chunks = list(stream)

    assert len(tracker.calls) == 1
    assert tracker.calls[0] == chunks[-1].impacts
