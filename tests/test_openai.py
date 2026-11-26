import os

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from pydantic import BaseModel
from ecologits.tracers.utils import AsyncImpactsStream, ImpactsStream

class GreetingDetection(BaseModel):
    is_greeting: bool

@pytest.mark.vcr
def test_openai_chat(tracer_init):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_chat(tracer_init):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_openai_stream_chat(tracer_init):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
def test_openai_stream_chat_with_options(tracer_init):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True,
        stream_options={"include_usage": True}
    )
    for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_stream_chat(tracer_init):
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    assert isinstance(stream, AsyncImpactsStream)
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_stream_chat_with_options(tracer_init):
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True,
        stream_options={"include_usage": True}
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
def test_openai_responses(tracer_init):
    client = OpenAI()
    response = client.responses.create(
        model="gpt-6-luna",
        input="Hello World!"
    )
    assert response.output is not None
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_responses(tracer_init):
    client = AsyncOpenAI()
    response = await client.responses.create(
        model="gpt-6-luna",
        input="Hello World!"
    )
    assert response.output is not None
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_openai_stream_responses(tracer_init):
    client = OpenAI()
    stream = client.responses.create(
        model="gpt-6-luna",
        input="Hello World!",
        stream=True
    )
    assert isinstance(stream, ImpactsStream)
    for event in stream:
        if event.type == "response.completed":
            assert event.impacts.energy.value > 0
            assert event.response.impacts.energy.value > 0
        else:
            assert not hasattr(event, "impacts")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_stream_responses(tracer_init):
    client = AsyncOpenAI()
    stream = await client.responses.create(
        model="gpt-6-luna",
        input="Hello World!",
        stream=True
    )
    async for event in stream:
        if event.type == "response.completed":
            assert event.impacts.energy.value > 0
            assert event.response.impacts.energy.value > 0
        else:
            assert not hasattr(event, "impacts")


@pytest.mark.vcr
def test_openai_responses_parse(tracer_init):
    """Test parsing of OpenAI responses with GreetingDetection."""
    client = OpenAI()
    response = client.responses.parse(
        model="gpt-6-luna",
        input=[{"role": "user", "content": "Hello World!"}],
        text_format=GreetingDetection
    )
    assert isinstance(response.output_parsed, GreetingDetection)
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_responses_parse(tracer_init):
    """Test async parsing of OpenAI responses with GreetingDetection."""
    client = AsyncOpenAI()
    response = await client.responses.parse(
        model="gpt-6-luna",
        input=[{"role": "user", "content": "Hello World!"}],
        text_format=GreetingDetection
    )

    assert isinstance(response.output_parsed, GreetingDetection)
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.skip(reason="Azure deployment needs verification before these tests can be re-recorded.")
def test_azure_openai_chat(tracer_init):
    client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello World!"}],
        model=os.getenv("AZURE_MODEL_DEPLOYMENT"),
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.skip(reason="Azure deployment needs verification before these tests can be re-recorded.")
async def test_azure_openai_async_chat(tracer_init):
    client = AsyncAzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    response = await client.chat.completions.create(
         model=os.getenv("AZURE_MODEL_DEPLOYMENT"),
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.skip(reason="Azure deployment needs verification before these tests can be re-recorded.")
def test_azure_openai_stream_chat(tracer_init):
    client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    stream = client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_DEPLOYMENT"),
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.skip(reason="Azure deployment needs verification before these tests can be re-recorded.")
async def test_azure_openai_async_stream_chat(tracer_init):
    client = AsyncAzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    stream = await client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_DEPLOYMENT"),
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.default_cassette("test_openai_chat.yaml")
def test_openai_chat_with_raw_response(tracer_init):
    client = OpenAI()
    raw = client.chat.completions.with_raw_response.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    response = raw.parse()
    assert len(response.choices) > 0


@pytest.mark.vcr
@pytest.mark.default_cassette("test_openai_async_chat.yaml")
@pytest.mark.asyncio
async def test_openai_async_chat_with_raw_response(tracer_init):
    client = AsyncOpenAI()
    raw = await client.chat.completions.with_raw_response.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    response = raw.parse()
    assert len(response.choices) > 0


@pytest.mark.vcr
@pytest.mark.default_cassette("test_openai_stream_chat.yaml")
def test_openai_stream_chat_with_raw_response(tracer_init):
    client = OpenAI()
    raw = client.chat.completions.with_raw_response.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    chunks = list(raw.parse())
    assert len(chunks) > 0


@pytest.mark.vcr
@pytest.mark.default_cassette("test_openai_chat.yaml")
def test_openai_chat_with_streaming_response(tracer_init):
    client = OpenAI()
    with client.chat.completions.with_streaming_response.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    ) as raw:
        response = raw.parse()
    assert len(response.choices) > 0


@pytest.mark.vcr
@pytest.mark.default_cassette("test_openai_responses.yaml")
def test_openai_responses_with_raw_response(tracer_init):
    client = OpenAI()
    raw = client.responses.with_raw_response.create(
        model="gpt-6-luna",
        input="Hello World!"
    )
    response = raw.parse()
    assert response.output is not None
