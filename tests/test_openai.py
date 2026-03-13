import os

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI


@pytest.mark.vcr
def test_openai_chat(tracer_init):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_chat(tracer_init):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_openai_stream_chat(tracer_init):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
def test_openai_stream_chat_with_options(tracer_init):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-5-nano",
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
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_async_stream_chat_with_options(tracer_init):
    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True,
        stream_options={"include_usage": True}
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
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
async def test_azure_openai_async_chat(tracer_init):
    client = AsyncAzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    response = await client.chat.completions.create(
         model=os.getenv("AZURE_MODEL_DEPLOYMENT"),
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
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
async def test_azure_openai_async_stream_chat(tracer_init):
    client = AsyncAzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    stream = await client.chat.completions.create(
        model=os.getenv("AZURE_MODEL_DEPLOYMENT"),
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0
