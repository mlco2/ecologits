import os

import pytest
from mistralai.client import Mistral


@pytest.mark.vcr
def test_mistralai_chat(tracer_init):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat(tracer_init):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = await client.chat.complete_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_mistralai_stream_chat(tracer_init):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    stream = client.chat.stream(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    for chunk in stream:
        assert chunk.data.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_stream_chat(tracer_init):
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    stream = await client.chat.stream_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    async for chunk in stream:
        assert chunk.data.impacts.energy.value >= 0
