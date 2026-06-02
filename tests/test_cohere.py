import pytest
from cohere import AsyncClient, Client


@pytest.mark.vcr
def test_cohere_chat(tracer_init):
    client = Client()
    chat = client.chat(
        message="Hello!",
        model="command-r-08-2024",
        max_tokens=100
    )
    assert len(chat.text) > 0
    assert chat.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_async_chat(tracer_init):
    client = AsyncClient()
    chat = await client.chat(
        message="Hello!",
        model="command-r-08-2024",
        max_tokens=100
    )
    assert len(chat.text) > 0
    assert chat.impacts.energy.value > 0


@pytest.mark.vcr
def test_cohere_stream_chat(tracer_init):
    client = Client()
    stream = client.chat_stream(
        message="Tell me a short story",
        model="command-r-08-2024",
        max_tokens=100
    )
    for event in stream:
        if event.event_type == "text-generation":
            assert len(event.text) > 0
        if event.event_type == "stream-end":
            assert event.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_cohere_async_stream_chat(tracer_init):
    client = AsyncClient()
    stream = client.chat_stream(
        message="Tell me a short story",
        model="command-r-08-2024",
        max_tokens=100
    )
    async for event in stream:
        if event.event_type == "text-generation":
            assert len(event.text) > 0
        if event.event_type == "stream-end":
            assert event.impacts.energy.value > 0
