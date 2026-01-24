import pytest
from anthropic import Anthropic, AsyncAnthropic


@pytest.mark.vcr
def test_anthropic_chat(tracer_init):
    client = Anthropic()
    response = client.messages.create(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    )
    assert len(response.content) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_chat(tracer_init):
    client = AsyncAnthropic()
    response = await client.messages.create(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    )
    assert len(response.content) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_anthropic_stream_chat(tracer_init):
    client = Anthropic()

    text_response = ""
    with client.messages.stream(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    ) as stream:
        for text in stream.text_stream:
            text_response += text
        assert stream.impacts.energy.value >= 0

    assert len(text_response) > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_stream_chat(tracer_init):
    client = AsyncAnthropic()

    text_response = ""
    async with client.messages.stream(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    ) as stream:
        async for text in stream.text_stream:
            text_response += text
        assert stream.impacts.energy.value >= 0

    assert len(text_response) > 0
