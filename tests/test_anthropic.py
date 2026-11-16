from unittest.mock import Mock

import pytest
from anthropic import Anthropic, AsyncAnthropic

from ecologits import EcoLogits
from ecologits.tracers import anthropic_tracer


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
@pytest.mark.default_cassette("test_anthropic_stream_chat.yaml")
@pytest.mark.parametrize("consume", ["text_stream", "events", "get_final_message", "get_final_text", "until_done"])
def test_anthropic_stream_chat(tracer_init, monkeypatch, consume):
    client = Anthropic()
    compute_impacts = Mock(wraps=anthropic_tracer.llm_impacts)
    monkeypatch.setattr(anthropic_tracer, "llm_impacts", compute_impacts)

    with EcoLogits.track_impacts(providers="anthropic") as tracker, client.messages.stream(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    ) as stream:
        if consume == "text_stream":
            assert "".join(stream.text_stream)
        elif consume == "events":
            assert list(stream)
        elif consume == "get_final_message":
            assert stream.get_final_message().content
        elif consume == "get_final_text":
            assert stream.get_final_text()
        else:
            stream.until_done()

        assert stream.impacts.energy.value >= 0
        impacts = stream.impacts

        # Reading the completed response again must not compute impacts twice.
        assert stream.get_final_message().content
        assert stream.impacts is impacts
        compute_impacts.assert_called_once()

    assert tracker.calls == [stream.impacts]
    assert tracker.impacts.energy == stream.impacts.energy


@pytest.mark.vcr
@pytest.mark.default_cassette("test_anthropic_async_stream_chat.yaml")
@pytest.mark.parametrize("consume", ["text_stream", "events", "get_final_message", "get_final_text", "until_done"])
@pytest.mark.asyncio
async def test_anthropic_async_stream_chat(tracer_init, monkeypatch, consume):
    client = AsyncAnthropic()
    compute_impacts = Mock(wraps=anthropic_tracer.llm_impacts)
    monkeypatch.setattr(anthropic_tracer, "llm_impacts", compute_impacts)

    async with EcoLogits.track_impacts(providers="anthropic") as tracker, client.messages.stream(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello World!"}],
        model="claude-haiku-4-5",
    ) as stream:
        if consume == "text_stream":
            assert "".join([text async for text in stream.text_stream])
        elif consume == "events":
            assert [event async for event in stream]
        elif consume == "get_final_message":
            assert (await stream.get_final_message()).content
        elif consume == "get_final_text":
            assert await stream.get_final_text()
        else:
            await stream.until_done()

        assert stream.impacts.energy.value >= 0
        impacts = stream.impacts

        assert (await stream.get_final_message()).content
        assert stream.impacts is impacts
        compute_impacts.assert_called_once()

    assert tracker.calls == [stream.impacts]
    assert tracker.impacts.energy == stream.impacts.energy
