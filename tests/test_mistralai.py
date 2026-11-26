import pytest


@pytest.mark.vcr
def test_mistralai_chat(tracer_init, mistralai_client):
    response = mistralai_client.chat.complete(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat(tracer_init, mistralai_client):
    response = await mistralai_client.chat.complete_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_mistralai_stream_chat(tracer_init, mistralai_client):
    stream = mistralai_client.chat.stream(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    for chunk in stream:
        assert chunk.data.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_stream_chat(tracer_init, mistralai_client):
    stream = await mistralai_client.chat.stream_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    )
    async for chunk in stream:
        assert chunk.data.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.default_cassette("test_mistralai_stream_chat.yaml")
def test_mistralai_stream_chat_as_context_manager(tracer_init, mistralai_client):
    with mistralai_client.chat.stream(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    ) as stream:
        chunks = list(stream)
    assert chunks
    assert chunks[-1].data.impacts.energy.value >= 0
    assert stream.response.is_closed


@pytest.mark.vcr
@pytest.mark.default_cassette("test_mistralai_async_stream_chat.yaml")
@pytest.mark.asyncio
async def test_mistralai_async_stream_chat_as_context_manager(tracer_init, mistralai_client):
    async with await mistralai_client.chat.stream_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="ministral-3b-2512"
    ) as stream:
        chunks = [chunk async for chunk in stream]
    assert chunks
    assert chunks[-1].data.impacts.energy.value >= 0
    assert stream.response.is_closed
