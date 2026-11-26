from unittest.mock import Mock

import pytest
from google import genai
from ecologits.tracers.utils import AsyncImpactsStream, ImpactsStream

from ecologits import EcoLogits

try:
    import httpx
except ModuleNotFoundError:
    import httpx2 as httpx


def normalize_request_method(request):
    # VCR records the SDK's POST as "post" with aiohttp and "POST" with HTTPX.
    request.method = request.method.upper()
    return request


replay_google_request = pytest.mark.vcr(before_record_request=normalize_request_method)


@pytest.mark.parametrize("config", [None, {"http_options": {"api_version": "v1beta"}}])
def test_google_genai_telemetry_endpoint(tracer_init, monkeypatch, config):
    telemetry = Mock()
    monkeypatch.setattr(EcoLogits.config, "opentelemetry", telemetry)
    handler = Mock(return_value=httpx.Response(200, json={
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
    }))
    client = genai.Client(api_key="test", http_options={"client_args": {"transport": httpx.MockTransport(handler)}})
    try:
        client.models.generate_content(model="gemini-2.5-flash-lite", contents="Hello", config=config)
    finally:
        if hasattr(client, "close"):
            client.close()

    telemetry.record_request.assert_called_once()
    assert telemetry.record_request.call_args.kwargs["endpoint"] == handler.call_args.args[0].url.path


@replay_google_request
def test_google_genai_content(tracer_init, google_genai_client):
    response = google_genai_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Tell me a joke!"
    )
    assert len(response.text) > 0
    assert response.impacts.energy.value > 0


@replay_google_request
@pytest.mark.asyncio
async def test_google_genai_async_content(tracer_init, google_genai_client):
    response = await google_genai_client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Tell me a joke!"
    )
    assert len(response.text) > 0
    assert response.impacts.energy.value > 0


@replay_google_request
def test_google_genai_content_stream(tracer_init, google_genai_client):
    stream = google_genai_client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents="Tell me a joke!"
    )
    assert isinstance(stream, ImpactsStream)
    for chunk in stream:
        assert len(chunk.text) > 0

        if chunk.candidates[0].finish_reason is not None:
            assert chunk.impacts.energy.value > 0


@replay_google_request
@pytest.mark.default_cassette("test_google_genai_content_stream.yaml")
@pytest.mark.asyncio
async def test_google_genai_async_content_stream(tracer_init, google_genai_client):
    stream = await google_genai_client.aio.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents="Tell me a joke!"
    )
    assert isinstance(stream, AsyncImpactsStream)
    async for chunk in stream:
        assert len(chunk.text) > 0

        if chunk.candidates[0].finish_reason is not None:
            assert chunk.impacts.energy.value > 0


@replay_google_request
def test_google_genai_chat(tracer_init, google_genai_client):
    chat = google_genai_client.chats.create(model="gemini-2.5-flash-lite")
    response = chat.send_message("Tell me a joke!")
    assert len(response.text) > 0
    assert response.impacts.energy.value > 0


@replay_google_request
@pytest.mark.asyncio
async def test_google_genai_async_chat(tracer_init, google_genai_client):
    chat = google_genai_client.aio.chats.create(model="gemini-2.5-flash-lite")
    response = await chat.send_message("Tell me a joke!")
    assert len(response.text) > 0
    assert response.impacts.energy.value > 0


@replay_google_request
def test_google_genai_chat_stream(tracer_init, google_genai_client):
    chat = google_genai_client.chats.create(model="gemini-2.5-flash-lite")
    stream = chat.send_message_stream("Tell me a joke!")
    for chunk in stream:
        assert len(chunk.text) > 0

        if chunk.candidates[0].finish_reason is not None:
            assert chunk.impacts.energy.value > 0


@replay_google_request
@pytest.mark.default_cassette("test_google_genai_chat_stream.yaml")
@pytest.mark.asyncio
async def test_google_genai_async_chat_stream(tracer_init, google_genai_client):
    chat = google_genai_client.aio.chats.create(model="gemini-2.5-flash-lite")
    stream = await chat.send_message_stream("Tell me a joke!")
    async for chunk in stream:
        assert len(chunk.text) > 0

        if chunk.candidates[0].finish_reason is not None:
            assert chunk.impacts.energy.value > 0
