import json

import pytest
from anthropic import Anthropic, AsyncAnthropic, AsyncStream, Stream

from ecologits import EcoLogits
from tests.sdk_http import sdk_httpx

httpx = sdk_httpx("anthropic")

MODEL = "claude-haiku-4-5"
MESSAGES = [{"role": "user", "content": "Hello World!"}]


class Server:
    """Answers stream calls like the Anthropic Messages API, and counts requests."""

    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        events = [
            ("message_start", {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL, "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "Hi"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 3}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


@pytest.fixture
def server():
    return Server()


@pytest.fixture
def client(server):
    return Anthropic(api_key="test", base_url="http://test",
                     http_client=httpx.Client(transport=httpx.MockTransport(server)))


@pytest.fixture
def async_client(server):
    return AsyncAnthropic(api_key="test", base_url="http://test",
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))


def test_create_stream(tracer_init, server, client):
    with EcoLogits.track_impacts(providers="anthropic") as tracker:
        stream = client.messages.create(model=MODEL, max_tokens=10, messages=MESSAGES, stream=True)

        assert server.requests == 1
        # Anthropic's Stream metaclass makes isinstance() true only for the exact type.
        assert stream.__class__ is Stream
        with stream as entered:
            events = list(entered)

    assert events[-1].type == "message_stop"
    assert events[-1].impacts.energy is not None
    assert stream.response.is_closed
    assert len(tracker.calls) == 1


@pytest.mark.asyncio
async def test_async_create_stream(tracer_init, server, async_client):
    async with EcoLogits.track_impacts(providers="anthropic") as tracker:
        stream = await async_client.messages.create(model=MODEL, max_tokens=10, messages=MESSAGES, stream=True)

        assert server.requests == 1
        assert stream.__class__ is AsyncStream
        async with stream as entered:
            events = [event async for event in entered]

    assert events[-1].type == "message_stop"
    assert events[-1].impacts.energy is not None
    assert stream.response.is_closed
    assert len(tracker.calls) == 1


def test_stream_helper_counts_once(tracer_init, client):
    with EcoLogits.track_impacts(providers="anthropic") as tracker, \
            client.messages.stream(model=MODEL, max_tokens=10, messages=MESSAGES) as stream:
        stream.get_final_message()

    assert len(tracker.calls) == 1
