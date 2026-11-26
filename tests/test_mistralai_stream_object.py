import json
from importlib.util import find_spec

import pytest

from ecologits import EcoLogits
from tests.sdk_http import sdk_httpx

if find_spec("mistralai.chat") is not None:
    from mistralai import Mistral
    from mistralai.utils.eventstreaming import EventStream, EventStreamAsync
else:
    from mistralai.client import Mistral
    from mistralai.client.utils.eventstreaming import EventStream, EventStreamAsync

httpx = sdk_httpx("mistralai")

MODEL = "mistral-small-latest"
MESSAGES = [{"role": "user", "content": "Hello World!"}]


class Server:
    """Answers chat stream calls like the Mistral API, and counts requests."""

    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        base = {"id": "cmpl-1", "object": "chat.completion.chunk", "created": 0, "model": MODEL}
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"},
                                  "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


@pytest.fixture
def server():
    return Server()


@pytest.fixture
def client(server):
    transport = httpx.MockTransport(server)
    return Mistral(api_key="test", client=httpx.Client(transport=transport),
                   async_client=httpx.AsyncClient(transport=transport))


def test_stream_is_sdk_event_stream(tracer_init, server, client):
    with EcoLogits.track_impacts(providers="mistralai") as tracker:
        stream = client.chat.stream(model=MODEL, messages=MESSAGES)

        # As without EcoLogits, the request is sent when stream() is called.
        assert server.requests == 1
        assert isinstance(stream, EventStream)
        with stream as entered:
            events = list(entered)

    assert events[-1].data.impacts.energy is not None
    assert stream.response.is_closed
    assert len(tracker.calls) == 1


@pytest.mark.asyncio
async def test_async_stream_is_sdk_event_stream(tracer_init, server, client):
    async with EcoLogits.track_impacts(providers="mistralai") as tracker:
        stream = await client.chat.stream_async(model=MODEL, messages=MESSAGES)

        assert server.requests == 1
        assert isinstance(stream, EventStreamAsync)
        async with stream as entered:
            events = [event async for event in entered]

    assert events[-1].data.impacts.energy is not None
    assert stream.response.is_closed
    assert len(tracker.calls) == 1
