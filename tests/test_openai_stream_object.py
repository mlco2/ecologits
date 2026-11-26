import json
import time

import pytest
from openai import AsyncOpenAI, OpenAI, Stream
from openai.resources.chat.completions import Completions
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

from ecologits import EcoLogits
from ecologits.tracers import openai_tracer
from tests.sdk_http import sdk_httpx

httpx = sdk_httpx("openai")

MODEL = "gpt-4o-mini"
MESSAGES = [{"role": "user", "content": "Hello World!"}]


def requires_chat_completions(method: str) -> pytest.MarkDecorator:
    # Older openai 1.x releases only have these helpers under `client.beta.chat.completions`.
    return pytest.mark.skipif(not hasattr(Completions, method), reason=f"`chat.completions.{method}` is beta-only")


class Answer(BaseModel):
    text: str


def _chat() -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"text": "Hi"}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _response(status: str = "completed") -> dict:
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": MODEL, "status": status,
        "output": [] if status != "completed" else [{
            "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
        }],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


def _sse(events: list[tuple[str, dict]], done: bool) -> bytes:
    body = "".join((f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n" for name, data in events)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


class Server:
    """Answers chat and responses calls like the OpenAI API, and counts requests."""

    def __init__(self) -> None:
        self.requests = 0
        self.delay = 0.0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        time.sleep(self.delay)
        stream = json.loads(request.content).get("stream", False)
        sse = {"content-type": "text/event-stream"}
        if request.url.path.endswith("/chat/completions"):
            if not stream:
                return httpx.Response(200, json=_chat())
            base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": MODEL}
            chunks = [
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"},
                                      "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            return httpx.Response(200, content=_sse([("", c) for c in chunks], done=True), headers=sse)
        if not stream:
            return httpx.Response(200, json=_response())
        events = [
            ("response.created", {"type": "response.created", "sequence_number": 0,
                                  "response": _response("in_progress")}),
            ("response.completed", {"type": "response.completed", "sequence_number": 1, "response": _response()}),
        ]
        return httpx.Response(200, content=_sse(events, done=False), headers=sse)


@pytest.fixture
def server():
    return Server()


@pytest.fixture
def client(server):
    return OpenAI(api_key="test", base_url="http://test/v1", http_client=httpx.Client(transport=httpx.MockTransport(server)))


@pytest.fixture
def async_client(server):
    return AsyncOpenAI(api_key="test", base_url="http://test/v1",
                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))


@requires_chat_completions("stream")
def test_chat_stream_helper(tracer_init, client):
    with EcoLogits.track_impacts(providers="openai") as tracker, \
            client.chat.completions.stream(model=MODEL, messages=MESSAGES) as stream:
        chunks = [event.chunk for event in stream if event.type == "chunk"]
        completion = stream.get_final_completion()

    assert completion.choices[0].message.content == "Hi"
    assert chunks[-1].impacts.energy is not None
    assert len(tracker.calls) == 1


@pytest.mark.asyncio
async def test_async_responses_stream_helper(tracer_init, async_client):
    async with EcoLogits.track_impacts(providers="openai") as tracker, \
            async_client.responses.stream(model=MODEL, input="Hello World!") as stream:
        events = [event async for event in stream]
        response = await stream.get_final_response()

    assert response.output_text == "Hi"
    assert events[-1].type == "response.completed"
    assert len(tracker.calls) == 1


def test_create_stream_is_sdk_stream(tracer_init, server, client, monkeypatch):
    latencies = []
    real_llm_impacts = openai_tracer._llm_impacts

    def spy(**kwargs):
        latencies.append(kwargs["request_latency"])
        return real_llm_impacts(**kwargs)

    monkeypatch.setattr(openai_tracer, "_llm_impacts", spy)
    server.delay = 0.2
    stream = client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)

    # As without EcoLogits, the request is sent when create() is called, and the latency includes it.
    assert server.requests == 1
    assert isinstance(stream, Stream)
    with stream as entered:
        items = list(entered)

    assert items[-1].impacts.energy is not None
    assert stream.response.is_closed
    assert latencies and min(latencies) >= 0.2


@pytest.fixture
def otel_endpoints(monkeypatch):
    endpoints = []

    class Recorder:
        def record_request(self, **kwargs):
            endpoints.append(kwargs["endpoint"])

    monkeypatch.setattr(EcoLogits.config, "opentelemetry", Recorder())
    return endpoints


@requires_chat_completions("parse")
def test_chat_parse(tracer_init, client, otel_endpoints):
    with EcoLogits.track_impacts(providers="openai") as tracker:
        completion = client.chat.completions.parse(model=MODEL, messages=MESSAGES, response_format=Answer)

    assert isinstance(completion, ParsedChatCompletion)
    assert completion.model_dump()["choices"][0]["message"]["parsed"] == {"text": "Hi"}
    assert completion.impacts.energy is not None
    assert len(tracker.calls) == 1
    assert otel_endpoints == ["/chat/completions/parse"]


@requires_chat_completions("parse")
@pytest.mark.asyncio
async def test_async_chat_parse(tracer_init, async_client, otel_endpoints):
    async with EcoLogits.track_impacts(providers="openai") as tracker:
        completion = await async_client.chat.completions.parse(model=MODEL, messages=MESSAGES, response_format=Answer)

    assert isinstance(completion, ParsedChatCompletion)
    assert completion.model_dump()["choices"][0]["message"]["parsed"] == {"text": "Hi"}
    assert completion.impacts.energy is not None
    assert len(tracker.calls) == 1
    assert otel_endpoints == ["/chat/completions/parse"]
