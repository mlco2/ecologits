import json
from inspect import Parameter, signature
from typing import Optional

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.resources.responses.responses import AsyncResponses, Responses

from ecologits.model_repository import models
from ecologits.tracers import openai_tracer
from tests.sdk_http import sdk_httpx

httpx = sdk_httpx("openai")

USAGE_CHAT = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
USAGE_RESPONSES = {
    "input_tokens": 5,
    "output_tokens": 3,
    "total_tokens": 8,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _chat(model: str) -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": USAGE_CHAT,
    }


def _chat_chunks(model: str, azure: bool) -> list[dict]:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    # Azure sends a first chunk with an empty model and no choices (content filter results).
    return [{**base, "model": "", "choices": []}, *chunks] if azure else chunks


def _response(model: str) -> dict:
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": model, "status": "completed",
        "output": [{
            "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
        }],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": USAGE_RESPONSES,
    }


def _sse(events: list[tuple[str, dict]], done: bool) -> bytes:
    body = "".join(
        (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n" for name, data in events
    )
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def _handler(response_model: str, azure: bool):
    def handle(request: httpx.Request) -> httpx.Response:
        stream = json.loads(request.content).get("stream", False)
        if request.url.path.endswith("/chat/completions"):
            if stream:
                chunks = _chat_chunks(response_model, azure)
                return httpx.Response(200, content=_sse([("", c) for c in chunks], done=True),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=_chat(response_model))
        if stream:
            completed = {"type": "response.completed", "sequence_number": 0, "response": _response(response_model)}
            return httpx.Response(200, content=_sse([("response.completed", completed)], done=False),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_response(response_model))
    return handle


async def _call(kind: str, response_model: str, requested_model: Optional[str], azure: bool = False):
    """Make one call through the real SDK and return the final object that carries impacts."""
    transport = httpx.MockTransport(_handler(response_model, azure))
    if azure:
        settings = {"api_key": "test", "api_version": "2024-10-21", "azure_endpoint": "http://test"}
        sync = AzureOpenAI(**settings, http_client=httpx.Client(transport=transport))
        aio = AsyncAzureOpenAI(**settings, http_client=httpx.AsyncClient(transport=transport))
    else:
        sync = OpenAI(api_key="test", base_url="http://test/v1", http_client=httpx.Client(transport=transport))
        aio = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=httpx.AsyncClient(transport=transport))
    model = {} if requested_model is None else {"model": requested_model}
    chat = {**model, "messages": [{"role": "user", "content": "Hello World!"}]}
    responses = {**model, "input": "Hello World!"}
    if kind == "chat":
        return sync.chat.completions.create(**chat)
    if kind == "chat_stream":
        return list(sync.chat.completions.create(**chat, stream=True))[-1]
    if kind == "async_chat":
        return await aio.chat.completions.create(**chat)
    if kind == "async_chat_stream":
        stream = await aio.chat.completions.create(**chat, stream=True)
        return [chunk async for chunk in stream][-1]
    if kind == "responses":
        return sync.responses.create(**responses)
    if kind == "responses_stream":
        return list(sync.responses.create(**responses, stream=True))[-1]
    if kind == "async_responses":
        return await aio.responses.create(**responses)
    if kind == "async_responses_stream":
        stream = await aio.responses.create(**responses, stream=True)
        return [event async for event in stream][-1]
    if kind == "responses_parse":
        return sync.responses.parse(**responses)
    if kind == "async_responses_parse":
        return await aio.responses.parse(**responses)
    raise AssertionError(kind)


CHAT_KINDS = ["chat", "chat_stream", "async_chat", "async_chat_stream"]
RESPONSES_KINDS = [
    "responses", "responses_stream", "async_responses", "async_responses_stream",
    "responses_parse", "async_responses_parse",
]


@pytest.fixture
def used_models(monkeypatch):
    used = []
    real_llm_impacts = openai_tracer._llm_impacts

    def spy(**kwargs):
        used.append(kwargs["model_name"])
        return real_llm_impacts(**kwargs)

    monkeypatch.setattr(openai_tracer, "_llm_impacts", spy)
    return used


def test_registry_preconditions():
    assert models.find_model("openai", "gpt-6-luna") is not None
    assert models.find_model("openai", "gpt-4o-mini") is not None
    assert models.find_model("openai", "gpt-6-luna-2026-09-22") is None
    assert models.find_model("openai", "my-deploy") is None
    assert models.find_model("openai", "unknown-model-a") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", CHAT_KINDS + RESPONSES_KINDS)
@pytest.mark.parametrize("response_model,requested_model,expected_model", [
    # Issue 291: Azure returns a dated id that is not registered for a registered deployment name.
    ("gpt-6-luna-2026-09-22", "gpt-6-luna", "gpt-6-luna"),
    # Both registered and different: the response id wins.
    ("gpt-4o-mini", "gpt-6-luna", "gpt-4o-mini"),
    # A deployment name that is not a model: the registered response id is used.
    ("gpt-4o-mini", "my-deploy", "gpt-4o-mini"),
    # Neither registered: keep the response id, so the call gets model-not-registered.
    ("unknown-model-a", "unknown-model-b", "unknown-model-a"),
])
async def test_openai_requested_model_fallback(tracer_init, used_models, kind, response_model, requested_model,
                                               expected_model):
    result = await _call(kind, response_model, requested_model)

    assert used_models and set(used_models) == {expected_model}
    if expected_model.startswith("unknown"):
        assert [e.code for e in result.impacts.errors] == ["model-not-registered"]
    else:
        assert result.impacts.energy is not None
    assert getattr(result, "response", result).model == response_model


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", CHAT_KINDS)
async def test_azure_dated_model_falls_back_to_deployment(tracer_init, used_models, kind):
    result = await _call(kind, "gpt-6-luna-2026-09-22", "gpt-6-luna", azure=True)

    assert used_models and set(used_models) == {"gpt-6-luna"}
    assert result.impacts.energy is not None
    assert result.model == "gpt-6-luna-2026-09-22"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", RESPONSES_KINDS)
async def test_responses_without_model_uses_response_model(tracer_init, used_models, kind):
    resource = AsyncResponses if kind.startswith("async_") else Responses
    method = resource.parse if kind.endswith("_parse") else resource.create
    if signature(method).parameters["model"].default is Parameter.empty:
        pytest.skip("This OpenAI SDK release requires a model for Responses requests.")

    result = await _call(kind, "gpt-4o-mini", None)

    assert used_models and set(used_models) == {"gpt-4o-mini"}
    assert result.impacts.energy is not None
