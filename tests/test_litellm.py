import litellm
import pytest

from litellm.types.utils import ModelResponse, Usage

from ecologits import EcoLogits
from ecologits.tracers.litellm_tracer import _match, litellm_chat_wrapper_non_stream, litellm_match_model


@pytest.mark.parametrize("model_name,expected_tuple", [
    ("gpt-4o", ("openai", "gpt-4o")),
    ("mistral/mistral-large-latest", ("mistralai", "mistral-large-latest")),
    ("command-r7b-12-2024", ("cohere", "command-r7b-12-2024")),
    ("huggingface/meta-llama/Llama-2-7b", ("huggingface_hub", "meta-llama/Llama-2-7b")),
    ("gemini/gemini-2.5-pro", ("google_genai", "gemini-2.5-pro")),
    ("gemini/gemini-2.5-flash", ("google_genai", "gemini-2.5-flash")),
    ("gemini/gemini-2.5-flash-lite", ("google_genai", "gemini-2.5-flash-lite"))
])
def test_litellm_match_model(model_name, expected_tuple):
    assert litellm_match_model(model_name) == expected_tuple


# Most names come from litellm.model_cost (LiteLLM 1.103.8).
@pytest.mark.parametrize("model_name,expected", [
    ("meta-llama/Llama-2-7b", ("huggingface_hub", "meta-llama/Llama-2-7b", False)),
    ("mistralai/Mistral-7B-Instruct-v0.1", ("huggingface_hub", "mistralai/Mistral-7B-Instruct-v0.1", False)),
    ("anthropic/claude-haiku-4-5-20251001", ("anthropic", "claude-haiku-4-5-20251001", False)),
    ("eu.anthropic.claude-opus-4-6", ("anthropic", "claude-opus-4-6", False)),
    ("us-gov.anthropic.claude-haiku-5-5", ("anthropic", "claude-haiku-5-5", False)),
    ("anthropic.claude-opus-4-6", ("anthropic", "claude-opus-4-6", False)),
    ("claude-opus-4-6-v1:0", ("anthropic", "claude-opus-4-6", False)),
    ("us.anthropic.claude-opus-4-6-v1", ("anthropic", "claude-opus-4-6", False)),
    ("claude-haiku-4-5@20251001", ("anthropic", "claude-haiku-4-5-20251001", False)),
    ("mistral-medium-3@001", ("mistralai", "mistral-medium-3", False)),
    ("mistral-small", ("mistralai", "mistral-small-latest", True)),
    ("gemini-pro", ("google_genai", "gemini-pro-latest", True)),
    ("mistral.mistral-7b-instruct-v0:2", ("huggingface_hub", "mistralai/Mistral-7B-Instruct-v0.2", True)),
    ("mistral.mixtral-8x7b-instruct-v0:1", ("huggingface_hub", "mistralai/Mixtral-8x7B-Instruct-v0.1", True)),
    ("databricks-meta-llama-3-1-8b-instruct", ("huggingface_hub", "meta-llama/Meta-Llama-3.1-8B-Instruct", True)),
])
def test_litellm_match_steps(model_name, expected):
    assert _match(model_name) == expected
    assert litellm_match_model(model_name) == expected[:2]


@pytest.mark.parametrize("model_name", [
    None,
    "",
    "openai/my-finetune",
    "openai/some-unknown-model",
    "anthropic/claude-made-up-9",
    "openai/glm-5.3",
    "anthropic/claude-fable",
    "glm-5",
    "o3-2025-04-16",
    "google.gemma-3-12b-it",
    "meta/llama-3-70b",
    "mistral.ministral-3-8b-instruct",
    "mistral-embed",
    "qwen-plus-latest",
    "perplexity/sonar-small-chat",
])
def test_litellm_match_model_unknown_other_version_or_no_digits_is_none(model_name):
    assert litellm_match_model(model_name) is None


@pytest.mark.parametrize("stream", [False, True])
def test_litellm_unknown_model_reports_model_not_registered(tracer_init, stream):
    response = litellm.completion(
        model="anthropic/claude-made-up-9",
        messages=[{"role": "user", "content": "Hello World!"}],
        mock_response="Hello there!",
        stream=stream,
    )
    last = list(response)[-1] if stream else response
    assert last.impacts.energy is None
    assert [e.code for e in last.impacts.errors] == ["model-not-registered"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_litellm_async_unknown_model_reports_model_not_registered(tracer_init, stream):
    response = await litellm.acompletion(
        model="anthropic/claude-made-up-9",
        messages=[{"role": "user", "content": "Hello World!"}],
        mock_response="Hello there!",
        stream=stream,
    )
    last = [chunk async for chunk in response][-1] if stream else response
    assert last.impacts.energy is None
    assert [e.code for e in last.impacts.errors] == ["model-not-registered"]


def test_litellm_response_without_model_reports_model_not_registered(tracer_init):
    response = litellm_chat_wrapper_non_stream(
        lambda: ModelResponse(model=None, usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
        None, (), {},
    )
    assert [e.code for e in response.impacts.errors] == ["model-not-registered"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model_name,fuzzy", [
    ("databricks/databricks-meta-llama-3-1-8b-instruct", True),
    ("gpt-4o", False),
])
def test_litellm_fuzzy_match_adds_warning(tracer_init, model_name, fuzzy, stream):
    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": "Hello World!"}],
        mock_response="Hello there!",
        stream=stream,
    )
    last = list(response)[-1] if stream else response
    assert last.impacts.energy is not None
    codes = [w.code for w in last.impacts.warnings or []]
    assert ("model-name-fuzzy-match" in codes) is fuzzy


def test_litellm_unknown_model_is_tracked_with_error(tracer_init):
    with EcoLogits.track_impacts(providers="litellm") as tracker:
        litellm.completion(
            model="anthropic/claude-made-up-9",
            messages=[{"role": "user", "content": "Hello World!"}],
            mock_response="Hello there!",
        )
    assert len(tracker.calls) == 1
    assert [e.code for e in tracker.calls[0].errors] == ["model-not-registered"]


@pytest.mark.vcr
def test_litellm_chat(tracer_init):
    response = litellm.completion(
        model="mistral/ministral-3b-2512",
        messages=[{"content": "Hello World!", "role": "user"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.skip("Broken test with latest version of LiteLLM")
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_chat(tracer_init):
    response = await litellm.acompletion(
        model="mistral/ministral-3b-2512",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_litellm_stream_chat(tracer_init):
    stream = litellm.completion(
        model="mistral/ministral-3b-2512",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_stream_chat(tracer_init):
    stream = await litellm.acompletion(
        model="mistral/ministral-3b-2512",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.default_cassette("../test_openai/test_openai_chat.yaml")
def test_litellm_chat_openai_model(tracer_init):
    response = litellm.completion(
        model="openai/gpt-6-luna",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0
