import litellm
import pytest

from ecologits.tracers.litellm_tracer import litellm_match_model


@pytest.mark.parametrize("model_name,expected_tuple", [
    ("gpt-4o", ("openai", "gpt-4o")),
    ("mistral/mistral-large-latest", ("mistralai", "mistral-large-latest")),
    ("command-r", ("cohere", "command-r")),
    ("huggingface/meta-llama/Llama-2-7b", ("huggingface_hub", "meta-llama/Llama-2-7b")),
    ("gemini/gemini-2.5-pro", ("google_genai", "gemini-2.5-pro")),
    ("gemini/gemini-2.5-flash", ("google_genai", "gemini-2.5-flash")),
    ("gemini/gemini-2.5-flash-lite", ("google_genai", "gemini-2.5-flash-lite"))
])
def test_litellm_match_model(model_name, expected_tuple):
    assert litellm_match_model(model_name) == expected_tuple


@pytest.mark.vcr
def test_litellm_chat(tracer_init):
    response = litellm.completion(
        model="mistral/mistral-small",
        messages=[{"content": "Hello World!", "role": "user"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.skip("Broken test with latest version of LiteLLM")
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_chat(tracer_init):
    response = await litellm.acompletion(
        model="mistral/mistral-small",
        messages=[{"role": "user", "content": "Hello World!"}]
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_litellm_stream_chat(tracer_init):
    stream = litellm.completion(
        model="mistral/mistral-small",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    for chunk in stream:
        assert chunk.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_stream_chat(tracer_init):
    stream = await litellm.acompletion(
        model="mistral/mistral-small",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value >= 0
