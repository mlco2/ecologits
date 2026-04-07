import pytest
from huggingface_hub import AsyncInferenceClient, InferenceClient

HF_INFERENCE_MODEL_URL = (
    "https://api-inference.huggingface.co/models/"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)


@pytest.mark.vcr
def test_huggingface_hub_chat(tracer_init):
    client = InferenceClient(model=HF_INFERENCE_MODEL_URL)
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Hello World!"}],
        max_tokens=15
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_huggingface_hub_async_chat(tracer_init):
    client = AsyncInferenceClient(model=HF_INFERENCE_MODEL_URL)
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello World!"}],
        max_tokens=15
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_huggingface_hub_stream_chat(tracer_init):
    client = InferenceClient(model=HF_INFERENCE_MODEL_URL)
    stream = client.chat_completion(
        messages=[{"role": "user", "content": "Hello World!"}],
        max_tokens=15,
        stream=True
    )
    for chunk in stream:
        assert chunk.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_huggingface_hub_async_stream_chat(tracer_init):
    client = AsyncInferenceClient(model=HF_INFERENCE_MODEL_URL)
    stream = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello World!"}],
        max_tokens=15,
        stream=True
    )
    async for chunk in stream:
        assert chunk.impacts.energy.value > 0
