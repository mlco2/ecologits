import mistralai
import pytest

from ecologits.model_repository import ModelRepository, models


@pytest.fixture(autouse=True)
def recorded_mistral_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the model used by the recorded responses available during these tests."""
    repository = ModelRepository(models=models.list_models())
    repository.add_model({
        "provider": "mistralai",
        "name": "mistral-tiny-latest",
        "architecture": {"type": "dense", "parameters": 7.3},
        "warnings": ["model-arch-not-released"],
        "deployment": {"tps": 98.8, "ttft": 0.29},
    })
    monkeypatch.setattr("ecologits.tracers.utils.models", repository)


@pytest.mark.vcr
def test_mistralai_chat(tracer_init):
    client = mistralai.Mistral()
    response = client.chat.complete(
        messages=[{"role": "user", "content": "Hello World!"}], model="mistral-tiny-latest"
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_chat(tracer_init):
    client = mistralai.Mistral()
    response = await client.chat.complete_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="mistral-tiny-latest"
    )
    assert len(response.choices) > 0
    assert response.impacts.energy.value > 0


@pytest.mark.vcr
def test_mistralai_stream_chat(tracer_init):
    client = mistralai.Mistral()
    stream = client.chat.stream(
        messages=[{"role": "user", "content": "Hello World!"}], model="mistral-tiny-latest"
    )
    for chunk in stream:
        assert chunk.data.impacts.energy.value >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_mistralai_async_stream_chat(tracer_init):
    client = mistralai.Mistral()
    stream = await client.chat.stream_async(
        messages=[{"role": "user", "content": "Hello World!"}], model="mistral-tiny-latest"
    )
    async for chunk in stream:
        assert chunk.data.impacts.energy.value >= 0
