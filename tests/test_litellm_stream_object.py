import litellm
import pytest
from litellm import CustomStreamWrapper

from ecologits import EcoLogits

MODEL = "gpt-4o-mini"
MESSAGES = [{"role": "user", "content": "Hello World!"}]


def test_stream_is_custom_stream_wrapper(tracer_init):
    with EcoLogits.track_impacts(providers="litellm") as tracker:
        stream = litellm.completion(model=MODEL, messages=MESSAGES, stream=True, mock_response="Hi there")

        assert isinstance(stream, CustomStreamWrapper)
        assert stream.model == MODEL
        chunks = list(stream)

    assert chunks[-1].impacts.energy is not None
    assert len(tracker.calls) == 1


@pytest.mark.asyncio
async def test_async_stream_is_custom_stream_wrapper(tracer_init):
    async with EcoLogits.track_impacts(providers="litellm") as tracker:
        stream = await litellm.acompletion(model=MODEL, messages=MESSAGES, stream=True, mock_response="Hi there")

        assert isinstance(stream, CustomStreamWrapper)
        assert stream.model == MODEL
        chunks = [chunk async for chunk in stream]

    assert chunks[-1].impacts.energy is not None
    assert len(tracker.calls) == 1
