import os

import pytest
import tiktoken
from aiohttp import ClientResponse, streams

from ecologits import EcoLogits

tiktoken.get_encoding("cl100k_base")

# Compatibility shim for vcrpy with aiohttp>=3.14.0. Remove once vcrpy releases
# https://github.com/kevin1024/vcrpy/pull/996 for
# https://github.com/kevin1024/vcrpy/issues/995.
if not hasattr(streams, "AsyncStreamReaderMixin"):
    class _AsyncStreamReaderMixinCompat:
        async def iter_chunked(self, n):
            while True:
                chunk = await self.read(n)
                if not chunk:
                    break
                yield chunk

    streams.AsyncStreamReaderMixin = _AsyncStreamReaderMixinCompat

_client_response_init = ClientResponse.__init__


class _VcrStreamWriterCompat:
    output_size = 0


def _client_response_init_compat(self, *args, **kwargs):
    kwargs.setdefault("stream_writer", _VcrStreamWriterCompat())
    return _client_response_init(self, *args, **kwargs)


ClientResponse.__init__ = _client_response_init_compat


@pytest.fixture(autouse=True)
def environment():
    set_envvar_if_unset("ANTHROPIC_API_KEY", "test-api-key")
    set_envvar_if_unset("MISTRAL_API_KEY", "test-api-key")
    set_envvar_if_unset("OPENAI_API_KEY", "test-api-key")
    set_envvar_if_unset("CO_API_KEY", "test-api-key")
    set_envvar_if_unset("GOOGLE_API_KEY", "test-api-key")
    set_envvar_if_unset("HF_TOKEN", "hf_test-token")
    set_envvar_if_unset("AZURE_OPENAI_API_KEY", "test-api-key")
    set_envvar_if_unset("AZURE_OPENAI_ENDPOINT", "https://ecologits-azure-openai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview")
    set_envvar_if_unset("OPENAI_API_VERSION", "2024-06-01")
    set_envvar_if_unset("AZURE_MODEL_DEPLOYMENT", "gpt-4o-mini")


def set_envvar_if_unset(name: str, value: str):
    if os.getenv(name) is None:
        os.environ[name] = value


@pytest.fixture(scope="session")
def vcr_config():
    return {"filter_headers": [
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key"
    ]}


@pytest.fixture(scope="session")
def tracer_init():
    EcoLogits.init(providers=[
        "anthropic",
        "cohere",
        "google_genai",
        "huggingface_hub",
        "litellm",
        "mistralai",
        "openai"
    ])
