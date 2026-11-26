import os

import pytest
import tiktoken

from ecologits import EcoLogits

tiktoken.get_encoding("cl100k_base")


@pytest.fixture(autouse=True)
def environment():
    set_envvar_if_unset("ANTHROPIC_API_KEY", "test-api-key")
    set_envvar_if_unset("MISTRAL_API_KEY", "test-api-key")
    set_envvar_if_unset("OPENAI_API_KEY", "test-api-key")
    set_envvar_if_unset("CO_API_KEY", os.getenv("COHERE_API_KEY", "test-api-key"))
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
    return {
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers,
        "filter_headers": ["authorization", "api-key", "x-api-key", "x-goog-api-key"],
    }


def scrub_response_headers(response):
    for header in list(response["headers"]):
        if header.lower() in {"set-cookie", "openai-organization", "openai-project"}:
            del response["headers"][header]
    return response


@pytest.fixture
def google_genai_client():
    from google import genai

    # The Gemini cassettes were recorded against the v1beta API.
    client = genai.Client(http_options={"api_version": "v1beta"})
    yield client
    if hasattr(client, "close"):
        client.close()


@pytest.fixture
def mistralai_client():
    from importlib.util import find_spec

    if find_spec("mistralai.chat") is not None:
        from mistralai import Mistral
    else:
        from mistralai.client import Mistral

    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


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
