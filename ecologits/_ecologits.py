from __future__ import annotations

import importlib.metadata
import importlib.util
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from packaging.version import Version

from ecologits.exceptions import EcoLogitsError
from ecologits.log import logger

if TYPE_CHECKING:
    from ecologits.measurement_repository import Deployment
    from ecologits.utils.opentelemetry import OpenTelemetry, OpenTelemetryLabels


def init_openai_instrumentor() -> None:
    if importlib.util.find_spec("openai") is not None:
        from ecologits.tracers.openai_tracer import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument()


def init_anthropic_instrumentor() -> None:
    if importlib.util.find_spec("anthropic") is not None:
        from ecologits.tracers.anthropic_tracer import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument()


def init_mistralai_instrumentor() -> None:
    if importlib.util.find_spec("mistralai") is not None:
        from ecologits.tracers.mistralai_tracer import MistralAIInstrumentor

        instrumentor = MistralAIInstrumentor()
        instrumentor.instrument()


def init_huggingface_instrumentor() -> None:
    if importlib.util.find_spec("huggingface_hub") is not None:
        version = Version(importlib.metadata.version("huggingface_hub"))
        if version >= Version("0.22.0"):
            from ecologits.tracers.huggingface_tracer import HuggingfaceInstrumentor

            instrumentor = HuggingfaceInstrumentor()
            instrumentor.instrument()


def init_cohere_instrumentor() -> None:
    if importlib.util.find_spec("cohere") is not None:
        from ecologits.tracers.cohere_tracer import CohereInstrumentor

        instrumentor = CohereInstrumentor()
        instrumentor.instrument()


def init_google_genai_instrumentor() -> None:
    if importlib.util.find_spec("google") is not None \
            and importlib.util.find_spec("google.genai") is not None:
        from ecologits.tracers.google_genai_tracer import GoogleGenaiInstrumentor

        instrumentor = GoogleGenaiInstrumentor()
        instrumentor.instrument()


def init_litellm_instrumentor() -> None:
    if importlib.util.find_spec("litellm") is not None:
        from ecologits.tracers.litellm_tracer import LiteLLMInstrumentor

        instrumentor = LiteLLMInstrumentor()
        instrumentor.instrument()


def init_local_instrumentor() -> None:
    """
    No-op: the `local` provider patches nothing.

    Self-hosted inference has no vendor client library to instrument. Impacts
    come from a measured reference, reached either by calling `llm_impacts`
    directly or through a tracer for whichever client is used to talk to the
    local endpoint (litellm, for instance).
    """
    return


_INSTRUMENTS = {
    "openai": init_openai_instrumentor,
    "anthropic": init_anthropic_instrumentor,
    "mistralai": init_mistralai_instrumentor,
    "huggingface_hub": init_huggingface_instrumentor,
    "cohere": init_cohere_instrumentor,
    "google_genai": init_google_genai_instrumentor,
    "litellm": init_litellm_instrumentor,
    "local": init_local_instrumentor,
}


class EcoLogits:
    """
    EcoLogits instrumentor to initialize function patching for each provider.

    Examples:
        EcoLogits initialization example with OpenAI.
        ```python
        from ecologits import EcoLogits
        from openai import OpenAI

        EcoLogits.init(providers=["openai"])

        client = OpenAI(api_key="<OPENAI_API_KEY>")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Tell me a funny joke!"}
            ]
        )

        # Get estimated environmental impacts of the inference
        print(f"Energy consumption: {response.impacts.energy.value} kWh")
        print(f"GHG emissions: {response.impacts.gwp.value} kgCO2eq")
        ```

    """
    @dataclass
    class _Config:
        providers: list[str] = field(default_factory=list)
        electricity_mix_zone: str | None = None
        opentelemetry: OpenTelemetry | None = None
        measurements: str | None = None
        deployment: Deployment | None = None

    config = _Config()

    @staticmethod
    def init(
        providers: str | list[str] | None = None,
        electricity_mix_zone: str | None = None,
        opentelemetry_endpoint: str | None = None,
        measurements: str | None = None,
        deployment: Deployment | None = None
    ) -> None:
        """
        Initialization static method. Will attempt to initialize all providers by default.

        Args:
            providers: list of providers to initialize (must select at least one provider).
            electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone of the datacenter.
            opentelemetry_endpoint: enable OpenTelemetry with the URL endpoint.
            measurements: path or URL of a curated snapshot of measured energy references,
                used by the `local` provider. Fetched once here, never per request.
            deployment: which measured deployment the `local` provider should use. Set it
                once here rather than passing hardware through every call site.
        """
        if isinstance(providers, str):
            providers = [providers]
        if providers is None:
            warnings.warn(
                "Initializing EcoLogits without defining providers will soon no longer be supported. For example "
                "with OpenAI, you should use `EcoLogits.init(providers=['openai'])` instead.",
                DeprecationWarning,
                stacklevel=2
            )

            providers = list(_INSTRUMENTS.keys())

        init_instruments(providers)

        EcoLogits.config.electricity_mix_zone = electricity_mix_zone
        EcoLogits.config.providers += providers
        EcoLogits.config.providers = list(set(EcoLogits.config.providers))

        if measurements is not None:
            # Loaded once, at init. A lookup per LLM call would put a network
            # round trip inside `client.chat.completions.create`.
            from ecologits.measurement_repository import load_measurements

            load_measurements(measurements)
            EcoLogits.config.measurements = measurements

        if deployment is not None:
            EcoLogits.config.deployment = deployment

        if opentelemetry_endpoint is not None:
            if not is_opentelemetry_installed():
                logger.error("OpenTelemetry package is not installed. Install with "
                             "`pip install ecologits[opentelemetry]`.")
                raise EcoLogitsError("OpenTelemetry package is not installed.")

            from ecologits.utils.opentelemetry import OpenTelemetry

            EcoLogits.config.opentelemetry = OpenTelemetry(endpoint=opentelemetry_endpoint)

    @staticmethod
    def label(**labels: str) -> OpenTelemetryLabels:
        """
        Create OpenTelemetry labels. Can be used as a context manager or as a function decorator.

        Args:
            **labels: Key-value pairs of OpenTelemetry labels.

        Returns:
            OpenTelemetryLabels instance.

        Examples:
            Context manager usage:
            ```python
            with EcoLogits.label(task="summarization"):
                response = client.chat.completions.create(...)

            # or in async mode
            async with EcoLogits.label(task="summarization"):
                response = await async_client.chat.completions.create(...)
            ```

            Decorator usage:
            ```python
            @EcoLogits.label(task="summarization")
            def text_summarization(text: str) -> str:
                response = client.chat.completions.create(...)
                ...

            # or in async mode
            @EcoLogits.label(task="summarization")
            async def text_summarization(text: str) -> str:
                response = await async_client.chat.completions.create(...)
                ...
            ```
        """
        if EcoLogits.config.opentelemetry is None:
            logger.error("You must enable OpenTelemetry to use labels. Initialize with "
                         "opentelemetry_endpoint='http://localhost:4318/v1/metrics' for instance.")
            raise EcoLogitsError("OpenTelemetry is not enabled.")

        from ecologits.utils.opentelemetry import OpenTelemetryLabels

        return OpenTelemetryLabels(**labels)


def init_instruments(providers: list[str]) -> None:
    for provider in providers:
        if provider not in _INSTRUMENTS:
            raise EcoLogitsError(f"Could not find tracer for the `{provider}` provider.")
        if provider not in EcoLogits.config.providers:
            init_func = _INSTRUMENTS[provider]
            init_func()


def is_opentelemetry_installed() -> bool:
    otel_pkgs = [
        "opentelemetry",
        "opentelemetry.sdk",
        "opentelemetry.exporter"
    ]
    return all(importlib.util.find_spec(p) is not None for p in otel_pkgs)
