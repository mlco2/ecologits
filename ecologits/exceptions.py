from __future__ import annotations

_PROVIDER_INSTALL_HINTS: dict[str, str] = {
    "openai": "pip install ecologits[openai]",
    "anthropic": "pip install ecologits[anthropic]",
    "mistralai": "pip install ecologits[mistralai]",
    "huggingface_hub": "pip install ecologits[huggingface-hub]",
    "cohere": "pip install ecologits[cohere]",
    "google_genai": "pip install ecologits[google-genai]",
    "litellm": "pip install ecologits[litellm]",
}

_VALID_PROVIDERS = list(_PROVIDER_INSTALL_HINTS.keys())


class EcoLogitsError(Exception):
    pass


class TracerInitializationError(EcoLogitsError):
    """Tracer is initialized twice."""
    pass


class ModelingError(EcoLogitsError):
    """Operation or computation not allowed."""
    pass


class ProviderNotInstalledError(EcoLogitsError):
    """Required provider package is not installed."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        hint = _PROVIDER_INSTALL_HINTS.get(provider, f"pip install {provider}")
        super().__init__(
            f"Provider '{provider}' is not installed. "
            f"Install it with: `{hint}`"
        )


class InvalidProviderError(EcoLogitsError):
    """Provider name is not recognised by EcoLogits."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        valid = ", ".join(f"'{p}'" for p in _VALID_PROVIDERS)
        super().__init__(
            f"Unknown provider '{provider}'. "
            f"Valid providers are: {valid}."
        )


class ConfigurationError(EcoLogitsError):
    """EcoLogits configuration is invalid or missing."""
    pass


class OpenTelemetryNotInstalledError(EcoLogitsError):
    """OpenTelemetry packages are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "OpenTelemetry package is not installed. "
            "Install it with: `pip install ecologits[opentelemetry]`"
        )
