"""Tests for exception classes and error messages (#100)."""
import pytest

from ecologits.exceptions import (
    ConfigurationError,
    EcoLogitsError,
    InvalidProviderError,
    ModelingError,
    OpenTelemetryNotInstalledError,
    ProviderNotInstalledError,
    TracerInitializationError,
)


class TestExceptionHierarchy:
    def test_tracer_initialization_error_is_ecologits_error(self):
        assert issubclass(TracerInitializationError, EcoLogitsError)

    def test_modeling_error_is_ecologits_error(self):
        assert issubclass(ModelingError, EcoLogitsError)

    def test_provider_not_installed_is_ecologits_error(self):
        assert issubclass(ProviderNotInstalledError, EcoLogitsError)

    def test_invalid_provider_is_ecologits_error(self):
        assert issubclass(InvalidProviderError, EcoLogitsError)

    def test_configuration_error_is_ecologits_error(self):
        assert issubclass(ConfigurationError, EcoLogitsError)

    def test_otel_not_installed_is_ecologits_error(self):
        assert issubclass(OpenTelemetryNotInstalledError, EcoLogitsError)


class TestProviderNotInstalledError:
    def test_message_contains_provider_name(self):
        err = ProviderNotInstalledError("openai")
        assert "openai" in str(err)

    def test_message_contains_install_hint(self):
        err = ProviderNotInstalledError("openai")
        assert "pip install" in str(err)

    def test_message_contains_extras_bracket(self):
        err = ProviderNotInstalledError("anthropic")
        assert "[anthropic]" in str(err)

    def test_unknown_provider_still_shows_hint(self):
        err = ProviderNotInstalledError("custom_provider")
        assert "custom_provider" in str(err)

    def test_provider_attribute(self):
        err = ProviderNotInstalledError("mistralai")
        assert err.provider == "mistralai"


class TestInvalidProviderError:
    def test_message_contains_bad_provider(self):
        err = InvalidProviderError("fakeai")
        assert "fakeai" in str(err)

    def test_message_lists_valid_providers(self):
        err = InvalidProviderError("fakeai")
        assert "openai" in str(err)

    def test_provider_attribute(self):
        err = InvalidProviderError("bad")
        assert err.provider == "bad"


class TestOpenTelemetryNotInstalledError:
    def test_message_contains_install_hint(self):
        err = OpenTelemetryNotInstalledError()
        assert "pip install" in str(err)
        assert "opentelemetry" in str(err)


class TestInvalidProviderViaInit:
    def test_init_raises_invalid_provider_error(self):
        from ecologits import EcoLogits
        with pytest.raises(InvalidProviderError):
            EcoLogits.init(providers=["definitely_not_a_provider"])
