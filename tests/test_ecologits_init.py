"""Tests for EcoLogits.init, init_from_config, init_from_env (#98, #100)."""
import json
import os
import pytest

from ecologits import EcoLogits
from ecologits.exceptions import ConfigurationError, InvalidProviderError


class TestInitFromConfig:
    def test_valid_json_config_initializes(self, tmp_path):
        cfg = {"providers": ["openai"]}
        p = tmp_path / "ecologits.json"
        p.write_text(json.dumps(cfg))
        EcoLogits.init_from_config(p)
        assert "openai" in EcoLogits.config.providers

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EcoLogits.init_from_config(tmp_path / "nope.json")

    def test_invalid_provider_in_config_raises(self, tmp_path):
        cfg = {"providers": ["not_a_real_provider"]}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(InvalidProviderError):
            EcoLogits.init_from_config(p)


class TestInitFromEnv:
    def test_raises_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("ECOLOGITS_PROVIDERS", raising=False)
        monkeypatch.delenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", raising=False)
        monkeypatch.delenv("ECOLOGITS_OPENTELEMETRY_ENDPOINT", raising=False)
        with pytest.raises(ConfigurationError):
            EcoLogits.init_from_env()

    def test_initializes_from_providers_env(self, monkeypatch):
        monkeypatch.setenv("ECOLOGITS_PROVIDERS", "openai")
        EcoLogits.init_from_env()
        assert "openai" in EcoLogits.config.providers

    def test_sets_electricity_mix_zone(self, monkeypatch):
        monkeypatch.setenv("ECOLOGITS_PROVIDERS", "openai")
        monkeypatch.setenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", "FRA")
        EcoLogits.init_from_env()
        assert EcoLogits.config.electricity_mix_zone == "FRA"


class TestInitValidation:
    def test_invalid_provider_raises_with_helpful_message(self):
        with pytest.raises(InvalidProviderError) as exc_info:
            EcoLogits.init(providers=["bad_provider"])
        assert "bad_provider" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_string_provider_accepted(self):
        EcoLogits.init(providers="openai")
        assert "openai" in EcoLogits.config.providers
