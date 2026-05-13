"""Tests for config loading helpers (#98)."""
import json
import os
import pytest
from pathlib import Path

from ecologits.config import load_config_from_json, load_config_from_env
from ecologits.exceptions import ConfigurationError


class TestLoadConfigFromJson:
    def test_loads_providers(self, tmp_path):
        cfg = {"providers": ["openai", "anthropic"]}
        p = tmp_path / "ecologits.json"
        p.write_text(json.dumps(cfg))
        result = load_config_from_json(p)
        assert result["providers"] == ["openai", "anthropic"]

    def test_loads_electricity_mix_zone(self, tmp_path):
        cfg = {"providers": ["openai"], "electricity_mix_zone": "FRA"}
        p = tmp_path / "ecologits.json"
        p.write_text(json.dumps(cfg))
        result = load_config_from_json(p)
        assert result["electricity_mix_zone"] == "FRA"

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config_from_json(tmp_path / "missing.json")

    def test_raises_value_error_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config_from_json(p)

    def test_accepts_string_path(self, tmp_path):
        cfg = {"providers": ["openai"]}
        p = tmp_path / "ecologits.json"
        p.write_text(json.dumps(cfg))
        result = load_config_from_json(str(p))
        assert result["providers"] == ["openai"]


class TestLoadConfigFromEnv:
    def test_returns_empty_dict_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("ECOLOGITS_PROVIDERS", raising=False)
        monkeypatch.delenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", raising=False)
        monkeypatch.delenv("ECOLOGITS_OPENTELEMETRY_ENDPOINT", raising=False)
        result = load_config_from_env()
        assert result == {}

    def test_parses_comma_separated_providers(self, monkeypatch):
        monkeypatch.setenv("ECOLOGITS_PROVIDERS", "openai, anthropic , mistralai")
        result = load_config_from_env()
        assert result["providers"] == ["openai", "anthropic", "mistralai"]

    def test_parses_electricity_mix_zone(self, monkeypatch):
        monkeypatch.setenv("ECOLOGITS_PROVIDERS", "openai")
        monkeypatch.setenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", "DEU")
        result = load_config_from_env()
        assert result["electricity_mix_zone"] == "DEU"

    def test_parses_opentelemetry_endpoint(self, monkeypatch):
        monkeypatch.setenv("ECOLOGITS_PROVIDERS", "openai")
        monkeypatch.setenv("ECOLOGITS_OPENTELEMETRY_ENDPOINT", "http://localhost:4318/v1/metrics")
        result = load_config_from_env()
        assert result["opentelemetry_endpoint"] == "http://localhost:4318/v1/metrics"

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MY_PROVIDERS", "cohere")
        result = load_config_from_env(prefix="MY_")
        assert result["providers"] == ["cohere"]
