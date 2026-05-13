"""Configuration loading utilities for EcoLogits (#98)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config_from_json(path: str | Path) -> dict[str, Any]:
    """
    Load EcoLogits configuration from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Dictionary of configuration values suitable for passing to ``EcoLogits.init``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.

    Example::

        config = load_config_from_json("ecologits.json")
        EcoLogits.init(**config)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc


def load_config_from_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load EcoLogits configuration from a YAML file.

    Requires ``pyyaml`` (``pip install pyyaml``).

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values suitable for passing to ``EcoLogits.init``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If ``pyyaml`` is not installed.

    Example::

        config = load_config_from_yaml("ecologits.yaml")
        EcoLogits.init(**config)
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config_from_env(prefix: str = "ECOLOGITS_") -> dict[str, Any]:
    """
    Load EcoLogits configuration from environment variables.

    Recognised variables (all prefixed with *prefix*):

    - ``ECOLOGITS_PROVIDERS`` – comma-separated list of providers, e.g. ``openai,anthropic``
    - ``ECOLOGITS_ELECTRICITY_MIX_ZONE`` – ISO 3166-1 alpha-3 zone code, e.g. ``FRA``
    - ``ECOLOGITS_OPENTELEMETRY_ENDPOINT`` – OpenTelemetry collector URL

    Args:
        prefix: Environment variable prefix (default ``"ECOLOGITS_"``).

    Returns:
        Dictionary of configuration values with only the keys that were set.

    Example::

        # With ECOLOGITS_PROVIDERS=openai,anthropic in the environment:
        config = load_config_from_env()
        EcoLogits.init(**config)
    """
    config: dict[str, Any] = {}

    raw_providers = os.environ.get(f"{prefix}PROVIDERS")
    if raw_providers:
        config["providers"] = [p.strip() for p in raw_providers.split(",") if p.strip()]

    zone = os.environ.get(f"{prefix}ELECTRICITY_MIX_ZONE")
    if zone:
        config["electricity_mix_zone"] = zone

    otel = os.environ.get(f"{prefix}OPENTELEMETRY_ENDPOINT")
    if otel:
        config["opentelemetry_endpoint"] = otel

    return config
