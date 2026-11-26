import warnings
from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock

import pytest

from ecologits import EcoLogits, _ecologits


@pytest.mark.parametrize(
    ("provider", "version", "deprecated_major"),
    [
        ("openai", "1.109.1", 1),
        ("openai", "2.54.0", None),
        ("openai", None, None),
        ("mistralai", "1.12.4", 1),
        ("mistralai", "2.10.1", None),
        ("huggingface_hub", "0.36.2", 0),
        ("huggingface_hub", "1.33.0", None),
        ("google_genai", "1.75.0", 1),
        ("google_genai", "2.34.0", None),
        ("anthropic", "0.125.0", None),
    ],
)
@pytest.mark.parametrize("entrypoint", [EcoLogits.init, EcoLogits.track_impacts])
def test_sdk_deprecation_warning(monkeypatch, provider, version, deprecated_major, entrypoint):
    package = provider.replace("_", "-")

    def installed_version(name):
        assert name == package
        if version is None:
            raise PackageNotFoundError(name)
        return version

    instrument = Mock()
    monkeypatch.setattr(_ecologits.importlib.metadata, "version", installed_version)
    monkeypatch.setattr(_ecologits, "_instrumented_providers", set())
    monkeypatch.setitem(_ecologits._INSTRUMENTS, provider, instrument)
    monkeypatch.setattr(EcoLogits, "config", EcoLogits._Config())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        entrypoint(providers=provider)
        entrypoint(providers=provider)

    instrument.assert_called_once()
    notices = [notice for notice in caught if notice.category is FutureWarning]
    if deprecated_major is None:
        assert notices == []
    else:
        [notice] = notices
        assert f"{package} {deprecated_major}.x in version 0.13" in str(notice.message)
        assert notice.filename == __file__
