from ecologits.status_messages import (
    ElectricityMixADPeDefaultWarning,
    ElectricityMixPEDefaultWarning,
    ElectricityMixWUEDefaultWarning,
    ModelArchMultimodalWarning,
    ModelArchNotReleasedWarning,
    ModelNotRegisteredError,
    WarningMessage,
    ZoneNotRegisteredError,
)
from ecologits.tracers.utils import llm_impacts


def test_warnings() -> None:
    impacts = llm_impacts(
        provider="openai",
        model_name="gpt-4o-mini",
        output_token_count=10,
        request_latency=10
    )
    assert impacts.energy.value > 0
    assert impacts.has_warnings
    assert isinstance(impacts.warnings[0], (ModelArchNotReleasedWarning, ModelArchMultimodalWarning))
    assert isinstance(impacts.warnings[1], (ModelArchNotReleasedWarning, ModelArchMultimodalWarning))


def test_electricity_mix_warnings() -> None:
    impacts = llm_impacts(
        provider="openai",
        model_name="gpt-4o-mini",
        output_token_count=10,
        request_latency=10,
        electricity_mix_zone="ABW"
    )
    assert impacts.energy.value > 0
    assert impacts.has_warnings
    assert any(isinstance(w, ElectricityMixADPeDefaultWarning) for w in impacts.warnings)
    assert any(isinstance(w, ElectricityMixPEDefaultWarning) for w in impacts.warnings)
    assert any(isinstance(w, ElectricityMixWUEDefaultWarning) for w in impacts.warnings)


def test_electricity_mix_no_warnings() -> None:
    impacts = llm_impacts(
        provider="openai",
        model_name="gpt-4o-mini",
        output_token_count=10,
        request_latency=10,
        electricity_mix_zone="ARG"
    )
    assert impacts.energy.value > 0
    assert impacts.has_warnings
    assert not any(w.code.startswith("electricity-mix-") for w in impacts.warnings)


def test_electricity_mix_warning_codes() -> None:
    assert isinstance(WarningMessage.from_code("electricity-mix-adpe-world"), ElectricityMixADPeDefaultWarning)
    assert isinstance(WarningMessage.from_code("electricity-mix-pe-world"), ElectricityMixPEDefaultWarning)
    assert isinstance(WarningMessage.from_code("electricity-mix-wue-world"), ElectricityMixWUEDefaultWarning)


def test_model_error() -> None:
    impacts = llm_impacts(
        provider="openai",
        model_name="unknown-model",
        output_token_count=10,
        request_latency=10
    )
    assert impacts.energy is None
    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ModelNotRegisteredError)


def test_zone_error() -> None:
    impacts = llm_impacts(
        provider="openai",
        model_name="gpt-4o-mini",
        output_token_count=10,
        request_latency=10,
        electricity_mix_zone="UNKNOWN-ZONE"
    )
    assert impacts.energy is None
    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ZoneNotRegisteredError)
