import pytest

from ecologits.status_messages import (
    ElectricityMixNotAvailableWarning,
    ImpactEstimateUncertainWarning,
    ModelArchDeprecatedWarning,
    ModelArchMultimodalWarning,
    ModelArchNotReleasedWarning,
    ModelNotRegisteredError,
    ProviderDataUnavailableWarning,
    WarningMessage,
    ErrorMessage,
    ZoneNotRegisteredError,
)
from ecologits.tracers.utils import llm_impacts


def test_warnings():
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


def test_model_error():
    impacts = llm_impacts(
        provider="openai",
        model_name="unknown-model",
        output_token_count=10,
        request_latency=10
    )
    assert impacts.energy is None
    assert impacts.has_errors
    assert isinstance(impacts.errors[0], ModelNotRegisteredError)


def test_zone_error():
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


class TestNewWarningCodes:
    def test_electricity_mix_not_available_from_code(self):
        w = WarningMessage.from_code("electricity-mix-not-available")
        assert isinstance(w, ElectricityMixNotAvailableWarning)
        assert "electricity" in str(w).lower()

    def test_model_arch_deprecated_from_code(self):
        w = WarningMessage.from_code("model-arch-deprecated")
        assert isinstance(w, ModelArchDeprecatedWarning)

    def test_provider_data_unavailable_from_code(self):
        w = WarningMessage.from_code("provider-data-unavailable")
        assert isinstance(w, ProviderDataUnavailableWarning)

    def test_impact_estimate_uncertain_from_code(self):
        w = WarningMessage.from_code("impact-estimate-uncertain")
        assert isinstance(w, ImpactEstimateUncertainWarning)

    def test_unknown_warning_code_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            WarningMessage.from_code("totally-unknown-code")

    def test_warning_str_contains_docs_url(self):
        w = ElectricityMixNotAvailableWarning()
        assert "ecologits.ai" in str(w)

    def test_all_new_warnings_are_warning_message(self):
        for cls in [
            ElectricityMixNotAvailableWarning,
            ModelArchDeprecatedWarning,
            ProviderDataUnavailableWarning,
            ImpactEstimateUncertainWarning,
        ]:
            assert issubclass(cls, WarningMessage)
