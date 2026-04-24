from ecologits.estimations import LLMEstimationResult, estimate_llm_impacts
from ecologits.status_messages import (
    ModelArchMultimodalWarning,
    ModelArchNotReleasedWarning,
    ModelNotRegisteredError,
    ZoneNotRegisteredError,
)
from ecologits.tracers.utils import ImpactsOutput, llm_impacts


def test_estimate_llm_impacts() -> None:
    estimation = estimate_llm_impacts(
        provider="cohere",
        model_name="c4ai-aya-expanse-8b",
        output_token_count=10,
        request_latency=10,
    )

    assert isinstance(estimation, LLMEstimationResult)
    assert estimation.energy.value > 0
    assert estimation.gwp.value > 0
    assert estimation.adpe.value > 0
    assert estimation.pe.value > 0
    assert estimation.wcf.value > 0
    assert estimation.details is None


def test_estimate_llm_impacts_with_details() -> None:
    estimation = estimate_llm_impacts(
        provider="cohere",
        model_name="c4ai-aya-expanse-8b",
        output_token_count=10,
        request_latency=10,
        include_details=True,
    )

    assert estimation.details is not None
    assert estimation.details.provider == "cohere"
    assert estimation.details.model_name == "c4ai-aya-expanse-8b"
    assert estimation.details.electricity_mix_zone == "USA"
    assert estimation.details.generation_latency > 0
    assert estimation.details.gpu_required_count > 0
    assert estimation.details.request_energy > 0
    assert estimation.details.request_usage_gwp > 0
    assert estimation.details.request_embodied_gwp > 0


def test_estimate_llm_impacts_uses_explicit_tps() -> None:
    default_estimation = estimate_llm_impacts(
        provider="cohere",
        model_name="c4ai-aya-expanse-8b",
        output_token_count=100,
    )
    fast_estimation = estimate_llm_impacts(
        provider="cohere",
        model_name="c4ai-aya-expanse-8b",
        output_token_count=100,
        tps=1000,
        ttft=0.1,
    )

    assert fast_estimation.energy.value < default_estimation.energy.value


def test_estimate_llm_impacts_model_error() -> None:
    estimation = estimate_llm_impacts(
        provider="openai",
        model_name="unknown-model",
        output_token_count=10,
    )

    assert estimation.energy is None
    assert estimation.has_errors
    assert isinstance(estimation.errors[0], ModelNotRegisteredError)


def test_estimate_llm_impacts_zone_error() -> None:
    estimation = estimate_llm_impacts(
        provider="openai",
        model_name="gpt-4o-mini",
        output_token_count=10,
        electricity_mix_zone="UNKNOWN-ZONE",
    )

    assert estimation.energy is None
    assert estimation.has_errors
    assert isinstance(estimation.errors[0], ZoneNotRegisteredError)


def test_estimate_llm_impacts_warnings() -> None:
    estimation = estimate_llm_impacts(
        provider="openai",
        model_name="gpt-4o-mini",
        output_token_count=10,
    )

    assert estimation.energy.value > 0
    assert estimation.has_warnings
    assert isinstance(estimation.warnings[0], (ModelArchNotReleasedWarning, ModelArchMultimodalWarning))
    assert isinstance(estimation.warnings[1], (ModelArchNotReleasedWarning, ModelArchMultimodalWarning))


def test_llm_impacts_wrapper_matches_estimation() -> None:
    estimation = estimate_llm_impacts(
        provider="cohere",
        model_name="c4ai-aya-expanse-8b",
        output_token_count=10,
        request_latency=10,
    )
    tracer_impacts = llm_impacts(
        provider="cohere",
        model_name="c4ai-aya-expanse-8b",
        output_token_count=10,
        request_latency=10,
    )

    assert isinstance(tracer_impacts, ImpactsOutput)
    assert tracer_impacts == estimation
