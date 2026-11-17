from ecologits.log import logger
from ecologits.model_repository import models
from ecologits.status_messages import ModelNotRegisteredError
from ecologits.tracers.utils import ImpactsOutput, llm_impacts
from ecologits.utils.range_value import RangeValue

# Compatibility with the current provider-specific registry. These namespaces
# identify model developers; provider identifies the host. Hosting combinations
# must be registered explicitly before they can be estimated.
_MODEL_NAMESPACES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "google_genai": "google",
    "mistralai": "mistralai",
}


def _registered_model_name(model_name: str, provider: str) -> str | None:
    if "/" not in model_name:
        return None
    if provider == "huggingface_hub":
        return model_name
    namespace = _MODEL_NAMESPACES.get(provider)
    if namespace is not None and model_name.startswith(f"{namespace}/"):
        return model_name.removeprefix(f"{namespace}/")
    return None


def text_impacts(
    model_name: str,
    provider: str,
    output_tokens: int,
    *,
    datacenter_location: str | None = None,
    datacenter_pue: float | RangeValue | None = None,
    datacenter_wue: float | RangeValue | None = None,
) -> ImpactsOutput:
    """Estimate the environmental impacts of a text generation request.

    Uses the same model repository and calculations as the provider integrations,
    without making an API request or requiring measured latency. Generation latency
    is estimated using registered throughput information when available, otherwise
    using the latency regression. Input tokens are not modeled by this estimator.

    Args:
        model_name: Namespaced model identifier, independent of its host (e.g.
            "openai/gpt-4.1-nano" or "mistralai/Mistral-7B-v0.1"). Registered
            aliases use the same namespace. Bare model names are not supported.
        provider: Hosting provider registered in EcoLogits: "openai", "anthropic",
            "cohere", "google_genai", "huggingface_hub", or "mistralai".
        output_tokens: Number of generated tokens, including reasoning tokens
            when applicable. Must be a non-negative integer.
        datacenter_location: Electricity mix zone, using ISO 3166-1 alpha-3 codes
            or an aggregate zone such as "WOR". Uses the provider's default
            location when omitted or None.
        datacenter_pue: Power Usage Effectiveness of the datacenter. Uses the
            provider default when None.
        datacenter_wue: Water Usage Effectiveness of the datacenter in L/kWh.
            Uses the provider default when None.

    Returns:
        Total, usage, and embodied impacts with model and electricity mix warnings.
        Unknown models, providers, or zones return errors in the result.

    Raises:
        ValueError: If output_tokens is not a non-negative integer.
    """
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
        raise ValueError("output_tokens must be a non-negative integer.")

    registered_name = _registered_model_name(model_name, provider)
    if registered_name is None or models.find_model(provider=provider, model_name=registered_name) is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    return llm_impacts(
        provider=provider,
        model_name=registered_name,
        output_token_count=output_tokens,
        request_latency=None,
        electricity_mix_zone=datacenter_location,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
    )
