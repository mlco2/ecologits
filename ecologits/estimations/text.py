import math

from ecologits.log import logger
from ecologits.model_repository import models
from ecologits.status_messages import ModelNotRegisteredError
from ecologits.tracers.utils import ImpactsOutput, _llm_impacts
from ecologits.utils.range_value import RangeValue

# TODO: Move model IDs and deployments to models.json.
# Public hosting-service names mapped to internal provider IDs. Each is also the namespace of the developer's own
# models, served by default.
_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "google": "google_genai",
    "mistralai": "mistralai",
}

# EcoLogits IDs for third-party models served under Mistral's API names.
_MISTRAL_DEPLOYMENTS = {
    "deepseek/deepseek-v4.1-flash-1017": "deepseek-v4-1-flash-1017",
    "zai/glm-5": "zai-glm-5",
    "zai/glm-5.2": "zai-glm-5-2",
    "zai/glm-5.3": "zai-glm-5-3",
    "zai/glm-5.3-flash": "zai-glm-5-3-flash",
}


def _resolve_deployment(model_name: str, provider: str | None) -> tuple[str, str] | None:
    """Return the internal provider and registered model name."""
    namespace, _, name = model_name.partition("/")
    if provider is None:
        provider = namespace
    if provider == "mistralai" and model_name in _MISTRAL_DEPLOYMENTS:
        name = _MISTRAL_DEPLOYMENTS[model_name]
    elif provider != namespace or provider not in _PROVIDERS:
        return None

    internal = _PROVIDERS[provider]
    if models.find_model(provider=internal, model_name=name) is None:
        return None
    return internal, name


def _validate_token_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def text_impacts(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    *,
    provider: str | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    datacenter_location: str | None = None,
    datacenter_pue: float | RangeValue | None = None,
    datacenter_wue: float | RangeValue | None = None,
    request_latency: float | None = None,
) -> ImpactsOutput:
    """Estimate the environmental impacts of a text generation request.

    Uses the same model repository and calculations as the provider integrations, without making an API request.
    Generation latency is estimated using registered throughput information when available, otherwise using the latency
    regression, and capped by request_latency when supplied. Input tokens and prompt caching are accepted and validated
    in preparation for future methodology; they do not affect estimates yet.

    Args:
        model_name: EcoLogits model ID: the developer's namespace and the model name, such as "openai/gpt-6-luna" or
            "zai/glm-5.2".
        input_tokens: Total input tokens, including tokens read from or written to the provider's prompt cache. Must be
            a non-negative integer. Currently does not affect estimates.
        output_tokens: Number of generated tokens, including reasoning tokens when applicable. Must be a non-negative
            integer.
        provider: Hosting service: "openai", "anthropic", "cohere", "google" or "mistralai". When omitted, the model
            developer's own API is used if registered. Hugging Face is not supported yet.
        cache_read_tokens: Input tokens whose computed state was reused from the provider's prompt cache. A subset of
            input_tokens. None means unknown or unreported; zero means no cache reads. Currently does not affect
            estimates.
        cache_write_tokens: Input tokens written to the provider's prompt cache by this request. A subset of
            input_tokens. None means unknown or unreported; zero means no cache writes. Currently does not affect
            estimates.
        datacenter_location: Electricity mix zone, using ISO 3166-1 alpha-3 codes or an aggregate zone such as "WOR".
            Uses the provider's default location when omitted or None.
        datacenter_pue: Power Usage Effectiveness of the datacenter. Values must be finite and at least 1. Uses the
            provider default when None.
        datacenter_wue: Water Usage Effectiveness of the datacenter in L/kWh. Values must be finite and at least 0. Uses
            the provider default when None.
        request_latency: Optional measured request latency in seconds. Improves precision when generation was faster
            than estimated: the estimated generation latency is capped by this value. Must be finite and positive.
            Estimates latency only when None.

    Returns:
        Total, usage, and embodied impacts with model and electricity mix warnings. Unknown models, providers, or zones
        return errors in the result.

    Raises:
        ValueError: If a token count is not a non-negative integer, or the sum of known cache reads and writes exceeds
            input_tokens, or a datacenter PUE or WUE override is invalid, or request_latency is not a finite positive
            number.
    """
    _validate_token_count("input_tokens", input_tokens)
    _validate_token_count("output_tokens", output_tokens)
    for name, count in (("cache_read_tokens", cache_read_tokens), ("cache_write_tokens", cache_write_tokens)):
        if count is not None:
            _validate_token_count(name, count)
    if (cache_read_tokens or 0) + (cache_write_tokens or 0) > input_tokens:
        raise ValueError("The sum of cache_read_tokens and cache_write_tokens must not exceed input_tokens.")
    if request_latency is not None and (
        isinstance(request_latency, bool)
        or not isinstance(request_latency, (int, float))
        or not math.isfinite(request_latency)
        or request_latency <= 0
    ):
        raise ValueError("request_latency must be a finite positive number of seconds.")

    if input_tokens > 0:
        logger.warning_once(
            "Input tokens and prompt caching are not modeled yet; input_tokens, "
            "cache_read_tokens and cache_write_tokens do not affect estimates."
        )

    deployment = _resolve_deployment(model_name, provider)
    if deployment is None:
        if provider is None:
            message = f"No default deployment for `{model_name}`. Supply a namespaced model ID and its provider."
        else:
            message = f"Could not find model `{model_name}` for {provider} provider."
        error = ModelNotRegisteredError(message=message)
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    internal_provider, registered_name = deployment
    return _llm_impacts(
        provider=internal_provider,
        model_name=registered_name,
        output_token_count=output_tokens,
        request_latency=request_latency,
        electricity_mix_zone=datacenter_location,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
    )
