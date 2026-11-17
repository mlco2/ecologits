# Text Generation

Estimate impacts for a text generation request without calling a provider API:

```python
from ecologits.estimations import text_impacts

impacts = text_impacts(
    model_name="openai/gpt-4.1-nano",
    provider="openai",
    output_tokens=1000,
)

if not impacts.has_errors:
    print(impacts.energy.value)  # kWh
    print(impacts.gwp.value)     # kgCO2eq
    print(impacts.wcf.value)     # L
```

The function returns an [`ImpactsOutput`][tracers.utils.ImpactsOutput] with total,
usage, and embodied impacts, plus warnings and errors. Values can be numbers or
[`RangeValue`][utils.range_value.RangeValue] intervals with `min`, `max`, and `mean`.

## Models and providers

`model_name` identifies the model with a developer namespace, independently of
`provider`, which identifies its host. For example, use `openai/gpt-4.1-nano`
with `provider="openai"`, or `mistralai/Mistral-7B-v0.1` with
`provider="huggingface_hub"`. Bare names such as `gpt-4.1-nano` are not accepted.
Registered aliases use the same namespace, for example `mistralai/mistral-small-latest`.

The current registry supports the namespaces `openai`, `anthropic`, `cohere`,
`google` (hosted by `google_genai`), and `mistralai` for their respective providers.
Hugging Face model IDs retain their existing developer namespace. The estimator
adapts these namespaced IDs to the current registry internally; each hosting
combination must be supported by the registry before it can be estimated.

Supported providers are `openai`, `anthropic`, `cohere`, `google_genai`,
`huggingface_hub`, and `mistralai`. Azure and AWS deployments are not supported yet.
An unknown model/provider pair or electricity mix zone returns a result with
`has_errors=True` and no impact values.

## Tokens and latency

`output_tokens` must be a non-negative integer. Include generated reasoning tokens
when the provider reports them separately from the visible response. Invalid token
counts raise `ValueError`.

The estimator uses the same calculations and model assumptions as the automatic
provider integrations. It estimates generation latency from registered throughput
and time to first token when available, falling back to the latency regression.
It does not require measured request latency. Input tokens are not modeled.
See the [LLM inference methodology](../methodology/llm_inference.md) for the modeling
assumptions and limitations.

## Datacenter assumptions

The default electricity mix is `WOR` (World). Pass `datacenter_location=None` to use
the provider's default location, or a zone such as `FRA` to select another mix.
PUE and WUE use provider defaults when omitted or set to `None`.

All datacenter overrides are keyword-only:

```python
from ecologits.utils.range_value import RangeValue

impacts = text_impacts(
    model_name="openai/gpt-4.1-nano",
    provider="openai",
    output_tokens=1000,
    datacenter_location="FRA",
    datacenter_pue=RangeValue(min=1.1, max=1.2),
    datacenter_wue=0.0,  # L/kWh
)
```

Check `impacts.has_warnings` and `impacts.has_errors` before using the estimates.
Warnings from model assumptions and electricity mix fallback factors are preserved.
