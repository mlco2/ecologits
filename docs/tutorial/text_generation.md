# Text Generation

Estimate impacts for a text generation request without calling a provider API:

```python
from ecologits.estimations import text_impacts

impacts = text_impacts(
    model_name="openai/gpt-6-luna",
    provider="openai",
    input_tokens=2000,
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
`provider`, which identifies its host. For example, use `openai/gpt-6-luna`
with `provider="openai"`, or `mistralai/Mistral-7B-v0.1` with
`provider="huggingface_hub"`. Bare names such as `gpt-6-luna` are not accepted.
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

`input_tokens` is required and must include all input tokens, including tokens read
from or written to the provider's prompt cache. `output_tokens` includes generated
reasoning tokens when the provider reports them separately from the visible response.
Both counts must be non-negative integers.

`cache_read_tokens` and `cache_write_tokens` are optional, keyword-only counts of
input tokens reused from or written to the provider's prompt cache by this request.
They default to `None` for unknown or unreported usage; use `0` for an observed zero.
When provided, they must be non-negative integers, and their sum must not exceed
`input_tokens`. Invalid counts, including booleans, raise `ValueError`.

For example, `input_tokens=10000`, `cache_read_tokens=8000`, and
`cache_write_tokens=1500` describe 8,000 reused tokens, 1,500 newly processed tokens
written to the cache, and 500 other input tokens. Cache writes still require input
processing.

Normalize provider usage before passing these counts. Anthropic's `input_tokens`
excludes cache reads and writes, so add `cache_read_input_tokens` and
`cache_creation_input_tokens` to obtain the total. OpenAI's input total and Gemini's
prompt total already include cached tokens.

The estimator uses the same calculations and model assumptions as the automatic
provider integrations. It estimates generation latency from registered throughput
and time to first token when available, falling back to the latency regression.
It does not require measured request latency. **Input tokens and prompt caching do
not affect estimates yet.** These arguments prepare the interface for future
methodology. A warning is logged once when a positive input count is supplied.
See the [LLM inference methodology](../methodology/llm_inference.md) for the modeling
assumptions and limitations.

## Datacenter assumptions

Location, PUE, and WUE use provider defaults when omitted or set to `None`.
For example, the default location is `USA` for OpenAI and `SWE` for Mistral AI.
Pass `datacenter_location="WOR"` to use the world mix explicitly, or a zone such
as `FRA` to select another mix. The world mix is also the fallback when the
provider's default location is unavailable.

All datacenter overrides are keyword-only:

```python
from ecologits.utils.range_value import RangeValue

impacts = text_impacts(
    model_name="openai/gpt-6-luna",
    provider="openai",
    input_tokens=2000,
    output_tokens=1000,
    datacenter_location="FRA",
    datacenter_pue=RangeValue(min=1.1, max=1.2),
    datacenter_wue=0.0,  # L/kWh
)
```

Check `impacts.has_warnings` and `impacts.has_errors` before using the estimates.
Warnings from model assumptions and electricity mix fallback factors are preserved.

## Migrating from llm_impacts

`ecologits.tracers.utils.llm_impacts` is deprecated. Use
`ecologits.estimations.text_impacts` for standalone text generation estimates.
The old function remains available with its original signature and calculations,
including support for measured request latency. Calling it emits a Python
`DeprecationWarning` at the caller's location. Automatic provider tracing and the
new estimation function use the internal implementation directly and do not emit
this deprecation warning.

Update your arguments as follows:

| Old `llm_impacts` argument | New `text_impacts` argument |
| --- | --- |
| `provider` | `provider` (the hosting provider) |
| `model_name="gpt-6-luna"` | `model_name="openai/gpt-6-luna"` (include the developer namespace) |
| `output_token_count` | `output_tokens` |
| `electricity_mix_zone` | `datacenter_location` |
| `datacenter_pue`, `datacenter_wue` | Same names; remain keyword-only |
| No input token argument | Required `input_tokens`, including cache reads and writes |
| `request_latency` | No equivalent; the estimator calculates latency internally |

Prefer keywords when migrating: the positional order changes from
`provider, model_name, output_token_count, request_latency, electricity_mix_zone`
to `model_name, provider, input_tokens, output_tokens`, followed by keyword-only
cache counts and datacenter overrides.

For example, replace:

```python
from ecologits.tracers.utils import llm_impacts

impacts = llm_impacts(
    provider="openai",
    model_name="gpt-6-luna",
    output_token_count=1000,
    request_latency=None,
    electricity_mix_zone="FRA",
)
```

with:

```python
from ecologits.estimations import text_impacts

impacts = text_impacts(
    model_name="openai/gpt-6-luna",
    provider="openai",
    input_tokens=2000,  # Supply the actual total input count for your request.
    output_tokens=1000,
    datacenter_location="FRA",
)
```

If you previously supplied measured `request_latency`, migrating can change your
estimates because `text_impacts` estimates latency instead. The deprecated
function continues to honor measured latency during the transition. Input and
cache counts are currently validated but do not affect estimates.
