# Text Generation

EcoLogits can estimate the environmental impacts of **text generation** requests with the [`text_impacts`][ecologits.estimations.text.text_impacts] function. To learn how this estimation is made, read the [text generation methodology](../methodology/llm_inference.md).

Estimates are computed locally, without making an API request. Use the model and token counts from a request you’ve already made.

```python
from ecologits.estimations import text_impacts

impacts = text_impacts(
    model_name="openai/gpt-6-luna",
    input_tokens=2000,
    output_tokens=1000,
)

if not impacts.has_errors:
    print(impacts.energy.value)  # kWh
    print(impacts.gwp.value)     # kgCO2eq
    print(impacts.wcf.value)     # L
```

The function returns an [`ImpactsOutput`][ecologits.tracers.utils.ImpactsOutput] with total, usage, and embodied impacts, plus warnings and errors. Values can be numbers or [`RangeValue`][ecologits.utils.range_value.RangeValue] intervals with `min`, `max`, and `mean`.

??? warning "Migrating from `llm_impacts`"

    `ecologits.tracers.utils.llm_impacts` is deprecated. To use the new interface:

    - Import `text_impacts` from `ecologits.estimations`.
    - Include the developer namespace in `model_name`, e.g. `openai/gpt-6-luna`.
    - Supply `input_tokens`, optionnally including cache reads and writes.
    - Rename `output_token_count` to `output_tokens` and `electricity_mix_zone` to `datacenter_location`.
    - The positional arguments are `model_name, input_tokens, output_tokens`. Pass `provider` by keyword if needed.
    - `request_latency` is optional and keyword-only. Pass your measured latency to keep the same estimates.

    The deprecated function continues to work with its original arguments and calculations, emitting a `DeprecationWarning` when called.

## Main parameters

### Model and provider

Pass a namespaced model ID, such as `openai/gpt-6-luna` or `zai/glm-5.2`. The optional `provider` keyword selects the hosting service: `"openai"`, `"anthropic"`, `"cohere"`, `"google"`, or `"mistralai"`. Each model/provider pair must be registered.

!!! warning "Hugging Face is not supported yet"

    Models served through Hugging Face will be supported in a future release.

| Model ID | Provider argument | Deployment used |
| --- | --- | --- |
| `openai/gpt-6-luna` | Omitted or `"openai"` | OpenAI |
| `zai/glm-5.2` | `"mistralai"` | GLM served by Mistral |

When `provider` is omitted, models in the `openai`, `anthropic`, `cohere`, `google`, and `mistralai` namespaces use their developer's own API. Otherwise, pass the provider explicitly. An unsupported combination, including Azure or OpenRouter, returns a structured error. Estimates use the registered deployment assumptions across service tiers.

### Input tokens and caching

`input_tokens` is the total input count, including tokens read from or written to the provider's prompt cache. It is required and must be a non-negative integer.

The optional, keyword-only cache counts are subsets of `input_tokens`:

- `cache_read_tokens`: tokens whose computed state was reused from the cache.
- `cache_write_tokens`: tokens written to the cache by this request.

They default to `None` for unreported usage; use `0` for an observed zero. Both must be non-negative integers, and their combined known count must not exceed `input_tokens`. For example, 2,000 input tokens can include 1,500 cache reads and 200 cache writes.

Normalize provider usage first: Anthropic's `input_tokens` excludes cache reads and writes, so add `cache_read_input_tokens` and `cache_creation_input_tokens`. OpenAI's input total and Gemini's prompt total already include cached tokens.

!!! warning "Input tokens and caching are not modeled yet"

    These parameters are validated but do not affect estimates yet. A warning is logged once when a positive input count is supplied.

### Output tokens

`output_tokens` is the number of generated tokens, including reasoning tokens when reported separately. It is required and must be a non-negative integer.

Generation latency is estimated from registered throughput and time to first token, falling back to the latency regression. No measured request latency is required.

## Extra parameters

### Data center configuration

Location, PUE, and WUE use provider defaults when omitted or set to `None`. All overrides are keyword-only:

| Parameter | Description |
| --- | --- |
| `datacenter_location` | Electricity mix zone, such as `FRA`. Use `WOR` explicitly for the world mix. |
| `datacenter_pue` | Power Usage Effectiveness, as a number or `RangeValue`. |
| `datacenter_wue` | Water Usage Effectiveness in L/kWh, as a number or `RangeValue`. |

PUE must be finite and at least 1. WUE must be finite and at least 0. These bounds apply to both endpoints of a range. Invalid overrides raise `ValueError`. Explicit values always take precedence over provider defaults.

```python
from ecologits.utils.range_value import RangeValue

impacts = text_impacts(
    model_name="openai/gpt-6-luna",
    provider="openai",
    input_tokens=2000,
    output_tokens=1000,
    datacenter_location="FRA",
    datacenter_pue=RangeValue(min=1.1, max=1.2),
    datacenter_wue=0.5,  # L/kWh
)
```

### Request latency

If you measured the request, pass `request_latency` in seconds to improve precision. The estimated generation latency is capped by this value, so it lowers the estimate when the request was faster than expected. It must be a finite positive number.

```python
impacts = text_impacts(
    model_name="openai/gpt-6-luna",
    input_tokens=2000,
    output_tokens=1000,
    request_latency=4.2,  # seconds
)
```

## Integration pattern

Make a request with the OpenAI SDK, then pass the returned model and token usage to `text_impacts`. This example requires `OPENAI_API_KEY` to be configured.

```python
from openai import OpenAI

from ecologits.estimations import text_impacts

client = OpenAI()
response = client.responses.create(
    model="gpt-6-luna",
    input="Explain how solar panels work in a few sentences.",
)

if response.usage is None:
    raise RuntimeError("The response did not include token usage.")

usage = response.usage
impacts = text_impacts(
    model_name=f"openai/{response.model}",
    provider="openai",
    input_tokens=usage.input_tokens,
    output_tokens=usage.output_tokens,
    cache_read_tokens=usage.input_tokens_details.cached_tokens,
    cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
)

if not impacts.has_errors:
    payload = {"response_id": response.id, "impacts": impacts.model_dump(mode="json")}
```

## Handling warnings and errors

Check warnings and errors before using estimates in reporting:

```python
if impacts.has_errors:
    for error in impacts.errors:
        print(error.message)

if impacts.has_warnings:
    for warning in impacts.warnings:
        print(warning.message)
```

Warnings indicate model assumptions or fallback values; errors mean no estimate could be computed. Unknown model/provider pairs or electricity mix zones return errors in the result. Invalid token counts raise `ValueError`. See [Warnings and Errors](warnings_and_errors.md) for details.
