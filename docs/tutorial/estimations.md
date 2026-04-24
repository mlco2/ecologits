# Manual Estimations

EcoLogits can estimate impacts without patching a provider client. This is useful when you already have usage data from another system, such as an agent report, an API gateway, or a calculator.

```python
from ecologits.estimations import estimate_llm_impacts

output_tokens = 12_782

estimation = estimate_llm_impacts(
    provider="openai",
    model_name="gpt-5-mini",
    output_token_count=output_tokens,
    tps=50,
)

print(estimation.energy.value)
print(estimation.gwp.value)
```

The returned object has the same impact fields as traced responses: `energy`, `gwp`, `adpe`, `pe`, `wcf`, `usage`, `embodied`, `warnings`, and `errors`.

!!! note "Generated tokens"

    The current methodology models generated tokens. If your tool only reports aggregate token totals, you can pass the aggregate value as `output_token_count` as a proxy, but EcoLogits does not yet distinguish prompt tokens from generated tokens in manual estimations.

## Latency and Throughput

If you know the request latency, pass it directly:

```python
estimation = estimate_llm_impacts(
    provider="openai",
    model_name="gpt-5-mini",
    output_token_count=12_782,
    request_latency=255.64,
)
```

If you do not know the latency, you can provide an average token throughput with `tps`. EcoLogits will use it to estimate generation latency.

```python
estimation = estimate_llm_impacts(
    provider="openai",
    model_name="gpt-5-mini",
    output_token_count=12_782,
    tps=50,
    ttft=0.5,
)
```

When `tps` or `ttft` are omitted, EcoLogits falls back to deployment metadata from the model repository when available, then to the methodology defaults.

## Electricity Mix

The `electricity_mix_zone` parameter represents the datacenter electricity mix, not the user's location. When it is omitted, EcoLogits uses the provider default datacenter zone when known, then falls back to the world average `WOR`.

```python
estimation = estimate_llm_impacts(
    provider="mistralai",
    model_name="mistral-large-latest",
    output_token_count=1_000,
    electricity_mix_zone="SWE",
)
```

## Intermediate Details

Set `include_details=True` to expose intermediate methodology values for explainability tools.

```python
estimation = estimate_llm_impacts(
    provider="cohere",
    model_name="c4ai-aya-expanse-8b",
    output_token_count=1_000,
    include_details=True,
)

print(estimation.details.generation_latency)
print(estimation.details.request_energy)
print(estimation.details.gpu_required_count)
```

For models represented by an interval, final impacts remain interval-aware. Intermediate details use representative mean parameter values to provide stable explanatory numbers.
