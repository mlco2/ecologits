# Video Generation

EcoLogits can estimate the environmental impacts of video generation requests with the [`video_impacts`][estimations.video.video_impacts] function.

This is useful when you call a video generation API and want to attach impact estimates to the request metadata, logs, analytics, or product reporting. You only need the model name, requested resolution, and requested duration.

!!! note "This page focuses on usage"

    To learn how the estimation is modeled, read the [video generation methodology](../methodology/video_generation.md).


## Quickstart

```python
from ecologits.estimations import video_impacts

impacts = video_impacts(
    model_name="openai/sora-2-pro",
    resolution="1080p",
    duration=10,
)

print(impacts.energy.value)
print(impacts.gwp.value)
```

The function returns an [`ImpactsOutput`][tracers.utils.ImpactsOutput], the same object used by EcoLogits provider integrations. It contains total impacts, usage impacts, embodied impacts, and possible warnings or errors.

```python
print(impacts.energy.value)       # Direct electricity consumption, in kWh
print(impacts.gwp.value)          # Global warming potential, in kgCO2eq
print(impacts.adpe.value)         # Abiotic resource depletion, in kgSbeq
print(impacts.pe.value)           # Primary energy, in MJ
print(impacts.wcf.value)          # Water consumption footprint, in L

print(impacts.usage.gwp.value)    # Usage phase only
print(impacts.embodied.gwp.value) # Embodied phase only
```

Video impact values are usually [`RangeValue`][utils.range_value.RangeValue] intervals. You can access the `mean`, `min`, and `max` values directly:

```python
print(impacts.gwp.value.mean) # Central estimate
print(impacts.gwp.value.min)  # Lower bound
print(impacts.gwp.value.max)  # Upper bound
```


## Function Inputs

### `model_name`

Use one of the video model identifiers supported by EcoLogits:

| Provider | Models |
|----------|--------|
| ByteDance | `bytedance/seedance-1.0`, `bytedance/seedance-1.5-pro` |
| Google | `google/veo-3.0`, `google/veo-3.0-fast`, `google/veo-3.1`, `google/veo-3.1-fast` |
| Kling AI | `klingai/kling-v1.6`, `klingai/kling-v2.6`, `klingai/kling-v3` |
| OpenAI | `openai/sora-2-pro` |
| Runway | `runway/gen-4.5` |

If the model is unknown, EcoLogits returns an `ImpactsOutput` with an error instead of raising an exception.

```python
impacts = video_impacts(
    model_name="unknown/model",
    resolution="720p",
    duration=5,
)

if impacts.has_errors:
    print(impacts.errors)
```

### `resolution`

You can use a named resolution:

```python
video_impacts(
    model_name="google/veo-3.1",
    resolution="720p",
    duration=8,
)
```

Supported named resolutions are:

| Name | Size |
|------|------|
| `720p` | 1280 x 720 |
| `1080p` | 1920 x 1080 |
| `4k` | 3840 x 2160 |

You can also pass an explicit `widthxheight` value:

```python
video_impacts(
    model_name="runway/gen-4.5",
    resolution="1024x576",
    duration=5,
)
```

### `duration`

Pass the requested video duration in seconds:

```python
video_impacts(
    model_name="klingai/kling-v3",
    resolution="1080p",
    duration=6,
)
```

Longer videos usually have higher impacts because they require more generated frames.

### `with_audio`

Set `with_audio=False` when the request generates video without audio:

```python
video_impacts(
    model_name="google/veo-3.0",
    resolution="1080p",
    duration=8,
    with_audio=False,
)
```

The default is `True`, which matches video models that generate audio with the video.


## Data Center Defaults

By default, EcoLogits uses provider-level assumptions for the model deployment location, PUE, and WUE. This keeps the function simple when you do not know where the provider served the request.

```python
impacts = video_impacts(
    model_name="bytedance/seedance-1.5-pro",
    resolution="720p",
    duration=5,
)
```

If you have better information for your context, you can override these values.

### Location

Use `datacenter_location` to select the electricity mix used for usage impacts:

```python
impacts = video_impacts(
    model_name="runway/gen-4.5",
    resolution="1080p",
    duration=10,
    datacenter_location="USA",
)
```

Available zones follow ISO 3166-1 alpha-3 codes, with extra aggregate zones such as `WOR` for World and `EEE` for Europe. When a zone is unknown, EcoLogits returns an `ImpactsOutput` with an error.

### PUE and WUE

Use `datacenter_pue` and `datacenter_wue` when you know the serving environment:

```python
impacts = video_impacts(
    model_name="openai/sora-2-pro",
    resolution="1080p",
    duration=10,
    datacenter_location="USA",
    datacenter_pue=1.2,
    datacenter_wue=0.569,
)
```

You can also pass ranges:

```python
from ecologits.utils.range_value import RangeValue

impacts = video_impacts(
    model_name="runway/gen-4.5",
    resolution="1080p",
    duration=10,
    datacenter_pue=RangeValue(min=1.09, max=1.14),
    datacenter_wue=RangeValue(min=0.13, max=0.999),
)
```


## Integration Pattern

The simplest integration is to compute impacts from the same request parameters you send to your video provider.

```python
from ecologits.estimations import video_impacts

model = "openai/sora-2-pro"
resolution = "1080p"
duration = 10

# response = client.videos.generate(
#     model=model,
#     prompt="A city street at sunrise",
#     resolution=resolution,
#     duration=duration,
# )

impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
)
```

Then serialize the result if you want to store it in logs, traces, or a database:

```python
impact_payload = impacts.model_dump(mode="json")
```

Or attach it to your own response payload:

```python
return {
    # "video_id": response.id,
    "model": model,
    "resolution": resolution,
    "duration": duration,
    "impacts": impact_payload,
}
```

This keeps the estimation close to the request parameters that matter without depending on a provider-specific response shape.


## Handling Warnings and Errors

Always check warnings and errors before using estimates in automated reporting.

```python
impacts = video_impacts(
    model_name="runway/gen-4.5",
    resolution="1080p",
    duration=10,
    datacenter_location="ABW",
)

if impacts.has_errors:
    for error in impacts.errors:
        print(error.message)

if impacts.has_warnings:
    for warning in impacts.warnings:
        print(warning.message)
```

Warnings usually mean EcoLogits completed the estimation but had to use a fallback value for part of the calculation. Errors mean the estimate could not be computed.
