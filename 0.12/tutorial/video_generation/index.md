# Video Generation

EcoLogits can estimate the environmental impacts of **video generation** requests with the [`video_impacts`](<https://ecologits.ai/latest/reference/ecologits/estimations/video/index.md#ecologits.estimations.video.video_impacts>) function. To learn how this estimation is made, read the [video generation methodology](<https://ecologits.ai/latest/methodology/video_generation/index.md>).

Estimates are computed locally, without making an API request. Use the model, resolution, and duration from a request you’ve already made.

```python
from ecologits.estimations import video_impacts

impacts = video_impacts(
    model_name="google/veo-3.1",
    resolution="1080p",
    duration=8,
)

if not impacts.has_errors:
    print(impacts.energy.value)  # kWh
    print(impacts.gwp.value)     # kgCO2eq
    print(impacts.wcf.value)     # L
```

The function returns an [`ImpactsOutput`](<https://ecologits.ai/latest/reference/ecologits/tracers/utils/index.md#ecologits.tracers.utils.ImpactsOutput>) with total, usage, and embodied impacts, plus warnings and errors. Values can be numbers or [`RangeValue`](<https://ecologits.ai/latest/reference/ecologits/utils/range_value/index.md#ecologits.utils.range_value.RangeValue>) intervals with `min`, `max`, and `mean`.

## Main parameters

### Model and provider

`model_name` identifies the model with its developer namespace, such as `google/veo-3.1`. The provider is determined from the registered model; there is no separate `provider` argument.

EcoLogits can estimate impacts for these registered video models:

| Provider | Models |
| --- | --- |
| Alibaba | `alibaba/wan2.1`, `alibaba/wan2.2` |
| ByteDance | `bytedance/seedance-1.0`, `bytedance/seedance-1.5-pro` |
| Google | `google/veo-3.0`, `google/veo-3.0-fast`, `google/veo-3.1`, `google/veo-3.1-fast` |
| Kling AI | `klingai/kling-v1.6`, `klingai/kling-v2.6`, `klingai/kling-v3` |
| Lightricks | `lightricks/ltx-2-t2v`, `lightricks/ltx-2-t2va` |
| OpenAI | `openai/sora-2-pro` |
| Runway | `runway/gen-4.5` |
| Tencent | `tencent/hunyuanvideo`, `tencent/hunyuanvideo-1.5` |

### Resolution

`resolution` is the requested video size. Use a named resolution or explicit width and height, such as `resolution="1024x576"`:

| Name | Size |
| --- | --- |
| `720p` | 1280 x 720 |
| `1080p` | 1920 x 1080 |
| `4k` | 3840 x 2160 |

### Duration

`duration` is the requested video length in seconds. Longer videos usually have higher impacts because they require more generated frames.

## Extra parameters

### Audio generation

`with_audio` defaults to `True`. Set `with_audio=False` when generating video without audio. The adjustment depends on the model; models without native audio support use the same estimate either way.

### Data center configuration

Location, PUE, and WUE use provider defaults when omitted or set to `None`. You can override them when you have information about the serving environment:

| Parameter | Description |
| --- | --- |
| `datacenter_location` | Electricity mix zone, such as `FRA`. Use `WOR` explicitly for the world mix. |
| `datacenter_pue` | Power Usage Effectiveness, as a number or `RangeValue`. |
| `datacenter_wue` | Water Usage Effectiveness in L/kWh, as a number or `RangeValue`. |

Default locations include `USA` for Google and `SGP` for Kling AI.

```python
from ecologits.utils.range_value import RangeValue

impacts = video_impacts(
    model_name="google/veo-3.1",
    resolution="1080p",
    duration=8,
    datacenter_location="FRA",
    datacenter_pue=RangeValue(min=1.1, max=1.2),
    datacenter_wue=0.5,  # L/kWh
)
```

## Integration pattern

Make a request with the [Google Gen AI SDK](<https://github.com/googleapis/python-genai#veo>), then pass the same generation settings to `video_impacts` once the video is ready. This example requires Google Cloud credentials, project, and location to be configured. The SDK model `veo-3.1-generate-001` corresponds to EcoLogits' `google/veo-3.1`.

```python
import time

from google import genai
from google.genai import types

from ecologits.estimations import video_impacts

client = genai.Client(vertexai=True)
config = types.GenerateVideosConfig(
    number_of_videos=1,
    resolution="1080p",
    duration_seconds=8,
    generate_audio=True,
)
operation = client.models.generate_videos(
    model="veo-3.1-generate-001",
    prompt="A city street at sunrise.",
    config=config,
)

while not operation.done:
    time.sleep(10)
    operation = client.operations.get(operation)

if operation.error or not operation.response or not operation.response.generated_videos:
    raise RuntimeError("Video generation did not return a video.")

impacts = video_impacts(
    model_name="google/veo-3.1",
    resolution=config.resolution,
    duration=config.duration_seconds,
    with_audio=config.generate_audio,
)

if not impacts.has_errors:
    payload = {"operation": operation.name, "impacts": impacts.model_dump(mode="json")}
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

Warnings indicate fallback values; errors mean no estimate could be computed. Unknown models or electricity mix zones return errors in the result. Unsupported or incorrectly formatted resolutions raise `ValueError`. See [Warnings and Errors](<https://ecologits.ai/latest/tutorial/warnings_and_errors/index.md>) for details.
