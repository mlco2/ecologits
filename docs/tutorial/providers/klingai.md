# Kling AI

!!! danger "Lack of transparency"

    Kling AI does not disclose detailed information about model architecture and inference infrastructure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on estimating video generation impacts for Kling AI models with the [`video_impacts`][estimations.video.video_impacts] function.


## Video Generation

Supported Kling AI video models:

| Model | Identifier |
|-------|------------|
| Kling 1.6 | `klingai/kling-v1.6` |
| Kling 2.6 | `klingai/kling-v2.6` |
| Kling 3 | `klingai/kling-v3` |

```python
from ecologits.estimations import video_impacts

model = "klingai/kling-v3"
resolution = "1080p"
duration = 6

impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
)

print(impacts.energy.value)
print(impacts.gwp.value)
```

Set `with_audio=False` when your request generates video without audio:

```python
impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
    with_audio=False,
)
```
