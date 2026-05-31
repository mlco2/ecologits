# Runway

!!! danger "Lack of transparency"

    Runway does not disclose detailed information about model architecture and inference infrastructure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on estimating video generation impacts for Runway models with the [`video_impacts`][estimations.video.video_impacts] function.


## Video Generation

Supported Runway video models:

| Model | Identifier |
|-------|------------|
| Gen-4.5 | `runway/gen-4.5` |

```python
from ecologits.estimations import video_impacts

model = "runway/gen-4.5"
resolution = "1080p"
duration = 10

impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
)

print(impacts.energy.value)
print(impacts.gwp.value)
```

Runway uses range-based provider defaults for PUE and WUE. If you have more specific deployment information, you can override those values:

```python
impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
    datacenter_location="USA",
    datacenter_pue=1.09,
    datacenter_wue=0.13,
)
```
