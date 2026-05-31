# ByteDance

!!! danger "Lack of transparency"

    ByteDance does not disclose detailed information about model architecture and inference infrastructure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on estimating video generation impacts for ByteDance models with the [`video_impacts`][estimations.video.video_impacts] function.


## Video Generation

Supported ByteDance video models:

| Model | Identifier |
|-------|------------|
| Seedance 1.0 | `bytedance/seedance-1.0` |
| Seedance 1.5 Pro | `bytedance/seedance-1.5-pro` |

```python
from ecologits.estimations import video_impacts

model = "bytedance/seedance-1.5-pro"
resolution = "1080p"
duration = 5

impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
)

print(impacts.energy.value)
print(impacts.gwp.value)
```

Use the same `model`, `resolution`, and `duration` values that you send to your video generation provider. You can then serialize the impacts object for logs, traces, or storage:

```python
impact_payload = impacts.model_dump(mode="json")
```
