# Warnings and Errors

EcoLogits may encounter situations where the calculation of environmental impacts has **high risk of inaccuracies or uncertainties** (reported as **warnings**), or where the calculation **fails** due to certain reasons like misconfiguration (reported as **errors**).

Warnings and errors are reported in the [`ImpactsOutput`][tracers.utils.ImpactsOutput] pydantic model within the `warnings` and `errors` fields respectively. Each warning or error contains a `code` (all listed below) and a `message` explaining the issue.

!!! note "Silent reporting of warnings and errors"

    By default, warnings and errors are reported **silently**. This means you won't see any warning logged or exception raised. This approach ensures your program continues to execute and avoids spamming the log output, especially when executing many requests.

Code example on how to determine **if your request resulted in any warnings or errors** and how to retrieve them.

```python
from ecologits import EcoLogits

EcoLogits.init()

response = ...  # Request code goes here

if response.impacts.has_warnings:
    for w in response.impacts.warnings:
        print(w)

if response.impacts.has_errors:
    for e in response.impacts.errors:
        print(e)
```


## Warnings

List of all the warnings that EcoLogits can report. 

### `model-arch-not-released`

This warning is reported when the model architecture is not disclosed by the provider. Thus, the estimation of environmental impacts is based on a assumption of the model architecture (e.g. dense or mixture of experts, number of parameters).

### `model-arch-multimodal`

This warning is reported when the model is multimodal. EcoLogits uses energy benchmarking data from open source LLMs that can only generate text. Models that can generate (or use as input) data from other modalities such as image, audio or video are currently not fully supported.


## Errors

List of all the errors that EcoLogits can report.

### `model-not-registered`

This error is reported when the selected model is not registered. This can happen when the model has been released recently or if you are using a custom model (such as fine-tuned models). In the first case you can try updating EcoLogits to the latest version, if the error persists, you can [open up an issue :octicons-link-external-16:](https://github.com/mlco2/ecologits/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml).

### `zone-not-registered`

This error is reported when the selected geographical zone is not registered. This can happen if the configured zone does not exist or if the custom zone is not properly registered.
