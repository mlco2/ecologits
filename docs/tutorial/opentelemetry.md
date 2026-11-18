# OpenTelemetry support

With the support of [OpenTelemetry :octicons-link-external-16:](https://opentelemetry.io/) in EcoLogits you can monitor request metrics such as the number of input/output tokens or the environmental impacts in time.

## Installation

To install EcoLogits along with all necessary dependencies for compatibility with OpenTelemetry, please use the `opentelemetry` extra-dependency option as follows:

```shell
pip install ecologits[opentelemetry] # (1)!
```

1. Also include the provider you use in the installation command.

This installation command ensures that EcoLogits is set up with the specific libraries required to work seamlessly with OpenTelemetry.


## Configuration

To configure OpenTelemetry with EcoLogits, you will need to provide the URL endpoint at the initialization.

```python
from ecologits import EcoLogits
from openai import OpenAI

# Configure OpenTelemetry endpoint in EcoLogits
EcoLogits.init(
    providers=["openai"],
    opentelemetry_endpoint='https://localhost:4318/v1/metrics'
)

client = OpenAI()

# All requests will log ecologits metrics
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a funny joke!"}]
)
```


## Metrics

EcoLogits will automatically report the following metrics when OpenTelemetry is enabled. 

!!! warning "Different units for impact metrics"
    
    Notice that the units used for impacts differ from the original [`ImpactsOutput`][ecologits.tracers.utils.ImpactsOutput] object, in order to respect [OpenTelemetry guidelines](https://opentelemetry.io/docs/specs/semconv/general/metrics/#units).


| Metric name                         | Type    | Description                                         |
|-------------------------------------|---------|-----------------------------------------------------|
| `ecologits_requests_total`          | Counter | Number of requests                                  |
| `ecologits_input_tokens_ratio`      | Gauge   | Number of input tokens                              |
| `ecologits_output_tokens_ratio`     | Gauge   | Number of output tokens                             |
| `ecologits_request_latency_seconds` | Gauge   | Request latency in seconds                          |
| `ecologits_energy_joules`           | Gauge   | Average energy consumption of the request in joules |
| `ecologits_gwp_gCO2eq`              | Gauge   | Average GWP impacts of the request in gCO2eq        |
| `ecologits_adpe_gSbeq`              | Gauge   | Average ADPe impacts of the request in gSbeq        |
| `ecologits_pe_joules`               | Gauge   | Average PE impacts of the request in joules         |


## Labels

By default, EcoLogits will report three labels:

- `provider`: Name of the provider
- `model`: Name of the model used
- `endpoint`: API endpoint of the request

### Configure custom labels

It is possible to add custom labels to any request using the [`EcoLogits.label`][ecologits._ecologits.EcoLogits.label] method.

#### With a context manager

```python hl_lines="10"
from ecologits import EcoLogits
from openai import OpenAI

EcoLogits.init(
    providers=["openai"], 
    opentelemetry_endpoint="http://localhost:4318/v1/metrics"
)
client = OpenAI()

with EcoLogits.label(my_custom_label="test"): # (1)! 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a funny joke"}]
    )
```

1. Also works with asynchronous code with `async with` syntax.

#### With a function decorator

```python hl_lines="10"
from ecologits import EcoLogits
from openai import OpenAI

EcoLogits.init(
    providers=["openai"], 
    opentelemetry_endpoint="http://localhost:4318/v1/metrics"
)
client = OpenAI()

@EcoLogits.label(task="summarization")  # (1)! 
def summarize(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize this:\n\n{text}"}]
    )
    return response.choices[0].message.content
```

1. Also works with asynchronous functions with `async def` syntax.
