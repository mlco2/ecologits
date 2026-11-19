#

[![EcoLogits](<https://ecologits.ai/latest/assets/logo_light.png#only-light>)](<https://ecologits.ai/latest/assets/logo_light.png#only-light>) [![EcoLogits](<https://ecologits.ai/latest/assets/logo_dark.png#only-dark>)](<https://ecologits.ai/latest/assets/logo_dark.png#only-dark>)

**EcoLogits** tracks the energy consumption and environmental impacts of using generative AI models through APIs. It supports major LLM providers such as OpenAI, Anthropic, Mistral AI and more (see [supported providers](<https://ecologits.ai/latest/tutorial/providers/index.md>)).

## Requirements

Python 3.11+

EcoLogits relies on key libraries to provide essential functionalities:

- for data modeling.
- for function patching.

## Installation

**Select providers**

☐ Anthropic  ☐ Cohere  ☐ Google Gemini  ☐ Hugging Face Inference Endpoints  ☐ LiteLLM  ☐ Mistral AI  ☑ OpenAI

**Select features**

☐ OpenTelemetry

**Run this command**

```
pip install ecologits[openai]
```

For detailed instructions on each provider, refer to the complete list of [supported providers and features](<https://ecologits.ai/latest/tutorial/providers/index.md>). It is also possible to install EcoLogits without any provider.

## Usage Example

Below are simple examples demonstrating how to use LLM models with **EcoLogits** to track environmental impacts. Go to the [tutorial page](<https://ecologits.ai/latest/tutorial/providers/index.md>) for more complete examples of each supported provider.

```python
from ecologits import EcoLogits
from openai import OpenAI

# Initialize EcoLogits
EcoLogits.init(providers=["openai"])

client = OpenAI(api_key="<OPENAI_API_KEY>")

response = client.chat.completions.create( # (1)!
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Tell me a funny joke!"}
    ]
)

# Get estimated environmental impacts of the inference
print(f"Energy consumption: {response.impacts.energy.value} kWh") # (2)!
print(f"GHG emissions: {response.impacts.gwp.value} kgCO2eq") # (3)!

# Get potential warnings
if response.impacts.has_warnings:
    for w in response.impacts.warnings:
        print(w) # (4)!

# Get potential errors
if response.impacts.has_errors:
    for w in response.impacts.errors:
        print(w) # (5)!
```

1. You don't need to change your code when making a request! 🎉
2. Total estimated energy consumption for the request in kilowatt-hour (kWh). You can expect an interval, see [example here](<https://ecologits.ai/latest/tutorial/impacts/index.md#example-with-a-rangevalue>).
3. Total estimated greenhouse gas emissions for the request in kilogram of CO2 equivalent (kgCO2eq). You can expect an interval, see [example here](<https://ecologits.ai/latest/tutorial/impacts/index.md#example-with-a-rangevalue>).
4. For `gpt-4o-mini`, you can expect two warnings: [`model-arch-not-released`](<https://ecologits.ai/latest/tutorial/warnings_and_errors/index.md#model-arch-not-released>) and [`model-arch-multimodal`](<https://ecologits.ai/latest/tutorial/warnings_and_errors/index.md#model-arch-multimodal>).
5. On this example you shouldn't get any error.

```python
from ecologits import EcoLogits
from anthropic import Anthropic

# Initialize EcoLogits
EcoLogits.init(providers=["anthropic"])

client = Anthropic(api_key="<ANTHROPIC_API_KEY>")

response = client.messages.create( # (1)!
    max_tokens=100,
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    model="claude-3-haiku-20240307",
)

# Get estimated environmental impacts of the inference
print(f"Energy consumption: {response.impacts.energy.value} kWh") # (2)!
print(f"GHG emissions: {response.impacts.gwp.value} kgCO2eq") # (3)!

# Get potential warnings
if response.impacts.has_warnings:
    for w in response.impacts.warnings:
        print(w) # (4)!

# Get potential errors
if response.impacts.has_errors:
    for w in response.impacts.errors:
        print(w) # (5)!
```

1. You don't need to change your code when making a request! 🎉
2. Total estimated energy consumption for the request in kilowatt-hour (kWh). You can expect an interval, see [example here](<https://ecologits.ai/latest/tutorial/impacts/index.md#example-with-a-rangevalue>).
3. Total estimated greenhouse gas emissions for the request in kilogram of CO2 equivalent (kgCO2eq). You can expect an interval, see [example here](<https://ecologits.ai/latest/tutorial/impacts/index.md#example-with-a-rangevalue>).
4. For `claude-3-haiku-20240307`, you can expect a [`model-arch-not-released`](<https://ecologits.ai/latest/tutorial/warnings_and_errors/index.md#model-arch-not-released>) warning.
5. On this example you shouldn't get any error.

```python
from ecologits import EcoLogits
from huggingface_hub import InferenceClient

# Initialize EcoLogits
EcoLogits.init(providers=["huggingface_hub"])

client = InferenceClient(model="meta-llama/Meta-Llama-3.1-8B")
response = client.chat_completion( # (1)!
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    max_tokens=15
)

# Get estimated environmental impacts of the inference
print(f"Energy consumption: {response.impacts.energy.value} kWh") # (2)!
print(f"GHG emissions: {response.impacts.gwp.value} kgCO2eq") # (3)!

# Get potential warnings
if response.impacts.has_warnings:
    for w in response.impacts.warnings:
        print(w) # (4)!

# Get potential errors
if response.impacts.has_errors:
    for w in response.impacts.errors:
        print(w) # (5)!
```

1. You don't need to change your code when making a request! 🎉
2. Total estimated energy consumption for the request in kilowatt-hour (kWh). You can expect an interval, see [example here](<https://ecologits.ai/latest/tutorial/impacts/index.md#example-with-a-rangevalue>).
3. Total estimated greenhouse gas emissions for the request in kilogram of CO2 equivalent (kgCO2eq). You can expect an interval, see [example here](<https://ecologits.ai/latest/tutorial/impacts/index.md#example-with-a-rangevalue>).
4. On this example you shouldn't get any warning.
5. On this example you shouldn't get any error.

### Impacts output in a nutshell

**[`ImpactsOutput`](<https://ecologits.ai/latest/reference/ecologits/tracers/utils/index.md#ecologits.tracers.utils.ImpactsOutput>)** object is returned for each request made to supported clients. It gathers, **[environmental impacts](<https://ecologits.ai/latest/tutorial/impacts/index.md>)** and potential **[warnings and errors](<https://ecologits.ai/latest/tutorial/warnings_and_errors/index.md>)**.

EcoLogits aims to give a comprehensive view of the environmental footprint of generative AI models at **inference**. Impacts are reported in total, but also **per life cycle phases**:

- **Usage**: related to the impacts of the energy consumption during model execution.
- **Embodied**: related to resource extraction, manufacturing and transportation of the hardware.

And, **multiple criteria**:

- **Energy**: related to the final electricity consumption in kWh.
- **Global Warming Potential** (GWP): related to climate change, commonly known as GHG emissions in kgCO2eq.
- **Abiotic Depletion Potential for Elements** (ADPe): related to the depletion of minerals and metals in kgSbeq.
- **Primary Energy** (PE): related to the energy consumed from primary sources like oil, gas or coal in MJ.
- **Water Consumption Footprint** (WCF): related to the water consumption of the data center and electricity generation power plant in liters.

For detailed instructions on how to use the library, see our **[tutorial section](<https://ecologits.ai/latest/tutorial/index.md>)**. If you want more details on the underlying impacts calculations, see our **[methodology section](<https://ecologits.ai/latest/methodology/index.md>)**.

## License

This project is licensed under the terms of the .
