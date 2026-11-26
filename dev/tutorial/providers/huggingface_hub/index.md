# Hugging Face Hub

> [!NOTE]
>
> **Lack of transparency**
>
> All models available on Hugging Face inference endpoints are open-weights, but there is not information on the inference infrastructure. Thus, the environmental impacts are estimated with a lower precision.

This guide focuses on the integration of 🌱 **EcoLogits** with the .

Official links:

- Repository:
- Documentation:

## Installation

EcoLogits supports versions 0.x (from 0.28.1), 1.x, and 2.x of the Hugging Face Hub Python SDK. Install the `huggingface-hub` extra to get a compatible version.

Hugging Face Hub 0.x emits a deprecation warning when EcoLogits initializes the integration. EcoLogits 0.13 will remove support for Hugging Face Hub 0.x.

```shell
pip install "ecologits[huggingface-hub]"
```

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

```python
from ecologits import EcoLogits
from huggingface_hub import InferenceClient

# Initialize EcoLogits
EcoLogits.init(providers=["huggingface_hub"])

client = InferenceClient(model="meta-llama/Meta-Llama-3.1-8B")
response = client.chat_completion(
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    max_tokens=15
)

# Get estimated environmental impacts of the inference
print(response.impacts)
```

```python
import asyncio
from ecologits import EcoLogits
from huggingface_hub import AsyncInferenceClient

# Initialize EcoLogits
EcoLogits.init(providers=["huggingface_hub"])

client = AsyncInferenceClient(model="meta-llama/Meta-Llama-3.1-8B")

async def main() -> None:
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Tell me a funny joke!"}],
        max_tokens=15
    )

    # Get estimated environmental impacts of the inference
    print(response.impacts)


asyncio.run(main())
```

### Streaming example

**In streaming mode, the impacts are calculated incrementally**, which means you don't need to sum the impacts from each data chunk. Instead, the impact information in the last chunk reflects the total cumulative environmental impacts for the entire request.

```python
from ecologits import EcoLogits
from huggingface_hub import InferenceClient

# Initialize EcoLogits
EcoLogits.init(providers=["huggingface_hub"])

client = InferenceClient(model="meta-llama/Meta-Llama-3.1-8B")
stream = client.chat_completion(
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    max_tokens=15,
    stream=True
)

for chunk in stream:
    # Get cumulative estimated environmental impacts of the inference
    print(chunk.impacts)
```

```python
import asyncio
from ecologits import EcoLogits
from huggingface_hub import AsyncInferenceClient

# Initialize EcoLogits
EcoLogits.init(providers=["huggingface_hub"])

client = AsyncInferenceClient(model="meta-llama/Meta-Llama-3.1-8B")

async def main() -> None:
    stream = await client.chat_completion(
        messages=[{"role": "user", "content": "Tell me a funny joke!"}],
        max_tokens=15,
        stream=True
    )

    async for chunk in stream:
        # Get cumulative estimated environmental impacts of the inference
        print(chunk.impacts)


asyncio.run(main())
```
