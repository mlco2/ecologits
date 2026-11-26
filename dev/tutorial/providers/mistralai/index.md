# Mistral AI

> [!WARNING]
>
> **Lack of transparency**
>
> Some models released by Mistral AI are not open-weights, plus there is no information on the inference infrastructure. Thus, the environmental impacts are estimated with a lower precision.

This guide focuses on the integration of 🌱 **EcoLogits** with the .

Official links:

- Repository:
- Documentation:

## Installation

EcoLogits supports versions 1.x (from 1.5.0), 2.x, and 3.x of the Mistral AI Python SDK. Install the `mistralai` extra to get a compatible version.

Mistral SDK 1.x emits a deprecation warning when EcoLogits initializes the integration. EcoLogits 0.13 will remove support for Mistral SDK 1.x. The examples below use `from mistralai.client import Mistral` for SDK 2.x and 3.x. With SDK 1.x, use `from mistralai import Mistral`.

```shell
pip install "ecologits[mistralai]"
```

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

```python
from ecologits import EcoLogits
from mistralai.client import Mistral

# Initialize EcoLogits
EcoLogits.init(providers=["mistralai"])

client = Mistral(api_key="<MISTRAL_API_KEY>")

response = client.chat.complete(
    messages=[
        {"role": "user", "content": "Tell me a funny joke!"}
    ],
    model="mistral-tiny"
)

# Get estimated environmental impacts of the inference
print(response.impacts)
```

```python
import asyncio
from ecologits import EcoLogits
from mistralai.client import Mistral

# Initialize EcoLogits
EcoLogits.init(providers=["mistralai"])

client = Mistral(api_key="<MISTRAL_API_KEY>")

async def main() -> None:
    response = await client.chat.complete_async(
        messages=[
            {"role": "user", "content": "Tell me a funny joke!"}
        ],
        model="mistral-tiny"
    )

    # Get estimated environmental impacts of the inference
    print(response.impacts)


asyncio.run(main())
```

### Streaming example

**In streaming mode, the impacts are calculated incrementally**, which means you don't need to sum the impacts from each data chunk. Instead, the impact information in the last chunk reflects the total cumulative environmental impacts for the entire request.

```python
from ecologits import EcoLogits
from mistralai.client import Mistral

# Initialize EcoLogits
EcoLogits.init(providers=["mistralai"])

client = Mistral(api_key="<MISTRAL_API_KEY>")

stream = client.chat.stream(
    messages=[
        {"role": "user", "content": "Tell me a funny joke!"}
    ],
    model="mistral-tiny"
)

for chunk in stream:
    # Get cumulative estimated environmental impacts of the inference
    print(chunk.data.impacts)
```

```python
import asyncio
from ecologits import EcoLogits
from mistralai.client import Mistral

# Initialize EcoLogits
EcoLogits.init(providers=["mistralai"])

client = Mistral(api_key="<MISTRAL_API_KEY>")

async def main() -> None:
    stream = await client.chat.stream_async(
        messages=[
            {"role": "user", "content": "Tell me a funny joke!"}
        ],
        model="mistral-tiny"
    )

    async for chunk in stream:
        # Get cumulative estimated environmental impacts of the inference
        print(chunk.data.impacts)


asyncio.run(main())
```
