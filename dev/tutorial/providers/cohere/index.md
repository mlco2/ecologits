# Cohere

> [!WARNING]
>
> **Lack of transparency**
>
> Majority of models released by Cohere are open-weights, but there is no information on the inference infrastructure. Thus, the environmental impacts are estimated with a lower precision.

This guide focuses on the integration of 🌱 **EcoLogits** with the .

Official links:

- Repository:
- Documentation:

## Installation

To install EcoLogits along with all necessary dependencies for compatibility with the Cohere client, please use the `cohere` extra-dependency option as follows:

```shell
pip install ecologits[cohere]
```

This installation command ensures that EcoLogits is set up with the specific libraries required to interface seamlessly with Cohere's Python client.

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

```python
from cohere import Client
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["cohere"])

client = Client(api_key="<COHERE_API_KEY>")

response = client.chat(
    message="Tell me a funny joke!", 
    max_tokens=100
)

# Get estimated environmental impacts of the inference
print(response.impacts)
```

```python
import asyncio
from cohere import AsyncClient
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["cohere"])

client = AsyncClient(api_key="<COHERE_API_KEY>")

async def main() -> None:
    response = await client.chat(
        message="Tell me a funny joke!", 
        max_tokens=100
    )

    # Get estimated environmental impacts of the inference
    print(response.impacts)


asyncio.run(main())
```

### Streaming example

**In streaming mode, the impacts are calculated in the last chunk for the entire request.**

```python
from cohere import Client
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["cohere"])

client = Client(api_key="<COHERE_API_KEY>")

stream = client.chat_stream(
    message="Tell me a funny joke!", 
    max_tokens=100
)

for chunk in stream:
    if chunk.event_type == "stream-end":
        # Get estimated environmental impacts of the inference
        print(chunk.impacts)
```

```python
import asyncio
from cohere import AsyncClient
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["cohere"])

client = AsyncClient(api_key="<COHERE_API_KEY>")

async def main() -> None:
    stream = client.chat_stream(
        message="Tell me a funny joke!", 
        max_tokens=100
    )

    async for chunk in stream:
        if chunk.event_type == "stream-end":
            # Get estimated environmental impacts of the inference
            print(chunk.impacts)


asyncio.run(main())
```
