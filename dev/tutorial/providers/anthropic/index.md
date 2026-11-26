# Anthropic

> [!CAUTION]
>
> **Lack of transparency**
>
> Anthropic does not disclose any information about model architecture and inference infrastrucure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on the integration of 🌱 **EcoLogits** with the .

Official links:

- Repository:
- Documentation:

## Installation

EcoLogits supports versions 0.x (from 0.45.2), 1.x, and 2.x of the Anthropic Python SDK. Install the `anthropic` extra to get a compatible version.

```shell
pip install "ecologits[anthropic]"
```

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

```python
from anthropic import Anthropic
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["anthropic"])

client = Anthropic(api_key="<ANTHROPIC_API_KEY>")

response = client.messages.create(
    max_tokens=100,
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    model="claude-3-haiku-20240307",
)

# Get estimated environmental impacts of the inference
print(response.impacts)
```

```python
import asyncio
from anthropic import AsyncAnthropic
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["anthropic"])

client = AsyncAnthropic(api_key="<ANTHROPIC_API_KEY>")

async def main() -> None:
    response = await client.messages.create(
        max_tokens=100,
        messages=[{"role": "user", "content": "Tell me a funny joke!"}],
        model="claude-3-haiku-20240307",
    )

    # Get estimated environmental impacts of the inference
    print(response.impacts)


asyncio.run(main())
```

### Streaming example

**In streaming mode, the impacts are calculated in the last chunk for the entire request.**

```python
from anthropic import Anthropic
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["anthropic"])

client = Anthropic(api_key="<ANTHROPIC_API_KEY>")

with client.messages.stream(
    max_tokens=100,
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    model="claude-3-haiku-20240307",
) as stream:
    for text in stream.text_stream:
        pass
    # Get estimated environmental impacts of the inference
    print(stream.impacts)
```

```python
import asyncio
from anthropic import AsyncAnthropic
from ecologits import EcoLogits

# Initialize EcoLogits
EcoLogits.init(providers=["anthropic"])

client = AsyncAnthropic(api_key="<ANTHROPIC_API_KEY>")

async def main() -> None:
    async with client.messages.stream(
        max_tokens=100,
        messages=[{"role": "user", "content": "Tell me a funny joke!"}],
        model="claude-3-haiku-20240307",
    ) as stream:
        async for text in stream.text_stream:
            pass
        # Get estimated environmental impacts of the inference
        print(stream.impacts)


asyncio.run(main())
```
