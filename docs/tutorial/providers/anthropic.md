# Anthropic

!!! danger "Lack of transparency"
    
    Anthropic does not disclose any information about model architecture and inference infrastrucure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on the integration of :seedling: **EcoLogits** with the [Anthropic official python client :octicons-link-external-16:](https://github.com/anthropics/anthropic-sdk-python).

Official links:

* Repository: [:simple-github: anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)
* Documentation: [:material-file-document: docs.anthropic.com](https://docs.anthropic.com)

## Installation

EcoLogits supports versions 0.x (from 0.45.2), 1.x, and 2.x of the Anthropic Python SDK. Install the `anthropic` extra to get a compatible version.

```shell
pip install "ecologits[anthropic]"
```

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

=== "Sync"

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

=== "Async"

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

=== "Sync"

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

=== "Async"

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


**With `client.messages.create(..., stream=True)`, the impacts are only available on the final `message_stop` event.** Other events are yielded as-is without impact data. The call returns the SDK stream object through a proxy, so you can use it with `with`, `close()` and `response`. `isinstance(stream, anthropic.Stream)` is `False` for this proxy, because the `Stream` class of the Anthropic SDK accepts only its exact type.

```python
from anthropic import Anthropic
from ecologits import EcoLogits

EcoLogits.init(providers=["anthropic"])

client = Anthropic(api_key="<ANTHROPIC_API_KEY>")

with client.messages.create(
    max_tokens=100,
    messages=[{"role": "user", "content": "Tell me a funny joke!"}],
    model="claude-haiku-4-5",
    stream=True,
) as stream:
    for event in stream:
        if event.type == "message_stop":
            # Get estimated environmental impacts of the inference
            print(event.impacts)
```
