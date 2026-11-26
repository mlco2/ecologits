# Cohere

!!! warning "Lack of transparency"
    
    Majority of models released by Cohere are open-weights, but there is no information on the inference infrastructure. Thus, the environmental impacts are estimated with a lower precision.

This guide focuses on the integration of :seedling: **EcoLogits** with the [Cohere official python client :octicons-link-external-16:](https://github.com/cohere-ai/cohere-python).

Official links:

* Repository: [:simple-github: cohere-ai/cohere-python](https://github.com/cohere-ai/cohere-python)
* Documentation: [:material-file-document: docs.cohere.com](https://docs.cohere.com/)


## Installation

EcoLogits supports the `Client` and `AsyncClient` chat methods in Cohere Python SDK 5.x (from 5.13.12), 6.x, and 7.x. Install the `cohere` extra to get a compatible version.

```shell
pip install "ecologits[cohere]"
```

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

=== "Sync"

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

=== "Async"

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

=== "Sync"

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

=== "Async"

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

