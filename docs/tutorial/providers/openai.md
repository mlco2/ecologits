# OpenAI

!!! danger "Lack of transparency"

    OpenAI does not disclose any information about model architecture and inference infrastrucure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on the integration of :seedling: **EcoLogits** with the [OpenAI official python client :octicons-link-external-16:](https://github.com/openai/openai-python).

Official links:

* Repository: [:simple-github: openai/openai-python](https://github.com/openai/openai-python)
* Documentation: [:material-file-document: platform.openai.com](https://platform.openai.com/docs/libraries/python-library)


## Installation

To install EcoLogits along with all necessary dependencies for compatibility with the OpenAI client, please use the `openai` extra-dependency option as follows:

```shell
pip install ecologits[openai]
```

This installation command ensures that EcoLogits is set up with the specific libraries required to interface seamlessly with OpenAI's Python client.


## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

=== "Sync"

    ```python
    from ecologits import EcoLogits
    from openai import OpenAI
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])
    
    client = OpenAI(api_key="<OPENAI_API_KEY>")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Tell me a funny joke!"}
        ]
    )
    
    # Get estimated environmental impacts of the inference
    print(response.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    from ecologits import EcoLogits
    from openai import AsyncOpenAI
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])
    
    client = AsyncOpenAI(api_key="<OPENAI_API_KEY>")
    
    async def main() -> None:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Tell me a funny joke!"}
            ]
        )
        
        # Get estimated environmental impacts of the inference
        print(response.impacts)
    
    
    asyncio.run(main())
    ```

### Streaming example

**In streaming mode, the impacts are calculated incrementally**, which means you don't need to sum the impacts from each data chunk. Instead, the impact information in the last chunk reflects the total cumulative environmental impacts for the entire request.

=== "Sync" 

    ```python
    from ecologits import EcoLogits
    from openai import OpenAI
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])
    
    client = OpenAI(api_key="<OPENAI_API_KEY>")
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello World!"}],
        stream=True
    )
    
    for chunk in stream:
        # Get cumulative estimated environmental impacts of the inference
        print(chunk.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    from ecologits import EcoLogits
    from openai import AsyncOpenAI
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])
    
    client = AsyncOpenAI(api_key="<OPENAI_API_KEY>")
    
    async def main() -> None:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Tell me a funny joke!"}
            ],
            stream=True
        )
        
        async for chunk in stream:
            # Get cumulative estimated environmental impacts of the inference
            print(chunk.impacts)
    
    
    asyncio.run(main())
    ```


## Responses

### Example

The Responses API works the same way as Chat Completions. EcoLogits enriches the `Response` object with the `Impacts` object containing environmental impact data.

=== "Sync"

    ```python
    from ecologits import EcoLogits
    from openai import OpenAI

    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])

    client = OpenAI(api_key="<OPENAI_API_KEY>")

    response = client.responses.create(
        model="gpt-4o-mini",
        input="Tell me a funny joke!"
    )

    # Get estimated environmental impacts of the inference
    print(response.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    from ecologits import EcoLogits
    from openai import AsyncOpenAI

    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])

    client = AsyncOpenAI(api_key="<OPENAI_API_KEY>")

    async def main() -> None:
        response = await client.responses.create(
            model="gpt-4o-mini",
            input="Tell me a funny joke!"
        )

        # Get estimated environmental impacts of the inference
        print(response.impacts)


    asyncio.run(main())
    ```

### Streaming example

**In streaming mode, the impacts are only available on the final `response.completed` event.** Non-completed events are yielded as-is without impact data.

=== "Sync"

    ```python
    from ecologits import EcoLogits
    from openai import OpenAI

    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])

    client = OpenAI(api_key="<OPENAI_API_KEY>")

    stream = client.responses.create(
        model="gpt-4o-mini",
        input="Tell me a funny joke!",
        stream=True
    )

    for event in stream:
        if event.type == "response.completed":
            # Get estimated environmental impacts of the inference
            print(event.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    from ecologits import EcoLogits
    from openai import AsyncOpenAI

    # Initialize EcoLogits
    EcoLogits.init(providers=["openai"])

    client = AsyncOpenAI(api_key="<OPENAI_API_KEY>")

    async def main() -> None:
        stream = await client.responses.create(
            model="gpt-4o-mini",
            input="Tell me a funny joke!",
            stream=True
        )

        async for event in stream:
            if event.type == "response.completed":
                # Get estimated environmental impacts of the inference
                print(event.impacts)


    asyncio.run(main())
    ```


## Compatibility with Azure OpenAI

EcoLogits is also compatible with [Azure OpenAI :octicons-link-external-16:](https://learn.microsoft.com/en-us/azure/ai-services/openai/).

```python
import os
from ecologits import EcoLogits
from openai import AzureOpenAI

# Initialize EcoLogits
EcoLogits.init(providers=["openai"])

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01"
)


response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "user", "content": "Tell me a funny joke!"}
    ]
)

# Get estimated environmental impacts of the inference
print(response.impacts)
```
