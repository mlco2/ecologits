# Google Gemini

!!! danger "Lack of transparency"
    
    Google does not disclose any information about model architecture and inference infrastrucure. Thus, the environmental impacts are estimated with a very low precision.

This guide focuses on the integration of :seedling: **EcoLogits** with the [Google Gemini official python client :octicons-link-external-16:](https://github.com/googleapis/python-genai).

Official links:

* Repository: [:simple-github: googleapis/python-genai](https://github.com/googleapis/python-genai)
* Documentation: [:material-file-document: ai.google.dev](https://ai.google.dev/gemini-api/docs)


## Installation

To install EcoLogits along with all necessary dependencies for compatibility with the Google Gemini client, please use the `google-genai` extra-dependency option as follows:

```shell
pip install ecologits[google-genai]
```

This installation command ensures that EcoLogits is set up with the specific libraries required to interface seamlessly with Google Gemini Python client.


## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data.

=== "Sync"

    ```python
    from ecologits import EcoLogits
    from google import genai
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["google_genai"])

    # Make generate content request
    client = genai.Client()
    response = client.models.generate_content( # (1)!
        model="gemini-2.0-flash-001",
        contents="Tell me a joke!"
    )

    # Get estimated environmental impacts of the inference
    print(response.impacts)
    ```

    1. Also compatible with the "chat API", see example below:
    ```python
    chat = client.chats.create(model='gemini-2.0-flash-001')
    response = chat.send_message('Tell me a joke!')
    print(response.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    from ecologits import EcoLogits
    from google import genai
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["google_genai"])

    # Make generate content request
    async def main() -> None:
        client = genai.Client()
        response = await client.aio.models.generate_content( # (1)!
            model="gemini-2.0-flash-001",
            contents="Tell me a joke!"
        )
    
        # Get estimated environmental impacts of the inference
        print(response.impacts)

    asyncio.run(main())
    ```
    
    1. Also compatible with the "chat API", see example below:
    ```python
    chat = client.aio.chats.create(model='gemini-2.0-flash-001')
    response = await chat.send_message('Tell me a joke!')
    print(response.impacts)
    ```

### Streaming example

**In streaming mode, the impacts are calculated in the last chunk for the entire request.**

=== "Sync" 

    ```python
    from ecologits import EcoLogits
    from google import genai

    # Initialize EcoLogits
    EcoLogits.init(providers=["google_genai"])

    # Make generate content request
    client = genai.Client()
    stream = client.models.generate_content_stream( # (1)!
        model="gemini-2.0-flash-001",
        contents="Tell me a joke!"
    )

    # Get cumulative estimated environmental impacts of the inference
    for chunk in stream:
        if chunk.impacts is not None:
            print(chunk.impacts)
    ```

    1. Also compatible with the "chat API", see example below:
    ```python
    chat = client.chats.create(model='gemini-2.0-flash-001')
    stream = chat.send_message_stream("Tell me long a joke!")
    for chunk in stream:
        if chunk.impacts is not None:
            print(chunk.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    from ecologits import EcoLogits
    from google import genai

    # Initialize EcoLogits
    EcoLogits.init(providers=["google_genai"])

    # Ask something to Google Gemini in streaming and async mode
    async def main() -> None:
    # Make generate content request
        client = genai.Client()
        stream = await client.aio.models.generate_content_stream( # (1)!
            model="gemini-2.0-flash-001",
            contents="Tell me a joke!"
        )

        # Get cumulative estimated environmental impacts of the inference
        async for chunk in stream:
            if chunk.impacts is not None:
                print(chunk.impacts)

    asyncio.run(main())
    ```
    
    1. Also compatible with the "chat API", see example below:
    ```python
    chat = client.aio.chats.create(model='gemini-2.0-flash-001')
    stream = await chat.send_message_stream("Tell me long a joke!")
    async for chunk in stream:
        if chunk.impacts is not None:
            print(chunk.impacts)
    ```


## Video Generation

Video generation impacts are estimated with the [`video_impacts`][estimations.video.video_impacts] function. This is independent from the Google Gemini client tracer: call it with the same model, resolution, and duration that you use for the video generation request.

Supported Google video models:

| Model | Identifier |
|-------|------------|
| Veo 3.0 | `google/veo-3.0` |
| Veo 3.0 Fast | `google/veo-3.0-fast` |
| Veo 3.1 | `google/veo-3.1` |
| Veo 3.1 Fast | `google/veo-3.1-fast` |

```python
from ecologits.estimations import video_impacts

model = "google/veo-3.1"
resolution = "1080p"
duration = 8

# Use these same values in your Google video generation request.

impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
)

print(impacts.gwp.value)
```

Set `with_audio=False` when your request generates video without audio:

```python
impacts = video_impacts(
    model_name=model,
    resolution=resolution,
    duration=duration,
    with_audio=False,
)
```
