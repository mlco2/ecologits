# LiteLLM

!!! warning "Lack of transparency"

    Depending on the provider, the level of transparency varies. Thus, the environmental impacts can be estimated with a low precision.

This guide focuses on the integration of :seedling: **EcoLogits** with the [LiteLLM official Python client :octicons-link-external-16:](https://github.com/BerriAI/litellm).

Official links:

* Repository: [:simple-github: BerriAI/litellm](https://github.com/BerriAI/litellm)
* Documentation: [:material-file-document: litellm.vercel.app](https://docs.litellm.ai/docs/)


## Installation

To install EcoLogits along with all necessary dependencies for compatibility with LiteLLM, please use the `litellm` extra-dependency option as follows:

```shell
pip install ecologits[litellm]
```

This installation command ensures that EcoLogits is set up with the specific libraries required to interface seamlessly with LiteLLM's Python client.


## Model names

EcoLogits finds the model from the name in the LiteLLM response, such as `gpt-4o` or `mistral/mistral-large-latest`. It looks for:

1. The exact model name, including the known aliases.
2. The name without host details: Bedrock regions, vendor prefixes and version suffixes, and Vertex `@` versions. For example, `us.anthropic.claude-opus-4-6-v1` becomes `claude-opus-4-6`, and `mistral-medium-3@001` becomes `mistral-medium-3`.
3. The `-latest` version of the name. For example, `mistral-small` becomes `mistral-small-latest`.
4. The most similar registered model, but only if the name contains numbers and the numbers in both names are the same. For example, `databricks-meta-llama-3-1-8b-instruct` becomes `meta-llama/Meta-Llama-3.1-8B-Instruct`, but `meta/llama-3-70b` does not become `Llama-2-70b`.

Steps 3 and 4 are guesses, so the impacts get a [`model-name-fuzzy-match`](../warnings_and_errors.md#model-name-fuzzy-match) warning. If no model matches, the impacts have no values and get a [`model-not-registered`](../warnings_and_errors.md#model-not-registered) error.

Before this lookup order, any name that was similar enough got the impacts of a registered model, with no warning. Some names that got impacts that way now get the `model-not-registered` error instead.

## Chat Completions

### Example

Integrating EcoLogits with your applications does not alter the standard outputs from the API responses. Instead, it enriches them by adding the `Impacts` object, which contains detailed environmental impact data. **Make sure you have the api key of the provider used in an .env file.** Make sure you call the litellm generation function as "litellm.completion" and not just "completion".

=== "Sync"

    ```python
    from ecologits import EcoLogits
    import litellm
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["litellm"])

    response = litellm.completion(
        model="gpt-4o-2024-05-13",
        messages=[{ "content": "Hello, how are you?","role": "user"}]
    )
    
    # Get estimated environmental impacts of the inference
    print(response.impacts)
    ```

=== "Async"

    ```python
    import asyncio
    import litellm
    from ecologits import EcoLogits
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["litellm"])

    async def main() -> None:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
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
    import litellm
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["litellm"])
    
    stream = litellm.completion(
        model="gpt-3.5-turbo",
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
    import litellm
    from ecologits import EcoLogits
    
    # Initialize EcoLogits
    EcoLogits.init(providers=["litellm"])
    
    async def main() -> None:
        stream = await litellm.acompletion(
            model="gpt-3.5-turbo",
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
