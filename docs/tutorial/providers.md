# Supported providers

## List of all providers

| Provider name    | Extra for installation | Guide                                                                                            | 
|------------------|------------------------|--------------------------------------------------------------------------------------------------| 
| Anthropic        | `anthropic`            | [Guide for Anthropic :octicons-link-16:](providers/anthropic.md)                                 |
| ByteDance        | -                      | [Guide for ByteDance :octicons-link-16:](providers/bytedance.md)                                 |
| Cohere           | `cohere`               | [Guide for Cohere :octicons-link-16:](providers/cohere.md)                                       |
| Google Gemini    | `google-genai`         | [Guide for Google Gemini :octicons-link-16:](providers/google_genai.md)                          |
| Hugging Face Hub | `huggingface-hub`      | [Guide for Hugging Face Hub :octicons-link-16:](providers/huggingface_hub.md)                    |
| Kling AI         | -                      | [Guide for Kling AI :octicons-link-16:](providers/klingai.md)                                    |
| LiteLLM          | `litellm`              | [Guide for LiteLLM :octicons-link-16:](providers/litellm.md)                                     |
| Mistral AI       | `mistralai`            | [Guide for Mistral AI :octicons-link-16:](providers/mistralai.md)                                |
| OpenAI           | `openai`               | [Guide for OpenAI :octicons-link-16:](providers/openai.md)                                       |
| Runway           | -                      | [Guide for Runway :octicons-link-16:](providers/runway.md)                                       |
| Azure OpenAI     | `openai`               | [Guide for Azure OpenAI :octicons-link-16:](providers/openai.md#compatibility-with-azure-openai) |


## Chat Completions

| Provider        |            Completions             |       Completions (stream)        |        Completions (async)        |   Completions (async + stream)    |
|-----------------|:----------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|
| Anthropic       | :material-checkbox-marked-circle:  |  :material-check-circle-outline:  | :material-checkbox-marked-circle: |  :material-check-circle-outline:  |
| Cohere          | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |
| Google Gemini   | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |
| HuggingFace Hub | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |
| LiteLLM         | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |
| Mistral AI      | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |
| OpenAI          | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |
| Azure OpenAI    | :material-checkbox-marked-circle:  | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: | :material-checkbox-marked-circle: |

Partial support for Anthropic streams, see full documentation: [Anthropic provider](providers/anthropic.md).


## Video Generation

Video generation impact estimation uses the [`video_impacts`][estimations.video.video_impacts] function. It does not require installing provider-specific SDK extras.

| Provider | Video impact estimation | Guide |
|----------|:-----------------------:|-------|
| ByteDance | :material-checkbox-marked-circle: | [ByteDance](providers/bytedance.md) |
| Google | :material-checkbox-marked-circle: | [Google Gemini](providers/google_genai.md#video-generation) |
| Kling AI | :material-checkbox-marked-circle: | [Kling AI](providers/klingai.md) |
| OpenAI | :material-checkbox-marked-circle: | [OpenAI](providers/openai.md#video-generation) |
| Runway | :material-checkbox-marked-circle: | [Runway](providers/runway.md) |
