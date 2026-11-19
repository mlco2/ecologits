# Supported providers

## List of all providers

| Provider name | Extra for installation | Guide |
| --- | --- | --- |
| Anthropic | `anthropic` |  |
| ByteDance | \- |  |
| Cohere | `cohere` |  |
| Google Gemini | `google-genai` |  |
| Hugging Face Hub | `huggingface-hub` |  |
| Kling AI | \- |  |
| LiteLLM | `litellm` |  |
| Mistral AI | `mistralai` |  |
| OpenAI | `openai` |  |
| Runway | \- |  |
| Azure OpenAI | `openai` |  |

## Chat Completions

| Provider | Completions | Completions (stream) | Completions (async) | Completions (async + stream) |
| --- | :---: | :---: | :---: | :---: |
| Anthropic |  |  |  |  |
| Cohere |  |  |  |  |
| Google Gemini |  |  |  |  |
| HuggingFace Hub |  |  |  |  |
| LiteLLM |  |  |  |  |
| Mistral AI |  |  |  |  |
| OpenAI |  |  |  |  |
| Azure OpenAI |  |  |  |  |

Partial support for Anthropic streams, see full documentation: [Anthropic provider](<https://ecologits.ai/latest/tutorial/providers/anthropic/index.md>).

## Video Generation

Video generation impact estimation uses the [`video_impacts`](<https://ecologits.ai/latest/reference/ecologits/estimations/video/index.md#ecologits.estimations.video.video_impacts>) function. It does not require installing provider-specific SDK extras.

| Provider | Video impact estimation | Guide |
| --- | :---: | --- |
| ByteDance |  | [ByteDance](<https://ecologits.ai/latest/tutorial/providers/bytedance/index.md>) |
| Google |  | [Google Gemini](<https://ecologits.ai/latest/tutorial/providers/google_genai/index.md#video-generation>) |
| Kling AI |  | [Kling AI](<https://ecologits.ai/latest/tutorial/providers/klingai/index.md>) |
| OpenAI |  | [OpenAI](<https://ecologits.ai/latest/tutorial/providers/openai/index.md#video-generation>) |
| Runway |  | [Runway](<https://ecologits.ai/latest/tutorial/providers/runway/index.md>) |
