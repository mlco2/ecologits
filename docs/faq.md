# Frequently Asked Questions


## Why are training impacts not included?

Even though the training impacts of generative AI models are substantial, we currently do not implement them in our methodologies and tools. EcoLogits is aimed at estimating the impacts of an API request made to a GenAI service. To make the impact assessment complete, we indeed should take into account training impacts. However, given that we focus on services that are used by millions of people, doing billions of requests annually the training impacts are in fact negligible.

For example, looking at Llama 3 70B, the estimated training greenhouse gas emissions are $1,900\ tCO2eq$. This is significant for an AI model but comparing it to running inference on that model for say 100 billion requests annually makes the share of impacts induced by training the model becomes very small. E.g., $\frac{1,900\ \text{tCO2eq}}{100\ \text{billion requests}} = 1.9e-8\ \text{tCO2eq per request}$ or $0.019\ \text{gCO2eq per request}$. This, compared to running a simple request to Llama 3 70B that would yield $1\ \text{to}\ 5\ \text{gCO2}$ (calculated with our methodology).

It does not mean that we do not plan to integrate training impacts, it is just not a priority right now due to the difference in order of magnitude. It is also worth mentioning that estimating the number of requests that will be ever made in the lifespan of a model is very difficult, both for open-source and proprietary models. You can join the discussion on [GitHub #70 :octicons-link-external-16:](https://github.com/mlco2/ecologits/discussions/70).


## What's the difference with CodeCarbon?

EcoLogits and [CodeCarbon :octicons-link-external-16:](https://github.com/mlco2/codecarbon) are two different tools that do not aim to address the same use case. CodeCarbon should be used when you control the execution environment of your model. This means that if you deploy models on your laptop, your server or in the cloud it is preferable to use CodeCarbon to get energy consumption and estimate carbon emissions associated with running your model (including training, fine-tuning or inference).

On the other hand EcoLogits is designed for scenarios where you do not have access to the execution environment of your GenAI model because it is managed by a third-party provider.  In such cases you can rely on EcoLogits to estimate energy consumption and environmental impacts for inference workloads. Both tools are complementary and can be used together to provide a comprehensive view of environmental impacts across different deployment scenarios.


## How can I estimate impacts of general use of GenAI models? 

If you want to estimate the environmental impacts of using generative AI models without coding or making request, we recommend you to use our online webapp [EcoLogits Calculator :octicons-link-external-16:](https://huggingface.co/spaces/genai-impact/ecologits-calculator).


## How do we assess impacts for proprietary models?

Environmental impacts are calculated based on model architecture and parameter count. For proprietary models, we lack transparency from providers, so we estimate parameter counts using available information. For GPT models, we based our estimates on leaked GPT-4 architecture and scaled parameters count for GPT-4-Turbo and GPT-4o based on pricing differences. For other proprietary models like Anthropic's Claude, we assume similar impacts for models released around the same time with similar performance on public benchmarks. Please note that these estimates are based on assumptions and may not be exact. Our methods are open-source and transparent, so you can always see the hypotheses we use.


## How to reduce my environmental impact?

First, you may want to assess [**indirect impacts** :octicons-link-external-16:](https://www.tsw.co.uk/blog/environmental/indirect-environment-impacts/) and [**rebound effects** :octicons-link-external-16:](https://en.wikipedia.org/wiki/Rebound_effect_(conservation)) of the project you are building. Does the finality of your product or service is impacting negatively the environment? Does the usage of your product or service drives up consumption and environmental impacts of previously existing technology?

Try to **be frugal** and question your usages or needs of AI:

- Do you really need AI to solve your problem?
- Do you really need GenAI to solve your problem? (you can read this [paper :octicons-link-external-16:](https://arxiv.org/pdf/2305.05862))
- Prefer fine-tuning of small and existing models over generalist models.
- Evaluate before, during and after the development of your project the environmental impacts with tools like EcoLogits or [CodeCarbon :octicons-link-external-16:](https://github.com/mlco2/codecarbon) ([see more tools :octicons-link-external-16:](https://github.com/samuelrince/awesome-green-ai))
- Restrict the use case and limit the usage of your tool or feature to the desired purpose.

**Do not buy new GPUs or hardware**. Hardware production for data centers is responsible for around 50% of the impacts compared to usage impacts. The share is even more bigger for consumer devices, around 80%.

Use cloud instances that are located in low emissions / high energy efficiency data centers (see [electricitymaps.com :octicons-link-external-16:](https://app.electricitymaps.com/map)).

**Optimize your models for production use cases.** You can look at model compression technics such as quantization, pruning or distillation. There are also inference optimization tricks available in some software.
