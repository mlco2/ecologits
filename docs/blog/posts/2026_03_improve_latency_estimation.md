---
title: Major update in our methodology
date: 2026-03-16
authors: 
  - samuelrince
slug: latency-estimation-with-openrouter
description: >
  Improvements on how EcoLogits estimates the generation latency using time-to-first-token and throughput data from OpenRouter.
categories:
  - Methodology
---

# Improvements in generation latency estimation

We have made an improvement regarding how we estimate the **generation latency** (i.e. estimated duration of a request excluding network latency) for a request. Our **new default estimation** is based on **time to first token** and **throughput** metrics collected on [OpenRouter](https://openrouter.ai/) for the supporter providers and models.

The new generation latency calculation is now:

$$
\text{generation latency} = \text{time to first token} + \text{throughput} \times \text{output tokens}
$$

With:

* Time-to-first-token (TTFT) represents the average duration in seconds of the pre-fill phase for an LLM.
* Throughput (TPS) represents the average number of output tokens generated per second, it helps estimate the duration of the decode phase for an LLM.

These two metrics are being collected from [OpenRouter](https://openrouter.ai/), a service that centralizes the access to many AI providers with a single API key. Since the service is widely adopted (**over 30 trillion tokens per month**) the average data should be representative of real-world conditions.

This work extends what was [previously done](2025_11_methodology_update.md#other-minor-changes) to patch the energy and impacts overestimations we had in [EcoLogits Calculator](https://huggingface.co/spaces/genai-impact/ecologits-calculator) compared to the Python library. Having this new estimation method in our core methodology makes it more reliable and reusable in all projects that depend on EcoLogits.

It is important to note that the **old method** to estimate generation latency using the [ML.ENERGY Leaderboard](https://ml.energy/leaderboard/?__theme=light) is **still being used when TTFT and TPS values are not available** on OpenRouter. This is the case for the Hugging Face inference provider that we support.
