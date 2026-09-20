---
title: Integrating Impact'IA into EcoLogits
date: 2026-09-20
authors: 
  - samuelrince
slug: impactia-integration
description: >
  EcoLogits integrates the work of the Impact'IA calculator from SNCF, Wavestone and Resilio: input tokens, data center network and building impacts, and an experimental estimation of training impacts.
categories:
  - Methodology
---

# Integrating Impact'IA into EcoLogits

[Impact'IA](https://github.com/SNCF-ImpactIA/ImpactIA) is an LLM environmental footprint calculator and a detailed methodology guide developed by **SNCF**, with **Wavestone** and **Resilio**, to help companies assess the environmental impacts of their generative AI projects. It builds on the EcoLogits methodology for the inference phase and extends it to a broader scope of the model life cycle. Its authors explicitly chose not to industrialize it and suggested integrating their work into open source tools such as EcoLogits. This is what this release does.

<!-- more -->

## What changes for the inference phase

Three improvements from Impact'IA are now part of the core [LLM inference methodology](../../methodology/llm_inference.md) and apply to every request:

* **Input tokens are taken into account.** The pre-fill latency (time-to-first-token) is now modeled as a linear function of the number of input tokens, scaled to each model with its median time-to-first-token measured on OpenRouter. Long prompts (large contexts, retrieval-augmented generation, agents) now increase the estimated latency, thus the server and embodied impacts of the request. All the tracers pass the number of input tokens reported by the providers.
* **Data center network equipment is included.** The firewalls, routers and switches attached to each server are accounted for both in the electricity consumption and in the embodied impacts (GWP) of the request.
* **Data center building embodied impacts are included.** The construction of the building and of the technical environment (electrical and cooling equipment) is allocated per kWh of IT electricity consumption (GWP).

These additions are modest in magnitude (a few percent of the impacts of a request) but they widen the scope of what EcoLogits accounts for, in line with life cycle assessment practices.

## An experimental training phase

The most discussed addition of Impact'IA is the **allocation of the training impacts** of a model to each request. Training impacts (final run, research and development experiments, dataset storage) are estimated top-down from the model size, its release date and the compute capacity of the provider, then shared between all the tokens the model is expected to generate over its lifetime.

EcoLogits now reports this estimation in a new `training` phase of the impacts, see the [LLM training methodology](../../methodology/llm_training.md).

```python
response = client.chat.completions.create(model="gpt-4.1-mini", messages=[...])

print(response.impacts.gwp.value)           # inference impacts (usage + embodied)
print(response.impacts.training.gwp.value)  # training impacts allocated to the request
```

We chose to report the training phase **separately** and to **not include it in the total impacts** of a request for two reasons. First, the estimation of the number of tokens a model generates during its lifetime is very uncertain (at least one order of magnitude), as providers do not publish these figures. Second, the training phase can only be estimated for models with a known release date and for providers with public compute capacity announcements, so including it in the totals would make models and providers harder to compare.

While reproducing the calculator, we also found a unit inconsistency in the Impact'IA spreadsheet that divides the estimated number of generated tokens by 1000, which explains most of the training share reported by the calculator (13% of the inference impacts in median). With consistent units, the training share estimated by EcoLogits is typically below 1% for the large providers. We reported it to the Impact'IA team and the methodology will keep evolving with better data on token volumes, such as the ones some providers started publishing.

## What we did not integrate

Impact'IA also models the impacts of the **retrieval-augmented generation embeddings** and of the **application platform** (the servers hosting the application around the model) of a project. The former is on our roadmap as an embeddings methodology. The latter depends on each application architecture and is out of the scope of an API-level library, it is better assessed with tools such as [e-footprint](https://e-footprint.boavizta.org/). Finally, Impact'IA reports an AWARE-weighted water use indicator (m³ eq) that differs from the water consumption footprint (L) reported by EcoLogits, so the embodied water values were not merged.

## Thanks

Many thanks to Xavier Verne and Thibaud Auzière (SNCF), Tatiana Fouquet, Benoit Durand and Feriel Sraieb (Wavestone), Rosendo Manas Faura and You Li (Resilio) for this work and for sharing it with the community. The Impact'IA methodology guide and calculator are published under the CC BY-NC-SA license.
