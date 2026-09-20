# Environmental Impacts of LLM Training

!!! warning "Experimental methodology"

    The training phase is reported **separately** from the usage and embodied phases of the inference and is **not included in the total impacts** of a request. The estimation relies on a top-down approach with very few public data points and should be considered as an **order of magnitude** with an uncertainty of at least one order of magnitude.

## Introduction

Training a model (data collection, experiments, final training run) happens once, before the model is used, but it is a prerequisite to every inference. Following the [Impact'IA](https://github.com/SNCF-ImpactIA/ImpactIA) methodology developed by SNCF, Wavestone and Resilio, EcoLogits estimates the impacts of the training of a model and allocates a share of them to each request, in proportion to the number of generated tokens.

The training impacts allocated to a request, $I_{\text{request}}^{\text{t}}$, are computed as:

$$
I_{\text{request}}^{\text{t}} = \frac{\#T_{\text{out}}}{\#T_{\text{model}}} \times \left( I_{\text{train}} + I_{\text{R\&D}} + I_{\text{storage}} \right),
$$

where $\#T_{\text{out}}$ is the number of output tokens of the request, $\#T_{\text{model}}$ is the total number of tokens the model is expected to generate during its lifetime and $I_{\text{train}}$, $I_{\text{R\&D}}$ and $I_{\text{storage}}$ are the impacts of the final training run, of the research and development experiments and of the training data storage respectively.

## Scope

The following stages of the model life cycle are modeled, based on the [macro-estimations of Impact'IA](https://github.com/SNCF-ImpactIA/ImpactIA/tree/main/doc):

| Stage        | Modeled | Comment                                                                        |
|--------------|---------|--------------------------------------------------------------------------------|
| Final training run | :material-check: | Compute estimated from the model size and release date.                    |
| Experiments and test runs (R&D) | :material-check: | Estimated as a multiple of the final training run.                   |
| Training data storage | :material-check: | Electricity consumption of the hard disk drives during the training run. |
| Data annotation | :material-close: | Negligible (below 0.1% of the inference impacts).                          |
| Workstations of the model developers | :material-close: | Negligible (below 0.1% of the inference impacts).            |

## Final training run

### Compute of the training run

The compute of the final training run, $C_{\text{train}}$ (in FLOPs), is estimated with a regression made by Impact'IA on the [Epoch AI](https://epoch.ai/data/notable-ai-models) notable models dataset. It depends on the total number of parameters of the model, $P_{\text{total}}$, and on the release date of the model, expressed as a number of days since January 1st, 2020, $D_{\text{release}}$. At equal size, recent models are trained with more compute:

$$
C_{\text{train}} = 10^{\alpha D_{\text{release}} + \beta} \times P_{\text{total}}^{\gamma},
$$

with $\alpha = 6 \times 10^{-4}$, $\beta = 17.151$ and $\gamma = 0.541$.

??? example "Example"

    For a 76B parameters model released on April 14, 2025 ($D_{\text{release}} = 1930$), the estimated compute is $C_{\text{train}} \approx 1.6 \times 10^{24}$ FLOPs.

### Energy consumption of the training run

The compute is converted into an electricity consumption with the compute efficiency of the training hardware, $\eta$ (in FLOPS per watt). We use $\eta = 1.4 \times 10^{12}$ FLOPS/W, the peak dense BF16 efficiency of an NVIDIA H100 GPU, to stay consistent with the hardware assumptions of the [inference methodology](llm_inference.md). To cover the same technical scope as the inference phase (GPUs, servers and network equipment), the GPU energy is multiplied by an infrastructure overhead ratio, $R_{\text{infra}}$, and by the PUE of the data center:

$$
E_{\text{train}} = \frac{C_{\text{train}}}{\eta \times 3.6 \times 10^{6}} \times R_{\text{infra}} \times \text{PUE}, \quad R_{\text{infra}} = \frac{\#\text{GPU}_{\text{installed}} \times W_{\text{GPU}} + W_{\text{server} \backslash \text{GPU}} + W_{\text{network}}}{\#\text{GPU}_{\text{installed}} \times W_{\text{GPU}}}.
$$

With the [server configuration of the inference methodology](llm_inference.md#modeling-server-energy-consumption) (8 GPUs of 700 W, 1.2 kW server and 161 W of network equipment), $R_{\text{infra}} \approx 1.24$.

## Research and development experiments

The experiments, test runs and ablations that precede the final training run are poorly documented. Impact'IA reviewed the available data (OpenAI 2024 compute spending, MiniMax and Z.ai technical reports) which suggest that the R&D compute is between 1 and 19 times the final training run. We use the conservative median value of 4:

$$
E_{\text{R\&D}} = 4 \times E_{\text{train}}.
$$

## Training data storage

The number of training tokens is derived from the compute with the approximation of [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361), $C_{\text{train}} = 6 \times P_{\text{total}} \times \#T_{\text{train}}$. Each token is assumed to be stored as 4 characters of 32 bits, on 30 TB hard disk drives of 9.5 W used at 20%, during a 100-day training run. The resulting electricity consumption, $E_{\text{storage}}$, is negligible compared to the training run (below 0.01%). The embodied impacts of the hard disk drives are not modeled for the same reason.

## Allocation to the request

### Lifetime number of generated tokens

The training impacts are shared between all the tokens the model generates during its lifetime. Since providers do not disclose these volumes, Impact'IA proposes a top-down estimation from the **AI compute capacity of the provider**, $W_{\text{provider}}$, gathered from public announcements:

$$
\#T_{\text{model}} = \frac{W_{\text{provider}} \times r_{\text{inference}}}{N_{\text{active}}} \times \frac{\eta \times \epsilon}{\text{PUE}} \times \frac{\Delta L_{\text{model}}}{2 \times P_{\text{active}}},
$$

where:

* $r_{\text{inference}} = 0.8$ is the share of the compute capacity dedicated to inference,
* $N_{\text{active}}$ is the number of active models of the provider, i.e. models released less than 2 years ago (dated snapshots and `latest` versions of the same model are counted once),
* $\epsilon = 0.5 \times 0.5 \times 0.7 = 0.175$ is an efficiency factor accounting for the GPU utilization, the memory-bound limitation of the decoding phase and the sharing of the GPUs,
* $\Delta L_{\text{model}} = 2$ years is the lifetime of the model,
* $2 \times P_{\text{active}}$ is the number of FLOPs required to generate one token.

The AI compute capacities currently used are:

| AI Provider | Compute capacity | Source                                                                                                                              |
|-------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Anthropic   | 1.4 GW           | [wccftech.com](https://wccftech.com/openai-sets-goal-to-scale-up-ai-compute-capacity-to-a-whopping-30gw-by-2030/)                    |
| Google      | 9 GW             | [cnbc.com](https://www.cnbc.com/2025/11/21/google-must-double-ai-serving-capacity-every-6-months-to-meet-demand.html)               |
| Mistral AI  | 200 MW           | [cnbc.com](https://www.cnbc.com/2026/03/30/mistral-ai-paris-data-center-cluster-debt-financing.html)                                |
| OpenAI      | 2 GW             | [finance.yahoo.com](https://finance.yahoo.com/sectors/technology/articles/coreweave-signs-multi-deal-anthropic-162247263.html)      |

The training phase is not reported for the other providers (Cohere, Hugging Face Hub) nor for models without a known release date, a [`training-not-modeled`](../tutorial/warnings_and_errors.md#training-not-modeled) warning is reported instead.

### Environmental impacts

The training location of the models being unknown, the usage impacts are computed with the **world average electricity mix**, $F_{\text{em}}^{\text{WOR}}$, and the PUE and WUE of the provider:

$$
I_{\text{request}}^{\text{t,u}} = \frac{\#T_{\text{out}}}{\#T_{\text{model}}} \times \left( E_{\text{train}} + E_{\text{R\&D}} + E_{\text{storage}} \right) \times F_{\text{em}}^{\text{WOR}}.
$$

The embodied impacts of the training infrastructure are estimated by applying the **embodied impact intensity of the inference phase** (embodied impacts per kWh of IT electricity, for each criteria) to the IT electricity consumption of the training allocated to the request. This assumes that the training and inference infrastructures have the same hardware and lifetime.

The water consumption footprint is computed as for the [inference phase](llm_inference.md#modeling-request-water-consumption-footprint-for-usage-phase) with the world average off-site WUE.

## Assumptions and limitations

* The compute regression is fitted on notable models and extrapolated to all models, including small ones for which it can overestimate the training compute.
* The compute efficiency $\eta$ is a peak theoretical value. Real-world training runs reach 30% to 50% of the peak (model FLOPs utilization), which underestimates the training energy by a factor of 2 to 3.
* The lifetime number of generated tokens is the most uncertain parameter: the compute capacities are announced targets rather than actual serving capacities, the share dedicated to inference and the utilization factors are assumptions, and models are not equally used. We estimate that the resulting per-token allocation is uncertain by **at least one order of magnitude**.
* **Difference with the Impact'IA calculator**: the Impact'IA spreadsheet expresses compute capacities in kW but uses them as W when converting them into FLOPS, which divides the lifetime number of tokens by 1000 and thus multiplies the training impacts per token by 1000. EcoLogits uses consistent units. As a consequence, the training share reported by EcoLogits (typically below 1% of the inference impacts for large providers) is much lower than the one reported by the Impact'IA calculator (13% in median). This has been reported to the Impact'IA team and the parameters of the methodology may be recalibrated in the future.
* The embodied impacts of the training infrastructure are approximated with the intensity of the inference infrastructure.
* Mixture of experts models use the mean number of active parameters to compute the number of generated tokens.

All the parameters of the training phase can be overridden when calling [`compute_llm_impacts`][impacts.llm.compute_llm_impacts], including the lifetime number of tokens (`model_lifetime_output_token_count`) if you have better data for your use case.

## References

- [Impact'IA](https://github.com/SNCF-ImpactIA/ImpactIA) methodology guide (SNCF, Wavestone, Resilio, 2026), from which the training, R&D, storage and allocation models are adapted.
- [Epoch AI](https://epoch.ai/data/notable-ai-models) notable AI models dataset used to fit the training compute regression.
- [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) for the relation between compute, parameters and training tokens.
- [Mistral AI (2025)](https://mistral.ai/news/our-contribution-to-a-global-environmental-standard-for-ai), [Epoch AI (2025)](https://epoch.ai/data-insights/openai-compute-spend) and [Aubakirova et al. (2025)](https://openrouter.ai/state-of-ai) for the orders of magnitude of training versus inference impacts.

## :material-scale-balance: License and credits

The Impact'IA methodology guide and calculator are published by SNCF, Wavestone and Resilio under the [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. This page is an original description of the adaptation of their methodology within EcoLogits and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) as the rest of the EcoLogits methodology. Please cite both Impact'IA and EcoLogits when using the training phase estimations.
