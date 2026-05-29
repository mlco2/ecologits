# Environmental Impacts of Video Generation

## Introduction

The environmental impacts of a video generation request, $I_{\text{request}}$, can be divided into two components: the usage impacts, $I_{\text{request}}^{\text{u}}$, which account for energy consumption, and the embodied impacts, $I_{\text{request}}^{\text{e}}$, which account for resource extraction, hardware manufacturing, and transportation:

$$
\begin{equation*}
\begin{split}
I_{\text{request}} &= I_{\text{request}}^{\text{u}} + I_{\text{request}}^{\text{e}} \\
&= E_{\text{request}} \times F_{\text{em}} + \frac{\Delta T}{\Delta L} \times I_{\text{server}}^{\text{e}},
\end{split}
\end{equation*}
$$

where $E_{\text{request}}$ is the electricity consumption of the hardware used for the request, $F_{\text{em}}$ is the impact factor of the electricity mix, $I_{\text{server}}^{\text{e}}$ is the embodied impact of the server and its accelerators, $\Delta T$ is the generation latency, and $\Delta L$ is the hardware lifetime.


## Usage impacts

To assess the usage impacts of a video generation request, we first estimate the generation latency of the request and the power draw of the hardware configuration used to serve it. We then account for data center overhead through the Power Usage Effectiveness (PUE) and transform electricity consumption into environmental impacts using country-level electricity-mix factors.

### Modeling the generation latency

[//]: # (TODO: Add paper link)

Our latency estimation follows the approach from [Jegham et al. (2026)](#). The main idea is that video generation is mostly compute-bound, so under fixed hardware the generation time is a good proxy for the computational work of the request. In practice, we use model-specific regressions fitted from observed generation latencies, directly derived from the paper.

We denote:

- $W$: video width in pixels,
- $H$: video height in pixels,
- $F$: number of generated frames,
- $T = \frac{W \times H \times F}{1000}$: spatiotemporal volume.

As in the paper, the frame count is derived from the target duration $D$ in seconds using a fixed frame rate of 24 fps:

$$
F(D) = \lfloor 24 \times D + 1 \rfloor.
$$

We then reuse the regression from the paper to estimate the generation latency as a function of the resolution and frame count:

$$
\Delta T_{\text{est}}(W, H, F) =
\left[
n \times F \times (W \times H)^2
+ m \times T
+ m_1 \times F
+ m_2 \times (W \times H)^2
+ n_1 \times F^2
+ n_2 \times T^2
+ g
\right] \times \omega_{\text{no-audio}},
$$

where $n$, $m$, $n_1$, $m_1$, $n_2$, $m_2$, and $g$ are model-specific regression coefficients, and $\omega_{\text{no-audio}}$ is a multiplicative factor applied when a model supports audio but the request is video-only.

This regression is used to capture the main components of the generation latency:

- the linear terms account for costs that scale approximately linearly with video size,
- the quadratic terms account for attention-like costs that grow faster with resolution or length,
- the constant term captures fixed per-request overheads.

When an observed request latency $\Delta T_{\text{request}}$ is available, we cap the estimate with:

$$
\Delta T = \min \left\{ \Delta T_{\text{est}}, \Delta T_{\text{request}} \right\}.
$$

??? note "On the no-audio factor"

    In the current implementation, regressions are stored with a model-specific `non_audio_weight`.

    - If a request is generated **with audio**, we force $\omega_{\text{no-audio}} = 1$.
    - If a request is generated **without audio**, we use the calibrated model value, typically lower than 1, depending on the model.

    This assumes that removing audio reduces generation latency by a constant multiplicative factor for a given model.

### Modeling server and accelerator energy consumption

Each video model is mapped to a fixed hardware configuration. The supported configurations currently include NVIDIA DGX GPU servers and TPU servers.

[//]: # (TODO: Add paper link)

We denote by $P_{\text{server}}$ the electrical power of the full machine, including the base server and all installed accelerators. The power consumptions for each server is derived from [Jegham et al. (2026)](#). The server energy consumed during the request is:

$$
E_{\text{server}} = \frac{\Delta T}{3600} \times P_{\text{server}},
$$

with $\Delta T$ in seconds and $P_{\text{server}}$ in kW, giving $E_{\text{server}}$ in kWh.

??? info "On power intervals"

    We use a range of power values between the 2.5th and 97.5th percentiles for each hardware configuration. This interval is propagated through the energy and impact equations, which is why EcoLogits returns interval estimates rather than a single deterministic value.

There is no additional allocation term for batch size or for a fraction of installed accelerators. In other words, the current methodology attributes the whole machine power to the request once the hardware configuration has been selected.

### Modeling request energy consumption

To account for data center overheads such as cooling and electrical distribution losses, we multiply the direct server energy by the Power Usage Effectiveness (PUE):

$$
E_{\text{request}} = \text{PUE} \times E_{\text{server}}.
$$

### Modeling request usage environmental impacts

To assess the environmental impacts of the request for the usage phase, we multiply the estimated electricity consumption by the impact factor of the electricity mix, $F_{\text{em}}$, specific to the target country and time. We use data from the [Our World in Data](https://ourworldindata.org/electricity-mix) for GWP impact and from the [ADEME Base Empreinte®](https://base-empreinte.ademe.fr/) for ADPe and PE impacts. It gives us:

$$
I^\text{u}_{\text{request}} = E_{\text{request}} \times F_{\text{em}}.
$$

#### Modeling request water consumption footprint for usage phase

To assess the Water Consumption Footprint (WCF) for the usage phase we use the modeling from [Li et al. (2025)](https://arxiv.org/abs/2304.03271). It uses the Water Usage Effectiveness (WUE) of both the data center $\text{WUE}_\text{on-site}$ and of the local electricity mix $\text{WUE}_\text{off-site}$. On-site data is assessed for each provider individually, whereas off-site data is averaged from each country according to the [World Resource Institute methodology](https://www.wri.org/research/guidance-calculating-water-use-embedded-purchased-electricity). It gives us:

$$
\text{WCF}^{\text{u}}_{\text{request}} =
E_{\text{server}} \times
\left[
\text{WUE}_{\text{on-site}} + \text{PUE} \times \text{WUE}_{\text{off-site}}
\right].
$$


## Embodied impacts

To determine the embodied impacts of a video generation request, we need to estimate the **hardware configuration** used to host the model and its lifetime. Embodied impacts account for resource extraction (e.g., minerals and metals), manufacturing, and transportation of the hardware.

### Modeling server embodied impacts

To estimate the embodied impacts of IT hardware, we use the [BoaviztAPI](https://github.com/Boavizta/boaviztapi) tool from the non-profit organization [Boavizta](https://boavizta.org/en/). This API embeds a bottom-up multicriteria environmental impact estimation engine for IT equipment and services. We focus here on estimating the embodied impacts of a server and an accelerator.

#### Server embodied impacts without accelerators

To assess the embodied impacts of the serving systems used in the methodology, we use reference values for the base server, without accelerators, denoted as $I^{\text{e}}_{\text{server} \backslash \text{acc}}$.

The embodied environmental impacts of the base server are:

|                   | DGX H800 | DGX H200 | TPU v6e |
|-------------------|----------|----------|---------|
| GWP (kgCO2eq)     | $6000$   | $6000$   | $3550$  |
| ADPe (kgSbeq)     | $0.42$   | $0.42$   | $0.42$  |
| PE (MJ)           | $8000$   | $8000$   | $8000$  |

??? note "On TPU server impacts"

    For TPU systems, the base-server GWP value comes from [Schneider et al. (2025)](https://arxiv.org/abs/2502.01671). Since this source only reports GWP, we approximate ADPe and PE using the values of the equivalent DGX server configuration. We also add storage impacts to the base TPU server, as they are absent from the original paper.

#### Accelerator embodied impacts

We use Boavizta-based values to estimate the embodied impacts of a single accelerator, denoted as $I^{\text{e}}_{\text{acc}}$.

|                   | H800      | H200      | TPU v6e   |
|-------------------|-----------|-----------|-----------|
| GWP (kgCO2eq)     | $273$     | $364$     | $323$     |
| ADPe (kgSbeq)     | $0.00895$ | $0.00895$ | $0.00895$ |
| PE (MJ)           | $3721$    | $4906$    | $4906$    |

??? info "On accelerator embodied values"

    In the current methodology, H800 systems are used for Chinese and more broadly Asia-based providers. The H800 embodied values reuse the H100 embodied values.

??? note "On TPU accelerator impacts"

    For TPU systems, the accelerator GWP value comes from [Schneider et al. (2025)](https://arxiv.org/abs/2502.01671). Since this source only reports GWP, we approximate ADPe and PE using the values of the equivalent DGX accelerator configuration.

#### Complete server embodied impacts

For the supported hardware configurations, we consider systems with $N_{\text{acc}} = 8$ accelerators. The complete embodied impacts of the serving system are then:

$$
I^{\text{e}}_{\text{server}} =
I^{\text{e}}_{\text{server} \backslash \text{acc}} + N_{\text{acc}} \times I^{\text{e}}_{\text{acc}}.
$$

### Modeling request embodied environmental impacts

To allocate the server embodied impacts to one request, we use an allocation based on the hardware utilization factor, $\frac{\Delta T}{\Delta L}$. In this case, $\Delta L$ represents the lifetime of the server and accelerator, which we fix at 3 years:

$$
I^{\text{e}}_{\text{request}} =
\frac{\Delta T}{\Delta L} \times I^{\text{e}}_{\text{server}}.
$$

!!! warning "Water consumption (WCF) is not modeled for the embodied phase due to a lack of data."


## Supplemental material

### Data center configuration

We use provider-level assumptions for the default **deployment location**, **PUE**, and **WUE** used in video generation impact estimates. These values are used when no request-specific data center location, PUE, or WUE is provided.

| AI Provider | Location | PUE         | WUE          |
|-------------|----------|-------------|--------------|
| ByteDance   | SGP      | 1.20        | 0.50         |
| Google      | USA      | 1.09        | 0.999        |
| Kling AI    | SGP      | 1.20        | 0.50         |
| OpenAI      | USA      | 1.20        | 0.569        |
| Runway      | USA      | 1.09 - 1.14 | 0.13 - 0.999 |


## References

[//]: # (TODO: add paper link)

- [Jegham et al. (2026)](#) for video generation latency and power consumption.
- [Schneider et al. (2025)](https://arxiv.org/abs/2502.01671) for TPU embodied GWP values.
- [Boavizta](https://boavizta.org/) and [BoaviztAPI](https://github.com/Boavizta/boaviztapi) for embodied impacts of the modeled hardware configurations.
- [Our World in Data](https://ourworldindata.org/), [ADEME Base Empreinte®](https://base-empreinte.ademe.fr/), and [World Resource Institute](https://www.wri.org/) for electricity-mix and water-use factors.
