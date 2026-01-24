# Methodology

## Evaluation methodologies

The following methodologies are **currently available and implemented in EcoLogits**:

- [x] [LLM Inference](llm_inference.md)

Upcoming methodologies ([join us](https://genai-impact.org/contact) to help speed up our progress):

- [ ] Embeddings
- [ ] Image Generation
- [ ] Multi-Modal

## Methodological background

EcoLogits employs the **Life Cycle Assessment (LCA) methodology**, as defined by ISO 14044, to estimate the environmental impacts of requests made to generative AI inference services. This approach focuses on **multiple phases** of the lifecycle, specifically raw material extraction, manufacturing, transportation (denoted as embodied impacts), usage and end-of-life. Notably, we do not cover the end-of-life phase due to data limitations on e-waste recycling.

Our assessment considers **four key environmental criteria**:

- Global Warming Potential (GWP): Evaluates the impact on global warming in terms of CO2 equivalents.
- Abiotic Resource Depletion for Elements (ADPe): Assesses the consumption of raw minerals and metals, expressed in antimony equivalents.
- Primary Energy (PE): Calculates energy consumed from natural sources, expressed in megajoules.
- Water Consumption Factor (WCF): Water consumed by data centers and electricity generation power plants not returned to its original sources.

Using a **bottom-up modeling approach**, we assess and aggregate the environmental impacts of all individual service components within scope. This method differs from top-down approaches by allowing precise allocation of each resource's impact to the overall environmental footprint. The key advantage of bottom-up modeling is that our methodology can be customized for each provider that share information.

Our method computes **high-confidence approximation intervals**, providing a range of values within which we are confident enough that the true consumption lies.

The methodology is grounded in **transparency** and **reproducibility**, utilizing open market and technical data to ensure our results are reliable and verifiable.

## Scope of the methodology

Our methodology focuses on **assessing the environmental impacts of GenAI inference tasks**. That is why we exclude impacts from training, networking and end-used devices, we thoroughly evaluate the impacts associated with hosting and running the model inferences.

<figure markdown="span">
  ![Figure: Energy consumption per output token vs. number of active parameters ](../assets/methodology/llm/figure_boundaries.png)
  <figcaption>Boundaries of our impact assessment methodology.</figcaption>
</figure>

Because evaluating the environmental footprint of GenAI services is hard we make some **assumptions** to simplify the assessment. In the following section we will describe general assumptions that we use, if you want to learn more about the specifics look at the according methodology page.  


## Assumptions and limitations

Estimating the environmental impacts of generative AI services at inference can be really challenging because of the **lack of open data and transparent information** from the key players (AI/cloud providers, hardware manufacturers, environmental impact databases, etc.) In the [LLM inference](llm_inference.md) methodology we explain at high-level all the [**assumptions and limitations**](llm_inference.md#assumptions-and-limitations) of our bottom-up approach.

!!! info "Regarding the assumptions we make on **proprietary models**, we have a [dedicated section](proprietary_models.md) for increased transparency and explainability."


## Licenses and citations

All the methodologies are **licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt="">

Please ensure that you adhere to the license terms and properly cite EcoLogits and its authors when using this work. Each methodology has an associated paper with specific citation requirements.
