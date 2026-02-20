# Why use EcoLogits?

Generative AI significantly impacts our environment, consuming electricity and contributing to global greenhouse gas emissions. In 2020, the ICT sector accounted for 2.1% to 3.9% of global emissions, with projections suggesting an increase to 6%-8% by 2025 due to continued growth and adoption [Freitag et al., 2021](https://doi.org/10.1016/j.patter.2021.100340). The advent of GenAI technologies like ChatGPT has further exacerbated this trend, causing a sharp rise in energy, water, and hardware costs for major tech companies. [[0](https://www.theguardian.com/technology/article/2024/jul/02/google-ai-emissions), [1](https://www.bloomberg.com/news/articles/2024-05-15/microsoft-s-ai-investment-imperils-climate-goal-as-emissions-jump-30)].

## Which is bigger: training or inference impacts?

The field of Green AI focuses on evaluating the environmental impacts of AI models. While many studies have concentrated on training impacts [[2](https://github.com/samuelrince/awesome-green-ai?tab=readme-ov-file)], they often overlook other critical phases like data collection, storage and processing phases, research experiments and inference. For GenAI, the inference phase can significantly overshadow training impacts when models are deployed at scale [[3](https://www.seldon.io/the-environmental-impact-of-ml-inference)]. EcoLogits specifically addresses this gap by focusing on the inference impacts of GenAI.

## How to assess impacts properly?

EcoLogits employs state-of-the-art methodologies based on Life Cycle Assessment and open data to assess environmental impacts across multiple phases and criteria. This includes usage impacts from electricity consumption and embodied impacts from the production and transportation of hardware. Our multi-criteria approach also evaluates carbon emissions, abiotic resource depletion, and primary energy consumption, providing a comprehensive view that informs decisions like model selection, hardware upgrades and cloud deployments.

## How difficult is it?

Assessing environmental impacts can be challenging with external providers due to lack of control over the execution environment. Meaning you can easily estimate usage impact regarding energy consumption with [CodeCarbon](https://github.com/mlco2/codecarbon) and also embodied impacts with [BoaviztAPI](https://github.com/Boavizta/boaviztapi), but these tools become less relevant with external service providers. EcoLogits simplifies this by basing calculations on well-founded assumptions about hardware, model size, and operational practices, making it easier to estimate impacts accurately. For more details, see our [methodology section](methodology/index.md).

## Easy to use

EcoLogits integrates seamlessly into existing GenAI providers, allowing you to assess the environmental impact of each API request with minimal code adjustments:

```python
from ecologits import EcoLogits

EcoLogits.init()    

# Then, you can make request to any supported provider.
```

See the list of supported providers and more code snippets in the [tutorial section](tutorial/index.md).

## Have more questions?

Feel free to ask question in our [GitHub discussions](https://github.com/mlco2/ecologits/discussions/categories/q-a) forum!
