<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/logo_dark.png">
    <img alt="EcoLogits" src="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/logo_light.png">
  </picture>
</p>

🌱 **EcoLogits** tracks the energy consumption and environmental impacts of using generative AI models through APIs.

[![PyPI version](https://img.shields.io/pypi/v/ecologits?color=00bf63)](https://pypi.org/project/ecologits/)
[![PyPI downloads](https://static.pepy.tech/badge/ecologits/month)](https://pepy.tech/projects/ecologits)
[![Python version](https://img.shields.io/pypi/pyversions/ecologits)](https://pypi.org/project/ecologits/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VxrpJ5xuWZKQLsSN12kdqUqkppHRct3G?usp=sharing)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15601289.svg)](https://doi.org/10.5281/zenodo.15601289)

Read the full **documentation on [ecologits.ai](https://ecologits.ai/)**.

EcoLogits is part of the **[CodeCarbon](https://codecarbon.io/) non-profit**.


## ⚙️ Installation

```shell
pip install ecologits
```

For integration with a specific provider, use `pip install ecologits[openai]`. We are currently supporting the following providers: `anthropic`, `cohere`, `google-genai`, `huggingface-hub`, `mistralai` and `openai`. See the full [list of providers](https://ecologits.ai/latest/tutorial/providers/).


## 🚀 Usage

```python
from ecologits import EcoLogits
from openai import OpenAI

# Initialize EcoLogits
EcoLogits.init(providers=["openai"])

client = OpenAI(api_key="<OPENAI_API_KEY>")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Tell me a funny joke!"}
    ]
)

# Get estimated environmental impacts of the inference
print(f"Energy consumption: {response.impacts.energy.value.mean} kWh")
print(f"GHG emissions: {response.impacts.gwp.value.mean} kgCO2eq")
```

See package documentation on [EcoLogits](https://ecologits.ai/)


## 💚 Sponsors & benefactors

<a href="https://resilio-solutions.com/" target="_blank">
<img src="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/sponsors/resilio.png" alt="Resilio" height="100" width="250">
</a>

<a href="https://www.terra-cognita.ai/" target="_blank">
<img src="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/sponsors/terra_cognita.png" alt="Terra Cognita" height="100" width="250">
</a>

<a href="https://sopht.com/" target="_blank">
<img src="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/sponsors/sopht.png" alt="Sopht" height="100" width="250">
</a>

<a href="https://www.avanade.com/" target="_blank">
<img src="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/sponsors/avanade.png" alt="Avanade" height="100" width="250">
</a>

<a href="https://www.culture.gouv.fr/" target="_blank">
<img src="https://raw.githubusercontent.com/genai-impact/ecologits/main/docs/assets/sponsors/ministere_culture.png" alt="Ministère de la Culture" height="100" width="132">
</a>


## 💪 Contributing

To get started with setting up a development environment and making a contribution to EcoLogits, see [Contributing to EcoLogits](https://ecologits.ai/latest/contributing/).


## ⚖️ License

This project is licensed under the terms of the [Mozilla Public License Version 2.0 (MPL-2.0)](https://www.mozilla.org/en-US/MPL/2.0/).
