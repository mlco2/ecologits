# Tutorial

The :seedling: **EcoLogits** library tracks the energy consumption and environmental impacts of generative AI models accessed through APIs and their official client libraries. 

It achieves this by **patching the Python client libraries**, ensuring that each API request is wrapped with an impact calculation function. This function computes the **environmental impact based on several request features**, such as the **chosen model**, the **number of tokens generated**, and the **request's latency**. The resulting data is then encapsulated in an [`ImpactsOutput`][tracers.utils.ImpactsOutput] object, which is added to the response, containing the environmental impacts for a specific request.


<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    Install [`ecologits`](#) with [`pip`](#) and get up
    and running in minutes.

    [:octicons-arrow-right-24: Getting started](../index.md)

-   :material-earth:{ .lg .middle } __Environmental impacts__

    ---

    Understand what environmental impacts and phases are reported.  

    [:octicons-arrow-right-24: Tutorial](impacts.md)

-   :fontawesome-regular-handshake:{ .lg .middle } __Supported providers__

    ---

    List of providers and tutorials on how to make requests.

    [:octicons-arrow-right-24: Providers](providers.md)

-   :material-test-tube:{ .lg .middle } __Methodology__

    ---
    
    Understand how we estimate environmental impacts.
    
    [:octicons-arrow-right-24: Methodology](../methodology/index.md)

</div>


## Initialization of EcoLogits

To use EcoLogits in your projects, you will need to initialize the client tracers that are used internally to intercept and enrich responses.

```python
from ecologits import EcoLogits

# Example for OpenAI and Anthropic
EcoLogits.init(providers=["openai", "anthropic"])
```

??? warning "Disabling a provider at runtime is not supported"

    **It is currently not possible to dynamically activate and deactivate a provider at runtime.** Each time that `EcoLogits` is re-initialized with another providers, the latter will be added to the list of already initialized providers. If you think that un-initializing a provider could be necessary for your use case, please [open an issue :octicons-link-external-16:](https://github.com/mlco2/ecologits/issues/new/choose)."


### Configure electricity mix

You can change the [electricity mix :octicons-link-external-16:](https://ourworldindata.org/electricity-mix) for server-side computation depending on a specific location. EcoLogits will automatically change the default impact factors for electricity consumption according to the selected zone. 

Available zones are listed in the [electricity_mixes.csv :octicons-link-external-16:](https://github.com/mlco2/ecologits/blob/main/ecologits/data/electricity_mixes.csv) file and are based on the [ISO 3166-1 alpha-3 :octicons-link-external-16:](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) convention with some extras like `WOR` for World or `EEE` for Europe. 

Electricity mixes for each geographic zone are sourced from the [ADEME Base Empreinte® :octicons-link-external-16:](https://base-empreinte.ademe.fr/) database and are based on yearly averages.

!!! info "Default electricity mix zone is `WOR` for World."

```python title="Select a different electricity mix"
from ecologits import EcoLogits

# Select the electricity mix of France
EcoLogits.init(providers=[...], electricity_mix_zone="FRA")
```
