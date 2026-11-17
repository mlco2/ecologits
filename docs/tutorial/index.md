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

Available zones are listed in the [electricity_mixes.json :octicons-link-external-16:](https://github.com/mlco2/ecologits/blob/main/ecologits/data/electricity_mixes.json) file and are based on the [ISO 3166-1 alpha-3 :octicons-link-external-16:](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) convention with some extras like `WOR` for World or `EEE` for Europe. 

Electricity mixes for each geographic zone are based on available data for the supported impact factors. When country-specific data is missing for ADPe, PE, or WUE, EcoLogits can use the world default value and report a warning in the request impacts.

!!! info "Default electricity mix zone is `WOR` for World."

```python title="Select a different electricity mix"
from ecologits import EcoLogits

# Select the electricity mix of France
EcoLogits.init(providers=[...], electricity_mix_zone="FRA")
```


## Tracking impacts in a block

If you only want to measure part of your code, use `EcoLogits.track_impacts()` instead of initializing EcoLogits for the whole process. Only the requests made inside the block are tracked, and the block gives you the total impacts of these requests.

```python title="Track the impacts of a block of code"
from ecologits import EcoLogits
from openai import OpenAI

client = OpenAI()

with EcoLogits.track_impacts() as tracker:
    client.chat.completions.create(...)
    client.chat.completions.create(...)

print(f"Energy consumption: {tracker.impacts.energy.value} kWh")
print(f"GHG emissions: {tracker.impacts.gwp.value} kgCO2eq")

# Impacts of each request
for impacts in tracker.calls:
    print(impacts.energy.value)
```

It also works in async code with `async with EcoLogits.track_impacts() as tracker:`.

The block only covers the current thread or async task, so other work running at the same time in the same process is not tracked. Requests made outside the block are left untouched, unless their provider was enabled with `EcoLogits.init()`. In that case they keep their impacts as usual, and only the requests made inside the block are counted by the tracker.

!!! info "Tracking in a block is currently supported for the OpenAI and Anthropic providers."
