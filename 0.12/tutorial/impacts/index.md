# Environmental Impacts

Environmental impacts are reported for each request in the **[`ImpactsOutput`](<https://ecologits.ai/latest/reference/ecologits/tracers/utils/index.md#ecologits.tracers.utils.ImpactsOutput>)** that features multiple [impact criteria](<#impact-criteria>) such as [energy consumption](<#energy>) or the [global warming potential](<#global-warming-potential-gwp>) per phase ([usage](<#usage>) or [embodied](<#embodied>)) as well as the total impacts. It also contains potential [warnings and errors](<https://ecologits.ai/latest/tutorial/warnings_and_errors/index.md>) that can occur during the calculation.

> [!NOTE]
>
> **To learn more on how we estimate the environmental impacts and what are our hypotheses go to the [methodology](<https://ecologits.ai/latest/methodology/index.md>) section.**

## Impacts Output

The [`ImpactsOutput`](<https://ecologits.ai/latest/reference/ecologits/tracers/utils/index.md#ecologits.tracers.utils.ImpactsOutput>) is structured the following way:

```python
from ecologits.tracers.utils import ImpactsOutput
from ecologits.impacts.modeling import ADPe, Embodied, Energy, GWP, PE, Usage


ImpactsOutput(
    energy=Energy(),    # Total energy consumed (electricity)
    gwp=GWP(),          # Total global warming potential (or GHG emissions)
    adpe=ADPe(),        # Total abiotic resource depletion
    pe=PE(),            # Total energy consumed from primary sources
    usage=Usage( # (1)!
        energy=Energy(),
        gwp=GWP(),
        adpe=ADPe(),
        pe=PE(),
    ),
    embodied=Embodied( # (2)!
        gwp=GWP(),
        adpe=ADPe(),
        pe=PE(),
    ),
    warnings=None, # (3)!
    errors=None
)
```

1. Usage impacts for the electricity consumption impacts. Note that the energy is equal to the "total" energy impact.
2. Embodied impacts for resource extract, manufacturing and transportation of hardware components allocated to the request.
3. List of [`WarningMessage`](<https://ecologits.ai/latest/reference/ecologits/status_messages/index.md#ecologits.status_messages.WarningMessage>) and [`ErrorMessage`](<https://ecologits.ai/latest/reference/ecologits/status_messages/index.md#ecologits.status_messages.ErrorMessage>).

### Example of an Impact Value

The impact objects named [`Energy`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.Energy>), [`GWP`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.GWP>), [`ADPe`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.ADPe>) or [`PE`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.PE>) all share the following structure:

```python
from ecologits.impacts.modeling import BaseImpact
from ecologits.utils.range_value import RangeValue

class GWP(BaseImpact):  # (1)!
    type: str = "GWP"
    name: str = "Global Warming Potential"
    unit: str = "kgCO2eq"
    value: float | RangeValue = 0.34 
```

You can retrieve the GWP impact value from a request with the following:

```python
>>> response.impacts.gwp.value #  (1)!
0.34    # Total GHG emissions in kgCO2eq.

>>> response.impacts.usage.gwp.value 
0.23    # or for the usage phase only (2)
```

1. Assuming you have made an inference and get the response in an `response` object.
2. Impacts of the usage phase corresponds to the impacts of electricity consumption. Learn more about [usage phase in the next section](<#usage>).

### Example with a `RangeValue`

Impact values can also be represented as **intervals with the [`RangeValue`](<https://ecologits.ai/latest/reference/ecologits/utils/range_value/index.md#ecologits.utils.range_value.RangeValue>)** object. They are used to give an estimate range of possible values between a `min` and a `max`.

> [!NOTE]
>
> **About `RangeValue` intervals**
>
> This range of values corresponds a **high-confidence approximation interval**, within which we are confident enough that the true consumption lies. This interval is defined by several approximations, such as the model size (if unknown) and the statistical regressions that we perform for estimating quantities. For more information, see the [methodology](<https://ecologits.ai/latest/methodology/llm_inference/index.md>).

Example of an impact with a `RangeValue`:

```python
>>> response.impacts.gwp.value
RangeValue(min=0.16, max=0.48) # in kgCO2eq (1)
```

1. [`RangeValue`](<https://ecologits.ai/latest/reference/ecologits/utils/range_value/index.md#ecologits.utils.range_value.RangeValue>) are used to define intervals. It corresponds to the 95% confidence interval of our approximation.

## Impact Criteria

To evaluate the impact of human activities on the planet or on the climate we use criteria that usually focus on a specific issue such as **GHG emissions for global warming**, **water consumption and pollution** or the **depletion of natural resources**. We currently support three environmental impact criteria in addition with the direct energy consumption.

**Monitoring multiple criteria is useful to avoid pollution shifting**, which is defined as the transfer of pollution from one medium to another. It is a common pitfall to optimize only one criterion like GHG emissions (e.g. buying new hardware that is more energy efficient), that can lead to higher impacts on minerals and metals depletion for example (see ).

### Energy

The [`Energy`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.Energy>) criterion refers to the **direct electricity consumption of GPUs, server and other equipments from the data center**. As defined the energy criteria is not an environmental impact, but it is used to estimate other impacts in the [usage phase](<#usage>). This criterion is expressed in **kilowatt-hour (kWh)**.

> [!NOTE]
>
> **Energy model attributes**
>
> Attributes:
>
> - **`type`** (`str`) –
>
>   energy
> - **`name`** (`str`) –
>
>   Energy
> - **`value`** (`ValueOrRange`) –
>
>   Energy value
> - **`unit`** (`str`) –
>
>   Kilowatt-hour (kWh)

### Global Warming Potential (GWP)

The Global Warming Potential ([`GWP`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.GWP>)) criterion is an index **measuring how much heat is absorbed by greenhouse gases in the atmosphere compared to carbon dioxide**. This criterion is expressed in **kilogram of carbon dioxide equivalent (kgCO2eq)**.

Learn more:

> [!NOTE]
>
> **GWP model attributes**
>
> Attributes:
>
> - **`type`** (`str`) –
>
>   GWP
> - **`name`** (`str`) –
>
>   Global Warming Potential
> - **`value`** (`ValueOrRange`) –
>
>   GWP value
> - **`unit`** (`str`) –
>
>   Kilogram Carbon Dioxide Equivalent (kgCO2eq)

### Abiotic Depletion Potential for Elements (ADPe)

The Abiotic Depletion Potential – elements ([`ADPe`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.ADPe>)) criterion represents the **reduction of non-renewable and non-living (abiotic) resources such as metals and minerals**. This criterion is expressed in **kilogram of antimony equivalent (kgSbeq)**.

Learn more:

> [!NOTE]
>
> **ADPe model attributes**
>
> Attributes:
>
> - **`type`** (`str`) –
>
>   ADPe
> - **`name`** (`str`) –
>
>   Abiotic Depletion Potential (elements)
> - **`value`** (`ValueOrRange`) –
>
>   ADPe value
> - **`unit`** (`str`) –
>
>   Kilogram Antimony Equivalent (kgSbeq)

### Primary Energy (PE)

The Primary Energy ([`PE`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.PE>)) criterion represents the amount of **energy consumed from natural sources such as raw fuels and other forms of energy, including waste**. This criterion is expressed in **megajoule (MJ)**.

Learn more:

> [!NOTE]
>
> **PE model attributes**
>
> Attributes:
>
> - **`type`** (`str`) –
>
>   PE
> - **`name`** (`str`) –
>
>   Primary Energy
> - **`value`** (`ValueOrRange`) –
>
>   PE value
> - **`unit`** (`str`) –
>
>   Megajoule (MJ)

## Impact Phases

Inspired from the Life Cycle Assessment methodology we classify impacts is two phases (usage and embodied). The **usage phase** is about the environmental impacts related to the energy consumption while using an AI model. The **embodied phase** encompasses upstream impacts such as resource extraction, manufacturing, and transportation. We currently do not support the third phase which is end-of-life due to a lack of open research and transparency on that matter.

Learn more:

Another **pitfall in environmental impact assessment is to only look at the usage phase and ignore upstream and downstream impacts**. This can lead to higher overall impacts on the entire life cycle. If you replace old hardware by newer that is more energy efficient, you will get a reduction of impacts on the usage phase, but it will increase the upstream impacts as well.

### Usage

The [`Usage`](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.Usage>) phase accounts for the environmental impacts while using AI models. We report all criteria in addition to direct energy consumption for this phase.

Note that we use the worldwide average [electricity mix](<#electricity-mix>) impact factor by default.

> [!NOTE]
>
> **Usage model attributes**
>
> Attributes:
>
> - **`type`** (`str`) –
>
>   usage
> - **`name`** (`str`) –
>
>   Usage
> - **`energy`** (`Energy`) –
>
>   Energy consumption
> - **`gwp`** (`GWP`) –
>
>   Global Warming Potential (GWP) usage impact
> - **`adpe`** (`ADPe`) –
>
>   Abiotic Depletion Potential for Elements (ADPe) usage impact
> - **`pe`** (`PE`) –
>
>   Primary Energy (PE) usage impact
> - **`wcf`** (`WCF`) –
>
>   Water Consumption Footprint (WCF) usage impact

### Embodied

The [Embodied](<https://ecologits.ai/latest/reference/ecologits/impacts/modeling/index.md#ecologits.impacts.modeling.Embodied>) phase accounts for the upstream environmental impacts such as resource extraction, manufacturing and transportation allocated to the request. We report all criteria (excluding energy consumption) for this phase.

> [!NOTE]
>
> **Embodied model attributes**
>
> Attributes:
>
> - **`type`** (`str`) –
>
>   embodied
> - **`name`** (`str`) –
>
>   Embodied
> - **`gwp`** (`GWP`) –
>
>   Global Warming Potential (GWP) embodied impact
> - **`adpe`** (`ADPe`) –
>
>   Abiotic Depletion Potential for Elements (ADPe) embodied impact
> - **`pe`** (`PE`) –
>
>   Primary Energy (PE) embodied impact

## Impact Factors

We use impact factors to quantify environmental harm from human activities, measuring the ratio of greenhouse gases, resource consumption, and other criteria resulting from activities like energy consumption, industrial processes, transportation, waster management and more.

### Electricity Mix

When initializing `EcoLogits`, you can choose a specific electricity mix zone from the database.

Select a different electricity mix

```python
from ecologits import EcoLogits

# Select the electricity mix of France
EcoLogits.init(electricity_mix_zone="FRA")
```

By default, the `WOR` World electricity mix is used, whose values are:

| Impact criteria | Value | Unit |
| --- | --- | --- |
| GWP | \\(5.904e-1\\) | kgCO2eq / kWh |
| ADPe | \\(7.378e-7\\) | kgSbeq / kWh |
| PE | \\(9.988\\) | MJ / kWh |
