from pydantic import BaseModel

from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Usage
from ecologits.status_messages import ErrorMessage, WarningMessage
from ecologits.utils.range_value import ValueOrRange


class LLMEstimationDetails(BaseModel):
    """
    Intermediate values used to estimate LLM inference impacts.

    Attributes:
        provider: Name of the provider.
        model_name: Name of the LLM.
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_total_parameter_count: Number of total parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds, when provided.
        tps: Number of generated tokens per second used for the estimate, when provided.
        ttft: Time-to-first-token latency used for the estimate, when provided.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone.
        datacenter_location: ISO 3166-1 alpha-3 code of the provider datacenter location.
        datacenter_pue: Power Usage Effectiveness of the datacenter.
        datacenter_wue: Water Usage Effectiveness of the datacenter.
        generation_latency: Token generation latency in seconds.
        gpu_required_count: Number of GPUs required to load the model.
        request_energy: Energy consumption of the request in kWh.
        request_usage_gwp: Usage Global Warming Potential in kgCO2eq.
        request_usage_adpe: Usage Abiotic Depletion Potential in kgSbeq.
        request_usage_pe: Usage Primary Energy in MJ.
        request_usage_wcf: Usage Water Consumption Footprint in L.
        request_embodied_gwp: Embodied Global Warming Potential in kgCO2eq.
        request_embodied_adpe: Embodied Abiotic Depletion Potential in kgSbeq.
        request_embodied_pe: Embodied Primary Energy in MJ.
    """
    provider: str
    model_name: str
    model_active_parameter_count: ValueOrRange
    model_total_parameter_count: ValueOrRange
    output_token_count: int
    request_latency: float | None = None
    tps: float | None = None
    ttft: float | None = None
    electricity_mix_zone: str
    datacenter_location: str | None = None
    datacenter_pue: ValueOrRange
    datacenter_wue: ValueOrRange
    generation_latency: ValueOrRange
    gpu_required_count: int
    request_energy: ValueOrRange
    request_usage_gwp: ValueOrRange
    request_usage_adpe: ValueOrRange
    request_usage_pe: ValueOrRange
    request_usage_wcf: ValueOrRange
    request_embodied_gwp: ValueOrRange
    request_embodied_adpe: ValueOrRange
    request_embodied_pe: ValueOrRange


class LLMEstimationResult(BaseModel):
    """
    LLM impacts estimation result.

    Attributes:
        energy: Total energy consumption.
        gwp: Total Global Warming Potential (GWP) impact.
        adpe: Total Abiotic Depletion Potential for Elements (ADPe) impact.
        pe: Total Primary Energy (PE) impact.
        wcf: Usage-only Water Consumption Footprint (WCF) impact.
        usage: Impacts for the usage phase.
        embodied: Impacts for the embodied phase.
        warnings: List of warnings.
        errors: List of errors.
        details: Intermediate estimation values.
    """
    energy: Energy | None = None
    gwp: GWP | None = None
    adpe: ADPe | None = None
    pe: PE | None = None
    wcf: WCF | None = None
    usage: Usage | None = None
    embodied: Embodied | None = None
    warnings: list[WarningMessage] | None = None
    errors: list[ErrorMessage] | None = None
    details: LLMEstimationDetails | None = None

    @property
    def has_warnings(self) -> bool:
        """
        Check whether the estimation result contains warnings.

        Returns:
            Whether warnings are present.
        """
        return isinstance(self.warnings, list) and len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        """
        Check whether the estimation result contains errors.

        Returns:
            Whether errors are present.
        """
        return isinstance(self.errors, list) and len(self.errors) > 0

    def add_warning(self, warning: WarningMessage) -> None:
        """
        Add a warning to the estimation result.

        Args:
            warning: Warning to add.
        """
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(warning)

    def add_errors(self, error: ErrorMessage) -> None:
        """
        Add an error to the estimation result.

        Args:
            error: Error to add.
        """
        if self.errors is None:
            self.errors = []
        self.errors.append(error)
