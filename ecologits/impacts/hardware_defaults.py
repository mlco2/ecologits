import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GPUEnergyRegression:
    """
    Coefficients of the exponential regression used to estimate GPU energy consumption per token.

    Attributes:
        alpha: Alpha coefficient of the energy regression.
        beta: Beta coefficient of the energy regression.
        gamma: Gamma coefficient of the energy regression.
    """
    alpha: float
    beta: float
    gamma: float


@dataclass
class LatencyRegression:
    """
    Coefficients of the linear regression used to estimate token generation latency.

    Attributes:
        alpha: Alpha coefficient of the latency regression.
        beta: Beta coefficient of the latency regression.
        gamma: Gamma coefficient of the latency regression.
    """
    alpha: float
    beta: float
    gamma: float


@dataclass
class GPUDefaults:
    """
    Default characteristics and embodied impacts of a single GPU.

    Attributes:
        memory_gb: Amount of memory available on a single GPU (in GB).
        embodied_gwp: Embodied Global Warming Potential of a single GPU (in kgCO2eq).
        embodied_adpe: Embodied Abiotic Depletion Potential of a single GPU (in kgSbeq).
        embodied_pe: Embodied Primary Energy of a single GPU (in MJ).
    """
    memory_gb: float
    embodied_gwp: float
    embodied_adpe: float
    embodied_pe: float


@dataclass
class ServerDefaults:
    """
    Default characteristics and embodied impacts of a server (excluding GPUs).

    Attributes:
        gpu_count: Number of GPUs available in the server.
        power_kw: Power consumption of the server (in kW).
        embodied_gwp: Embodied Global Warming Potential of the server (in kgCO2eq).
        embodied_adpe: Embodied Abiotic Depletion Potential of the server (in kgSbeq).
        embodied_pe: Embodied Primary Energy of the server (in MJ).
    """
    gpu_count: int
    power_kw: float
    embodied_gwp: float
    embodied_adpe: float
    embodied_pe: float


@dataclass
class HardwareDefaults:
    """
    Default hardware and serving assumptions used to estimate LLM inference impacts.

    Attributes:
        model_quantization_bits: Number of bits used to represent the model weights by default.
        gpu_energy_regression: Coefficients of the GPU energy regression.
        latency_regression: Coefficients of the token generation latency regression.
        gpu: Default GPU characteristics and embodied impacts.
        server: Default server characteristics and embodied impacts.
        hardware_lifespan_seconds: Default hardware amortization lifespan (in seconds).
        batch_size: Default number of requests handled concurrently by the server.
    """
    model_quantization_bits: int
    gpu_energy_regression: GPUEnergyRegression
    latency_regression: LatencyRegression
    gpu: GPUDefaults
    server: ServerDefaults
    hardware_lifespan_seconds: float
    batch_size: int

    @classmethod
    def from_json(cls, filepath: Optional[str] = None) -> "HardwareDefaults":
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..", "data", "hardware_defaults.json"
            )
        with open(filepath) as fd:
            data: dict[str, Any] = json.load(fd)

        return cls(
            model_quantization_bits=int(data["model_quantization_bits"]),
            gpu_energy_regression=GPUEnergyRegression(**data["gpu_energy_regression"]),
            latency_regression=LatencyRegression(**data["latency_regression"]),
            gpu=GPUDefaults(**data["gpu"]),
            server=ServerDefaults(**data["server"]),
            hardware_lifespan_seconds=float(data["hardware_lifespan_seconds"]),
            batch_size=int(data["batch_size"]),
        )


# Loaded once at import time (triggered by `EcoLogits.init()`) to avoid IO delay at runtime.
hardware_defaults = HardwareDefaults.from_json()
