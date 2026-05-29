import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ecologits.status_messages import WarningMessage


@dataclass
class ElectricityMix:
    """
    Electricity mix of a country

    Attributes:
        zone: ISO 3166-1 alpha-3 code of the electricity mix zone
        adpe: Abiotic Depletion Potential of the mix (in kgSbeq / kWh)
        pe: Primary Energy of the mix (in MJ / kWh)
        gwp: Global Warming Potential of the mix (in kgCO2eq / kWh)
        wue: Water Usage Effectiveness of the mix (in L / kWh)
        warnings: Warnings linked to the electricity mix
    """
    zone: str
    adpe: float
    pe: float
    gwp: float
    wue: float
    warnings: list[WarningMessage] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ElectricityMix":
        warnings = []
        if "warnings" in data and data["warnings"] is not None:
            warnings = [WarningMessage.from_code(code) for code in data["warnings"]]

        return cls(
            zone=data["name"],
            adpe=float(data["adpe"]),
            pe=float(data["pe"]),
            gwp=float(data["gwp"]),
            wue=float(data["wue"]),
            warnings=warnings,
        )


class ElectricityMixRepository:
    """
    Repository of electricity mixes.
    """

    def __init__(self, electricity_mixes: list[ElectricityMix]) -> None:
        self.__electricity_mixes: dict[str, ElectricityMix] = {}
        for electricity_mix in electricity_mixes:
            if electricity_mix.zone in self.__electricity_mixes:
                raise ValueError(f"duplicated electricity mix with: {electricity_mix.zone}")
            self.__electricity_mixes[electricity_mix.zone] = electricity_mix

    def find_electricity_mix(self, zone: str) -> Optional[ElectricityMix]:
        return self.__electricity_mixes.get(zone)

    def list_electricity_mixes(self) -> list[ElectricityMix]:
        return list(self.__electricity_mixes.values())

    @classmethod
    def from_json(cls, filepath: Optional[str] = None) -> "ElectricityMixRepository":
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "data", "electricity_mixes.json"
            )
        with open(filepath) as fd:
            data = json.load(fd)

        electricity_mixes = []
        if "electricity_mixes" in data and data["electricity_mixes"] is not None:
            for electricity_mix in data["electricity_mixes"]:
                electricity_mixes.append(ElectricityMix.from_json(electricity_mix))

        if len(electricity_mixes) == 0:
            raise ValueError("Cannot initialize on an empty electricity mix repository.")
        return cls(electricity_mixes)


electricity_mixes = ElectricityMixRepository.from_json()
