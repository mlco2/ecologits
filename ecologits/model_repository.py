import json
import os
import re
from datetime import date
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel

from ecologits.status_messages import WarningMessage
from ecologits.utils.range_value import ValueOrRange

ACTIVE_MODEL_PERIOD_DAYS = 2 * 365

_MODEL_FAMILY_SUFFIXES = re.compile(
    r"(-latest|-\d{4}-\d{2}-\d{2}|-\d{8}|-\d{4}(-rc\d+)?|-\d{3})$"
)


def model_family(model_name: str) -> str:
    """
    Strip version suffixes (dates, snapshots, `latest`) from a model name to identify its family.

    Examples:
        `gpt-4o-2024-08-06` -> `gpt-4o`, `claude-opus-4-1-20250805` -> `claude-opus-4-1`,
        `mistral-medium-2508` -> `mistral-medium`, `codestral-latest` -> `codestral`.

    Args:
        model_name: Name of the model.

    Returns:
        The model family name.
    """
    previous = None
    while previous != model_name:
        previous = model_name
        model_name = _MODEL_FAMILY_SUFFIXES.sub("", model_name)
    return model_name


class Providers(Enum):
    anthropic = "anthropic"
    mistralai = "mistralai"
    openai = "openai"
    huggingface_hub = "huggingface_hub"
    cohere = "cohere"
    google_genai = "google_genai"


class ArchitectureTypes(Enum):
    DENSE = "dense"
    MOE = "moe"


class ParametersMoE(BaseModel):
    total: ValueOrRange
    active: ValueOrRange


class Architecture(BaseModel):
    type: ArchitectureTypes
    parameters: Union[ValueOrRange, ParametersMoE]


class Deployment(BaseModel):
    tps: float | None = None
    ttft: float | None = None


class Alias(BaseModel):
    provider: Providers
    name: str
    alias: str


class Model(BaseModel):
    """
    LLM Model

    Attributes:
        provider: Provider of the model (e.g. "OpenAI")
        name: Name of the model (e.g. "gpt-4o-mini")
        architecture: Architecture type (dense or mixture-of-experts)
        warnings: Warnings linked to the model (e.g. "model-arch-not-released" or "model-arch-multimodal")
        sources: Source of the model information (website link)
        deployment: Deployment information (tps, ttft)
        release_date: Release date of the model (used to estimate the training impacts)
    """

    provider: Providers
    name: str
    architecture: Architecture
    warnings: list[WarningMessage] = []
    sources: list[str] = []
    deployment: Deployment | None = None
    release_date: date | None = None

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Model":
        warnings = []
        sources = []
        if "warnings" in data and data["warnings"] is not None:
            warnings = [WarningMessage.from_code(code) for code in data["warnings"]]
        if "source" in data and data["sources"] is not None:
            sources = data["sources"]
        deployment = None
        if "deployment" in data and data["deployment"] is not None:
            deployment = Deployment.model_validate(data["deployment"])
        release_date = None
        if "release_date" in data and data["release_date"] is not None:
            release_date = date.fromisoformat(data["release_date"])
        return cls(
            provider=Providers(data["provider"]),
            name=data["name"],
            architecture=Architecture.model_validate(data["architecture"]),
            warnings=warnings,
            sources=sources,
            deployment=deployment,
            release_date=release_date,
        )


class ModelRepository:
    """
    Repository of models
    """

    def __init__(self, models: Optional[list[Model]] = None, aliases: Optional[list[Alias]] = None) -> None:
        self.__models: dict[tuple[str, str], Model] = {}
        self.__aliases: set[str] = set()
        if models is not None:
            for m in models:
                key = m.provider.value, m.name
                if key in self.__models:
                    raise ValueError(f"duplicated models with: {key}")
                self.__models[key] = m

        if aliases is not None:
            for a in aliases:
                model_key = a.provider.value, a.alias
                if model_key not in self.__models:
                    raise ValueError(f"model alias not found: {model_key}")
                alias_key = a.provider.value, a.name
                model = self.__models[model_key].model_copy()
                model.name = a.name
                self.__models[alias_key] = model
                self.__aliases.add(a.name)

    def add_model(self, data: dict[str, Any]) -> None:
        model = Model.from_json(data)
        key = model.provider.value, model.name
        if key in self.__models:
            raise ValueError(f"duplicated models with: {key}")
        self.__models[key] = model

    def find_model(self, provider: str, model_name: str) -> Optional[Model]:
        return self.__models.get((provider, model_name))

    def list_models(self) -> list[Model]:
        return list(self.__models.values())

    def count_active_models(self, provider: str, reference_date: Optional[date] = None,
                            active_period_days: int = ACTIVE_MODEL_PERIOD_DAYS) -> int:
        """
        Count the models of a provider that are actively serving requests.

        A model is considered active during a fixed period after its release date (2 years by default), as in the
        Impact'IA methodology. Dated snapshots and `latest` versions of the same model family are counted once (see
        `model_family`), aliases are not counted and models without a release date are ignored.

        Args:
            provider: Name of the provider.
            reference_date: Date at which the count is made (today by default).
            active_period_days: Duration in days after its release during which a model is considered active.

        Returns:
            The number of active models of the provider.
        """
        if reference_date is None:
            reference_date = date.today()
        families = set()
        for (model_provider, name), model in self.__models.items():
            if model_provider != provider or name in self.__aliases or model.release_date is None:
                continue
            if 0 <= (reference_date - model.release_date).days < active_period_days:
                families.add(model_family(name))
        return len(families)

    @classmethod
    def from_json(cls, filepath: Optional[str] = None) -> "ModelRepository":
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "data", "models.json"
            )
        with open(filepath) as fd:
            data = json.load(fd)

            alias_list = []
            if "aliases" in data and data["aliases"] is not None:
                for alias in data["aliases"]:
                    alias_list.append(Alias.model_validate(alias))

            model_list = []
            if "models" in data and data["models"] is not None:
                for model in data["models"]:
                    model_list.append(Model.from_json(model))

        if len(model_list) == 0:
            raise ValueError("Cannot initialize on an empty model repository.")
        return cls(models=model_list, aliases=alias_list)


models = ModelRepository.from_json()
