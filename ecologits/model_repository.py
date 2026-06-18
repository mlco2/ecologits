import json
import os
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel

from ecologits.status_messages import WarningMessage
from ecologits.utils.range_value import ValueOrRange


class Providers(Enum):
    anthropic = "anthropic"
    mistralai = "mistralai"
    openai = "openai"
    huggingface_hub = "huggingface_hub"
    cohere = "cohere"
    google_genai = "google_genai"
    stabilityai = "stabilityai"
    black_forest_labs = "black_forest_labs"
    midjourney = "midjourney"
    kling = "kling"
    runway = "runway"
    luma = "luma"


class ArchitectureTypes(Enum):
    DENSE = "dense"
    MOE = "moe"


class ModalityTypes(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class ParametersMoE(BaseModel):
    total: ValueOrRange
    active: ValueOrRange


class Architecture(BaseModel):
    type: ArchitectureTypes
    parameters: Union[ValueOrRange, ParametersMoE]


class DiffusionParameters(BaseModel):
    """
    Diffusion model description used for image and video generation.

    Attributes:
        flops_denoise_per_step: FLOPs for one denoiser forward pass at the model native resolution.
        default_steps: Default number of denoising steps (sampler dependent).
        default_guidance_scale: Default guidance scale; CFG double-pass applies when > 1.0.
            Flow-matching models (Flux, SD3) store their typical value without triggering double-pass.
        latent_downscale: Spatial downscale factor between pixel and latent space (e.g. 8).
        is_video: Whether the model generates video.
        default_frames: Default number of generated frames (video only).
        default_fps: Default frames per second (video only).
        frame_attn_fraction: Share of per-step FLOPs from full-sequence 3D attention (scales as F²).
            0.0 for image models and sparse-attention video models; 0.5–0.64 for full-3D-attention
            DiT video models. Used in the mixed frame scaling: (1-f)*(F/F_default) + f*(F/F_default)².
    """

    flops_denoise_per_step: ValueOrRange
    default_steps: int
    default_guidance_scale: float = 7.5
    cfg_double_pass: bool = True
    latent_downscale: int = 8
    is_video: bool = False
    default_frames: int | None = None
    default_fps: int | None = None
    frame_attn_fraction: float = 0.0


class Deployment(BaseModel):
    tps: float | None = None
    ttft: float | None = None


class Alias(BaseModel):
    provider: Providers
    name: str
    alias: str


class Model(BaseModel):
    """
    Generative AI Model.

    Attributes:
        provider: Provider of the model (e.g. "OpenAI")
        name: Name of the model (e.g. "gpt-4o-mini")
        modality: Output modality produced by the model (text, image, or video).
        architecture: Architecture type (dense or mixture-of-experts)
        warnings: Warnings linked to the model (e.g. "model-arch-not-released" or "model-arch-multimodal")
        sources: Source of the model information (website link)
        deployment: Deployment information (tps, ttft)
        diffusion: Optional diffusion description for image and video generation models
    """

    provider: Providers
    name: str
    modality: ModalityTypes = ModalityTypes.TEXT
    architecture: Architecture
    warnings: list[WarningMessage] = []
    sources: list[str] = []
    deployment: Deployment | None = None
    diffusion: DiffusionParameters | None = None

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
        diffusion = None
        if "diffusion" in data and data["diffusion"] is not None:
            diffusion = DiffusionParameters.model_validate(data["diffusion"])
        modality = ModalityTypes.TEXT
        if "modality" in data and data["modality"] is not None:
            modality = ModalityTypes(data["modality"])
        return cls(
            provider=Providers(data["provider"]),
            name=data["name"],
            modality=modality,
            architecture=Architecture.model_validate(data["architecture"]),
            warnings=warnings,
            sources=sources,
            deployment=deployment,
            diffusion=diffusion,
        )


class ModelRepository:
    """
    Repository of models
    """

    def __init__(self, models: Optional[list[Model]] = None, aliases: Optional[list[Alias]] = None) -> None:
        self.__models: dict[tuple[str, str], Model] = {}
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
