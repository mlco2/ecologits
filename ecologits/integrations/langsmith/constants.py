"""
Constants for the LangSmith integration.

All string keys, normalisation maps, scaling factors and feedback metadata used
by the integration are centralised here so the rest of the package contains no
string literals tied to LangSmith / ecologits conventions.
"""

from __future__ import annotations

DEFAULT_RUN_TYPE: str = "chain"
LLM_RUN_TYPE: str = "llm"

KEY_EXTRA: str = "extra"
KEY_METADATA: str = "metadata"
KEY_INVOCATION_PARAMS: str = "invocation_params"
KEY_SERIALIZED: str = "serialized"
KEY_SERIALIZED_ID: str = "id"
KEY_RUN_TYPE: str = "_type"
KEY_LS_PROVIDER: str = "ls_provider"
KEY_LS_MODEL_NAME: str = "ls_model_name"
KEY_MODEL_NAME: str = "model_name"
KEY_MODEL: str = "model"
KEY_AZURE_DEPLOYMENT: str = "azure_deployment"
KEY_USAGE_METADATA: str = "usage_metadata"
KEY_USAGE: str = "usage"
KEY_TOKEN_USAGE: str = "token_usage"
KEY_LLM_OUTPUT: str = "llm_output"
KEY_OUTPUT_TOKENS: str = "output_tokens"
KEY_COMPLETION_TOKENS: str = "completion_tokens"

LS_PROVIDER_TO_ECOLOGITS: dict[str, str] = {
    "openai": "openai",
    "azure": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "mistral": "mistralai",
    "mistralai": "mistralai",
    "google": "google_genai",
    "google_genai": "google_genai",
    "google_vertexai": "google_genai",
    "huggingface": "huggingface_hub",
    "huggingface_hub": "huggingface_hub",
}

TYPE_TO_PROVIDER: dict[str, str] = {
    "azure-openai-chat": "openai",
    "azure-openai": "openai",
    "openai-chat": "openai",
    "openai": "openai",
    "anthropic": "anthropic",
    "anthropic-chat": "anthropic",
    "cohere": "cohere",
    "cohere-chat": "cohere",
    "mistral": "mistralai",
    "mistral-chat": "mistralai",
    "google-palm": "google_genai",
    "google-genai": "google_genai",
    "vertexai": "google_genai",
    "huggingface": "huggingface_hub",
}

SERIALIZED_MODULE_TO_PROVIDER: dict[str, str] = {
    "azure_openai": "openai",
    "openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "mistralai": "mistralai",
    "google_genai": "google_genai",
    "google_vertexai": "google_genai",
    "huggingface_hub": "huggingface_hub",
}

# TODO: Move DEPLOYMENT_TO_MODEL entries to models.json "aliases" section so
# ModelRepository.find_model() resolves them natively. Until then, extend this
# dict manually with Azure deployment names (dots stripped/replaced by dashes,
# e.g. gpt-4.1 -> gpt-41) for every OpenAI model added to models.json.
DEPLOYMENT_TO_MODEL: dict[str, str] = {
    "gpt-51": "gpt-5.1",
    "gpt-41-mini": "gpt-4.1-mini",
    "gpt-41": "gpt-4.1",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "gpt-35-turbo": "gpt-3.5-turbo",
    "gpt-35-turbo-16k": "gpt-3.5-turbo-16k",
}

IMPACT_FIELDS: tuple[str, ...] = ("energy", "gwp", "adpe", "pe", "wcf")

# Feedback key (display) name per ecologits impact field, with the unit baked in
# after scaling so chart axes can simply read the key.
FEEDBACK_KEY_BASE: dict[str, str] = {
    "energy": "energy_wh",
    "gwp": "gwp_gco2eq",
    "adpe": "adpe_mgsbeq",
    "pe": "pe_kj",
    "wcf": "wcf_ml",
}

# Scale raw ecologits units → values with >=4 significant figures after
# LangSmith's 4-decimal-place rounding (matches onejo-ml convention).
SCORE_SCALE: dict[str, float] = {
    "energy": 1e3,  # kWh -> Wh
    "gwp": 1e3,  # kgCO2eq -> gCO2eq
    "adpe": 1e6,  # kgSbeq -> mgSbeq
    "pe": 1e3,  # MJ -> kJ
    "wcf": 1e3,  # L -> mL
}

VARIANTS: tuple[str, ...] = ("min", "max", "avg")
VARIANT_AVG: str = "avg"
VARIANT_MIN: str = "min"
VARIANT_MAX: str = "max"

SCORE_ROUND_DIGITS: int = 4

# LangSmith's FeedbackSourceType enum only accepts "api" or "model"
# (see langsmith/schemas.py FeedbackSourceType). The ecologits origin is
# surfaced via source_info instead (FEEDBACK_SOURCE_INFO below).
LANGSMITH_FEEDBACK_SOURCE_TYPE: str = "api"

FEEDBACK_SOURCE_KEY_ORIGIN: str = "source"
FEEDBACK_SOURCE_KEY_VERSION: str = "version"
FEEDBACK_SOURCE_VALUE_ORIGIN: str = "ecologits"

# ---------------------------------------------------------------------------
# SCI for AI (ISO/IEC 21031:2024) — Consumer Boundary
# ---------------------------------------------------------------------------
SCI_BOUNDARY_CONSUMER: str = "consumer"
SCI_ISO_REFERENCE: str = "ISO/IEC 21031:2024"
SCI_METHODOLOGY: str = "sci-ai-v1.0"
SCI_OFFSETS_EXCLUDED: str = "true"

FUNCTIONAL_UNIT_PER_TOKEN: str = "token"
FUNCTIONAL_UNIT_PER_WORKFLOW: str = "workflow"

# Per-token SCI (mgCO2eq/token — completion tokens, Consumer functional unit for LLMs).
SCI_KEY_PER_TOKEN: str = "sci_per_token_mgco2eq"
# Per-workflow SCI (gCO2eq/workflow — Consumer functional unit for Agentic AI).
SCI_KEY_PER_WORKFLOW: str = "sci_per_workflow_gco2eq"
# Regional grid carbon intensity surfaced from codecarbon (gCO2eq/kWh).
SCI_KEY_CARBON_INTENSITY: str = "sci_I_gco2eq_per_kwh"

# Scale: raw kgCO2eq → display units matching the onejo-ml convention.
SCI_SCALE_PER_TOKEN: float = 1e9   # kgCO2eq/token  → mgCO2eq/token  (×1e6 kg→g, ×1e3 g→mg = 1e9)
SCI_SCALE_PER_WORKFLOW: float = 1e3  # kgCO2eq/workflow → gCO2eq/workflow

SCI_FEEDBACK_METADATA: dict[str, str] = {
    "sci_boundary": SCI_BOUNDARY_CONSUMER,
    "sci_iso_ref": SCI_ISO_REFERENCE,
    "sci_methodology": SCI_METHODOLOGY,
    "sci_offsets_excluded": SCI_OFFSETS_EXCLUDED,
}

# ---------------------------------------------------------------------------
# codecarbon runtime tracking
# ---------------------------------------------------------------------------
CODECARBON_FIELD_TO_KEY: dict[str, str] = {
    "energy_consumed": "runtime_energy_wh",
    "emissions": "runtime_gwp_gco2eq",
    "water_consumed": "runtime_water_ml",
    "cpu_energy": "runtime_cpu_energy_wh",
    "gpu_energy": "runtime_gpu_energy_wh",
    "ram_energy": "runtime_ram_energy_wh",
    "duration": "runtime_duration_s",
}

# Scale raw codecarbon units → chart-friendly values (same convention as SCORE_SCALE).
CODECARBON_FIELD_SCALE: dict[str, float] = {
    "energy_consumed": 1e3,  # kWh -> Wh
    "emissions": 1e3,  # kgCO2eq -> gCO2eq
    "water_consumed": 1e3,  # L -> mL
    "cpu_energy": 1e3,  # kWh -> Wh
    "gpu_energy": 1e3,  # kWh -> Wh
    "ram_energy": 1e3,  # kWh -> Wh
    "duration": 1.0,  # seconds (no conversion)
}

DEFAULT_MEASURE_POWER_SECS: float = 1.0
DEFAULT_TRACKING_MODE: str = "process"
CODECARBON_PROJECT_NAME: str = "ecologits-langsmith"
