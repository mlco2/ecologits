"""
LangSmith integration for EcoLogits.

Provides `environment_traceable`, a drop-in replacement for LangSmith's
`@traceable` that additionally computes ecologits environmental impacts for
every captured LLM child run and posts them as LangSmith Feedback scores,
plus codecarbon runtime emissions on every run in the trace tree.

Requires the optional `langsmith` extra: `pip install ecologits[langsmith]`.
"""

from ecologits.integrations.langsmith.decorator import environment_traceable
from ecologits.integrations.langsmith.impacts import (
    build_feedback_payloads,
    compute_call_impact,
    post_impacts_as_feedback,
)
from ecologits.integrations.langsmith.payload_builder import (
    build_sci_per_token_payloads,
    build_sci_per_workflow_payloads,
)
from ecologits.integrations.langsmith.runtime import (
    build_default_tracker,
    build_runtime_feedback_payloads,
    build_sci_intensity_payload,
    post_runtime_as_feedback,
)

__all__ = [
    "build_default_tracker",
    "build_feedback_payloads",
    "build_runtime_feedback_payloads",
    "build_sci_intensity_payload",
    "build_sci_per_token_payloads",
    "build_sci_per_workflow_payloads",
    "compute_call_impact",
    "environment_traceable",
    "post_impacts_as_feedback",
    "post_runtime_as_feedback",
]
