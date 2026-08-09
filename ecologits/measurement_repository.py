from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Concurrency of the measurement that best matches EcoLogits' own modelling
# assumption (`impacts.llm.BATCH_SIZE`). Used only to break ties when the caller
# does not state a concurrency; the chosen value is always reported back in a
# warning so the assumption is never silent.
PREFERRED_CONCURRENCY = 64

CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "ecologits"


class Measurement(BaseModel):
    """
    A measured energy cost per output token for a served model.

    One row of the curated snapshot produced by CodeCarbon's LLM energy
    benchmark. It describes a specific deployment — a model, on stated
    hardware, served by a stated engine, at a stated concurrency — and is only
    evidence about deployments resembling that one.

    Attributes:
        model_name: Name of the measured model (usually a Hugging Face repo id)
        it_energy_per_token: IT energy per output token in kWh, PUE-free
        latency_per_token: Token generation latency in seconds, if measured
        concurrency: Requests in flight during the measurement
        gpu_model: Name of the GPU used
        gpu_count: Number of GPUs the model was served on
        engine: Serving engine (e.g. "vllm")
        engine_version: Version of the serving engine
        quantization: Weight quantization (e.g. "fp16")
        model_revision: Revision or commit hash of the measured weights
        spec_version: Version of the benchmark spec the measurement follows
        id: Identifier of the source record
    """

    model_name: str
    it_energy_per_token: float
    latency_per_token: float | None = None
    concurrency: int
    gpu_model: str | None = None
    gpu_count: int = 1
    engine: str | None = None
    engine_version: str | None = None
    quantization: str | None = None
    model_revision: str | None = None
    spec_version: str | None = None
    id: str | None = None
    deployment_id: str | None = None
    deployment_label: str | None = None

    @property
    def summary(self) -> str:
        """Human-readable statement of the conditions the measurement holds under."""
        parts = [f"concurrency {self.concurrency}"]
        if self.gpu_model:
            parts.append(f"{self.gpu_count}x {self.gpu_model}")
        if self.engine:
            engine = self.engine
            if self.engine_version:
                engine += f" {self.engine_version}"
            parts.append(engine)
        if self.quantization:
            parts.append(self.quantization)
        if self.deployment_label or self.deployment_id:
            parts.append(f"deployment {self.deployment_label or self.deployment_id}")
        return ", ".join(parts)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Measurement:
        # Accepts the carbonserver `/model-benchmarks/export` payload, whose
        # latency field carries its unit in the name.
        return cls(
            model_name=data["model_name"],
            it_energy_per_token=float(data["it_energy_per_token"]),
            latency_per_token=data.get("latency_per_token_s")
            or data.get("latency_per_token"),
            concurrency=int(data["concurrency"]),
            gpu_model=data.get("gpu_model"),
            gpu_count=int(data.get("gpu_count") or 1),
            engine=data.get("engine"),
            engine_version=data.get("engine_version"),
            quantization=data.get("quantization"),
            model_revision=data.get("model_revision"),
            spec_version=data.get("spec_version"),
            id=str(data["id"]) if data.get("id") else None,
            deployment_id=data.get("deployment_id"),
            deployment_label=data.get("deployment_label"),
        )


class Deployment(BaseModel):
    """
    Selector describing which measured deployment to use.

    Every field narrows the search and every one of them changes the answer, so
    all of them are selectable. Unset fields are simply not filtered on.

    Two distinct identifiers, because they answer different questions:

    * `id` names a *deployment* — a machine and its serving configuration. It
      stays the same when that deployment is re-benchmarked, so selecting by it
      means "whatever the current measurement for my box is".
    * `measurement_id` pins one *record*: a single benchmark run. Use it to
      reproduce a published figure or to keep a number stable across snapshot
      updates.

    Attributes:
        id: Identifier of the deployment
        label: Human-readable name of the deployment
        gpu_model: Name of the GPU, matched exactly
        gpu_count: Number of GPUs
        engine: Serving engine (e.g. "vllm")
        engine_version: Version of the serving engine
        quantization: Weight quantization (e.g. "fp16")
        concurrency: Requests in flight
        model_revision: Revision or commit hash of the weights
        measurement_id: Identifier of one exact benchmark record
    """

    id: str | None = None
    label: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None
    engine: str | None = None
    engine_version: str | None = None
    quantization: str | None = None
    concurrency: int | None = None
    model_revision: str | None = None
    measurement_id: str | None = None

    def matches(self, measurement: Measurement) -> bool:
        """Whether a measurement satisfies every criterion that was set."""
        criteria = (
            (self.measurement_id, measurement.id),
            (self.id, measurement.deployment_id),
            (self.label, measurement.deployment_label),
            (self.gpu_model, measurement.gpu_model),
            (self.gpu_count, measurement.gpu_count),
            (self.engine, measurement.engine),
            (self.engine_version, measurement.engine_version),
            (self.quantization, measurement.quantization),
            (self.concurrency, measurement.concurrency),
            (self.model_revision, measurement.model_revision),
        )
        return all(wanted is None or wanted == actual for wanted, actual in criteria)


class MeasurementSelection(BaseModel):
    """
    Outcome of resolving a `Deployment` against the repository.

    Carries the candidates as well as the choice so that an ambiguous result
    can name what it could not choose between, rather than picking one.
    """

    measurement: Measurement | None = None
    candidates: list[Measurement] = []
    used_concurrency_preference: bool = False

    @property
    def is_ambiguous(self) -> bool:
        return self.measurement is None and len(self.candidates) > 1

    @property
    def is_empty(self) -> bool:
        return not self.candidates

    def describe_candidates(self) -> str:
        return "; ".join(
            f"{m.summary} -> {m.it_energy_per_token:.3g} kWh/token" for m in self.candidates
        )


class MeasurementRepository:
    """
    Repository of measured energy references.

    Empty by default: EcoLogits ships no measurements, because a measurement is
    only meaningful with the deployment it describes. Load a curated snapshot
    explicitly with `EcoLogits.init(measurements=...)`.
    """

    def __init__(self, measurements: list[Measurement] | None = None) -> None:
        self.__measurements: list[Measurement] = list(measurements or [])

    def __len__(self) -> int:
        return len(self.__measurements)

    def add_measurement(self, measurement: Measurement) -> None:
        self.__measurements.append(measurement)

    def replace_all(self, measurements: list[Measurement]) -> None:
        """
        Swap the contents in place.

        In place rather than by rebinding the module global, because consumers
        import the singleton by value (`from ... import measurements`, as they
        do for `models` and `electricity_mixes`). Rebinding would leave every
        importer holding the old, empty repository.
        """
        self.__measurements = list(measurements)

    def list_measurements(self) -> list[Measurement]:
        return list(self.__measurements)

    def find_measurements(
        self, model_name: str, deployment: Deployment | None = None
    ) -> list[Measurement]:
        """
        Every measurement matching the model and the deployment criteria.

        Filters are applied strictly: a caller asking for concurrency 8 never
        gets a measurement taken at 64, because per-token energy varies by
        roughly an order of magnitude across that range and a near miss is not
        a better answer than no answer.
        """
        candidates = [m for m in self.__measurements if m.model_name == model_name]
        if deployment is not None:
            candidates = [m for m in candidates if deployment.matches(m)]
        return candidates

    def select(
        self, model_name: str, deployment: Deployment | None = None
    ) -> MeasurementSelection:
        """
        Resolve a deployment to exactly one measurement, or report why not.

        Only one tie is broken automatically: when the remaining candidates
        differ solely in concurrency, `PREFERRED_CONCURRENCY` wins, because
        that is the concurrency EcoLogits' own model assumes and the choice is
        reported back to the caller.

        Anything else that leaves more than one candidate -- two quantizations,
        two engines, two boxes with the same GPU -- is ambiguous and is
        returned as such. Picking one silently would hand back a number that
        can be several times wrong with nothing to indicate a choice was made.
        """
        candidates = self.find_measurements(model_name, deployment)
        if len(candidates) <= 1:
            return MeasurementSelection(
                measurement=candidates[0] if candidates else None,
                candidates=candidates,
            )

        preferred = [m for m in candidates if m.concurrency == PREFERRED_CONCURRENCY]
        if len(preferred) == 1 and self._differ_only_by_concurrency(candidates):
            return MeasurementSelection(
                measurement=preferred[0],
                candidates=candidates,
                used_concurrency_preference=True,
            )
        return MeasurementSelection(measurement=None, candidates=candidates)

    @staticmethod
    def _differ_only_by_concurrency(candidates: list[Measurement]) -> bool:
        signatures = {
            (
                m.gpu_model,
                m.gpu_count,
                m.engine,
                m.engine_version,
                m.quantization,
                m.model_revision,
                m.deployment_id,
            )
            for m in candidates
        }
        return len(signatures) == 1

    def find_measurement(
        self, model_name: str, deployment: Deployment | None = None
    ) -> Measurement | None:
        """Convenience wrapper returning the selected measurement, or None."""
        return self.select(model_name, deployment).measurement

    @classmethod
    def from_json(cls, filepath: str) -> MeasurementRepository:
        with open(filepath) as fd:
            data = json.load(fd)
        return cls(cls._parse(data))

    @classmethod
    def from_url(cls, url: str, timeout: float = 10.0) -> MeasurementRepository:
        """
        Fetch a snapshot once and cache it on disk.

        Deliberately not called per request: a network round trip inside a
        library that wraps `client.chat.completions.create` would add latency
        to every LLM call and couple the caller's availability to a service
        they did not choose. A stale cached snapshot is preferable to that, so
        a failed fetch falls back to the cache when one exists.
        """
        cache_path = cls._cache_path(url)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(payload, encoding="utf-8")
        except Exception:
            if not cache_path.exists():
                raise
            payload = cache_path.read_text(encoding="utf-8")
        return cls(cls._parse(json.loads(payload)))

    @staticmethod
    def _cache_path(url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return CACHE_DIR / f"measurements-{digest}.json"

    @staticmethod
    def _parse(data: Any) -> list[Measurement]:
        # Accepts either a bare list (the export endpoint) or an object with a
        # "measurements" key (a checked-in snapshot file).
        if isinstance(data, dict):
            data = data.get("measurements") or []
        return [Measurement.from_json(entry) for entry in data]


measurements = MeasurementRepository()


def load_measurements(source: str) -> MeasurementRepository:
    """
    Load a curated snapshot into the global repository.

    Args:
        source: Path to a snapshot file, or an http(s) URL to fetch once.

    Returns:
        The populated repository.
    """
    if source.startswith("http://") or source.startswith("https://"):
        loaded = MeasurementRepository.from_url(source)
    else:
        loaded = MeasurementRepository.from_json(source)
    measurements.replace_all(loaded.list_measurements())
    return measurements
