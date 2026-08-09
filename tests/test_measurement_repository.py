import json

import pytest

from ecologits.impacts.llm import compute_llm_impacts, compute_llm_impacts_measured
from ecologits.measurement_repository import (
    Deployment,
    Measurement,
    MeasurementRepository,
    load_measurements,
)
from ecologits.measurement_repository import measurements as global_measurements
from ecologits.tracers.utils import llm_impacts

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def make_measurement(
    concurrency=64,
    gpu_model="NVIDIA A100-SXM4-80GB",
    energy=1.0e-07,
    quantization="fp16",
    deployment_id=None,
    deployment_label=None,
    measurement_id=None,
):
    return Measurement(
        model_name=MODEL,
        it_energy_per_token=energy,
        latency_per_token=0.024,
        concurrency=concurrency,
        gpu_model=gpu_model,
        gpu_count=1,
        engine="vllm",
        engine_version="0.6.3",
        quantization=quantization,
        spec_version="1.0.0-draft",
        id=measurement_id,
        deployment_id=deployment_id,
        deployment_label=deployment_label,
    )


@pytest.fixture
def loaded(tmp_path):
    """Populate the global repository, then restore it."""
    original = global_measurements.list_measurements()
    snapshot = [
        {
            "id": "b1b9d5e0-58e8-45f0-9ef5-4549b3d6f3f0",
            "model_name": MODEL,
            "it_energy_per_token": 1.0e-07,
            "latency_per_token_s": 0.024,
            "concurrency": 64,
            "gpu_model": "NVIDIA A100-SXM4-80GB",
            "gpu_count": 1,
            "engine": "vllm",
            "engine_version": "0.6.3",
            "quantization": "fp16",
            "spec_version": "1.0.0-draft",
        }
    ]
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(snapshot))
    load_measurements(str(path))
    yield
    global_measurements.replace_all(original)


class TestParsing:
    def test_reads_the_export_payload_shape(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "model_name": MODEL,
                        "it_energy_per_token": 2.0e-07,
                        "latency_per_token_s": 0.03,
                        "concurrency": 8,
                    }
                ]
            )
        )
        repo = MeasurementRepository.from_json(str(path))
        assert len(repo) == 1
        assert repo.list_measurements()[0].latency_per_token == 0.03

    def test_reads_a_wrapped_snapshot_file(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                {
                    "measurements": [
                        {
                            "model_name": MODEL,
                            "it_energy_per_token": 2.0e-07,
                            "concurrency": 8,
                        }
                    ]
                }
            )
        )
        assert len(MeasurementRepository.from_json(str(path))) == 1

    def test_ships_empty(self):
        """
        EcoLogits carries no measurements of its own: a measurement is only
        meaningful alongside the deployment it describes.
        """
        assert len(MeasurementRepository()) == 0


class TestSelection:
    def test_unknown_model_returns_none(self):
        repo = MeasurementRepository([make_measurement()])
        assert repo.find_measurement("not-a-model") is None

    def test_concurrency_filter_is_strict(self):
        """
        Per-token energy varies by ~10x between concurrency 1 and 64. A near
        miss is not a better answer than no answer.
        """
        repo = MeasurementRepository([make_measurement(concurrency=64)])
        assert repo.find_measurement(MODEL, Deployment(concurrency=8)) is None

    def test_concurrency_preference_reports_itself(self):
        repo = MeasurementRepository(
            [make_measurement(concurrency=1), make_measurement(concurrency=64)]
        )
        selection = repo.select(MODEL)
        assert selection.used_concurrency_preference
        assert selection.measurement.concurrency == 64

    def test_prefers_the_concurrency_matching_ecologits_batch_size(self):
        repo = MeasurementRepository(
            [
                make_measurement(concurrency=1),
                make_measurement(concurrency=64),
                make_measurement(concurrency=128),
            ]
        )
        assert repo.find_measurement(MODEL).concurrency == 64

    def test_no_preferred_concurrency_available_is_ambiguous(self):
        """
        The tie-break only applies when concurrency 64 is among the candidates.
        Otherwise there is no principled choice, so the caller makes it.
        """
        repo = MeasurementRepository(
            [make_measurement(concurrency=1), make_measurement(concurrency=32)]
        )
        assert repo.select(MODEL).is_ambiguous

    def test_gpu_filter(self):
        repo = MeasurementRepository(
            [make_measurement(gpu_model="NVIDIA A100-SXM4-80GB")]
        )
        assert (
            repo.find_measurement(MODEL, Deployment(gpu_model="NVIDIA RTX 4090"))
            is None
        )

    def test_quantization_is_selectable(self):
        """
        int8 is roughly 2.5x cheaper per token than fp16. Before this was a
        filter, the two were indistinguishable to the selector.
        """
        repo = MeasurementRepository(
            [
                make_measurement(quantization="fp16", energy=1.0e-07),
                make_measurement(quantization="int8", energy=4.0e-08),
            ]
        )
        picked = repo.find_measurement(MODEL, Deployment(quantization="int8"))
        assert picked.it_energy_per_token == 4.0e-08

    def test_two_quantizations_are_ambiguous_not_guessed(self):
        repo = MeasurementRepository(
            [
                make_measurement(quantization="fp16", energy=1.0e-07),
                make_measurement(quantization="int8", energy=4.0e-08),
            ]
        )
        selection = repo.select(MODEL)
        assert selection.is_ambiguous
        assert selection.measurement is None
        assert len(selection.candidates) == 2

    def test_two_boxes_with_the_same_gpu_are_ambiguous(self):
        """Same hardware class, different numbers: the caller must choose."""
        repo = MeasurementRepository(
            [
                make_measurement(energy=1.0e-07, deployment_id="box-1"),
                make_measurement(energy=1.6e-07, deployment_id="box-2"),
            ]
        )
        assert repo.select(MODEL).is_ambiguous

    def test_deployment_id_resolves_the_ambiguity(self):
        repo = MeasurementRepository(
            [
                make_measurement(energy=1.0e-07, deployment_id="box-1"),
                make_measurement(energy=1.6e-07, deployment_id="box-2"),
            ]
        )
        picked = repo.find_measurement(MODEL, Deployment(id="box-2"))
        assert picked.it_energy_per_token == 1.6e-07

    def test_deployment_label_also_selects(self):
        repo = MeasurementRepository(
            [
                make_measurement(energy=1.0e-07, deployment_label="prod"),
                make_measurement(energy=1.6e-07, deployment_label="staging"),
            ]
        )
        picked = repo.find_measurement(MODEL, Deployment(label="staging"))
        assert picked.it_energy_per_token == 1.6e-07

    def test_measurement_id_pins_one_record(self):
        """
        deployment id follows a box across re-benchmarks; measurement id pins
        one historical run.
        """
        repo = MeasurementRepository(
            [
                make_measurement(energy=1.0e-07, measurement_id="run-a", deployment_id="box-1"),
                make_measurement(energy=1.2e-07, measurement_id="run-b", deployment_id="box-1"),
            ]
        )
        assert repo.select(MODEL, Deployment(id="box-1")).is_ambiguous
        picked = repo.find_measurement(MODEL, Deployment(measurement_id="run-a"))
        assert picked.it_energy_per_token == 1.0e-07

    def test_candidate_description_names_the_energies(self):
        repo = MeasurementRepository(
            [
                make_measurement(quantization="fp16", energy=1.0e-07),
                make_measurement(quantization="int8", energy=4.0e-08),
            ]
        )
        described = repo.select(MODEL).describe_candidates()
        assert "fp16" in described and "int8" in described
        assert "kWh/token" in described

    def test_summary_states_the_conditions(self):
        summary = make_measurement().summary
        assert "concurrency 64" in summary
        assert "NVIDIA A100-SXM4-80GB" in summary
        assert "vllm 0.6.3" in summary


class TestMeasuredComputation:
    def _impacts(self, energy=1.0e-07, output_token_count=1000, **kwargs):
        params = dict(
            it_energy_per_output_token=energy,
            latency_per_output_token=0.024,
            output_token_count=output_token_count,
            gpu_count=1,
            concurrency=64,
            if_electricity_mix_adpe=5.75e-08,
            if_electricity_mix_pe=9.988,
            if_electricity_mix_gwp=0.478,
            if_electricity_mix_wue=1.8,
            datacenter_pue=1.0,
            datacenter_wue=0.0,
        )
        params.update(kwargs)
        return compute_llm_impacts_measured(**params)

    def test_energy_comes_straight_from_the_measurement(self):
        """
        The substitution happens at request_it_energy, so with PUE 1.0 the
        reported energy is exactly measured-energy x tokens.
        """
        impacts = self._impacts(energy=1.0e-07, output_token_count=1000)
        assert impacts.energy.value == pytest.approx(1.0e-04)

    def test_pue_is_applied_downstream_not_folded_in(self):
        base = self._impacts(datacenter_pue=1.0)
        with_pue = self._impacts(datacenter_pue=1.2)
        assert with_pue.energy.value == pytest.approx(base.energy.value * 1.2)

    def test_gwp_uses_the_callers_electricity_mix(self):
        impacts = self._impacts(if_electricity_mix_gwp=0.5)
        assert impacts.usage.gwp.value == pytest.approx(impacts.energy.value * 0.5)

    def test_energy_scales_linearly_with_tokens(self):
        one = self._impacts(output_token_count=100)
        ten = self._impacts(output_token_count=1000)
        assert ten.energy.value == pytest.approx(one.energy.value * 10)

    def test_embodied_impacts_are_still_computed(self):
        """Measurement replaces the energy estimate, not the whole model."""
        impacts = self._impacts()
        assert impacts.embodied.gwp.value > 0
        assert impacts.gwp.value > impacts.usage.gwp.value

    def test_request_latency_caps_generation_latency(self):
        unbounded = self._impacts()
        capped = self._impacts(request_latency=0.5)
        # Embodied impact is amortized over generation time, so a shorter
        # attributed latency means a smaller embodied share.
        assert capped.embodied.gwp.value < unbounded.embodied.gwp.value

    def test_missing_latency_does_not_crash(self):
        impacts = self._impacts(latency_per_output_token=None, request_latency=None)
        assert impacts.energy.value > 0
        assert impacts.embodied.gwp.value == 0

    def test_measured_and_estimated_paths_agree_in_shape(self):
        """Both paths must produce the same populated impact structure."""
        estimated = compute_llm_impacts(
            model_active_parameter_count=7.0,
            model_total_parameter_count=7.0,
            output_token_count=1000,
            request_latency=10.0,
            if_electricity_mix_adpe=5.75e-08,
            if_electricity_mix_pe=9.988,
            if_electricity_mix_gwp=0.478,
            if_electricity_mix_wue=1.8,
            datacenter_pue=1.1,
            datacenter_wue=0.2,
        )
        measured = self._impacts()
        assert set(estimated.model_dump()) == set(measured.model_dump())


class TestLlmImpactsIntegration:
    def test_local_provider_uses_the_measurement(self, loaded):
        impacts = llm_impacts(
            provider="local",
            model_name=MODEL,
            output_token_count=1000,
            request_latency=25.0,
            electricity_mix_zone="FRA",
        )
        assert not impacts.has_errors
        assert impacts.energy.value == pytest.approx(1.0e-04)

    def test_provenance_warning_names_the_conditions(self, loaded):
        """
        Selecting a measurement silently would hide the caveat that matters
        most: which deployment the number actually describes.
        """
        impacts = llm_impacts(
            provider="local",
            model_name=MODEL,
            output_token_count=100,
            request_latency=5.0,
        )
        codes = [w.code for w in impacts.warnings]
        assert "model-energy-measured" in codes
        message = next(w.message for w in impacts.warnings if w.code == "model-energy-measured")
        assert "concurrency 64" in message
        assert "NVIDIA A100-SXM4-80GB" in message

    def test_unknown_model_reports_measurement_not_found(self, loaded):
        impacts = llm_impacts(
            provider="local",
            model_name="unmeasured-model",
            output_token_count=100,
            request_latency=5.0,
        )
        assert impacts.has_errors
        assert impacts.errors[0].code == "measurement-not-found"

    def test_measured_path_needs_no_models_json_entry(self, loaded):
        """
        The measurement supplies energy, latency and GPU count, so the model
        repository is never consulted — which is why a self-hosted model that
        EcoLogits has never heard of still works.
        """
        from ecologits.model_repository import models

        assert models.find_model(provider="local", model_name=MODEL) is None
        impacts = llm_impacts(
            provider="local",
            model_name=MODEL,
            output_token_count=100,
            request_latency=5.0,
        )
        assert not impacts.has_errors

    def test_ambiguity_is_reported_with_the_candidates(self):
        """A silent pick between candidates that differ 2.5x is the worst outcome."""
        original = global_measurements.list_measurements()
        global_measurements.replace_all(
            [
                make_measurement(quantization="fp16", energy=1.0e-07),
                make_measurement(quantization="int8", energy=4.0e-08),
            ]
        )
        try:
            impacts = llm_impacts(
                provider="local",
                model_name=MODEL,
                output_token_count=100,
                request_latency=5.0,
            )
        finally:
            global_measurements.replace_all(original)

        assert impacts.has_errors
        assert impacts.errors[0].code == "measurement-ambiguous"
        assert "fp16" in impacts.errors[0].message
        assert "int8" in impacts.errors[0].message

    def test_deployment_set_at_init_is_used_by_default(self):
        """The target is bound once, not threaded through every call site."""
        from ecologits import EcoLogits

        original = global_measurements.list_measurements()
        original_deployment = EcoLogits.config.deployment
        global_measurements.replace_all(
            [
                make_measurement(energy=1.0e-07, deployment_id="box-1"),
                make_measurement(energy=1.6e-07, deployment_id="box-2"),
            ]
        )
        EcoLogits.config.deployment = Deployment(id="box-2")
        try:
            impacts = llm_impacts(
                provider="local",
                model_name=MODEL,
                output_token_count=1000,
                request_latency=25.0,
            )
        finally:
            global_measurements.replace_all(original)
            EcoLogits.config.deployment = original_deployment

        assert not impacts.has_errors
        assert impacts.energy.value == pytest.approx(1.6e-04)

    def test_per_call_deployment_overrides_the_default(self):
        from ecologits import EcoLogits

        original = global_measurements.list_measurements()
        original_deployment = EcoLogits.config.deployment
        global_measurements.replace_all(
            [
                make_measurement(energy=1.0e-07, deployment_id="box-1"),
                make_measurement(energy=1.6e-07, deployment_id="box-2"),
            ]
        )
        EcoLogits.config.deployment = Deployment(id="box-2")
        try:
            impacts = llm_impacts(
                provider="local",
                model_name=MODEL,
                output_token_count=1000,
                request_latency=25.0,
                deployment=Deployment(id="box-1"),
            )
        finally:
            global_measurements.replace_all(original)
            EcoLogits.config.deployment = original_deployment

        assert impacts.energy.value == pytest.approx(1.0e-04)

    def test_hosted_providers_are_untouched(self):
        impacts = llm_impacts(
            provider="openai",
            model_name="gpt-4o-mini",
            output_token_count=100,
            request_latency=5.0,
        )
        assert not impacts.has_errors
        codes = [w.code for w in (impacts.warnings or [])]
        assert "model-energy-measured" not in codes

    def test_concurrency_mismatch_reports_not_found(self, loaded):
        impacts = llm_impacts(
            provider="local",
            model_name=MODEL,
            output_token_count=100,
            request_latency=5.0,
            deployment=Deployment(concurrency=1),
        )
        assert impacts.has_errors
        assert impacts.errors[0].code == "measurement-not-found"
