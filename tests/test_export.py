"""Tests for CSV export utility (#118)."""
import csv
import io
import pytest

from ecologits.impacts.llm import compute_llm_impacts
from ecologits.utils.export import impacts_to_csv


def _make_impacts():
    return compute_llm_impacts(
        model_active_parameter_count=7.0,
        model_total_parameter_count=7.0,
        output_token_count=100,
        if_electricity_mix_adpe=1e-8,
        if_electricity_mix_pe=9.0,
        if_electricity_mix_gwp=0.4,
        if_electricity_mix_wue=0.5,
        datacenter_pue=1.2,
        datacenter_wue=1.8,
        request_latency=2.0,
    )


def _parse_csv(csv_str: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(csv_str)))


class TestImpactsToCsv:
    def test_returns_string(self):
        assert isinstance(impacts_to_csv(_make_impacts()), str)

    def test_has_header(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        assert "phase" in rows[0]
        assert "metric" in rows[0]
        assert "value" in rows[0]
        assert "unit" in rows[0]

    def test_total_rows_present(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        phases = [r["phase"] for r in rows]
        assert "total" in phases

    def test_usage_rows_present_by_default(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        phases = [r["phase"] for r in rows]
        assert "usage" in phases

    def test_embodied_rows_present_by_default(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        phases = [r["phase"] for r in rows]
        assert "embodied" in phases

    def test_no_phase_rows_when_disabled(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts(), include_phases=False))
        phases = {r["phase"] for r in rows}
        assert phases == {"total"}

    def test_energy_unit_is_kwh(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        energy_rows = [r for r in rows if r["phase"] == "total" and r["metric"] == "energy"]
        assert len(energy_rows) == 1
        assert energy_rows[0]["unit"] == "kWh"

    def test_value_is_numeric(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        for row in rows:
            float(row["value"])

    def test_five_total_metrics(self):
        rows = _parse_csv(impacts_to_csv(_make_impacts()))
        total_rows = [r for r in rows if r["phase"] == "total"]
        metrics = {r["metric"] for r in total_rows}
        assert metrics == {"energy", "gwp", "adpe", "pe", "wcf"}
