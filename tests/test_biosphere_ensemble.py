from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from map_maker.pipeline.biosphere_ensemble import (
    BiosphereEnsembleConfig,
    BiosphereEnsembleResult,
    BiosphereEnsembleThresholds,
    BiosphereSeedReport,
    EnsembleGate,
    evaluate_biosphere_ensemble,
    run_biosphere_ensemble,
)
from map_maker.pipeline.stages.biosphere_validation import CLIMATE_STRATA


def _seed_report(
    seed: int,
    *,
    land_fraction: float = 0.29,
    npp: float = 62.0,
    biomass: float = 900.0,
    cover: float = 0.75,
    vegetated: float = 0.90,
) -> BiosphereSeedReport:
    values = {
        "global_land_surface_fraction": land_fraction,
        "global_potential_npp_pg_c_year": npp,
        "global_potential_biomass_pg_c": biomass,
        "land_mean_potential_vegetation_cover_fraction": cover,
        "potentially_vegetated_land_fraction": vegetated,
    }
    kpi_rows = [
        {
            "kpi_id": kpi_id,
            "value": value,
            "gate_kind": "earth_diagnostic" if kpi_id.startswith("global_") else "diagnostic",
            "comparison_status": "within_reference",
        }
        for kpi_id, value in values.items()
    ]
    kpi_rows.append(
        {
            "kpi_id": "warm_humid_to_warm_dry_npp_ratio",
            "value": 4.0,
            "gate_kind": "earth_structure",
            "comparison_status": "within_reference",
        }
    )
    zone_rows = []
    for index, definition in enumerate(CLIMATE_STRATA):
        zone_rows.append(
            {
                "zone_id": definition["zone_id"],
                "zone_land_area_fraction": 1.0 / len(CLIMATE_STRATA),
                "metric_id": "annual_potential_npp",
                "statistic": "mean",
                "value": 0.1 + index * 0.1,
                "reportable": True,
            }
        )
    return BiosphereSeedReport(
        seed,
        pa.Table.from_pylist(kpi_rows),
        pa.Table.from_pylist(zone_rows),
        {
            "hard_gate_pass": 1,
            "reference_profile_version": "earth_biosphere_v1",
        },
    )


def _thresholds() -> BiosphereEnsembleThresholds:
    return BiosphereEnsembleThresholds(minimum_seed_count=2)


def test_ensemble_accepts_stable_multi_seed_earth_profile():
    evaluation = evaluate_biosphere_ensemble(
        [_seed_report(1), _seed_report(2, npp=64.0, biomass=940.0)],
        _thresholds(),
    )

    assert evaluation.passed
    assert evaluation.hard_gate_pass
    assert evaluation.stability_pass
    assert evaluation.earth_profile_pass
    assert evaluation.metric_catalog.num_rows >= 12


def test_ensemble_keeps_stability_failure_separate_from_earth_range_status():
    evaluation = evaluate_biosphere_ensemble(
        [_seed_report(1, npp=30.0), _seed_report(2, npp=120.0)],
        _thresholds(),
    )

    assert not evaluation.stability_pass
    assert evaluation.earth_profile_pass
    assert any(
        gate.name == "stability.global_potential_npp_pg_c_year" and not gate.passed
        for gate in evaluation.gates
    )


def test_ensemble_rejects_mixed_reference_profile_versions():
    reports = [_seed_report(1), _seed_report(2)]
    reports[1].metadata["reference_profile_version"] = "earth_biosphere_v2"

    evaluation = evaluate_biosphere_ensemble(reports, _thresholds())

    assert not evaluation.hard_gate_pass
    assert any(
        gate.name == "reference_profile_version_match" and not gate.passed
        for gate in evaluation.gates
    )


def test_ensemble_config_requires_versioned_profile_and_enough_unique_seeds(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("topology: cubed_sphere\nresolutions: [{face_resolution: 8}]\n")
    config_path = tmp_path / "ensemble.yaml"
    config_path.write_text(
        """
format_version: 1
reference_profile: earth_biosphere_v1
base_config: base.yaml
seeds: [1, 2]
face_resolution: 8
ensemble_tolerances:
  minimum_seed_count: 2
""".strip()
    )

    config = BiosphereEnsembleConfig.from_file(config_path)

    assert config.seeds == (1, 2)
    assert config.face_resolution == 8
    assert config.thresholds.minimum_seed_count == 2

    config_path.write_text(config_path.read_text().replace("[1, 2]", "[1, 1]"))
    with pytest.raises(ValueError, match="unique"):
        BiosphereEnsembleConfig.from_file(config_path)


def test_biosphere_validation_cli_distinguishes_calibration_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from map_maker.cli import main
    from map_maker.pipeline import biosphere_ensemble

    base = tmp_path / "base.yaml"
    base.write_text("topology: cubed_sphere\nresolutions: [{face_resolution: 8}]\n")
    config_path = tmp_path / "ensemble.yaml"
    config_path.write_text(
        """
base_config: base.yaml
seeds: [1, 2]
ensemble_tolerances:
  minimum_seed_count: 2
""".strip()
    )
    result = BiosphereEnsembleResult(
        passed=False,
        execution_valid=True,
        earth_profile_pass=False,
        report_path=tmp_path / "report.json",
        metric_catalog_path=tmp_path / "ensemble.parquet",
        seed_count=2,
        gates=(
            EnsembleGate(
                "earth_profile.global_potential_npp_pg_c_year",
                "earth_profile",
                False,
                0.0,
                "within-reference seed fraction >= 0.8",
            ),
        ),
    )
    monkeypatch.setattr(biosphere_ensemble, "run_biosphere_ensemble", lambda config: result)

    assert main(["validate-biosphere", "--config", str(config_path)]) == 1
    assert "OUTSIDE REFERENCE" in capsys.readouterr().out


def test_ensemble_persists_seed_execution_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from map_maker.pipeline import biosphere_ensemble

    base = tmp_path / "base.yaml"
    base.write_text("topology: cubed_sphere\nresolutions: [{face_resolution: 8}]\n")
    config_path = tmp_path / "ensemble.yaml"
    config_path.write_text(
        """
base_config: base.yaml
output_dir: report
seeds: [1, 2]
ensemble_tolerances:
  minimum_seed_count: 2
""".strip()
    )

    class FailingEngine:
        def __init__(self, config, *, generate_visuals):
            del config, generate_visuals

        def run(self, stages):
            del stages
            raise RuntimeError("synthetic upstream failure")

    monkeypatch.setattr(biosphere_ensemble, "ExecutionEngine", FailingEngine)
    result = run_biosphere_ensemble(BiosphereEnsembleConfig.from_file(config_path))
    report = json.loads(result.report_path.read_text())

    assert not result.execution_valid
    assert report["successful_seed_count"] == 0
    assert len(report["execution_failures"]) == 2
    assert report["status"] == "invalid"
