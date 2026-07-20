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
    FunctionalVegetationSeedReport,
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


def _functional_report(*, outside_reference: bool = False) -> FunctionalVegetationSeedReport:
    global_values = {
        "land_mean_functional_vegetated_fraction": 0.52,
        "land_mean_functional_woody_fraction": 0.19,
        "land_mean_functional_herbaceous_fraction": 0.19,
        "land_mean_functional_xeric_low_stature_fraction": 0.12,
        "land_mean_functional_hydrophytic_fraction": 0.02,
        "land_mean_nonvegetated_ground_fraction": 0.46,
        "land_mean_inland_open_water_fraction": 0.02,
        "land_fire_tendency_resource_p90": 0.18,
        "land_grazing_resource_p90": 0.33,
        "land_forest_resource_p90": 0.36,
        "land_pasture_resource_p90": 0.19,
        "land_crop_resource_p90": 0.42,
    }
    kpi_rows = [
        {
            "kpi_id": kpi_id,
            "value": value,
            "gate_kind": "earth_diagnostic",
            "comparison_status": "within_reference",
        }
        for kpi_id, value in global_values.items()
    ]
    kpi_rows.append(
        {
            "kpi_id": "warm_humid_to_warm_dry_woody_ratio",
            "value": 3.0,
            "gate_kind": "earth_structure",
            "comparison_status": "outside_reference" if outside_reference else "within_reference",
        }
    )
    zone_rows = []
    for definition in CLIMATE_STRATA:
        for metric_id, value in (
            ("functional_woody_fraction", 0.20),
            ("functional_herbaceous_fraction", 0.20),
            ("functional_hydrophytic_fraction", 0.02),
            ("resource_fire_tendency", 0.15),
            ("resource_forest", 0.25),
        ):
            zone_rows.append(
                {
                    "zone_id": definition["zone_id"],
                    "zone_land_area_fraction": 1.0 / len(CLIMATE_STRATA),
                    "metric_id": metric_id,
                    "statistic": "mean",
                    "value": value,
                    "reportable": True,
                }
            )
    return FunctionalVegetationSeedReport(
        pa.Table.from_pylist(kpi_rows),
        pa.Table.from_pylist(zone_rows),
        {
            "hard_gate_pass": 1,
            "reference_profile_version": "earth_functional_vegetation_v1",
        },
    )


def _with_functional(
    report: BiosphereSeedReport, *, outside_reference: bool = False
) -> BiosphereSeedReport:
    return BiosphereSeedReport(
        report.seed,
        report.kpis,
        report.climate_distributions,
        report.metadata,
        _functional_report(outside_reference=outside_reference),
    )


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


def test_ensemble_accepts_stable_functional_vegetation_profile():
    evaluation = evaluate_biosphere_ensemble(
        [_with_functional(_seed_report(1)), _with_functional(_seed_report(2))],
        _thresholds(),
    )

    assert evaluation.passed
    assert evaluation.functional_profile_pass
    assert any(
        gate.name == "functional_profile.warm_humid_to_warm_dry_woody_ratio" and gate.passed
        for gate in evaluation.gates
    )


def test_ensemble_separates_functional_profile_miss_from_stability():
    evaluation = evaluate_biosphere_ensemble(
        [
            _with_functional(_seed_report(1)),
            _with_functional(_seed_report(2), outside_reference=True),
        ],
        _thresholds(),
    )

    assert evaluation.stability_pass
    assert not evaluation.functional_profile_pass
    assert not evaluation.passed


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

    invalid_profile = config_path.read_text() + "\nfunctional_reference_profile: experimental_v2\n"
    config_path.write_text(invalid_profile)
    with pytest.raises(ValueError, match="functional_reference_profile"):
        BiosphereEnsembleConfig.from_file(config_path)

    config_path.write_text(
        invalid_profile.replace("experimental_v2", "earth_functional_vegetation_v1")
    )
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
