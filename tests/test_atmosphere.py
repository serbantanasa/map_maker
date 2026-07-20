from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.stages.atmosphere import AtmosphereConfig


@pytest.fixture(autouse=True)
def _ensure_atmosphere_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "planet",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "sea_level",
        "atmosphere",
        "climate",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(
    tmp_path: Path,
    run_id: str,
    *,
    carbon_dioxide_ppm: float = 280.0,
    validation_profile: str = "earthlike",
) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 16}],
            "rng_seed": 29,
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 12,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 2,
                },
                "world_age": {"world_age": 4.1},
                "atmosphere": {
                    "validation_profile": validation_profile,
                    "carbon_dioxide_ppm": carbon_dioxide_ppm,
                },
                "climate": {
                    "spinup_years": 6,
                    "moisture_spinup_years": 1,
                    "moisture_steps_per_month_at_face_128": 16,
                },
            },
        }
    )


def _array(result, name: str) -> np.ndarray:
    return np.asarray(result.artifact_records[name].value.array())


def test_atmosphere_persists_composition_and_hydrostatic_pressure(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "atmosphere-earth"))
    results = engine.run(["atmosphere"])
    atmosphere = results["atmosphere"]
    pressure = _array(atmosphere, "SurfacePressureKPa")
    oxygen = _array(atmosphere, "OxygenPartialPressureKPa")
    co2 = _array(atmosphere, "CO2PartialPressurePa")
    elevation = _array(results["sea_level"], "SurfaceElevationM")

    assert pressure.shape == engine.context.topology.face_shape
    assert pressure.dtype == np.float32
    assert np.all(np.isfinite(pressure)) and np.all(pressure > 0.0)
    np.testing.assert_allclose(oxygen, pressure * 0.20946, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(co2, pressure * 0.280, rtol=2e-6, atol=1e-5)
    assert float(np.mean(pressure[elevation > 1_000.0])) < float(
        np.mean(pressure[elevation <= 0.0])
    )

    catalog = atmosphere.artifact_records["AtmosphericCompositionCatalog"].value
    assert isinstance(catalog, pa.Table)
    assert catalog.column_names == [
        "gas",
        "dry_mole_fraction",
        "reference_partial_pressure_pa",
        "role",
    ]
    assert catalog.num_rows == 4
    assert float(np.sum(np.asarray(catalog["dry_mole_fraction"]))) == pytest.approx(1.0)

    metadata = atmosphere.artifact_records["AtmosphereMetadata"].value
    assert metadata["hard_gate_pass"] == 1
    assert metadata["earth_diagnostic_pass"] == 1
    assert metadata["co2_greenhouse_temperature_offset_c"] == pytest.approx(0.0)
    assert 7_000.0 < metadata["scale_height_m"] < 10_000.0


def test_profile_diagnostics_do_not_clamp_or_reject_non_earth_state(tmp_path: Path):
    config = _config(
        tmp_path,
        "atmosphere-hothouse",
        carbon_dioxide_ppm=28_000.0,
        validation_profile="hothouse",
    )
    results = ExecutionEngine(config).run(["climate"])
    atmosphere_metadata = results["atmosphere"].artifact_records["AtmosphereMetadata"].value
    climate_metadata = results["climate"].artifact_records["ClimateMetadata"].value

    assert atmosphere_metadata["earth_diagnostic_pass"] == 0
    assert atmosphere_metadata["hard_gate_pass"] == 1
    assert atmosphere_metadata["profile_has_reference_diagnostics"] == 0
    assert atmosphere_metadata["profile_calibration_status"] == "uncalibrated"
    assert atmosphere_metadata["co2_greenhouse_temperature_offset_c"] > 10.0
    assert climate_metadata["composition_greenhouse_offset_c"] == pytest.approx(
        atmosphere_metadata["co2_greenhouse_temperature_offset_c"]
    )


def test_atmosphere_config_rejects_only_invalid_physical_contracts():
    config = AtmosphereConfig.from_mapping(
        {"validation_profile": "snowball", "carbon_dioxide_ppm": 50.0}
    )
    assert config.validation_profile == "snowball"
    with pytest.raises(ValueError, match="Unknown atmosphere controls"):
        AtmosphereConfig.from_mapping({"paint_habitable": True})
    with pytest.raises(ValueError, match="fractions exceed one"):
        AtmosphereConfig.from_mapping({"oxygen_dry_fraction": 0.999, "carbon_dioxide_ppm": 2_000.0})
    with pytest.raises(ValueError, match="one of"):
        AtmosphereConfig.from_mapping({"validation_profile": "earth_only"})
