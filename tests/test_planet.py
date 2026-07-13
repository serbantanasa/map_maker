from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._planet_native import run_cubed_sphere_planet
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.planet import PlanetConfig
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_planet_registered():
    registry().clear()
    for module_name in ("geometry", "planet"):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(tmp_path: Path, run_id: str, **planet_overrides: float) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 24}],
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {"planet": planet_overrides},
        }
    )


def _array(result, name: str) -> np.ndarray:
    return np.asarray(result.artifact_records[name].value.array())


def test_planet_outputs_earth_forcing_and_visuals(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "earth-forcing"), generate_visuals=True)
    results = engine.run(["planet"])
    planet = results["planet"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)

    monthly = _array(planet, "MonthlyInsolationWm2")
    daylight = _array(planet, "MonthlyDaylightHours")
    annual = _array(planet, "AnnualMeanInsolationWm2")
    seasonality = _array(planet, "InsolationSeasonalityWm2")
    polar_extreme = _array(planet, "PolarLightExtremeFraction")
    orbital_distance = _array(planet, "OrbitalDistanceAU")
    declination = _array(planet, "SolarDeclinationRad")

    assert monthly.shape == (12, *grid.face_shape)
    assert daylight.shape == monthly.shape
    for field in (monthly, daylight, annual, seasonality, polar_extreme):
        assert field.dtype == np.float32
        assert np.all(np.isfinite(field))
    assert np.all(monthly >= 0.0)
    assert np.all((daylight >= 0.0) & (daylight <= 24.0))
    assert np.all((polar_extreme >= 0.0) & (polar_extreme <= 1.0))
    np.testing.assert_allclose(annual, np.mean(monthly, axis=0), rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(seasonality, np.ptp(monthly, axis=0), rtol=0.0, atol=1e-4)

    global_mean = float(np.average(annual, weights=grid.cell_areas))
    assert global_mean == pytest.approx(340.3, abs=1.0)
    equatorial = np.abs(grid.latitude) <= np.deg2rad(15.0)
    polar = np.abs(grid.latitude) >= np.deg2rad(66.5)
    assert float(np.mean(annual[equatorial])) > float(np.mean(annual[polar])) + 150.0
    assert float(np.max(polar_extreme[polar])) > 0.0
    assert float(np.max(polar_extreme[equatorial])) == 0.0

    latitude_flat = grid.latitude.reshape(-1)
    north_index = int(np.argmin(np.abs(latitude_flat - np.deg2rad(45.0))))
    south_index = int(np.argmin(np.abs(latitude_flat + np.deg2rad(45.0))))
    north_peak = int(np.argmax(monthly.reshape(12, -1)[:, north_index]))
    south_peak = int(np.argmax(monthly.reshape(12, -1)[:, south_index]))
    separation = abs(north_peak - south_peak)
    assert min(separation, 12 - separation) in {5, 6}

    assert orbital_distance.shape == (12,)
    assert declination.shape == (12,)
    assert float(np.min(orbital_distance)) == pytest.approx(0.9837, abs=0.002)
    assert float(np.max(orbital_distance)) == pytest.approx(1.0163, abs=0.002)
    assert float(np.max(np.abs(declination))) == pytest.approx(np.deg2rad(23.44), abs=0.01)

    metadata = planet.artifact_records["PlanetMetadata"].value
    assert metadata["model"] == "keplerian_orbital_forcing_v1"
    assert metadata["forcing_semantics"] == "twelve_equal_time_monthly_daily_means"
    assert metadata["global_mean_insolation_w_m2"] == pytest.approx(global_mean, abs=1e-3)
    assert metadata["tide_strength_index"] == pytest.approx(1.0)
    assert metadata["obliquity_stability_index"] > 0.5

    visual_dir = engine.context.config.run_visual_dir() / "planet"
    assert (visual_dir / "annual_mean_insolation.png").is_file()
    assert (visual_dir / "insolation_seasonality.png").is_file()
    assert (visual_dir / "monthly_insolation.png").is_file()


def test_planet_is_deterministic_and_cacheable(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "planet-first")).run(["planet"])["planet"]
    second = ExecutionEngine(_config(tmp_path, "planet-second")).run(["planet"])["planet"]
    for name in (
        "MonthlyInsolationWm2",
        "MonthlyDaylightHours",
        "AnnualMeanInsolationWm2",
        "InsolationSeasonalityWm2",
        "PolarLightExtremeFraction",
        "OrbitalDistanceAU",
        "SolarDeclinationRad",
    ):
        np.testing.assert_array_equal(_array(first, name), _array(second, name))
    assert second.stats is not None and second.stats.cache_hit


def _ffi_arguments(face_resolution: int = 6) -> dict[str, object]:
    grid = CubedSphereGrid.create(face_resolution)
    shape = grid.face_shape
    return {
        "star_luminosity_solar": 1.0,
        "semi_major_axis_au": 1.0,
        "eccentricity": 0.0167,
        "obliquity_radians": np.deg2rad(23.44),
        "rotation_period_hours": 24.0,
        "orbital_period_days": 365.2422,
        "perihelion_day": 3.0,
        "northern_vernal_equinox_day": 79.0,
        "moon_mass_lunar": 1.0,
        "moon_distance_km": 384_400.0,
        "areas": grid.cell_areas,
        "latitudes": grid.latitude,
        "monthly_insolation_out": np.empty((12, *shape), dtype=np.float32),
        "monthly_daylight_out": np.empty((12, *shape), dtype=np.float32),
        "annual_mean_out": np.empty(shape, dtype=np.float32),
        "seasonality_out": np.empty(shape, dtype=np.float32),
        "polar_extreme_fraction_out": np.empty(shape, dtype=np.float32),
        "orbital_distance_out": np.empty(12, dtype=np.float32),
        "solar_declination_out": np.empty(12, dtype=np.float32),
    }


def test_planet_ffi_rejects_overlapping_and_invalid_buffers():
    arguments = _ffi_arguments()
    arguments["seasonality_out"] = arguments["annual_mean_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_cubed_sphere_planet(**arguments)

    arguments = _ffi_arguments()
    arguments["monthly_insolation_out"] = np.empty((11, 6, 6, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        run_cubed_sphere_planet(**arguments)

    arguments = _ffi_arguments()
    arguments["latitudes"] = np.asarray(arguments["latitudes"], dtype=np.float32)
    with pytest.raises(ValueError, match="must be float64"):
        run_cubed_sphere_planet(**arguments)


def test_planet_config_and_cli_reject_invalid_controls(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown planet controls"):
        PlanetConfig.from_mapping({"painted_latitude_bands": 3})
    with pytest.raises(ValueError, match="must be in"):
        PlanetConfig.from_mapping({"eccentricity": 0.7})
    with pytest.raises(ValueError, match="must be in"):
        PlanetConfig.from_mapping({"star_luminosity_solar": 0.2, "semi_major_axis_au": 1.0})

    rectangular = PipelineConfig.from_mapping(
        {
            "topology": "sphere",
            "resolutions": [{"height": 16, "width": 32}],
            "run_id": "rectangular-planet",
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
        }
    )
    with pytest.raises(NotImplementedError, match="cubed_sphere"):
        ExecutionEngine(rectangular).run(["planet"])
    with pytest.raises(SystemExit):
        pipeline_tools_main(["--stage", "planet"])
