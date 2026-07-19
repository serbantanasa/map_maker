from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._climate_native import run_cubed_sphere_climate
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.climate import ClimateConfig
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_climate_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "planet",
        "atmosphere",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "climate",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(tmp_path: Path, run_id: str, *, seed: int = 17) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 20}],
            "rng_seed": seed,
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 14,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 3,
                },
                "world_age": {"world_age": 4.1},
                "climate": {
                    "spinup_years": 12,
                    "moisture_spinup_years": 2,
                    "moisture_steps_per_month_at_face_128": 16,
                    "moisture_diffusion_substeps_at_face_128": 2,
                    "synoptic_mixing_passes_at_face_128": 16,
                },
            },
        }
    )


def _array(result, name: str) -> np.ndarray:
    return np.asarray(result.artifact_records[name].value.array())


def _cross_face_gradient_ratio(values: np.ndarray, neighbors: np.ndarray) -> float:
    flat = values.reshape(-1)
    flat_neighbors = neighbors.reshape(-1, 4)
    face_size = values.shape[1] * values.shape[2]
    interior: list[float] = []
    cross_face: list[float] = []
    for source in range(flat.size):
        source_face = source // face_size
        for target_value in flat_neighbors[source]:
            target = int(target_value)
            if source >= target:
                continue
            delta = abs(float(flat[source] - flat[target]))
            if target // face_size == source_face:
                interior.append(delta)
            else:
                cross_face.append(delta)
    return float(np.mean(cross_face)) / max(float(np.mean(interior)), 1e-6)


def test_climate_outputs_seasonal_causal_fields_and_visuals(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "climate-basic"), generate_visuals=True)
    results = engine.run(["climate"])
    climate = results["climate"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)

    monthly_names = (
        "MonthlySurfaceTemperatureC",
        "MonthlyWindSpeedMps",
        "MonthlyPrecipitationMm",
        "MonthlyRelativeHumidity",
        "MonthlySnowfallMm",
        "MonthlySnowmeltMm",
        "MonthlySnowWaterEquivalentMm",
        "MonthlyEvaporationMm",
        "MonthlyRunoffPotentialMm",
    )
    monthly = {name: _array(climate, name) for name in monthly_names}
    for field in monthly.values():
        assert field.shape == (12, *grid.face_shape)
        assert field.dtype == np.float32
        assert np.all(np.isfinite(field))
    for name in monthly_names[1:]:
        assert np.all(monthly[name] >= 0.0)
    assert np.all(monthly["MonthlyRelativeHumidity"] <= 1.0)
    assert np.all(monthly["MonthlySnowfallMm"] <= monthly["MonthlyPrecipitationMm"] + 1e-4)
    assert np.all(
        monthly["MonthlyRunoffPotentialMm"]
        <= monthly["MonthlyPrecipitationMm"] + monthly["MonthlySnowmeltMm"] + 1e-4
    )

    wind = _array(climate, "MonthlyWindVectorXYZMps")
    assert wind.shape == (12, *grid.face_shape, 3)
    speed = monthly["MonthlyWindSpeedMps"]
    np.testing.assert_allclose(np.linalg.norm(wind, axis=-1), speed, rtol=2e-5, atol=2e-5)
    radial_leak = np.sum(wind * grid.xyz[None, ...], axis=-1)
    assert float(np.max(np.abs(radial_leak))) < 1e-4
    assert float(np.percentile(speed, 95)) > 5.0

    annual_temperature = _array(climate, "AnnualMeanTemperatureC")
    annual_precipitation = _array(climate, "AnnualPrecipitationMm")
    aridity = _array(climate, "AnnualAridityIndex")
    orography = _array(climate, "ClimateOrographyM")
    np.testing.assert_allclose(
        annual_temperature,
        np.mean(monthly["MonthlySurfaceTemperatureC"], axis=0),
        atol=1e-4,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        annual_precipitation,
        np.sum(monthly["MonthlyPrecipitationMm"], axis=0),
        atol=1e-3,
        rtol=1e-6,
    )
    assert np.all(np.isfinite(aridity)) and np.all(aridity >= 0.0)

    ocean = _array(results["world_age"], "BaseOceanMask") >= 0.5
    assert np.all(orography[ocean] == 0.0)
    assert np.all(orography[~ocean] >= 0.0)
    assert float(np.max(orography[~ocean])) > 500.0
    assert 100.0 < float(np.mean(annual_precipitation[~ocean])) < 2500.0

    land = ~ocean
    predictors = np.column_stack(
        (np.ones(np.count_nonzero(land)), np.sin(grid.latitude[land]) ** 2)
    )
    coefficients, *_ = np.linalg.lstsq(predictors, annual_temperature[land], rcond=None)
    temperature_residual = annual_temperature[land] - predictors @ coefficients
    assert float(np.corrcoef(temperature_residual, orography[land])[0, 1]) < -0.25

    midlatitude = (np.abs(grid.latitude) >= np.deg2rad(20.0)) & (
        np.abs(grid.latitude) <= np.deg2rad(50.0)
    )
    seasonal_range = np.ptp(monthly["MonthlySurfaceTemperatureC"], axis=0)
    assert float(np.median(seasonal_range[midlatitude & ocean])) < float(
        np.median(seasonal_range[midlatitude & land])
    )
    north_index = int(np.argmin(np.abs(grid.latitude.reshape(-1) - np.deg2rad(45.0))))
    south_index = int(np.argmin(np.abs(grid.latitude.reshape(-1) + np.deg2rad(45.0))))
    flattened_temperature = monthly["MonthlySurfaceTemperatureC"].reshape(12, -1)
    separation = abs(
        int(np.argmax(flattened_temperature[:, north_index]))
        - int(np.argmax(flattened_temperature[:, south_index]))
    )
    assert min(separation, 12 - separation) in {5, 6}

    assert _cross_face_gradient_ratio(annual_temperature, grid.neighbor_indices) < 2.5
    assert _cross_face_gradient_ratio(annual_precipitation, grid.neighbor_indices) < 3.0
    metadata = climate.artifact_records["ClimateMetadata"].value
    assert metadata["model"] == "seasonal_energy_moisture_climate_v5"
    assert metadata["composition_greenhouse_offset_c"] == pytest.approx(0.0)
    assert metadata["effective_greenhouse_offset_c"] == pytest.approx(0.0)
    assert metadata["effective_transport_steps_per_month"] == 2
    assert metadata["effective_moisture_diffusion_substeps"] == 1
    assert metadata["effective_moisture_steps_per_month"] == 2
    assert metadata["effective_moisture_advection_fraction"] == pytest.approx(0.38)
    assert metadata["effective_synoptic_mixing_passes"] == 1
    assert metadata["transport_reference_face_resolution"] == 128
    assert -10.0 < metadata["global_mean_temperature_c"] < 30.0
    assert 100.0 < metadata["global_mean_annual_precipitation_mm"] < 2500.0
    assert 0.0 < metadata["dry_land_area_fraction"] < 0.98

    visual_dir = engine.context.config.run_visual_dir() / "climate"
    for filename in (
        "annual_temperature.png",
        "annual_precipitation.png",
        "monthly_temperature.png",
        "monthly_precipitation.png",
        "prevailing_wind.png",
    ):
        assert (visual_dir / filename).is_file()


def test_climate_is_deterministic_and_cacheable(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "climate-first")).run(["climate"])["climate"]
    second = ExecutionEngine(_config(tmp_path, "climate-second")).run(["climate"])["climate"]
    for name in (
        "MonthlySurfaceTemperatureC",
        "MonthlyWindVectorXYZMps",
        "MonthlyPrecipitationMm",
        "MonthlyRunoffPotentialMm",
        "AnnualMeanTemperatureC",
        "AnnualPrecipitationMm",
    ):
        np.testing.assert_array_equal(_array(first, name), _array(second, name))
    assert second.stats is not None and second.stats.cache_hit


def _ffi_arguments(face_resolution: int = 6) -> dict[str, object]:
    grid = CubedSphereGrid.create(face_resolution)
    shape = grid.face_shape
    monthly_shape = (12, *shape)
    ocean = np.ascontiguousarray((grid.longitude < 0.0).astype(np.float32))
    elevation = np.ascontiguousarray(
        np.where(ocean >= 0.5, -3000.0, 1800.0 * np.maximum(grid.xyz[..., 2], 0.0)),
        dtype=np.float32,
    )
    insolation = np.empty(monthly_shape, dtype=np.float32)
    for month in range(12):
        seasonal = 80.0 * np.sin((month + 0.5) / 12.0 * 2.0 * np.pi) * np.sin(grid.latitude)
        insolation[month] = 330.0 + 90.0 * np.cos(grid.latitude) + seasonal
    outputs = {
        "climate_orography_out": np.empty(shape, dtype=np.float32),
        "temperature_out": np.empty(monthly_shape, dtype=np.float32),
        "wind_xyz_out": np.empty((*monthly_shape, 3), dtype=np.float32),
        "wind_speed_out": np.empty(monthly_shape, dtype=np.float32),
        "precipitation_out": np.empty(monthly_shape, dtype=np.float32),
        "humidity_out": np.empty(monthly_shape, dtype=np.float32),
        "snowfall_out": np.empty(monthly_shape, dtype=np.float32),
        "snowmelt_out": np.empty(monthly_shape, dtype=np.float32),
        "snowpack_out": np.empty(monthly_shape, dtype=np.float32),
        "evaporation_out": np.empty(monthly_shape, dtype=np.float32),
        "runoff_out": np.empty(monthly_shape, dtype=np.float32),
        "annual_temperature_out": np.empty(shape, dtype=np.float32),
        "annual_precipitation_out": np.empty(shape, dtype=np.float32),
        "aridity_out": np.empty(shape, dtype=np.float32),
    }
    return {
        "spinup_years": 4,
        "moisture_spinup_years": 2,
        "moisture_steps_per_month": 4,
        "synoptic_mixing_passes": 4,
        "greenhouse_offset_c": 0.0,
        "land_albedo": 0.30,
        "ocean_albedo": 0.28,
        "olr_intercept_w_m2": 203.0,
        "olr_slope_w_m2_c": 2.09,
        "heat_transport_w_m2": 35.0,
        "land_thermal_response": 0.22,
        "ocean_thermal_response": 0.10,
        "atmospheric_exchange": 0.16,
        "lapse_rate_c_per_km": 6.5,
        "wind_scale": 1.0,
        "moisture_advection_fraction": 0.38,
        "moisture_diffusion_fraction": 0.50,
        "orographic_factor": 0.18,
        "rain_shadow_factor": 0.70,
        "runoff_base_fraction": 0.35,
        "areas": grid.cell_areas,
        "neighbors": grid.neighbor_indices,
        "xyz": grid.xyz,
        "latitudes": grid.latitude,
        "elevation": elevation,
        "relief": np.ascontiguousarray(np.maximum(elevation, 100.0), dtype=np.float32),
        "ocean": ocean,
        "insolation": insolation,
        "declination": np.zeros(12, dtype=np.float32),
        **outputs,
    }


def test_climate_ffi_is_orography_sensitive_and_rejects_invalid_buffers():
    mountainous = _ffi_arguments()
    run_cubed_sphere_climate(**mountainous)
    flat = _ffi_arguments()
    flat["elevation"] = np.where(np.asarray(flat["ocean"]) >= 0.5, -3000.0, 0.0).astype(np.float32)
    run_cubed_sphere_climate(**flat)
    assert not np.array_equal(mountainous["temperature_out"], flat["temperature_out"])
    assert not np.array_equal(mountainous["precipitation_out"], flat["precipitation_out"])

    overlapping = _ffi_arguments()
    overlapping["annual_precipitation_out"] = overlapping["annual_temperature_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_cubed_sphere_climate(**overlapping)

    invalid_shape = _ffi_arguments()
    invalid_shape["wind_xyz_out"] = np.empty((12, 6, 6, 6, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        run_cubed_sphere_climate(**invalid_shape)


def test_climate_config_and_cli_reject_invalid_controls():
    with pytest.raises(ValueError, match="Unknown climate controls"):
        ClimateConfig.from_mapping({"paint_climate_zones": True})
    with pytest.raises(ValueError, match="lower than land"):
        ClimateConfig.from_mapping({"land_thermal_response": 0.1, "ocean_thermal_response": 0.2})
    with pytest.raises(ValueError, match="advection plus diffusion"):
        ClimateConfig.from_mapping(
            {"moisture_advection_fraction": 0.8, "moisture_diffusion_fraction": 0.4}
        )
    with pytest.raises(SystemExit):
        pipeline_tools_main(["--stage", "climate"])
