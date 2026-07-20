from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._sea_level_native import run_cubed_sphere_sea_level
from map_maker.pipeline.cubed_sphere import CubedSphereGrid


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    reg = registry()
    reg.clear()
    for module_name in (
        "geometry",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "sea_level",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield
    reg.clear()


def _outputs(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        name: np.empty(shape, dtype=np.float32)
        for name in (
            "ocean_mask_out",
            "ocean_fraction_out",
            "surface_elevation_out",
            "ocean_depth_out",
            "shelf_fraction_out",
            "coastal_mask_out",
            "inland_below_sea_level_out",
        )
    }


def _config(tmp_path: Path, *, face_resolution: int = 32, seed: int = 42) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": face_resolution}],
            "rng_seed": seed,
            "run_id": f"sea-level-test-{seed}",
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 20,
                    "continental_fraction": 0.42,
                    "lloyd_iterations": 4,
                },
                "world_age": {"world_age": 4.1},
                "sea_level": {"target_ocean_area_fraction": 0.65},
            },
        }
    )


def test_native_sea_level_hits_fraction_and_preserves_isolated_basin():
    grid = CubedSphereGrid.create(24)
    x = grid.xyz[..., 0]
    y = grid.xyz[..., 1]
    z = grid.xyz[..., 2]
    elevation = np.ascontiguousarray(
        1_500.0 * x + 500.0 * np.sin(5.0 * y) * np.cos(3.0 * z), dtype=np.float32
    )
    isolated = (x > 0.55) & (y > 0.25) & (z > 0.15)
    elevation[isolated] -= 2_000.0
    relief = np.full(grid.face_shape, 180.0, dtype=np.float32)
    outputs = _outputs(grid.face_shape)
    metadata = run_cubed_sphere_sea_level(
        target_ocean_area_fraction=0.65,
        shelf_depth_m=200.0,
        minimum_coastal_relief_m=40.0,
        coastal_relief_scale=0.45,
        areas=grid.cell_areas,
        neighbors=grid.neighbor_indices,
        elevation=elevation,
        relief=relief,
        **outputs,
    )
    ocean_fraction = outputs["ocean_fraction_out"]
    actual = float(np.sum(ocean_fraction * grid.cell_areas) / np.sum(grid.cell_areas))
    assert abs(actual - 0.65) < 1e-6
    assert np.all((ocean_fraction >= 0.0) & (ocean_fraction <= 1.0))
    np.testing.assert_allclose(
        outputs["surface_elevation_out"], elevation - metadata["sea_level_m"], atol=1e-5
    )
    assert metadata["below_level_component_count"] >= 1


def test_stage_separates_crust_from_surface_geography(tmp_path: Path):
    config = _config(tmp_path)
    engine = ExecutionEngine(config, generate_visuals=True)
    results = engine.run(["sea_level"])
    sea_level = results["sea_level"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)

    surface_ocean = np.asarray(sea_level.artifact_records["SurfaceOceanMask"].value.array())
    ocean_fraction = np.asarray(sea_level.artifact_records["SurfaceOceanFraction"].value.array())
    surface_elevation = np.asarray(sea_level.artifact_records["SurfaceElevationM"].value.array())
    crust_ocean = np.asarray(results["world_age"].artifact_records["BaseOceanMask"].value.array())
    metadata = sea_level.artifact_records["SeaLevelMetadata"].value

    assert surface_ocean.shape == grid.face_shape
    assert surface_elevation.shape == grid.face_shape
    np.testing.assert_array_equal(ocean_fraction >= 0.5, surface_ocean >= 0.5)
    assert not np.array_equal(surface_ocean >= 0.5, crust_ocean >= 0.5)
    actual = float(np.sum(ocean_fraction * grid.cell_areas) / np.sum(grid.cell_areas))
    assert abs(actual - 0.65) <= 1e-6
    assert metadata["significant_land_component_count"] >= 3
    assert metadata["largest_land_component_share"] < 0.75
    assert metadata["largest_land_component_coastline_complexity"] > 2.0
    assert metadata["coastline_edge_count"] > 100
    visual_dir = config.run_visual_dir() / "sea_level"
    assert (visual_dir / "surface_geography_global.png").is_file()
    assert (visual_dir / "shelves_and_inland_basins.png").is_file()


def test_stage_is_deterministic(tmp_path: Path):
    config = _config(tmp_path, face_resolution=24)
    first = ExecutionEngine(config).run(["sea_level"])["sea_level"]
    second = ExecutionEngine(config).run(["sea_level"])["sea_level"]
    for name in ("SurfaceOceanMask", "SurfaceOceanFraction", "SurfaceElevationM"):
        first_array = np.asarray(first.artifact_records[name].value.array())
        second_array = np.asarray(second.artifact_records[name].value.array())
        np.testing.assert_array_equal(first_array, second_array)


@pytest.mark.parametrize("seed", (7, 42, 101, 202, 404, 808))
def test_earthlike_seed_ensemble_avoids_compact_blob_worlds(tmp_path: Path, seed: int):
    result = ExecutionEngine(_config(tmp_path, seed=seed)).run(["sea_level"])["sea_level"]
    metadata = result.artifact_records["SeaLevelMetadata"].value

    assert abs(metadata["ocean_fractional_area_fraction"] - 0.65) <= 1e-6
    assert 3 <= metadata["significant_land_component_count"] <= 24
    assert metadata["largest_land_component_share"] < 0.78
    assert metadata["largest_land_component_coastline_complexity"] > 2.0
    assert metadata["coastline_edge_count"] > 300
    assert 0.01 < metadata["continental_shelf_area_fraction"] < 0.10
    assert metadata["largest_inland_basin_area_fraction"] < 0.02
