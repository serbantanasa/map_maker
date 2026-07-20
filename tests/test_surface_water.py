from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline import _surface_water_native as native
from map_maker.pipeline.stages.surface_water import SurfaceWaterConfig


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "planet",
        "atmosphere",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "sea_level",
        "climate",
        "cryosphere",
        "hydrology",
        "basin_refinement",
        "basin_erosion",
        "hydrology_pass2",
        "surface_water",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(tmp_path: Path, run_id: str, *, root: str = "primary") -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 16}],
            "rng_seed": 22,
            "run_id": run_id,
            "output_dir": str(tmp_path / root / "out"),
            "cache_dir": str(tmp_path / root / "cache"),
            "log_dir": str(tmp_path / root / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 14,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 3,
                },
                "world_age": {"world_age": 4.1},
                "climate": {
                    "spinup_years": 10,
                    "moisture_spinup_years": 2,
                    "moisture_steps_per_month_at_face_128": 16,
                },
                "basin_refinement": {
                    "refinement_factor": 4,
                    "terrain_noise_fraction": 0.4,
                },
                "basin_erosion": {
                    "minimum_bed_slope": 1e-5,
                    "maximum_deposition_fraction": 0.35,
                    "deposition_slope_scale": 0.001,
                    "maximum_deposition_depth_m": 10.0,
                },
                "hydrology_pass2": {
                    "minimum_depression_depth_m": 5.0,
                    "maximum_receiver_change_fraction": 0.15,
                    "maximum_receiver_change_cell_fraction": 0.15,
                    "maximum_new_depression_area_fraction": 0.02,
                },
                "surface_water": {
                    "minimum_solver_iterations": 8,
                    "maximum_solver_iterations": 64,
                    "maximum_connected_inundation_fraction": 0.25,
                    "outlet_erosion_score_threshold": 0.30,
                    "outlet_erosion_depth_scale_m": 200.0,
                    "minimum_outlet_erosion_discharge_m3s": 0.10,
                },
            },
        }
    )


def _table(result, name: str) -> pa.Table:
    value = result.artifact_records[name].value
    assert isinstance(value, pa.Table)
    return value


def _native_fixture() -> dict[str, object]:
    return {
        "controls": {
            "refinement_factor": 4,
            "minimum_solver_iterations": 4,
            "maximum_solver_iterations": 64,
            "transient_max_months": 3,
            "permanent_min_months": 12,
            "convergence_tolerance_fraction": 1e-8,
            "open_water_evaporation_factor": 1.0,
            "seepage_mm_year": 0.0,
            "subgrid_relief_scale": 1.0,
            "minimum_subgrid_relief_m": 10.0,
            "maximum_connected_inundation_fraction": 1.0,
            "minimum_wet_area_fraction": 0.01,
            "wetland_max_mean_depth_m": 3.0,
        },
        "cell_ids": np.array([0, 1, 2], dtype=np.int32),
        "receiver_ids": np.array([1, 2, -1], dtype=np.int32),
        "depression_ids": np.array([-1, 1, -1], dtype=np.int32),
        "source_active": np.ones(3, dtype=np.uint8),
        "area_km2": np.ones(3, dtype=np.float64),
        "terrain_elevation_m": np.array([2.0, -10.0, 2.0], dtype=np.float64),
        "hydrologic_elevation_m": np.array([2.0, 0.0, 2.0], dtype=np.float64),
        "parent_relief_m": np.full(3, 20.0, dtype=np.float32),
        "monthly_runoff_mm": np.full((12, 3), 100.0, dtype=np.float32),
        "monthly_evaporation_mm": np.zeros((12, 3), dtype=np.float32),
        "sediment_accommodation": np.full(3, 0.5, dtype=np.float32),
        "candidate_ids": np.array([1], dtype=np.int32),
        "spill_receiver_ids": np.array([2], dtype=np.int32),
    }


def test_native_surface_water_solves_periodic_fill_spill_across_cffi():
    candidates, cells, metadata = native.run_surface_water_balance(**_native_fixture())

    assert metadata["graph_valid"] == 1
    assert metadata["convergence_valid"] == 1
    assert metadata["water_balance_relative_error"] < 1e-12
    assert candidates["class_code"].tolist() == [3]
    assert candidates["wet_month_count"].tolist() == [12]
    assert candidates["annual_overflow_km3"][0] > 0.0
    assert candidates["solver_iterations"][0] <= 64
    assert cells["fine_cell_id"].tolist() == [1]
    assert np.all((cells["monthly_inundation_fraction"] >= 0.0))
    assert np.all((cells["monthly_inundation_fraction"] <= 1.0))


def test_native_surface_water_layout_and_sentinels_match_cffi():
    for kind, c_name in enumerate(
        (
            "SurfaceWaterConfig",
            "SurfaceWaterCandidateRecord",
            "SurfaceWaterCellRecord",
            "SurfaceWaterStats",
        )
    ):
        assert native._lib.surface_water_native_struct_size(kind) == native._ffi.sizeof(c_name)

    candidate = native._ffi.new("SurfaceWaterCandidateRecord*")
    candidate.depression_id = 17
    candidate.downstream_depression_id = 23
    candidate.annual_total_inflow_km3 = 1234.5
    candidate.monthly_water_area_km2[11] = 987.25
    candidate_view = native._ffi.buffer(
        candidate, native._ffi.sizeof("SurfaceWaterCandidateRecord")
    )
    parsed_candidate = np.frombuffer(candidate_view, dtype=native.CANDIDATE_DTYPE, count=1)[0]
    assert parsed_candidate["depression_id"] == 17
    assert parsed_candidate["downstream_depression_id"] == 23
    assert parsed_candidate["annual_total_inflow_km3"] == 1234.5
    assert parsed_candidate["monthly_water_area_km2"][11] == 987.25

    cell = native._ffi.new("SurfaceWaterCellRecord*")
    cell.fine_cell_id = 31
    cell.depression_id = 17
    cell.mean_inundation_fraction = 0.375
    cell.monthly_inundation_fraction[11] = 0.625
    cell_view = native._ffi.buffer(cell, native._ffi.sizeof("SurfaceWaterCellRecord"))
    parsed_cell = np.frombuffer(cell_view, dtype=native.CELL_DTYPE, count=1)[0]
    assert parsed_cell["fine_cell_id"] == 31
    assert parsed_cell["depression_id"] == 17
    assert parsed_cell["mean_inundation_fraction"] == 0.375
    assert parsed_cell["monthly_inundation_fraction"][11] == 0.625


def test_native_surface_water_rejects_zero_material_wet_area_threshold():
    fixture = _native_fixture()
    fixture["controls"] = {**fixture["controls"], "minimum_wet_area_fraction": 0.0}

    with pytest.raises(RuntimeError, match="invalid controls"):
        native.run_surface_water_balance(**fixture)


def test_zero_overflow_never_requests_outlet_erosion():
    module = importlib.import_module("map_maker.pipeline.stages.surface_water")
    config = SurfaceWaterConfig.from_mapping(
        {
            "outlet_erosion_score_threshold": 0.0,
            "minimum_outlet_erosion_discharge_m3s": 0.0,
        }
    )
    cells = pa.table(
        {
            "fine_cell_id": pa.array([1], type=pa.int32()),
            "area_km2": pa.array([1.0], type=pa.float64()),
        }
    )
    candidates = pa.table(
        {
            "depression_id": pa.array([7], type=pa.int32()),
            "maximum_fill_depth_m": pa.array([100.0], type=pa.float64()),
        }
    )
    candidate_records = np.zeros(1, dtype=native.CANDIDATE_DTYPE)
    candidate_records["class_code"] = 3
    candidate_records["mean_water_area_km2"] = 1.0
    cell_records = np.zeros(1, dtype=native.CELL_DTYPE)
    cell_records["fine_cell_id"] = 1
    cell_records["depression_id"] = 7
    cell_records["potential_inundation_fraction"] = 1.0

    _, _, erosion_required, recommended_incision, _, _ = module._outlet_erosion_feedback(
        config,
        cells,
        candidates,
        candidate_records,
        cell_records,
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )

    assert not erosion_required[0]
    assert recommended_incision[0] == 0.0


def test_surface_water_persists_fractional_monthly_state_and_feedback(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "surface-water"), generate_visuals=True)
    results = engine.run(["surface_water"])
    result = results["surface_water"]
    pass2 = results["hydrology_pass2"]
    metadata = result.artifact_records["SurfaceWaterMetadata"].value
    candidates = _table(result, "SurfaceWaterCandidateCatalog")
    cells = _table(result, "SeasonalSurfaceWaterCellCatalog")
    monthly = _table(result, "SurfaceWaterMonthlyStateCatalog")
    pass2_cells = _table(pass2, "StabilizedBasinCellCatalog")
    pass2_candidates = _table(pass2, "LocalDepressionCandidateCatalog")

    assert candidates.num_rows == pass2_candidates.num_rows == metadata["candidate_count"]
    assert monthly.num_rows == 12 * candidates.num_rows
    assert cells.num_rows == int(
        np.count_nonzero(np.asarray(pass2_cells["stabilized_depression_id"]) >= 0)
    )
    assert metadata["graph_valid"] == 1
    assert metadata["convergence_valid"] == 1
    assert metadata["direct_catchment_valid"] == 1
    assert metadata["runoff_inheritance_relative_error"] < 1e-12
    assert metadata["water_balance_relative_error"] < 1e-9
    assert metadata["independent_water_balance_relative_error"] < 1e-9
    assert metadata["maximum_area_reconstruction_relative_error"] < 1e-6
    assert metadata["downstream_candidate_mismatch_count"] == 0
    assert sum(metadata["published_class_counts"].values()) == candidates.num_rows
    assert metadata["surface_water_ready_for_soils"] == int(
        metadata["outlet_erosion_required_count"] == 0
    )

    fractions = np.asarray(
        cells["monthly_inundation_fraction"].combine_chunks().values, dtype=np.float32
    ).reshape(cells.num_rows, 12)
    assert np.all(np.isfinite(fractions))
    assert np.all((fractions >= 0.0) & (fractions <= 1.0))
    assert np.max(np.asarray(cells["potential_inundation_fraction"])) <= 0.25 + 1e-6
    assert set(candidates["surface_water_class"].to_pylist()) <= set(
        native.SURFACE_WATER_CLASSES.values()
    )
    required = np.asarray(candidates["outlet_erosion_required"])
    assert np.all(np.asarray(candidates["class_code"])[required] == 1)
    assert np.all(np.asarray(candidates["recommended_outlet_incision_m"])[~required] == 0.0)
    residual = (
        np.asarray(candidates["annual_total_inflow_km3"])
        - np.asarray(candidates["annual_evaporation_km3"])
        - np.asarray(candidates["annual_seepage_km3"])
        - np.asarray(candidates["annual_overflow_km3"])
        - np.asarray(candidates["annual_storage_change_km3"])
    )
    assert np.max(np.abs(residual)) < 1e-10
    visual = engine.context.config.run_visual_dir() / "surface_water"
    assert (visual / "seasonal_surface_water.png").is_file()


def test_surface_water_is_deterministic_and_cacheable(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "first", root="first")).run(["surface_water"])[
        "surface_water"
    ]
    second = ExecutionEngine(_config(tmp_path, "second", root="second")).run(["surface_water"])[
        "surface_water"
    ]
    for artifact in (
        "SurfaceWaterCandidateCatalog",
        "SeasonalSurfaceWaterCellCatalog",
        "SurfaceWaterMonthlyStateCatalog",
    ):
        assert _table(first, artifact).equals(_table(second, artifact))
    assert (
        first.artifact_records["SurfaceWaterMetadata"].value
        == second.artifact_records["SurfaceWaterMetadata"].value
    )
    cached = ExecutionEngine(_config(tmp_path, "second", root="second")).run(["surface_water"])[
        "surface_water"
    ]
    assert cached.stats is not None and cached.stats.cache_hit


def test_stage_rejects_corrupted_native_direct_inflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = importlib.import_module("map_maker.pipeline.stages.surface_water")
    native_run = module.run_surface_water_balance

    def corrupted_native_run(**kwargs):
        candidates, cells, metadata = native_run(**kwargs)
        candidates = candidates.copy()
        candidates["monthly_direct_inflow_km3"][0, 0] += 1.0
        return candidates, cells, metadata

    monkeypatch.setattr(module, "run_surface_water_balance", corrupted_native_run)
    with pytest.raises(RuntimeError, match="direct catchment inflow"):
        ExecutionEngine(_config(tmp_path, "corrupt")).run(["surface_water"])


def test_stage_rejects_corrupted_native_upstream_inflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = importlib.import_module("map_maker.pipeline.stages.surface_water")
    native_run = module.run_surface_water_balance

    def corrupted_native_run(**kwargs):
        candidates, cells, metadata = native_run(**kwargs)
        candidates = candidates.copy()
        candidates["monthly_upstream_inflow_km3"][0, 0] += 1.0
        return candidates, cells, metadata

    monkeypatch.setattr(module, "run_surface_water_balance", corrupted_native_run)
    with pytest.raises(RuntimeError, match="upstream overflow transfer"):
        ExecutionEngine(_config(tmp_path, "corrupt-upstream")).run(["surface_water"])


def test_surface_water_config_rejects_unknown_and_invalid_controls():
    with pytest.raises(ValueError, match="Unknown surface-water controls"):
        SurfaceWaterConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="solver iterations"):
        SurfaceWaterConfig.from_mapping({"maximum_solver_iterations": 0})
    with pytest.raises(ValueError, match="month thresholds"):
        SurfaceWaterConfig.from_mapping({"transient_max_months": 12})
    with pytest.raises(ValueError, match="month thresholds"):
        SurfaceWaterConfig.from_mapping({"permanent_min_months": 11})
    with pytest.raises(ValueError, match="maximum_connected_inundation_fraction"):
        SurfaceWaterConfig.from_mapping({"maximum_connected_inundation_fraction": 1.1})
    with pytest.raises(ValueError, match="minimum_wet_area_fraction"):
        SurfaceWaterConfig.from_mapping({"minimum_wet_area_fraction": 0.0})
