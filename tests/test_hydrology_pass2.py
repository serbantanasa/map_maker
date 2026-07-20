from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline import _hydrology_pass2_native as native
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.hydrology_pass2 import HydrologyPass2Config, _depression_catalog


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
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(
    tmp_path: Path,
    run_id: str,
    *,
    root: str = "primary",
    face_resolution: int = 16,
    rng_seed: int = 22,
) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": face_resolution}],
            "rng_seed": rng_seed,
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
            },
        }
    )


def _table(result, name: str) -> pa.Table:
    value = result.artifact_records[name].value
    assert isinstance(value, pa.Table)
    return value


def _native_fixture() -> dict[str, object]:
    grid = CubedSphereGrid.create(4)
    ids = np.array([row * 4 + col for row in range(3) for col in range(3)], dtype=np.int32)
    xyz = grid.xyz.reshape(-1, 3)[ids]
    anchors = np.zeros(9, dtype=np.uint8)
    anchors[[1, 4, 7]] = 1
    fixed = np.full(9, np.iinfo(np.int32).min, dtype=np.int32)
    fixed[1] = ids[4]
    fixed[4] = ids[7]
    fixed[7] = -1
    return {
        "controls": {
            "fine_resolution": 4,
            "minimum_depression_depth_m": 0.5,
            "planet_radius_m": 6_371_000.0,
        },
        "cell_ids": ids,
        "terrain_before_m": np.array([6, 5, 6, 0, 4, 5, 6, 3, 6], dtype=np.float64),
        "routing_surface_after_m": np.array([6, 5, 6, 0, -5, 5, 6, -10, 6], dtype=np.float64),
        "cell_areas_km2": np.ones(9, dtype=np.float64),
        "cell_xyz": xyz,
        "anchor_kinds": anchors,
        "source_active": np.ones(9, dtype=np.uint8),
        "fixed_receiver_ids": fixed,
    }


def test_native_pass2_preserves_trunk_and_conserves_area_across_cffi():
    records, metadata = native.run_hydrology_pass2(**_native_fixture())

    assert metadata["graph_valid"] == 1
    assert metadata["trunk_preserved_valid"] == 1
    assert metadata["physical_trunk_edge_count"] == 2
    assert metadata["baseline_depression_count"] == 1
    assert metadata["stabilized_depression_count"] == 0
    assert metadata["terminal_accumulated_area_km2"] == pytest.approx(9.0)
    assert metadata["contributing_area_residual_km2"] == pytest.approx(0.0, abs=1e-12)
    by_id = {int(record["fine_cell_id"]): record for record in records}
    fixture = _native_fixture()
    ids = fixture["cell_ids"]
    assert by_id[int(ids[1])]["stabilized_receiver_id"] == ids[4]
    assert by_id[int(ids[4])]["stabilized_receiver_id"] == ids[7]
    assert by_id[int(ids[7])]["stabilized_receiver_id"] == -1
    assert by_id[int(ids[3])]["baseline_fill_depth_m"] == pytest.approx(4.0)
    assert by_id[int(ids[3])]["stabilized_fill_depth_m"] == pytest.approx(0.0)
    assert by_id[int(ids[3])]["depression_changed"] == 1


def test_native_record_layout_round_trips_sentinel_values():
    record = native._ffi.new("StabilizedCellRecord*")
    record.fine_cell_id = 17
    record.stabilized_receiver_id = 23
    record.stabilized_hydrologic_elevation_m = 1000.00004
    record.contributing_area_km2 = 1234.5
    view = native._ffi.buffer(record, native._ffi.sizeof("StabilizedCellRecord"))
    parsed = np.frombuffer(view, dtype=native.STABILIZED_CELL_DTYPE, count=1)[0]

    assert parsed["fine_cell_id"] == 17
    assert parsed["stabilized_receiver_id"] == 23
    assert parsed["stabilized_hydrologic_elevation_m"] == pytest.approx(
        1000.00004, rel=0.0, abs=1e-12
    )
    assert parsed["contributing_area_km2"] == 1234.5


def test_native_pass2_rejects_nonadjacent_fixed_trunk_edge():
    fixture = _native_fixture()
    fixture["fixed_receiver_ids"] = fixture["fixed_receiver_ids"].copy()
    fixture["fixed_receiver_ids"][1] = fixture["cell_ids"][8]

    with pytest.raises(RuntimeError, match="invalid anchor or fixed trunk receiver"):
        native.run_hydrology_pass2(**fixture)


def test_depression_catalog_uses_final_exit_after_reentry():
    cells = pa.table(
        {
            "fine_cell_id": pa.array([0, 1, 2, 3], type=pa.int32()),
            "stabilized_receiver_id": pa.array([1, 2, 3, -1], type=pa.int32()),
            "baseline_depression_id": pa.array([-1, -1, -1, -1], type=pa.int32()),
            "stabilized_depression_id": pa.array([7, -1, 7, -1], type=pa.int32()),
            "area_km2": pa.array([1.0, 1.0, 1.0, 1.0], type=pa.float64()),
            "stabilized_fill_depth_m": pa.array([5.0, 0.0, 4.0, 0.0], type=pa.float64()),
            "stabilized_hydrologic_elevation_m": pa.array(
                [10.0, 8.0, 10.0, 7.0], type=pa.float64()
            ),
        }
    )

    catalog, _ = _depression_catalog(cells)

    assert catalog.num_rows == 1
    assert catalog["spill_cell_id"][0].as_py() == 2
    assert catalog["spill_receiver_id"][0].as_py() == 3


def test_depression_catalog_accepts_an_empty_candidate_set():
    cells = pa.table(
        {
            "fine_cell_id": pa.array([0, 1], type=pa.int32()),
            "stabilized_receiver_id": pa.array([1, -1], type=pa.int32()),
            "baseline_depression_id": pa.array([7, -1], type=pa.int32()),
            "stabilized_depression_id": pa.array([-1, -1], type=pa.int32()),
            "area_km2": pa.array([2.0, 3.0], type=pa.float64()),
            "stabilized_fill_depth_m": pa.array([0.0, 0.0], type=pa.float64()),
            "stabilized_hydrologic_elevation_m": pa.array([2.0, 1.0], type=pa.float64()),
        }
    )

    catalog, metadata = _depression_catalog(cells)

    assert catalog.num_rows == 0
    assert metadata == {
        "new_depression_area_km2": 0.0,
        "removed_depression_area_km2": 2.0,
        "new_depression_count": 0,
        "changed_depression_count": 0,
        "stable_depression_count": 0,
    }


def test_hydrology_pass2_stabilizes_real_connector_basin(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "pass2", rng_seed=4), generate_visuals=True)
    results = engine.run(["basin_erosion", "hydrology_pass2"])
    erosion = results["basin_erosion"]
    result = results["hydrology_pass2"]
    metadata = result.artifact_records["HydrologyPass2Metadata"].value
    cells = _table(result, "StabilizedBasinCellCatalog")
    reaches = _table(result, "StabilizedRiverReachCatalog")
    depressions = _table(result, "LocalDepressionCandidateCatalog")
    corrections = _table(result, "HydrologyCorrectionCatalog")
    profiles = _table(erosion, "ChannelBedProfileCatalog")

    assert metadata["graph_valid"] == 1
    assert metadata["independent_graph_valid"] == 1
    assert metadata["trunk_preserved_valid"] == 1
    assert metadata["process_exclusion_valid"] == 1
    assert metadata["independent_process_exclusion_valid"] == 1
    assert metadata["invalid_terminal_receiver_count"] == 0
    assert metadata["trunk_receiver_mismatch_count"] == 0
    assert metadata["baseline_uncovered_cell_count"] == 0
    assert metadata["stabilized_uncovered_cell_count"] == 0
    assert abs(metadata["independent_contributing_area_residual_km2"]) < 1e-8
    assert metadata["receiver_changed_area_fraction"] <= 0.15
    assert metadata["independent_receiver_changed_area_fraction"] <= 0.15
    assert (
        metadata["receiver_changed_cell_count"]
        == metadata["independent_receiver_changed_cell_count"]
    )
    assert metadata["receiver_changed_area_km2"] == pytest.approx(
        metadata["independent_receiver_changed_area_km2"]
    )
    assert metadata["new_depression_area_fraction"] <= 0.02
    assert cells.num_rows == metadata["cell_count"]
    assert reaches.num_rows > 0

    changed = np.asarray(cells["receiver_changed"]) | np.asarray(cells["depression_changed"])
    assert corrections.num_rows == np.count_nonzero(changed)
    excluded = np.asarray(cells["process_excluded"])
    assert np.any(excluded)
    assert np.all(np.asarray(cells["stabilized_depression_id"])[excluded] < 0)
    assert np.all(np.asarray(cells["stabilized_receiver_id"])[excluded] == -2)

    channel_ids = np.unique(np.asarray(profiles["fine_cell_id"], dtype=np.int32))
    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    cell_order = np.argsort(cell_ids)
    channel_rows = cell_order[np.searchsorted(cell_ids[cell_order], channel_ids)]
    assert np.all(np.asarray(cells["routing_anchor_kind"])[channel_rows] == "channel")
    assert not np.any(excluded[channel_rows])
    expected_beds = np.array(
        [
            np.asarray(profiles["bed_elevation_m"])[
                np.asarray(profiles["fine_cell_id"]) == cell_id
            ][0]
            for cell_id in channel_ids
        ]
    )
    assert np.allclose(
        np.asarray(cells["routing_surface_after_m"])[channel_rows],
        expected_beds,
        rtol=0.0,
        atol=1e-12,
    )
    if depressions.num_rows:
        assert np.all(np.asarray(depressions["area_km2"]) > 0.0)
        assert np.all(np.asarray(depressions["potential_fill_volume_km3"]) > 0.0)
    visual = engine.context.config.run_visual_dir() / "hydrology_pass2"
    assert (visual / "stabilized_drainage.png").is_file()


def test_hydrology_pass2_is_independently_deterministic_and_cacheable(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "first", root="first")).run(["hydrology_pass2"])[
        "hydrology_pass2"
    ]
    second = ExecutionEngine(_config(tmp_path, "second", root="second")).run(["hydrology_pass2"])[
        "hydrology_pass2"
    ]
    for artifact in (
        "StabilizedBasinCellCatalog",
        "StabilizedRiverReachCatalog",
        "LocalDepressionCandidateCatalog",
        "HydrologyCorrectionCatalog",
    ):
        assert _table(first, artifact).equals(_table(second, artifact))
    assert (
        first.artifact_records["HydrologyPass2Metadata"].value
        == second.artifact_records["HydrologyPass2Metadata"].value
    )

    cached = ExecutionEngine(_config(tmp_path, "second", root="second")).run(["hydrology_pass2"])[
        "hydrology_pass2"
    ]
    assert cached.stats is not None and cached.stats.cache_hit


def test_stage_rejects_corrupted_native_trunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("map_maker.pipeline.stages.hydrology_pass2")
    native_run = module.run_hydrology_pass2

    def corrupted_native_run(**kwargs):
        records, metadata = native_run(**kwargs)
        records = records.copy()
        channel = np.flatnonzero(records["anchor_kind"] == 1)
        records["stabilized_receiver_id"][channel[0]] = -2
        return records, metadata

    monkeypatch.setattr(module, "run_hydrology_pass2", corrupted_native_run)
    with pytest.raises(RuntimeError, match="changed the accepted physical trunk"):
        ExecutionEngine(_config(tmp_path, "corrupt")).run(["hydrology_pass2"])


def test_stage_rejects_corrupted_native_correction_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = importlib.import_module("map_maker.pipeline.stages.hydrology_pass2")
    native_run = module.run_hydrology_pass2

    def corrupted_native_run(**kwargs):
        records, metadata = native_run(**kwargs)
        metadata = {
            **metadata,
            "receiver_changed_cell_count": metadata["receiver_changed_cell_count"] + 1,
        }
        return records, metadata

    monkeypatch.setattr(module, "run_hydrology_pass2", corrupted_native_run)
    with pytest.raises(RuntimeError, match="correction budgets disagree"):
        ExecutionEngine(_config(tmp_path, "corrupt-budget")).run(["hydrology_pass2"])


def test_hydrology_pass2_config_rejects_unknown_and_invalid_controls():
    with pytest.raises(ValueError, match="Unknown Hydrology Pass 2 controls"):
        HydrologyPass2Config.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="minimum_depression_depth_m"):
        HydrologyPass2Config.from_mapping({"minimum_depression_depth_m": -1.0})
    with pytest.raises(ValueError, match="minimum_depression_depth_m"):
        HydrologyPass2Config.from_mapping({"minimum_depression_depth_m": 0.0})
    with pytest.raises(ValueError, match="maximum_receiver_change_fraction"):
        HydrologyPass2Config.from_mapping({"maximum_receiver_change_fraction": 1.1})
