from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.basin_refinement import BasinRefinementConfig


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "planet",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "climate",
        "hydrology",
        "basin_refinement",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(tmp_path: Path, run_id: str, *, seed: int = 37) -> PipelineConfig:
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
                    "spinup_years": 10,
                    "moisture_spinup_years": 2,
                    "moisture_steps_per_month_at_face_128": 16,
                },
                "basin_refinement": {
                    "refinement_factor": 4,
                    "terrain_noise_fraction": 0.4,
                },
            },
        }
    )


def _table(result, name: str) -> pa.Table:
    value = result.artifact_records[name].value
    assert isinstance(value, pa.Table)
    return value


def _collapsed(values: np.ndarray) -> list[int]:
    return [
        int(value) for index, value in enumerate(values) if index == 0 or value != values[index - 1]
    ]


def test_refines_complete_basin_without_cell_wide_rivers(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "basin-refinement"), generate_visuals=True)
    result = engine.run(["basin_refinement"])["basin_refinement"]
    metadata = result.artifact_records["BasinRefinementMetadata"].value
    cells = _table(result, "RefinedBasinCellCatalog")
    parents = _table(result, "RefinedBasinParentCatalog")
    reaches = _table(result, "RefinedRiverReachCatalog")
    memberships = _table(result, "RefinedReachCellCatalog")

    factor = int(metadata["refinement_factor"])
    fine_resolution = int(metadata["fine_resolution"])
    assert factor == 4
    assert fine_resolution == 80
    assert metadata["path_topology_valid"] == 1
    assert metadata["junction_merge_valid"] == 1
    assert metadata["reverse_directed_edge_conflict_count"] == 0
    assert metadata["directed_path_dag_valid"] == 1
    assert metadata["directed_path_graph_valid"] == 1
    assert metadata["corridor_cell_capacity_valid"] == 1
    assert metadata["nested_corridor_support_valid"] == 1
    assert metadata["process_exclusion_valid"] == 1
    assert metadata["inherited_discharge_relative_error"] == 0.0
    assert metadata["parent_count"] == parents.num_rows
    assert metadata["child_count"] == cells.num_rows == parents.num_rows * factor * factor
    assert metadata["reach_count"] == reaches.num_rows > 0
    assert metadata["channel_reach_count"] + metadata["connector_reach_count"] == reaches.num_rows
    assert metadata["reach_cell_count"] == memberships.num_rows > 0
    assert metadata["maximum_parent_area_relative_error"] < 1e-10
    assert metadata["maximum_parent_elevation_error_m"] < 1e-3

    assert len(set(cells["fine_cell_id"].to_pylist())) == cells.num_rows
    assert np.max(np.abs(np.asarray(parents["area_relative_error"]))) < 1e-10
    assert np.max(np.abs(np.asarray(parents["elevation_error_m"]))) < 1e-3
    assert np.any(np.asarray(cells["terrain_offset_m"]) > 0.0)
    assert np.any(np.asarray(cells["terrain_offset_m"]) < 0.0)
    assert "process_excluded" in cells.column_names
    assert "process_excluded" in parents.column_names
    assert (
        int(np.count_nonzero(np.asarray(parents["process_excluded"])))
        == metadata["process_excluded_parent_count"]
    )

    fine_grid = CubedSphereGrid.create(fine_resolution)
    fine_to_parent = fine_grid.parent_map(factor).reshape(-1)
    fine_neighbors = fine_grid.neighbor_indices.reshape(-1, 4)
    path_lengths_by_reach: dict[int, float] = {}
    directed_edges: set[tuple[int, int]] = set()
    for row in range(reaches.num_rows):
        fine_path = np.asarray(reaches["fine_cell_path"][row].as_py(), dtype=np.int32)
        parent_path = reaches["parent_cell_path"][row].as_py()
        assert len(fine_path) >= len(parent_path)
        assert _collapsed(fine_to_parent[fine_path]) == parent_path
        assert np.all(
            [
                target in fine_neighbors[source]
                for source, target in zip(fine_path[:-1], fine_path[1:])
            ]
        )
        reach_id = int(reaches["reach_id"][row].as_py())
        path_lengths_by_reach[reach_id] = float(reaches["path_length_km"][row].as_py()) * 1_000.0
        directed_edges.update(
            (int(source), int(target)) for source, target in zip(fine_path[:-1], fine_path[1:])
        )
    assert not any((target, source) in directed_edges for source, target in directed_edges)

    channel_fraction = np.asarray(memberships["channel_fraction"])
    valley_fraction = np.asarray(memberships["valley_fraction"])
    floodplain_fraction = np.asarray(memberships["floodplain_fraction"])
    assert np.all((channel_fraction >= 0.0) & (channel_fraction < 0.01))
    assert np.all(channel_fraction <= floodplain_fraction + 1e-7)
    assert np.all(floodplain_fraction <= valley_fraction + 1e-7)
    assert np.all(channel_fraction <= valley_fraction)
    assert np.all((floodplain_fraction >= 0.0) & (floodplain_fraction <= 1.0))

    membership_reach = np.asarray(memberships["reach_id"], dtype=np.int32)
    membership_length = np.asarray(memberships["reach_length_m"], dtype=np.float64)
    potential_volume = np.asarray(memberships["potential_incised_volume_m3"], dtype=np.float64)
    width_by_reach = dict(
        zip(
            np.asarray(reaches["reach_id"], dtype=np.int32),
            np.asarray(reaches["channel_width_m"], dtype=np.float64),
            strict=True,
        )
    )
    incision_by_reach = dict(
        zip(
            np.asarray(reaches["reach_id"], dtype=np.int32),
            np.asarray(reaches["incision_m"], dtype=np.float64),
            strict=True,
        )
    )
    for reach_id, expected_length in path_lengths_by_reach.items():
        selected = membership_reach == reach_id
        actual_length = float(np.sum(membership_length[selected]))
        row = int(np.flatnonzero(np.asarray(reaches["reach_id"]) == reach_id)[0])
        physical_length = float(reaches["physical_channel_length_km"][row].as_py()) * 1_000.0
        assert physical_length <= expected_length
        assert actual_length == pytest.approx(physical_length, rel=1e-12)
        assert float(np.sum(potential_volume[selected])) == pytest.approx(
            width_by_reach[reach_id] * actual_length * incision_by_reach[reach_id], rel=1e-6
        )

    cell_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    cell_areas = np.asarray(cells["area_km2"], dtype=np.float64)
    cell_order = np.argsort(cell_ids)
    sorted_cell_ids = cell_ids[cell_order]
    membership_cell_ids = np.asarray(memberships["fine_cell_id"], dtype=np.int32)
    membership_areas = cell_areas[cell_order[np.searchsorted(sorted_cell_ids, membership_cell_ids)]]
    for corridor in ("channel", "valley", "floodplain"):
        represented = float(
            np.sum(np.asarray(memberships[f"{corridor}_fraction"]) * membership_areas)
        )
        assert metadata[f"represented_{corridor}_area_km2"] == pytest.approx(represented, rel=1e-12)
        assert metadata[f"requested_{corridor}_area_km2"] >= represented
        assert 1.0 - 1e-6 < metadata[f"{corridor}_area_retention_fraction"] <= 1.0 + 1e-7
        _, inverse = np.unique(membership_cell_ids, return_inverse=True)
        per_cell = np.bincount(
            inverse, weights=np.asarray(memberships[f"{corridor}_fraction"], dtype=np.float64)
        )
        assert np.max(per_cell, initial=0.0) <= 1.0 + 1e-6

    assert {
        "terminal_kind",
        "downstream_join_fine_cell",
        "reach_kind",
        "terminal_parent_cell",
        "terminal_receiver_cell",
        "terminal_sink_type",
        "terminal_resolved",
    }.issubset(reaches.column_names)
    assert "support_role" in memberships.column_names
    lateral = np.asarray(memberships["support_role"]) == "lateral"
    assert np.all(membership_length[lateral] == 0.0)
    assert np.all(channel_fraction[lateral] == 0.0)
    assert np.all(potential_volume[lateral] == 0.0)
    connector_ids = set(
        np.asarray(reaches["reach_id"], dtype=np.int32)[
            np.asarray(reaches["reach_kind"]) == "connector"
        ].tolist()
    )
    if connector_ids:
        connector_memberships = np.isin(membership_reach, list(connector_ids))
        assert not np.any(connector_memberships)
        connector_parent_cells: set[int] = set()
        for path, kind in zip(
            reaches["parent_cell_path"].to_pylist(), reaches["reach_kind"].to_pylist(), strict=True
        ):
            if kind == "connector":
                connector_parent_cells.update(int(cell) for cell in path[:-1])
        assert not connector_parent_cells.intersection(
            np.asarray(memberships["parent_cell_id"], dtype=np.int32).tolist()
        )
        connector_rows = np.asarray(reaches["reach_kind"]) == "connector"
        assert np.all(np.asarray(reaches["physical_channel_length_km"])[connector_rows] == 0.0)
    unresolved = [
        kind for kind in reaches["terminal_kind"].to_pylist() if kind.startswith("unresolved_")
    ]
    assert metadata["source_to_sink_ready"] == int(not unresolved)

    visual = engine.context.config.run_visual_dir() / "basin_refinement" / "refined_basin.png"
    assert visual.is_file()


def test_refinement_is_deterministic_and_cacheable(tmp_path: Path):
    first_config = _config(tmp_path / "independent-first", "refinement-first")
    second_config = _config(tmp_path / "independent-second", "refinement-second")
    first = ExecutionEngine(first_config).run(["basin_refinement"])["basin_refinement"]
    second = ExecutionEngine(second_config).run(["basin_refinement"])["basin_refinement"]
    for name in (
        "RefinedBasinCellCatalog",
        "RefinedBasinParentCatalog",
        "RefinedRiverReachCatalog",
        "RefinedReachCellCatalog",
    ):
        assert _table(first, name).equals(_table(second, name))
    assert (
        first.artifact_records["BasinRefinementMetadata"].value
        == second.artifact_records["BasinRefinementMetadata"].value
    )
    assert second.stats is not None and not second.stats.cache_hit
    cached = ExecutionEngine(second_config).run(["basin_refinement"])["basin_refinement"]
    assert cached.stats is not None and cached.stats.cache_hit


def test_refinement_config_rejects_invalid_controls():
    with pytest.raises(ValueError, match="Unknown basin refinement controls"):
        BasinRefinementConfig.from_mapping({"carve_whole_cells": True})
    with pytest.raises(ValueError, match="power of two"):
        BasinRefinementConfig.from_mapping({"refinement_factor": 3})
    with pytest.raises(ValueError, match="basin_id"):
        BasinRefinementConfig.from_mapping({"basin_id": -1})
    with pytest.raises(ValueError, match="terrain_noise_fraction"):
        BasinRefinementConfig.from_mapping({"terrain_noise_fraction": 1.1})
