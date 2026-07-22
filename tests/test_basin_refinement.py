from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.basin_refinement import (
    BasinRefinementConfig,
    _bounded_unresolved_basin_targets,
    _expand_parent_halo,
    _extend_hydraulic_surface_over_submerged_approaches,
)


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
                    # The face-20 fixture has intentionally abrupt, continent-scale
                    # parent means; this exercises convergence rather than the
                    # face-128 Earthlike correction-amplitude calibration.
                    "maximum_center_correction_scale_fraction": 10.0,
                    "maximum_tile_bubble_correlation_p50": 0.50,
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
    assert metadata["conditioning_converged"] == 1
    assert metadata["terrain_parent_mean_conditioning_valid"] == 1
    assert metadata["maximum_parent_elevation_error_m"] <= metadata["maximum_parent_mean_error_m"]
    assert (
        metadata["maximum_parent_elevation_error_relief_fraction"]
        <= metadata["maximum_parent_mean_error_relief_fraction"]
    )
    assert metadata["terrain_tile_motif_valid"] == 1
    assert metadata["terrain_local_relief_envelope_valid"] == 1
    assert metadata["terrain_parent_offset_span_max_m"] <= metadata["maximum_parent_offset_span_m"]
    assert (
        metadata["terrain_parent_offset_span_relief_fraction_max"]
        <= metadata["maximum_parent_offset_span_relief_fraction"]
    )
    assert metadata["unresolved_basin_depth_bound_valid"] == 1
    assert metadata["terrain_parent_boundary_edge_count"] > 0
    assert metadata["terrain_parent_boundary_continuity_valid"] == 1
    assert metadata["terrain_parent_boundary_residual_p95_ratio"] <= 2.0

    assert len(set(cells["fine_cell_id"].to_pylist())) == cells.num_rows
    assert np.max(np.abs(np.asarray(parents["area_relative_error"]))) < 1e-10
    assert (
        np.max(np.abs(np.asarray(parents["elevation_error_m"])))
        <= metadata["maximum_parent_mean_error_m"]
    )
    assert np.any(np.asarray(cells["terrain_offset_m"]) > 0.0)
    assert np.any(np.asarray(cells["terrain_offset_m"]) < 0.0)
    assert "process_excluded" in cells.column_names
    assert "process_excluded" in parents.column_names
    assert "standing_water_fraction" in cells.column_names
    assert "channel_surface_prior_m" in cells.column_names
    assert "hydraulic_surface_controlled" in cells.column_names
    assert "standing_water_fraction" in parents.column_names
    assert "channel_surface_prior_m" in parents.column_names
    assert "hydraulic_surface_controlled" in parents.column_names
    assert "source_parent_elevation_m" in parents.column_names
    assert "unresolved_basin_depth_adjusted" in parents.column_names
    assert (
        int(np.count_nonzero(np.asarray(parents["process_excluded"])))
        == metadata["process_excluded_parent_count"]
    )
    channel_surface = np.asarray(cells["channel_surface_prior_m"], dtype=np.float32)
    terrain_elevation = np.asarray(cells["terrain_elevation_m"], dtype=np.float32)
    assert np.all(np.isfinite(channel_surface))
    controlled = np.asarray(cells["hydraulic_surface_controlled"], dtype=bool)
    np.testing.assert_array_equal(channel_surface[~controlled], terrain_elevation[~controlled])
    assert metadata["fractional_water_channel_prior_parent_count"] == int(
        np.count_nonzero(np.asarray(parents["standing_water_fraction"]) > 0.0)
    )
    assert metadata["hydraulic_surface_prior_parent_count"] == int(
        np.count_nonzero(np.asarray(parents["hydraulic_surface_controlled"]))
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
    with pytest.raises(ValueError, match="terrain_base_wavelength_m"):
        BasinRefinementConfig.from_mapping({"terrain_base_wavelength_m": 0.0})
    with pytest.raises(ValueError, match="terrain_octave_count"):
        BasinRefinementConfig.from_mapping({"terrain_octave_count": 1})
    with pytest.raises(ValueError, match="conditioning_sigma_parent_cells"):
        BasinRefinementConfig.from_mapping({"conditioning_sigma_parent_cells": 3.0})
    with pytest.raises(ValueError, match="halo_parent_rings"):
        BasinRefinementConfig.from_mapping({"halo_parent_rings": 9})
    with pytest.raises(ValueError, match="maximum_parent_boundary"):
        BasinRefinementConfig.from_mapping({"maximum_parent_boundary_residual_p95_ratio": 0.0})


def test_parent_halo_expands_over_topology_neighbors():
    neighbors = np.asarray(
        [
            [1, 2, 1, 2],
            [0, 3, 0, 3],
            [0, 3, 0, 3],
            [1, 2, 1, 2],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(_expand_parent_halo(np.asarray([0]), neighbors, 0), [0])
    np.testing.assert_array_equal(_expand_parent_halo(np.asarray([0]), neighbors, 1), [0, 1, 2])
    np.testing.assert_array_equal(_expand_parent_halo(np.asarray([0]), neighbors, 2), [0, 1, 2, 3])


def test_unresolved_hydraulic_basin_depth_is_bounded_without_losing_source() -> None:
    source = np.asarray([-4_000.0, -500.0, -4_000.0, -4_000.0], dtype=np.float32)
    relief = np.asarray([300.0, 300.0, 800.0, 300.0], dtype=np.float32)
    hydraulic_surface = np.asarray([200.0, 200.0, 200.0, 200.0], dtype=np.float32)
    controlled = np.asarray([True, True, True, False])
    target, adjusted, depth_limit = _bounded_unresolved_basin_targets(
        source,
        relief,
        hydraulic_surface,
        controlled,
        maximum_depth_m=1_200.0,
        relief_depth_multiplier=4.0,
    )
    np.testing.assert_array_equal(adjusted, [True, False, True, False])
    np.testing.assert_allclose(depth_limit, [1_200.0, 1_200.0, 3_200.0, 1_200.0])
    np.testing.assert_allclose(target, [-1_000.0, -500.0, -3_000.0, -4_000.0])
    np.testing.assert_allclose(source, [-4_000.0, -500.0, -4_000.0, -4_000.0])


def test_submerged_channel_approach_inherits_receiving_water_surface():
    fine_cell_ids = np.asarray([10, 11, 12, 13, 20, 21, 29, 30, 31], dtype=np.int32)
    prior = np.asarray([-50.0, 20.0, -100.0, 0.0, -100.0, 0.0, -100.0, -50.0, 0.0])
    controlled = np.asarray([False, False, False, True, False, True, False, False, True])
    reaches = pa.table(
        {
            "reach_kind": pa.array(["channel", "connector", "channel", "channel"]),
            "fine_cell_path": pa.array(
                [[10, 11, 12, 13], [20, 21], [29, 30], [30, 31]],
                type=pa.list_(pa.int32()),
            ),
        }
    )

    conditioned, updated_control, approach_count = (
        _extend_hydraulic_surface_over_submerged_approaches(
            fine_cell_ids, prior, controlled, reaches
        )
    )

    np.testing.assert_array_equal(
        conditioned,
        np.asarray([-50.0, 20.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        updated_control,
        np.asarray([False, False, True, True, False, True, True, True, True]),
    )
    assert approach_count == 3
