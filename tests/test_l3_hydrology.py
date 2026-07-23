from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from map_maker.cli import main
from map_maker.pipeline._hydrology_native import run_regional_hydrology
from map_maker.pipeline.l3_hydrology import (
    L3HydrologyConfig,
    L3HydrologyResult,
    _adjacent_d8,
    _crop_receiver_grid,
    _d8_neighbors,
    _discharge_continuity_metrics,
    _downscale_monthly,
    _forcing_error,
    _flow_arrow_cells,
    _hillshaded_terrain,
    _outer_boundary_mask,
    _reach_support_masks,
    _refined_target_basin,
    _realize_parent_fraction,
    _topographic_weights,
)


def test_cli_reports_process_coverage_honestly(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        "output_dir: target\nterrain_output_dir: terrain\nhydrology_output_dir: hydro\n",
        encoding="utf8",
    )
    result = L3HydrologyResult(
        output_dir=tmp_path / "hydro",
        manifest_path=tmp_path / "hydro/manifest.json",
        validation_path=tmp_path / "hydro/validation.json",
        zarr_path=tmp_path / "hydro/hydrology.zarr",
        preview_path=tmp_path / "hydro/hydrology.png",
        target_id="test-catchment",
        cell_count=1_000,
        process_cell_count=1_000,
        display_cell_count=600,
        hidden_routing_halo_cell_count=400,
        river_reach_count=42,
        lake_count=7,
        validation_passed=True,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.l3_hydrology.generate_l3_hydrology",
        lambda _config: result,
    )

    assert main(["l3-hydrology", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "all 600 displayed cells" in output
    assert "1000 routed stored cells" in output
    assert "400 hidden-halo cells" in output


def _controls() -> dict[str, int | float]:
    return {
        "planet_radius_m": 6_371_000.0,
        "minimum_depression_depth_m": 1.0,
        "wetland_mean_depth_m": 2.0,
        "endorheic_aridity_threshold": 0.35,
        "maximum_fill_time_years": 10_000.0,
        "lake_seepage_mm_year": 30.0,
        "subgrid_relief_scale": 0.5,
        "subgrid_connected_basin_fraction": 1.0,
        "breach_score_threshold": 1.0,
        "maximum_breach_incision_m": 20.0,
        "breach_length_cells": 4,
        "river_discharge_threshold_m3s": 0.01,
        "river_contributing_area_threshold_km2": 0.1,
        "river_minimum_discharge_m3s": 0.0,
    }


def test_config_accepts_hydrology_controls(tmp_path: Path) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        "\n".join(
            (
                "format_version: 2",
                "output_dir: target",
                "terrain_output_dir: terrain",
                "hydrology_output_dir: hydro",
                "grid:",
                "  display_boundary_halo_l2_cells: 4",
                "hydrology:",
                "  river_contributing_area_threshold_km2: 12",
                "  maximum_routed_to_inherited_area_ratio: 1.2",
                "limits:",
                "  maximum_peak_memory_gb: 8",
                "  maximum_hydrology_storage_gb: 2",
            )
        ),
        encoding="utf8",
    )
    config = L3HydrologyConfig.from_file(config_path)
    assert config.river_contributing_area_threshold_km2 == 12.0
    assert config.display_boundary_halo_l2_cells == 4
    assert config.output_dir == (tmp_path / "hydro").resolve()


def test_refined_target_basin_selects_most_inherited_overlap() -> None:
    receiver = np.asarray([-1, 4, 4, -1, -1, -1], dtype=np.int32)
    discharge = np.asarray([0.0, 8.0, 12.0, 0.0, 0.0, 0.0], dtype=np.float32)
    basin = np.asarray([3, 7, 8, 7, -1, 8], dtype=np.int32)
    terminal = np.asarray([0, 0, 0, 0, 3, 0], dtype=np.uint8)
    entry, basin_id, core = _refined_target_basin(
        {
            "FlowReceiverID": receiver,
            "MeanDischargeM3s": discharge,
            "BasinID": basin,
        },
        terminal,
        np.asarray([False, False, True, False, False, True]),
        np.ones(6, dtype=np.float64),
    )
    assert entry == 2
    assert basin_id == 8
    assert np.array_equal(core, np.asarray([False, False, True, False, False, True]))


def test_d8_adjacency_includes_diagonals() -> None:
    source = np.zeros((3, 3), dtype=bool)
    source[1, 1] = True
    adjacent = _adjacent_d8(source)
    assert np.count_nonzero(adjacent) == 8
    assert not adjacent[1, 1]


def test_reach_support_distinguishes_waterbody_connectors() -> None:
    reaches = pa.table(
        {
            "cell_path": pa.array([[0, 1, 2], [2, 3, 4]], type=pa.list_(pa.int32())),
            "reach_kind": pa.array(["channel", "connector"], type=pa.string()),
        }
    )
    network, connector = _reach_support_masks(reaches, 6)
    assert np.array_equal(network, [True, True, True, True, True, False])
    assert np.array_equal(connector, [False, False, True, True, True, False])


def test_below_datum_land_is_not_colored_as_ocean() -> None:
    colors = _hillshaded_terrain(np.full((3, 3), -150.0, dtype=np.float32), 200.0)
    red, green, blue = colors[1, 1]
    assert green > red
    assert green > blue


def test_flow_arrows_follow_receiver_direction() -> None:
    river = np.zeros((30, 30), dtype=bool)
    discharge = np.zeros((30, 30), dtype=np.float32)
    receiver = np.full(30 * 30, -1, dtype=np.int32)
    for row in range(8, 22):
        source = row * 30 + 15
        river[row, 15] = True
        discharge[row, 15] = 200.0
        receiver[source] = source + 30
    source, target = _flow_arrow_cells(river, discharge, receiver)
    assert len(source) == 1
    assert int(target[0] - source[0]) == 30


def test_discharge_continuity_allows_only_waterbody_losses() -> None:
    receiver = np.asarray([1, 2, 3, -1, 5, -1], dtype=np.int32)
    core = np.ones(6, dtype=bool)
    discharge = np.asarray([10.0, 15.0, 14.0, 20.0, 8.0, 6.0], dtype=np.float32)
    lake = np.asarray([0.0, 0.7, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    wetland = np.zeros(6, dtype=np.float32)
    metrics = _discharge_continuity_metrics(receiver, core, discharge, lake, wetland)
    assert metrics["material_mean_discharge_loss_edge_count"] == 2
    assert metrics["waterbody_loss_edge_count"] == 1
    assert metrics["unexplained_mean_discharge_loss_edge_count"] == 1


def test_d8_neighbors_include_diagonals_without_edge_wrap() -> None:
    neighbors = _d8_neighbors(3, 4)
    center = 1 * 4 + 1
    assert set(map(int, neighbors[center])) == {0, 1, 2, 4, 6, 8, 9, 10}
    corner = 0
    assert set(map(int, neighbors[corner])) == {0, 1, 4, 5}
    assert 3 not in neighbors[corner]


def test_outer_boundary_and_cropped_receivers_keep_hidden_halo_off_map() -> None:
    rows, columns = np.indices((6, 7), dtype=np.int32)
    boundary = _outer_boundary_mask(rows.reshape(-1), columns.reshape(-1)).reshape(6, 7)
    assert np.count_nonzero(boundary) == 22
    receiver = np.arange(42, dtype=np.int32).reshape(6, 7)
    receiver[:, :-1] += 1
    receiver[:, -1] = -1
    cropped = _crop_receiver_grid(receiver.reshape(-1), 6, 7, slice(1, 5), slice(1, 6))
    cropped = cropped.reshape(4, 5)
    assert np.all(cropped[:, :-1] == np.arange(20, dtype=np.int32).reshape(4, 5)[:, :-1] + 1)
    assert np.all(cropped[:, -1] == -1)


def test_fraction_realization_conserves_parent_area() -> None:
    parent_fraction = np.asarray([0.25, 0.625], dtype=np.float32)
    area = np.asarray([1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 4.0], dtype=np.float64)
    elevation = np.asarray([3.0, 1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 6.0], dtype=np.float32)
    realized = _realize_parent_fraction(parent_fraction, area, elevation, 4)
    for parent, target in enumerate(parent_fraction):
        rows = slice(parent * 4, (parent + 1) * 4)
        actual = float(np.sum(realized[rows] * area[rows]))
        expected = float(target * np.sum(area[rows]))
        assert abs(actual - expected) < 1e-7


def test_orographic_downscale_conserves_monthly_parent_volume() -> None:
    elevation = np.asarray([0.0, 100.0, 200.0, 300.0, 50.0, 150.0], dtype=np.float32)
    area = np.asarray([1.0, 2.0, 1.0, 2.0, 1.5, 1.5], dtype=np.float64)
    parent_id = np.asarray([7, 7, 7, 7, 8, 8], dtype=np.int32)
    eligible = np.ones(6, dtype=bool)
    weights = _topographic_weights(elevation, area, eligible, parent_id, 0.3)
    source = np.asarray([[10.0 + month for month in range(12)] for _ in range(2)], dtype=np.float32)
    source_rows = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32)
    downscaled = _downscale_monthly(source, source_rows, weights, eligible)
    error = _forcing_error(
        downscaled,
        source,
        source_rows,
        parent_id,
        area,
        eligible,
        eligible,
    )
    assert error < 1e-6
    assert np.ptp(downscaled[:, :4], axis=1).min() > 0.0


def test_native_regional_d8_route_is_acyclic_and_conservative() -> None:
    height = width = 9
    count = height * width
    row, column = np.indices((height, width), dtype=np.float32)
    elevation = (90.0 - 3.0 * row - 2.0 * column).reshape(-1).astype(np.float32)
    elevation[4 * width + 4] -= 12.0
    terminal = np.zeros((height, width), dtype=np.uint8)
    terminal[[0, -1], :] = 1
    terminal[:, [0, -1]] = 1
    xyz = np.stack(
        (
            np.ones(count, dtype=np.float64),
            (column.reshape(-1) - 4.0) * 1e-4,
            (row.reshape(-1) - 4.0) * 1e-4,
        ),
        axis=1,
    )
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    radius_km = 6_371.0
    outputs, lakes, _breaches, reaches, metadata = run_regional_hydrology(
        controls=_controls(),
        areas_steradians=np.full(count, 0.04 / (radius_km * radius_km), dtype=np.float64),
        neighbors=_d8_neighbors(height, width),
        xyz=np.ascontiguousarray(xyz, dtype=np.float32),
        elevation=np.ascontiguousarray(elevation),
        relief=np.full(count, 5.0, dtype=np.float32),
        rock_strength=np.full(count, 0.5, dtype=np.float32),
        accommodation=np.full(count, 0.5, dtype=np.float32),
        terminal=np.ascontiguousarray(terminal.reshape(-1)),
        runoff=np.full((12, count), 80.0, dtype=np.float32),
        evaporation=np.full((12, count), 20.0, dtype=np.float32),
        aridity=np.full(count, 0.8, dtype=np.float32),
    )
    assert metadata["topology_valid"] == 1
    assert metadata["conservation_relative_error"] < 1e-6
    assert reaches.num_rows > 0
    assert lakes.num_rows >= 0
    receiver = outputs["FlowReceiverID"]
    for start in np.flatnonzero(terminal.reshape(-1) == 0):
        seen: set[int] = set()
        cell = int(start)
        while receiver[cell] >= 0 and terminal.reshape(-1)[cell] == 0:
            assert cell not in seen
            seen.add(cell)
            cell = int(receiver[cell])
