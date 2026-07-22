from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from map_maker.cli import main
from map_maker.pipeline._l3_terrain_native import run_l3_terrain_chunk
from map_maker.pipeline.l3_terrain import (
    L3TerrainConfig,
    L3TerrainResult,
    _conditioning_basis,
    _interpolated_center_correction,
    _tile_motif_metrics,
)


def test_l3_terrain_config_resolves_paths_and_rejects_invalid_conditioning(tmp_path: Path):
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
output_dir: target
terrain_output_dir: terrain
grid:
  base_cell_size_m: 200
  l3_refinement_factor: 22
terrain:
  chunk_parent_count: 32
  conditioning_damping: 0.7
limits:
  maximum_base_cell_count: 3000000
""",
        encoding="utf8",
    )

    config = L3TerrainConfig.from_file(config_path)

    assert config.target_dir == tmp_path / "target"
    assert config.output_dir == tmp_path / "terrain"
    assert config.refinement_factor == 22
    assert config.chunk_parent_count == 32
    with pytest.raises(ValueError, match="conditioning_damping"):
        L3TerrainConfig(
            target_dir=tmp_path,
            output_dir=tmp_path,
            conditioning_damping=1.1,
        ).validate()


def test_bilinear_conditioning_basis_partitions_unity_and_crosses_parent_edges():
    factor = 22
    center, neighbors = _conditioning_basis(factor)
    np.testing.assert_allclose(center + np.sum(neighbors, axis=0), 1.0, atol=1e-15)

    resolution = 2_048
    face = 4
    rows = range(999, 1_003)
    columns = range(999, 1_004)
    context_ids = np.asarray(
        sorted(
            face * resolution * resolution + row * resolution + column
            for row in rows
            for column in columns
        ),
        dtype=np.int32,
    )
    context_values = np.asarray(
        [
            11.0 * ((cell_id % (resolution * resolution)) // resolution)
            + 7.0 * (cell_id % resolution)
            for cell_id in context_ids
        ],
        dtype=np.float64,
    )
    west = face * resolution * resolution + 1_000 * resolution + 1_000
    east = west + 1
    west_values = _interpolated_center_correction(
        np.asarray([west], dtype=np.int32),
        context_ids,
        context_values,
        resolution,
        factor,
    ).reshape(factor, factor)
    east_values = _interpolated_center_correction(
        np.asarray([east], dtype=np.int32),
        context_ids,
        context_values,
        resolution,
        factor,
    ).reshape(factor, factor)

    boundary_jump = east_values[:, 0] - west_values[:, -1]
    np.testing.assert_allclose(boundary_jump, 7.0 / factor, atol=1e-12)


def test_tile_motif_metric_rejects_repeated_parent_bubbles():
    factor = 22
    unit = (np.arange(factor) + 0.5) / factor
    bubble = np.outer(np.sin(np.pi * unit), np.sin(np.pi * unit)).reshape(-1)
    patterns = np.stack([bubble * sign for sign in (1.0, -1.0) for _ in range(20)])

    metrics = _tile_motif_metrics(patterns.reshape(-1), factor)

    assert metrics["tile_bubble_absolute_correlation_p50"] > 0.95
    assert metrics["tile_bubble_absolute_correlation_p95"] > 0.95


def test_native_l3_chunk_is_partition_invariant_and_uses_uint64_ids():
    resolution = 2_048
    face = 4
    context_ids = np.asarray(
        sorted(
            face * resolution * resolution + row * resolution + column
            for row in range(999, 1_003)
            for column in range(999, 1_003)
        ),
        dtype=np.int32,
    )
    context_count = len(context_ids)
    context_elevation = np.linspace(200.0, 420.0, context_count, dtype=np.float32)
    controls = {
        "parent_resolution": resolution,
        "factor": 22,
        "planet_radius_m": 6_371_000.0,
        "terrain_seed": 42,
        "relief_realization_fraction": 0.42,
        "base_wavelength_m": 16_000.0,
        "octave_count": 5,
        "persistence": 0.52,
        "domain_warp_fraction": 0.22,
        "orogenic_ridge_fraction": 0.32,
    }
    common = {
        "controls": controls,
        "context_parent_ids": context_ids,
        "context_elevation_m": context_elevation,
        "context_relief_m": np.full(context_count, 300.0, dtype=np.float32),
        "context_rock_strength": np.full(context_count, 0.7, dtype=np.float32),
        "context_orogenic_strength": np.full(context_count, 0.5, dtype=np.float32),
        "context_ridge_direction_xyz": np.tile(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (context_count, 1)
        ),
    }
    selected = np.asarray([context_ids[5], context_ids[6]], dtype=np.int32)

    combined, combined_stats = run_l3_terrain_chunk(**common, chunk_parent_ids=selected)
    first, _ = run_l3_terrain_chunk(**common, chunk_parent_ids=selected[:1].copy())
    second, _ = run_l3_terrain_chunk(**common, chunk_parent_ids=selected[1:].copy())

    assert combined_stats["missing_context_neighbor_count"] == 0
    assert np.all(combined["cell_id"] > np.iinfo(np.uint32).max)
    np.testing.assert_array_equal(
        combined["cell_id"], np.concatenate((first["cell_id"], second["cell_id"]))
    )
    np.testing.assert_array_equal(
        combined["elevation_m"],
        np.concatenate((first["elevation_m"], second["elevation_m"])),
    )


def test_l3_terrain_cli(tmp_path: Path, monkeypatch, capsys):
    result = L3TerrainResult(
        output_dir=tmp_path / "terrain",
        manifest_path=tmp_path / "terrain/manifest.json",
        validation_path=tmp_path / "terrain/validation.json",
        zarr_path=tmp_path / "terrain/terrain.zarr",
        preview_path=tmp_path / "terrain/terrain.png",
        target_id="selected",
        parent_count=12,
        cell_count=5_808,
        actual_cell_size_m=198.4,
        chunk_count=3,
        resumed_chunk_count=0,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.l3_terrain.L3TerrainConfig.from_file",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr("map_maker.pipeline.l3_terrain.generate_l3_terrain", lambda config: result)

    assert main(["l3-terrain", "--config", str(tmp_path / "l3.yaml")]) == 0
    output = capsys.readouterr().out
    assert "L3 terrain" in output
    assert "5808 cells" in output
    assert "198.4 m" in output
