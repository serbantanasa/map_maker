from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from map_maker.cli import main
import map_maker.pipeline.l3_terrain as l3_terrain_module
from map_maker.pipeline._l3_terrain_native import run_l3_terrain_chunk
from map_maker.pipeline.l3_terrain import (
    L3TerrainConfig,
    L3TerrainResult,
    _TerrainSources,
    _conditioning_basis,
    _interpolated_center_correction,
    _nice_scale_km,
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


def test_l3_generate_resumes_validates_and_rejects_corrupt_cache(tmp_path: Path, monkeypatch):
    resolution = 2_048
    factor = 22
    face = 4
    context_ids = np.asarray(
        sorted(
            face * resolution * resolution + row * resolution + column
            for row in range(999, 1_003)
            for column in range(999, 1_003)
        ),
        dtype=np.int32,
    )
    domain_ids = np.asarray(
        sorted(
            face * resolution * resolution + row * resolution + column
            for row in range(1_000, 1_002)
            for column in range(1_000, 1_002)
        ),
        dtype=np.int32,
    )
    context_count = len(context_ids)
    context_elevation = np.asarray(
        [
            4.0 * ((int(cell_id) % (resolution * resolution)) // resolution)
            + 3.0 * (int(cell_id) % resolution)
            - 6_500.0
            for cell_id in context_ids
        ],
        dtype=np.float32,
    )
    controls = {
        "parent_resolution": resolution,
        "factor": factor,
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
        "context_orogenic_strength": np.full(context_count, 0.4, dtype=np.float32),
        "context_ridge_direction_xyz": np.tile(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (context_count, 1)
        ),
    }
    context_geometry, _ = run_l3_terrain_chunk(
        **common,
        chunk_parent_ids=context_ids,
    )
    context_area = context_geometry["area_km2"].reshape(context_count, factor * factor).sum(axis=1)
    target_manifest = tmp_path / "target-manifest.json"
    handoff_manifest = tmp_path / "handoff-manifest.json"
    target_manifest.write_text("{}", encoding="utf8")
    handoff_manifest.write_text("{}", encoding="utf8")
    sources = _TerrainSources(
        target_id="tiny-continuous-window",
        target_manifest_path=target_manifest,
        handoff_dir=tmp_path,
        handoff_manifest_path=handoff_manifest,
        parent_resolution=resolution,
        planet_radius_m=6_371_000.0,
        context_ids=context_ids,
        context_elevation_m=context_elevation,
        context_relief_m=common["context_relief_m"],
        context_area_km2=context_area,
        context_rock_strength=common["context_rock_strength"],
        context_orogenic_strength=common["context_orogenic_strength"],
        context_ridge_direction_xyz=common["context_ridge_direction_xyz"],
        domain_ids=domain_ids,
        domain_handoff_rows=np.arange(len(domain_ids), dtype=np.int32),
        domain_inside_core=np.asarray([True, False, False, False]),
        domain_inside_process_halo=np.asarray([False, True, True, False]),
        domain_outside_process=np.asarray([False, False, False, True]),
        domain_lake_fraction=np.zeros(len(domain_ids), dtype=np.float32),
        domain_wetland_fraction=np.zeros(len(domain_ids), dtype=np.float32),
        domain_ocean_fraction=np.zeros(len(domain_ids), dtype=np.float32),
    )
    config = L3TerrainConfig(
        target_dir=tmp_path,
        output_dir=tmp_path / "terrain",
        refinement_factor=factor,
        chunk_parent_count=2,
        maximum_parent_mean_error_m=100.0,
        maximum_parent_mean_error_relief_fraction=0.5,
        maximum_parent_area_relative_error=1e-8,
        maximum_parent_boundary_residual_p95_ratio=100.0,
        maximum_chunk_boundary_residual_p95_ratio=100.0,
        maximum_cell_size_relative_error=0.5,
        minimum_terrain_offset_std_m=0.01,
        maximum_tile_bubble_correlation_p50=1.0,
        maximum_tile_bubble_correlation_p95=1.0,
        maximum_center_correction_relief_fraction=10.0,
        maximum_base_cell_count=5_000,
        maximum_peak_memory_gb=1.0,
        maximum_storage_gb=0.1,
    )
    monkeypatch.setattr(l3_terrain_module, "_load_sources", lambda _: sources)
    native = l3_terrain_module.run_l3_terrain_chunk
    calls = 0

    def interrupt_second_chunk(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return native(**kwargs)

    monkeypatch.setattr(l3_terrain_module, "run_l3_terrain_chunk", interrupt_second_chunk)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        l3_terrain_module.generate_l3_terrain(config)

    monkeypatch.setattr(l3_terrain_module, "run_l3_terrain_chunk", native)
    result = l3_terrain_module.generate_l3_terrain(config)
    assert result.resumed_chunk_count == 1
    assert result.cell_count == len(domain_ids) * factor * factor
    preview = Image.open(result.preview_path)
    assert preview.width > factor * 2
    assert preview.height > factor * 2
    assert (result.output_dir / "terrain_domain.png").is_file()

    cached = l3_terrain_module.generate_l3_terrain(config)
    assert cached.zarr_path == result.zarr_path
    chunk_path = result.zarr_path / "terrain/elevation_m/0"
    chunk_path.write_bytes(chunk_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="integrity check failed for terrain_zarr"):
        l3_terrain_module.generate_l3_terrain(config)


def test_scale_bar_uses_readable_metric_steps():
    assert _nice_scale_km(1_000, 200.0) == 50.0


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
