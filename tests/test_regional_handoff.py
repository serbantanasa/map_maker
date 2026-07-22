from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from map_maker.cli import main
from map_maker.pipeline.regional_handoff import (
    RegionalHandoffConfig,
    RegionalHandoffResult,
    _realize_surface_fractions,
    _restrict_grid_array,
    export_regional_handoff,
)


def test_config_resolves_paths_and_applies_cli_overrides(tmp_path: Path):
    config_path = tmp_path / "handoff.yaml"
    config_path.write_text(
        """\
format_version: 1
world_config: world.yaml
output_dir: products/region
region:
  basin_id: auto
  halo_parent_rings: 2
  refinement_factor: 8
  chunk_rows: 1024
""",
        encoding="utf8",
    )

    config = RegionalHandoffConfig.from_file(
        config_path,
        output_dir=tmp_path / "override",
        basin_id=17,
        halo_parent_rings=3,
        refinement_factor=16,
    )

    assert config.source_config == config_path
    assert config.world_config == tmp_path / "world.yaml"
    assert config.output_dir == tmp_path / "override"
    assert config.basin_id == 17
    assert config.halo_parent_rings == 3
    assert config.refinement_factor == 16
    assert config.chunk_rows == 1024

    config_path.write_text(
        "world_config: world.yaml\nregion:\n  refinement_factor: 3\n",
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="power of two"):
        RegionalHandoffConfig.from_file(config_path)


def test_grid_restriction_preserves_leading_and_trailing_dimensions():
    face_resolution = 3
    cells = 6 * face_resolution * face_resolution
    parent_ids = np.asarray([0, 7, cells - 1], dtype=np.int32)
    base = np.arange(cells, dtype=np.float32).reshape(6, face_resolution, face_resolution)

    plain, plain_axis = _restrict_grid_array(base, parent_ids, face_resolution) or (None, None)
    leading, leading_axis = _restrict_grid_array(
        np.stack((base, base + 100.0)), parent_ids, face_resolution
    ) or (None, None)
    trailing, trailing_axis = _restrict_grid_array(
        np.stack((base, base + 100.0), axis=-1), parent_ids, face_resolution
    ) or (None, None)

    assert plain_axis == 0
    np.testing.assert_array_equal(plain, base.reshape(-1)[parent_ids])
    assert leading_axis == 1
    np.testing.assert_array_equal(
        leading,
        np.stack((base.reshape(-1)[parent_ids], base.reshape(-1)[parent_ids] + 100.0), axis=1),
    )
    assert trailing_axis == 0
    np.testing.assert_array_equal(trailing, leading)


def test_surface_realization_is_deterministic_and_parent_area_conserving():
    parent_row = np.repeat(np.arange(2, dtype=np.int32), 4)
    area = np.asarray([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    terrain = np.asarray([30.0, 20.0, 10.0, 0.0, -3.0, -2.0, -1.0, 0.0])
    fine_ids = np.arange(8, dtype=np.int32)
    targets = {
        "ocean_fraction": np.asarray([0.25, 0.60]),
        "lake_fraction": np.asarray([0.15, 0.10]),
        "wetland_fraction": np.asarray([0.20, 0.05]),
    }

    first = _realize_surface_fractions(targets, parent_row, area, terrain, fine_ids, parent_count=2)
    second = _realize_surface_fractions(
        targets, parent_row, area, terrain, fine_ids, parent_count=2
    )

    parent_area = np.bincount(parent_row, weights=area)
    for name, target in targets.items():
        np.testing.assert_array_equal(first[name], second[name])
        represented = np.bincount(parent_row, weights=first[name] * area) / parent_area
        np.testing.assert_allclose(represented, target, atol=1e-7)
    occupancy = sum(first.values())
    assert np.min(occupancy) >= 0.0
    assert np.max(occupancy) <= 1.0


def test_regional_handoff_cli(tmp_path: Path, monkeypatch, capsys):
    config_path = tmp_path / "handoff.yaml"
    config_path.write_text("world_config: world.yaml\n", encoding="utf8")
    result = RegionalHandoffResult(
        output_dir=tmp_path / "region",
        manifest_path=tmp_path / "region/manifest.json",
        validation_path=tmp_path / "region/validation.json",
        preview_path=tmp_path / "region/preview.png",
        zarr_path=tmp_path / "region/region.zarr",
        basin_id=23,
        parent_count=31,
        core_parent_count=20,
        child_count=7936,
        validation_passed=True,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.regional_handoff.export_regional_handoff", lambda config: result
    )

    assert main(["regional-handoff", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Regional handoff" in output
    assert "basin 23" in output
    assert "7936 sparse L2 cells" in output


def test_small_world_handoff_package_is_complete_and_valid(tmp_path: Path):
    world_path = tmp_path / "world.yaml"
    world_path.write_text(
        f"""\
topology: cubed_sphere
resolutions:
  - face_resolution: 20
rng_seed: 37
run_id: small-handoff-world
output_dir: {tmp_path / "world-out"}
cache_dir: {tmp_path / "cache"}
log_dir: {tmp_path / "logs"}
stage_overrides:
  tectonics:
    num_plates: 14
    continental_fraction: 0.35
    lloyd_iterations: 3
  world_age:
    world_age: 4.1
  climate:
    spinup_years: 10
    moisture_spinup_years: 2
    moisture_steps_per_month_at_face_128: 16
  basin_refinement:
    terrain_noise_fraction: 0.4
""",
        encoding="utf8",
    )
    config_path = tmp_path / "handoff.yaml"
    config_path.write_text(
        """\
format_version: 1
world_config: world.yaml
output_dir: package
region:
  basin_id: auto
  halo_parent_rings: 1
  refinement_factor: 4
  chunk_rows: 256
""",
        encoding="utf8",
    )

    config = RegionalHandoffConfig.from_file(config_path)
    result = export_regional_handoff(config)

    assert result.validation_passed
    assert result.parent_count > result.core_parent_count > 0
    assert result.child_count == result.parent_count * 16
    manifest = json.loads(result.manifest_path.read_text(encoding="utf8"))
    validation = json.loads(result.validation_path.read_text(encoding="utf8"))
    assert manifest["status"] == "complete"
    assert manifest["validation_passed"]
    assert manifest["semantics"]["rivers"].endswith("no applied L2 incision")
    assert "RefinedBasinCellCatalog" in manifest["source"]["artifacts"]["basin_refinement"]
    assert manifest["software"]["native_libraries"]
    assert validation["passed"]
    assert validation["maximum_surface_fraction_error"] <= 1e-6
    assert validation["all_refined_path_cells_packaged"] == 1
    assert validation["context_parent_count"] == (
        validation["reach_path_support_parent_count"] + validation["halo_parent_count"]
    )
    assert manifest["selection"]["maximum_halo_ring"] <= config.halo_parent_rings
    region = zarr.open_group(str(result.zarr_path), mode="r")
    assert region.attrs["child_level"] == "L2"
    assert region["l2/geometry/cell_id"].shape == (result.child_count,)
    assert region["l2/surface/ocean_fraction"].shape == (result.child_count,)
    assert (result.output_dir / "tables/refined_river_reaches.parquet").is_file()
    assert result.preview_path.is_file()

    repeated = export_regional_handoff(config)
    repeated_manifest = json.loads(repeated.manifest_path.read_text(encoding="utf8"))
    assert repeated_manifest == manifest
