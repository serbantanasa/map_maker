from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from map_maker.cli import main
from map_maker.pipeline.l3_target import (
    L3TargetConfig,
    L3TargetResult,
    _context_selection,
    _rectangular_l2_window,
    _upstream_closure,
)


def test_l3_target_config_resolves_paths_and_validates_grid(tmp_path: Path):
    path = tmp_path / "target.yaml"
    path.write_text(
        """\
format_version: 2
handoff_dir: handoff
output_dir: target
target:
  id: first-catchment
  outlet_parent_cell_id: 17
  context_parent_rings: 2
  minimum_area_km2: 10000
  maximum_area_km2: 50000
grid:
  base_cell_size_m: 200
  adaptive_minimum_cell_size_m: 25
  adaptive_maximum_cell_size_m: 50
limits:
  maximum_base_cell_count: 2000000
""",
        encoding="utf8",
    )

    config = L3TargetConfig.from_file(path)

    assert config.handoff_dir == tmp_path / "handoff"
    assert config.output_dir == tmp_path / "target"
    assert config.target_id == "first-catchment"
    assert config.outlet_parent_cell_id == 17
    assert config.base_cell_size_m == 200
    assert config.terrain_window_halo_l2_cells == 4
    assert config.terrain_source_context_l2_cells == 1
    assert config.maximum_peak_memory_gb == 24
    with pytest.raises(ValueError, match="base_cell_size_m"):
        L3TargetConfig(
            handoff_dir=tmp_path,
            output_dir=tmp_path,
            target_id="bad-grid",
            outlet_parent_cell_id=1,
            base_cell_size_m=500,
        ).validate()


def test_upstream_closure_and_context_rings_are_exact():
    cells = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32)
    receivers = np.asarray([3, 3, 4, 9, 6, 9], dtype=np.int32)

    closure = _upstream_closure(4, cells, receivers)

    np.testing.assert_array_equal(closure, [1, 2, 3, 4])
    parent_ids = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    neighbors = np.asarray(
        [
            [2, -1, -1, -1],
            [1, 3, -1, -1],
            [2, 4, 5, -1],
            [3, 6, -1, -1],
            [3, 7, -1, -1],
            [4, -1, -1, -1],
            [5, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    selected, rings, missing = _context_selection(
        np.asarray([3], dtype=np.int32), parent_ids, neighbors, rings=2
    )
    ring_by_id = dict(zip(selected.tolist(), rings.tolist(), strict=True))
    assert ring_by_id == {1: 2, 2: 1, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}
    assert not missing


def test_rectangular_l2_window_is_complete_and_roles_are_exclusive():
    resolution = 10
    face = 1
    cell_ids = np.asarray(
        [
            face * resolution * resolution + row * resolution + column
            for row in range(resolution)
            for column in range(resolution)
        ],
        dtype=np.int32,
    )
    core_ids = {
        face * resolution * resolution + 4 * resolution + 4,
        face * resolution * resolution + 5 * resolution + 5,
    }
    core = np.isin(cell_ids, list(core_ids))

    selected, roles, metadata = _rectangular_l2_window(
        cell_ids,
        core,
        resolution,
        halo_cells=1,
        source_context_cells=1,
    )

    assert np.count_nonzero(selected) == 36
    assert metadata["terrain_l2_cell_count"] == 16
    inside_window = roles["inside_terrain_window"]
    role_count = sum(
        roles[name].astype(np.int8)
        for name in (
            "inside_target_core",
            "inside_process_halo",
            "outside_process_domain",
        )
    )
    assert np.all(role_count[inside_window] == 1)
    assert np.all(role_count[~inside_window] == 0)
    assert np.count_nonzero(roles["inside_target_core"]) == 2
    assert np.count_nonzero(roles["outside_process_domain"]) == 2

    with pytest.raises(RuntimeError, match="continuous L3 window"):
        _rectangular_l2_window(
            np.delete(cell_ids, np.flatnonzero(cell_ids == min(core_ids))[0] - 11),
            np.delete(core, np.flatnonzero(cell_ids == min(core_ids))[0] - 11),
            resolution,
            halo_cells=1,
            source_context_cells=1,
        )


def test_l3_target_cli(tmp_path: Path, monkeypatch, capsys):
    config_path = tmp_path / "target.yaml"
    config_path.write_text(
        "format_version: 2\nhandoff_dir: handoff\noutput_dir: target\n"
        "target:\n  id: selected\n  outlet_parent_cell_id: 9\n",
        encoding="utf8",
    )
    result = L3TargetResult(
        output_dir=tmp_path / "target",
        manifest_path=tmp_path / "target/manifest.json",
        validation_path=tmp_path / "target/validation.json",
        preview_path=tmp_path / "target/preview.png",
        target_id="selected",
        outlet_parent_cell_id=9,
        core_parent_count=12,
        context_parent_count=8,
        core_area_km2=52_000.0,
        estimated_base_cell_count=1_300_000,
    )
    monkeypatch.setattr("map_maker.pipeline.l3_target.export_l3_target", lambda config: result)

    assert main(["l3-target", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "L3 target" in output
    assert "selected" in output
    assert "1300000" in output
