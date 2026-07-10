from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from map_maker.cli import main
from map_maker.pipeline._cubed_sphere_native import generate_cubed_sphere
from map_maker.pipeline.cubed_sphere import CubedSphereGrid, run_cubed_sphere_diagnostic


@pytest.fixture(scope="module")
def grid() -> CubedSphereGrid:
    return CubedSphereGrid.create(24)


def test_geometry_shapes_units_and_area(grid: CubedSphereGrid):
    assert grid.face_shape == (6, 24, 24)
    assert grid.cell_count == 6 * 24 * 24
    assert grid.xyz.shape == (6, 24, 24, 3)
    assert grid.neighbor_indices.shape == (6, 24, 24, 4)
    np.testing.assert_allclose(np.linalg.norm(grid.xyz, axis=-1), 1.0, atol=1e-6)
    assert math.isclose(float(np.sum(grid.cell_areas)), 4.0 * math.pi, abs_tol=1e-12)
    assert float(np.max(grid.cell_areas) / np.min(grid.cell_areas)) < 1.5
    for face in range(6):
        assert math.isclose(
            float(np.sum(grid.cell_areas[face])), 4.0 * math.pi / 6.0, abs_tol=1e-12
        )


def test_cross_face_neighbors_are_unique_reciprocal_and_uniform(grid: CubedSphereGrid):
    neighbors = grid.neighbor_indices.reshape(grid.cell_count, 4)
    assert np.all((neighbors >= 0) & (neighbors < grid.cell_count))
    assert all(len(set(row.tolist())) == 4 for row in neighbors)

    for index, adjacent in enumerate(neighbors):
        for neighbor in adjacent:
            assert index in neighbors[int(neighbor)]

    face_size = grid.face_resolution * grid.face_resolution
    source_faces = np.repeat(np.arange(6), face_size * 4)
    target_faces = neighbors.reshape(-1) // face_size
    cross_face_links = int(np.count_nonzero(source_faces != target_faces))
    assert cross_face_links == 24 * grid.face_resolution

    xyz = grid.xyz.reshape(grid.cell_count, 3)
    source = np.repeat(xyz, 4, axis=0)
    target = xyz[neighbors.reshape(-1)]
    angles = np.arccos(np.clip(np.sum(source * target, axis=1), -1.0, 1.0))
    assert float(np.max(angles) / np.min(angles)) < 1.5


def test_global_index_round_trip_and_read_only_arrays(grid: CubedSphereGrid):
    for face, row, col in ((0, 0, 0), (2, 11, 7), (5, 23, 23)):
        index = grid.global_index(face, row, col)
        assert grid.decode_index(index) == (face, row, col)
    assert not grid.xyz.flags.writeable
    assert not grid.cell_areas.flags.writeable
    with pytest.raises(IndexError):
        grid.global_index(6, 0, 0)
    with pytest.raises(IndexError):
        grid.decode_index(grid.cell_count)
    with pytest.raises(ValueError, match="int32 global-index capacity"):
        generate_cubed_sphere(20_000)


def test_diagnostic_writes_cube_net_and_report(tmp_path: Path):
    net_path, report_path = run_cubed_sphere_diagnostic(face_resolution=16, output_dir=tmp_path)

    with Image.open(net_path) as image:
        assert image.mode == "RGB"
        assert image.size == (64, 48)
        assert np.asarray(image).std() > 0.0
    report = json.loads(report_path.read_text(encoding="utf8"))
    assert report["topology"] == "equiangular_cubed_sphere"
    assert report["cell_count"] == 6 * 16 * 16
    assert report["area_error_from_4pi"] < 1e-12
    assert report["native_library"]["abi_version"] == 1
    assert len(report["native_library"]["sha256"]) == 64


def test_topology_cli(tmp_path: Path):
    assert (
        main(
            [
                "topology",
                "--face-resolution",
                "8",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "cube_net.png").exists()
    assert (tmp_path / "topology.json").exists()
