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


def test_parent_children_maps_are_inverse_and_row_major():
    coarse = CubedSphereGrid.create(5)
    fine = CubedSphereGrid.create(10)
    parents = fine.parent_map(factor=2).reshape(-1)
    children = coarse.children_map(factor=2).reshape(coarse.cell_count, 4)

    assert not fine.parent_map().flags.writeable
    assert not coarse.children_map().flags.writeable
    for parent, child_ids in enumerate(children):
        np.testing.assert_array_equal(parents[child_ids], parent)

    parent = coarse.global_index(2, 1, 3)
    np.testing.assert_array_equal(
        children[parent],
        [
            fine.global_index(2, 2, 6),
            fine.global_index(2, 2, 7),
            fine.global_index(2, 3, 6),
            fine.global_index(2, 3, 7),
        ],
    )


def test_restriction_conserves_area_extensive_and_intensive_fields():
    fine = CubedSphereGrid.create(16)
    coarse = CubedSphereGrid.create(8)
    rng = np.random.default_rng(42)
    extensive = rng.random(fine.face_shape)
    restricted_extensive = fine.restrict_extensive(extensive)
    assert math.isclose(
        float(np.sum(restricted_extensive)), float(np.sum(extensive)), abs_tol=1e-12
    )

    restricted_areas = fine.restrict_extensive(fine.cell_areas)
    np.testing.assert_allclose(restricted_areas, coarse.cell_areas, rtol=0.0, atol=1e-15)

    intensive = rng.normal(size=fine.face_shape)
    restricted_intensive = fine.restrict_intensive(intensive)
    fine_integral = float(np.sum(intensive * fine.cell_areas))
    coarse_integral = float(np.sum(restricted_intensive * coarse.cell_areas))
    assert math.isclose(fine_integral, coarse_integral, abs_tol=1e-14)
    np.testing.assert_allclose(fine.restrict_intensive(np.full(fine.face_shape, 7.5)), 7.5)


def test_constant_prolongation_copies_parent_values():
    coarse = CubedSphereGrid.create(6)
    fine = CubedSphereGrid.create(18)
    values = np.arange(coarse.cell_count, dtype=np.float64).reshape(coarse.face_shape)
    prolonged = coarse.prolongate_constant(values, factor=3)
    assert prolonged.shape == fine.face_shape
    np.testing.assert_array_equal(
        prolonged.reshape(-1), values.reshape(-1)[fine.parent_map(3).reshape(-1)]
    )


def test_d4_halo_matches_cross_face_neighbors_and_has_nan_corners(grid: CubedSphereGrid):
    values = np.arange(grid.cell_count, dtype=np.float32).reshape(grid.face_shape)
    halo = grid.with_d4_halo(values)
    resolution = grid.face_resolution
    flat_values = values.reshape(-1)

    assert halo.shape == (6, resolution + 2, resolution + 2)
    np.testing.assert_array_equal(halo[:, 1:-1, 1:-1], values)
    for face in range(6):
        np.testing.assert_array_equal(
            halo[face, 0, 1:-1], flat_values[grid.neighbor_indices[face, 0, :, 0]]
        )
        np.testing.assert_array_equal(
            halo[face, -1, 1:-1], flat_values[grid.neighbor_indices[face, -1, :, 1]]
        )
        np.testing.assert_array_equal(
            halo[face, 1:-1, 0], flat_values[grid.neighbor_indices[face, :, 0, 2]]
        )
        np.testing.assert_array_equal(
            halo[face, 1:-1, -1], flat_values[grid.neighbor_indices[face, :, -1, 3]]
        )
    assert np.isnan(halo[:, (0, 0, -1, -1), (0, -1, 0, -1)]).all()

    precise_values = values.astype(np.float64) + 2**40
    precise_halo = grid.with_d4_halo(precise_values)
    assert precise_halo.dtype == np.float64
    np.testing.assert_array_equal(precise_halo[:, 1:-1, 1:-1], precise_values)


def test_cross_face_edge_orientation_matches_explicit_cube_contract():
    grid = CubedSphereGrid.create(4)
    # Per source face, entries are N/S/W/E: target face, target edge, reversed.
    expected = (
        ((4, "S", False), (5, "N", False), (3, "E", False), (2, "W", False)),
        ((4, "N", True), (5, "S", True), (2, "E", False), (3, "W", False)),
        ((4, "E", True), (5, "E", False), (0, "E", False), (1, "W", False)),
        ((4, "W", False), (5, "W", True), (1, "E", False), (0, "W", False)),
        ((1, "N", True), (0, "N", False), (3, "N", False), (2, "N", True)),
        ((0, "S", False), (1, "S", True), (3, "S", True), (2, "S", False)),
    )

    def source_edges(face: int) -> tuple[np.ndarray, ...]:
        return (
            grid.neighbor_indices[face, 0, :, 0],
            grid.neighbor_indices[face, -1, :, 1],
            grid.neighbor_indices[face, :, 0, 2],
            grid.neighbor_indices[face, :, -1, 3],
        )

    for face, contracts in enumerate(expected):
        for ids, (target_face, target_edge, reversed_order) in zip(
            source_edges(face), contracts, strict=True
        ):
            offsets = list(range(grid.face_resolution))
            if reversed_order:
                offsets.reverse()
            if target_edge == "N":
                coordinates = [(target_face, 0, offset) for offset in offsets]
            elif target_edge == "S":
                coordinates = [
                    (target_face, grid.face_resolution - 1, offset) for offset in offsets
                ]
            elif target_edge == "W":
                coordinates = [(target_face, offset, 0) for offset in offsets]
            else:
                coordinates = [
                    (target_face, offset, grid.face_resolution - 1) for offset in offsets
                ]
            assert [grid.decode_index(int(index)) for index in ids] == coordinates


def test_hierarchy_rejects_invalid_shapes_factors_and_values(grid: CubedSphereGrid):
    with pytest.raises(ValueError, match="must divide"):
        grid.parent_map(5)
    with pytest.raises(ValueError, match="greater than one"):
        grid.children_map(1)
    with pytest.raises(ValueError, match="must have shape"):
        grid.restrict_extensive(np.zeros((24, 24)))
    values = np.ones(grid.face_shape)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        grid.restrict_intensive(values)
    with pytest.raises(ValueError, match="finite values"):
        grid.restrict_extensive(np.full(grid.face_shape, np.finfo(np.float64).max))
    with pytest.raises(TypeError, match="float32 or float64"):
        grid.with_d4_halo(np.ones(grid.face_shape, dtype=np.int32))


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
    assert report["native_library"]["abi_version"] == 2
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
