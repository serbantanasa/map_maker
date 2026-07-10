"""Canonical cubed-sphere topology and diagnostic rendering."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .._native import native_library_info
from ._cubed_sphere_native import (
    D4_NEIGHBORS,
    FACE_COUNT,
    children_map as native_children_map,
    fill_d4_halo as native_fill_d4_halo,
    generate_cubed_sphere,
    parent_map as native_parent_map,
    prolongate_constant as native_prolongate_constant,
    restrict_extensive as native_restrict_extensive,
    restrict_intensive as native_restrict_intensive,
)

FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


@dataclass(frozen=True)
class CubedSphereGrid:
    """Six equiangular raster faces with topology-aware global neighbors."""

    face_resolution: int
    xyz: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    cell_areas: np.ndarray
    neighbor_indices: np.ndarray

    @classmethod
    def create(cls, face_resolution: int) -> "CubedSphereGrid":
        arrays = generate_cubed_sphere(face_resolution)
        for array in arrays:
            array.setflags(write=False)
        return cls(face_resolution, *arrays)

    @property
    def cell_count(self) -> int:
        return FACE_COUNT * self.face_resolution * self.face_resolution

    @property
    def face_shape(self) -> tuple[int, int, int]:
        return FACE_COUNT, self.face_resolution, self.face_resolution

    def global_index(self, face: int, row: int, col: int) -> int:
        resolution = self.face_resolution
        if not 0 <= face < FACE_COUNT:
            raise IndexError(f"face must be in [0, {FACE_COUNT})")
        if not 0 <= row < resolution or not 0 <= col < resolution:
            raise IndexError(f"row and col must be in [0, {resolution})")
        return face * resolution * resolution + row * resolution + col

    def decode_index(self, index: int) -> tuple[int, int, int]:
        if not 0 <= index < self.cell_count:
            raise IndexError(f"global index must be in [0, {self.cell_count})")
        face_size = self.face_resolution * self.face_resolution
        face, within_face = divmod(index, face_size)
        row, col = divmod(within_face, self.face_resolution)
        return face, row, col

    def neighbors(self, face: int, row: int, col: int) -> np.ndarray:
        return self.neighbor_indices[face, row, col]

    def angular_distance(self, first: int, second: int) -> float:
        face_a, row_a, col_a = self.decode_index(first)
        face_b, row_b, col_b = self.decode_index(second)
        vector_a = self.xyz[face_a, row_a, col_a]
        vector_b = self.xyz[face_b, row_b, col_b]
        dot = float(np.clip(np.dot(vector_a, vector_b), -1.0, 1.0))
        return math.acos(dot)

    def _require_field(self, values: np.ndarray, *, name: str = "values") -> np.ndarray:
        array = np.asarray(values)
        if array.shape != self.face_shape:
            raise ValueError(f"{name} must have shape {self.face_shape}, got {array.shape}")
        return array

    def parent_map(self, factor: int = 2) -> np.ndarray:
        """Return the same-face coarse parent global ID for every cell."""

        parents = native_parent_map(self.face_resolution, factor)
        parents.setflags(write=False)
        return parents

    def children_map(self, factor: int = 2) -> np.ndarray:
        """Return row-major child global IDs, treating this grid as coarse."""

        children = native_children_map(self.face_resolution, factor)
        children.setflags(write=False)
        return children

    def restrict_extensive(self, values: np.ndarray, factor: int = 2) -> np.ndarray:
        """Sum child quantities into same-face parent cells."""

        return native_restrict_extensive(self._require_field(values), factor)

    def restrict_intensive(self, values: np.ndarray, factor: int = 2) -> np.ndarray:
        """Area-weight child values into same-face parent cells."""

        return native_restrict_intensive(self._require_field(values), self.cell_areas, factor)

    def prolongate_constant(self, values: np.ndarray, factor: int = 2) -> np.ndarray:
        """Copy each coarse prior to its row-major same-face children."""

        return native_prolongate_constant(self._require_field(values), factor)

    def with_d4_halo(self, values: np.ndarray) -> np.ndarray:
        """Add topology-aware width-one edge halos with undefined corners."""

        return native_fill_d4_halo(self._require_field(values))


def _direction_colors(grid: CubedSphereGrid) -> np.ndarray:
    colors = ((grid.xyz.astype(np.float64) + 1.0) * 127.5).clip(0.0, 255.0)
    return colors.astype(np.uint8)


def render_cube_net(grid: CubedSphereGrid, output_path: Path | str) -> Path:
    """Render globally continuous XYZ colors on an unrotated cube net."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = grid.face_resolution
    net = np.zeros((resolution * 3, resolution * 4, 3), dtype=np.uint8)
    colors = _direction_colors(grid)
    placements = {
        3: (1, 0),  # -Y
        0: (1, 1),  # +X
        2: (1, 2),  # +Y
        1: (1, 3),  # -X
        4: (0, 1),  # +Z
        5: (2, 1),  # -Z
    }
    for face, (net_row, net_col) in placements.items():
        row_start = net_row * resolution
        col_start = net_col * resolution
        net[row_start : row_start + resolution, col_start : col_start + resolution] = colors[face]
    Image.fromarray(net, mode="RGB").save(output_path)
    return output_path


def write_topology_report(grid: CubedSphereGrid, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_neighbors = grid.neighbor_indices.reshape(grid.cell_count, D4_NEIGHBORS)
    flat_xyz = grid.xyz.reshape(grid.cell_count, 3).astype(np.float64)
    source = np.repeat(flat_xyz, D4_NEIGHBORS, axis=0)
    target = flat_xyz[flat_neighbors.reshape(-1)]
    distances = np.arccos(np.clip(np.sum(source * target, axis=1), -1.0, 1.0))
    native_info = native_library_info("topology_native")
    report = {
        "format_version": 1,
        "topology": "equiangular_cubed_sphere",
        "face_order": list(FACE_NAMES),
        "face_resolution": grid.face_resolution,
        "cell_count": grid.cell_count,
        "area_total": float(np.sum(grid.cell_areas)),
        "area_error_from_4pi": float(abs(np.sum(grid.cell_areas) - 4.0 * math.pi)),
        "cell_area_min": float(np.min(grid.cell_areas)),
        "cell_area_max": float(np.max(grid.cell_areas)),
        "neighbor_angle_min": float(np.min(distances)),
        "neighbor_angle_mean": float(np.mean(distances)),
        "neighbor_angle_max": float(np.max(distances)),
        "native_library": {
            "abi_version": native_info["abi_version"],
            "sha256": native_info["sha256"],
        },
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    return output_path


def run_cubed_sphere_diagnostic(
    *, face_resolution: int = 64, output_dir: Path | str = Path("out/topology")
) -> tuple[Path, Path]:
    grid = CubedSphereGrid.create(face_resolution)
    output_dir = Path(output_dir).expanduser().resolve()
    net_path = render_cube_net(grid, output_dir / "cube_net.png")
    report_path = write_topology_report(grid, output_dir / "topology.json")
    return net_path, report_path


__all__ = [
    "CubedSphereGrid",
    "FACE_NAMES",
    "render_cube_net",
    "run_cubed_sphere_diagnostic",
    "write_topology_report",
]
