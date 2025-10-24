"""High-performance topology abstractions backed by native kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from ._topology_native import TopologyKind, compute_cell_areas, compute_neighbors
from .config import GridInfo

NEIGHBOR_OFFSETS = np.array(
    [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class NeighborView:
    indices: np.ndarray
    weights: np.ndarray


class Topology:
    """Base topology interface backed by precomputed native data."""

    def __init__(self, grid: GridInfo, kind: TopologyKind) -> None:
        self.shape = (grid.height, grid.width)
        self.grid = grid
        self.kind = kind

        self._cell_area = compute_cell_areas(kind, grid.height, grid.width)
        self._neighbor_indices, self._neighbor_weights = compute_neighbors(kind, grid.height, grid.width)

        self._wrap_rows = kind == "torus"
        self._wrap_cols = True

        self._lon, self._lat = self._compute_lon_lat(kind, grid.height, grid.width)
        self._xyz = self._compute_xyz(kind)

    @staticmethod
    def _compute_lon_lat(kind: TopologyKind, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        j = (np.arange(width, dtype=np.float64) + 0.5) / width
        i = (np.arange(height, dtype=np.float64) + 0.5) / height
        lon = (j - 0.5) * (2.0 * math.pi)
        if kind == "sphere":
            lat = (0.5 - i) * math.pi
        elif kind == "cylinder":
            lat = 1.0 - 2.0 * i  # range [-1, 1]
        else:  # torus
            lat = 2.0 * math.pi * i
        lon_grid = np.broadcast_to(lon, (height, width))
        lat_grid = np.broadcast_to(lat[:, None], (height, width))
        return lon_grid.astype(np.float64), lat_grid.astype(np.float64)

    def _compute_xyz(self, kind: TopologyKind) -> np.ndarray:
        if kind == "sphere":
            lat = self._lat
            lon = self._lon
            cos_lat = np.cos(lat)
            x = cos_lat * np.cos(lon)
            y = cos_lat * np.sin(lon)
            z = np.sin(lat)
            return np.stack((x, y, z), axis=-1).astype(np.float32)
        if kind == "cylinder":
            lon = self._lon
            lat = self._lat
            x = np.cos(lon)
            y = np.sin(lon)
            z = lat
            return np.stack((x, y, z), axis=-1).astype(np.float32)
        # torus parameterization
        major = 1.0
        minor = 0.5
        u = (self._lon + math.pi) % (2 * math.pi)  # shift to [0, 2pi]
        v = self._lat
        cos_v = np.cos(v)
        sin_v = np.sin(v)
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        x = (major + minor * cos_v) * cos_u
        y = (major + minor * cos_v) * sin_u
        z = minor * sin_v
        return np.stack((x, y, z), axis=-1).astype(np.float32)

    def cell_area(self, i: int, j: int) -> float:
        return float(self._cell_area[i, j])

    def neighbors(self, i: int, j: int, mode: str = "D8") -> NeighborView:
        indices = self._neighbor_indices[i, j]
        weights = self._neighbor_weights[i, j]
        if mode.upper() == "D4":
            indices = indices[:4]
            weights = weights[:4]
        mask = indices >= 0
        return NeighborView(indices=indices[mask], weights=weights[mask])

    def distance(self, coord_a: Tuple[int, int], coord_b: Tuple[int, int]) -> float:
        ax, ay = coord_a
        bx, by = coord_b
        vec_a = self._xyz[ax, ay]
        vec_b = self._xyz[bx, by]
        if self.kind == "sphere":
            dot = float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
            return math.acos(dot)
        return float(np.linalg.norm(vec_a - vec_b))

    def wrap(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        i, j = coord
        rows, cols = self.shape
        if self._wrap_rows:
            i %= rows
        else:
            i = min(max(i, 0), rows - 1)
        if self._wrap_cols:
            j %= cols
        else:
            j = min(max(j, 0), cols - 1)
        return i, j

    def to_xyz(self, i: int, j: int) -> np.ndarray:
        return self._xyz[i, j]

    def child_mapping(self, level: int) -> np.ndarray:
        if level < 0:
            raise ValueError("level must be non-negative")
        scale = 1 << level
        rows = self.shape[0] // scale
        cols = self.shape[1] // scale
        if rows == 0 or cols == 0 or self.shape[0] % scale or self.shape[1] % scale:
            raise ValueError("Resolution not divisible by requested level")
        mapping = np.zeros((rows, cols, 2), dtype=np.int32)
        for r in range(rows):
            for c in range(cols):
                mapping[r, c] = (r * scale, c * scale)
        return mapping

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def xyz(self) -> np.ndarray:
        return self._xyz

    @property
    def cell_areas(self) -> np.ndarray:
        return self._cell_area.copy()


class SphereTopology(Topology):
    def __init__(self, grid: GridInfo) -> None:
        super().__init__(grid, "sphere")


class CylinderTopology(Topology):
    def __init__(self, grid: GridInfo) -> None:
        super().__init__(grid, "cylinder")


class TorusTopology(Topology):
    def __init__(self, grid: GridInfo) -> None:
        super().__init__(grid, "torus")


def load_topology(name: str, grid: GridInfo) -> Topology:
    key = name.lower()
    if key == "sphere":
        return SphereTopology(grid)
    if key == "cylinder":
        return CylinderTopology(grid)
    if key == "torus":
        return TorusTopology(grid)
    raise ValueError(f"Unsupported topology '{name}'")


__all__ = [
    "Topology",
    "NeighborView",
    "SphereTopology",
    "CylinderTopology",
    "TorusTopology",
    "load_topology",
]
