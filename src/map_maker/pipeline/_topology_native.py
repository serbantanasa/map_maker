"""Deterministic topology precomputation used until cubed-sphere Rust lands."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

TopologyKind = Literal["sphere", "cylinder", "torus"]

_OFFSETS = np.array(
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
    dtype=np.int32,
)
_WEIGHTS = np.array(
    [1.0, 1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)],
    dtype=np.float32,
)


def _validate_dimensions(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("topology dimensions must be positive")


def compute_cell_areas(kind: TopologyKind, height: int, width: int) -> np.ndarray:
    _validate_dimensions(height, width)
    if kind == "sphere":
        latitude_edges = math.pi / 2.0 - np.arange(height + 1, dtype=np.float64) * (
            math.pi / height
        )
        strip_areas = np.abs(np.diff(np.sin(latitude_edges))) * (2.0 * math.pi / width)
        return np.broadcast_to(strip_areas[:, None], (height, width)).copy()
    if kind in {"cylinder", "torus"}:
        return np.full((height, width), 1.0 / (height * width), dtype=np.float64)
    raise ValueError(f"Unsupported topology kind {kind!r}")


def compute_neighbors(
    kind: TopologyKind,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_dimensions(height, width)
    if kind not in {"sphere", "cylinder", "torus"}:
        raise ValueError(f"Unsupported topology kind {kind!r}")

    rows, cols = np.indices((height, width), dtype=np.int32)
    neighbor_rows = rows[..., None] + _OFFSETS[:, 0]
    neighbor_cols = (cols[..., None] + _OFFSETS[:, 1]) % width

    if kind == "torus":
        neighbor_rows %= height
        valid = np.ones_like(neighbor_rows, dtype=bool)
    else:
        valid = (neighbor_rows >= 0) & (neighbor_rows < height)

    neighbor_rows = np.clip(neighbor_rows, 0, height - 1)
    indices = neighbor_rows * width + neighbor_cols
    indices = np.where(valid, indices, -1).astype(np.int32, copy=False)

    weights = np.broadcast_to(_WEIGHTS, indices.shape).copy()
    weights[~valid] = 0.0
    return indices, weights


__all__ = ["TopologyKind", "compute_cell_areas", "compute_neighbors"]
