"""Canonical geometry and topology artifact stage."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from ..cubed_sphere import FACE_NAMES, CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


def _artifact_array(result, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    if hasattr(value, "array"):
        return np.asarray(value.array())
    return np.asarray(value)


def _geometry_visualizer(
    result, request: VisualizationRequest
) -> Optional[list[VisualizationResult]]:
    xyz = _artifact_array(result, "GeometryXYZ")
    if xyz is None:
        return None

    if xyz.ndim == 4 and xyz.shape[0] == 6 and xyz.shape[-1] == 3:
        resolution = xyz.shape[1]
        colors = ((xyz.astype(np.float64) + 1.0) * 127.5).clip(0.0, 255.0).astype(np.uint8)
        net = np.zeros((resolution * 3, resolution * 4, 3), dtype=np.uint8)
        placements = {
            3: (1, 0),
            0: (1, 1),
            2: (1, 2),
            1: (1, 3),
            4: (0, 1),
            5: (2, 1),
        }
        for face, (net_row, net_col) in placements.items():
            row = net_row * resolution
            col = net_col * resolution
            net[row : row + resolution, col : col + resolution] = colors[face]
        path = request.output_dir / "cube_net.png"
        Image.fromarray(net, mode="RGB").save(path)
        return [VisualizationResult(path=path, artifact_name="GeometryXYZ", metadata={})]

    return None


@stage(
    "geometry",
    outputs=(
        "GeometryXYZ",
        "Longitude",
        "Latitude",
        "CellArea",
        "NeighborsD4",
        "TopologyMetadata",
    ),
    version="v1",
    visualizer=_geometry_visualizer,
)
def geometry_stage(context, deps, config):
    del deps, config
    topology = context.topology
    metadata = {
        "kind": str(topology.kind),
        "shape": list(topology.shape),
        "cell_count": int(np.prod(topology.shape, dtype=np.int64)),
        "area_total_steradians": float(np.sum(topology.cell_areas)),
    }
    if isinstance(topology, CubedSphereGrid):
        metadata.update(
            {
                "canonical": True,
                "face_count": 6,
                "face_order": list(FACE_NAMES),
                "face_resolution": topology.face_resolution,
                "neighbor_order": ["north", "south", "west", "east"],
            }
        )
    else:
        metadata["canonical"] = False

    return {
        "GeometryXYZ": topology.xyz,
        "Longitude": topology.lon,
        "Latitude": topology.lat,
        "CellArea": topology.cell_areas,
        "NeighborsD4": topology.neighbor_indices,
        "TopologyMetadata": metadata,
    }


__all__ = ["geometry_stage"]
