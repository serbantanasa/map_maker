from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline.config import GridInfo
from map_maker.pipeline.topology import (
    CylinderTopology,
    SphereTopology,
    TorusTopology,
)
from map_maker.pipeline.tools import run_topology_visualizer


def _sphere_topology() -> SphereTopology:
    return SphereTopology(GridInfo(height=12, width=24))


def _cylinder_topology() -> CylinderTopology:
    return CylinderTopology(GridInfo(height=10, width=20))


def _torus_topology() -> TorusTopology:
    return TorusTopology(GridInfo(height=8, width=16))


def test_sphere_area_integrates_surface():
    topo = _sphere_topology()
    total_area = topo._cell_area.sum()
    assert math.isclose(total_area, 4 * math.pi, rel_tol=1e-3)


def test_lat_lon_monotonic_and_no_clipping():
    topo = _sphere_topology()
    lat = topo.lat[:, 0]
    assert np.all(np.diff(lat) < 0), "Latitude should decrease from north to south"
    assert lat[0] < math.pi / 2 and lat[-1] > -math.pi / 2


def test_neighbors_wrap_longitude_on_sphere():
    topo = _sphere_topology()
    neighbors = topo.neighbors(5, 0).indices
    wrapped_index = 5 * topo.shape[1] + (topo.shape[1] - 1)
    assert wrapped_index in neighbors


def test_cylinder_wraps_only_longitude():
    topo = _cylinder_topology()
    wrapped = topo.wrap((-3, -1))
    assert wrapped == (0, topo.shape[1] - 1)
    lower = topo.wrap((topo.shape[0] + 2, 5))
    assert lower == (topo.shape[0] - 1, 5)


def test_torus_wraps_both_axes_and_neighbors_connect_edges():
    topo = _torus_topology()
    wrapped = topo.wrap((-1, -1))
    assert wrapped == (topo.shape[0] - 1, topo.shape[1] - 1)
    neighbors = topo.neighbors(0, 0).indices
    expected = (topo.shape[0] - 1) * topo.shape[1] + (topo.shape[1] - 1)
    assert expected in neighbors


def test_neighbors_modes_and_weights():
    topo = _sphere_topology()
    view_d8 = topo.neighbors(6, 6)
    view_d4 = topo.neighbors(6, 6, mode="D4")
    assert len(view_d4.indices) == 4
    assert len(view_d8.indices) >= len(view_d4.indices)
    assert all(weight in (1.0, math.sqrt(2)) for weight in view_d8.weights if weight)


def test_distance_symmetry():
    topo = _torus_topology()
    a = (0, 0)
    b = (3, 7)
    assert math.isclose(topo.distance(a, b), topo.distance(b, a))


def test_child_mapping_requires_divisible_resolution():
    topo = _sphere_topology()
    mapping = topo.child_mapping(1)
    assert mapping.shape == (6, 12, 2)
    with pytest.raises(ValueError):
        topo.child_mapping(4)


def test_run_topology_visualizer_generates_png(tmp_path: Path):
    visuals_dir, elapsed_ms = run_topology_visualizer(
        topology="sphere",
        width=64,
        height=32,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
        run_id="preview-test",
    )
    assert visuals_dir.exists()
    expected = {
        "lon_gradient.png",
        "lat_gradient.png",
        "wrap_checker.png",
        "wrap_gradient.png",
        "z_height.png",
    }
    files = {path.name for path in visuals_dir.glob("*.png")}
    assert expected.issubset(files)
    assert elapsed_ms >= 0.0
