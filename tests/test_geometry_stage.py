from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.cubed_sphere import CubedSphereGrid


@pytest.fixture(autouse=True)
def _register_geometry_stage():
    reg = registry()
    reg.clear()
    module = importlib.import_module("map_maker.pipeline.stages.geometry")
    importlib.reload(module)
    yield
    reg.clear()


def _config(tmp_path: Path, run_id: str) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 12}],
            "rng_seed": 41,
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
        }
    )


def test_cubed_sphere_resolution_config_contract(tmp_path: Path):
    config = _config(tmp_path, "geometry-config")
    level = config.resolution_set.native
    assert level.face_resolution == 12
    assert level.height == 12
    assert level.width == 12
    assert level.to_dict()["face_count"] == 6

    with pytest.raises(ValueError, match="require face_resolution"):
        PipelineConfig.from_mapping(
            {"topology": "cubed_sphere", "resolutions": [{"height": 12, "width": 12}]}
        )
    with pytest.raises(ValueError, match="only valid"):
        PipelineConfig.from_mapping(
            {"topology": "sphere", "resolutions": [{"face_resolution": 12}]}
        )


def test_geometry_stage_persists_canonical_topology(tmp_path: Path):
    config = _config(tmp_path, "geometry-stage")
    engine = ExecutionEngine(config, generate_visuals=True)
    assert isinstance(engine.context.topology, CubedSphereGrid)
    result = engine.run(["geometry"])["geometry"]

    xyz = result.artifact_records["GeometryXYZ"].value
    areas = result.artifact_records["CellArea"].value
    neighbors = result.artifact_records["NeighborsD4"].value
    metadata = result.artifact_records["TopologyMetadata"].value
    assert isinstance(xyz, np.ndarray)
    assert isinstance(areas, np.ndarray)
    assert isinstance(neighbors, np.ndarray)
    assert xyz.shape == (6, 12, 12, 3)
    assert areas.shape == (6, 12, 12)
    assert neighbors.shape == (6, 12, 12, 4)
    assert math.isclose(float(np.sum(areas)), 4.0 * math.pi, abs_tol=1e-12)
    assert metadata["canonical"] is True
    assert metadata["face_resolution"] == 12
    assert (config.run_visual_dir() / "geometry" / "cube_net.png").is_file()


def test_geometry_stage_cache_replay_preserves_arrays(tmp_path: Path):
    config = _config(tmp_path, "geometry-cache")
    first = ExecutionEngine(config).run(["geometry"])["geometry"]
    second = ExecutionEngine(config).run(["geometry"])["geometry"]
    assert second.stats is not None and second.stats.cache_hit
    np.testing.assert_array_equal(
        first.artifact_records["NeighborsD4"].value,
        second.artifact_records["NeighborsD4"].value,
    )
