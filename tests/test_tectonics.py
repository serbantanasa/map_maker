from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._tectonics_native import PLATE_FIELD_COMPONENTS


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    reg = registry()
    reg.clear()
    module = importlib.import_module("map_maker.pipeline.stages.tectonics")
    importlib.reload(module)
    yield
    reg.clear()


def _make_config(tmp_path: Path, run_id: str, overrides: dict | None = None) -> PipelineConfig:
    base = {
        "topology": "sphere",
        "resolutions": [{"height": 64, "width": 128}],
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "log_dir": str(tmp_path / "logs"),
        "run_id": run_id,
    }
    if overrides:
        base["stage_overrides"] = overrides
    return PipelineConfig.from_mapping(base)


def test_tectonics_outputs_and_shapes(tmp_path: Path):
    config = _make_config(
        tmp_path,
        "tectonics-test",
        overrides={
            "tectonics": {
                "num_plates": 12,
                "velocity_scale": 0.8,
                "hotspot_density": 0.05,
            },
        },
    )
    engine = ExecutionEngine(config, generate_visuals=True)
    results = engine.run(["tectonics"])
    tectonics_result = results["tectonics"]

    plate_handle = tectonics_result.artifact_records["PlateField"].value
    assert plate_handle is not None
    plate_array = np.array(plate_handle.array(), copy=False)
    assert plate_array.shape == (64, 128, PLATE_FIELD_COMPONENTS)
    assert plate_array.dtype == np.float32

    conv = tectonics_result.artifact_records["BoundaryConvergence"].value
    assert conv is not None
    conv_array = np.array(conv.array(), copy=False)
    assert conv_array.shape == (64, 128)
    assert conv_array.dtype == np.float32

    hotspot = tectonics_result.artifact_records["HotspotMap"].value
    assert hotspot is not None
    hotspot_array = np.array(hotspot.array(), copy=False)
    assert hotspot_array.min() >= 0.0
    assert hotspot_array.max() <= 1.0 + 1e-6

    subduction = tectonics_result.artifact_records["BoundarySubduction"].value
    assert subduction is not None
    subduction_array = np.array(subduction.array(), copy=False)
    assert subduction_array.shape == (64, 128)
    assert np.all((subduction_array >= 0.0) & (subduction_array <= 1.0 + 1e-6))

    metadata_record = tectonics_result.artifact_records["TectonicsMetadata"]
    meta_dict = metadata_record.value
    assert isinstance(meta_dict, dict)
    assert meta_dict.get("num_plates") == 12
    assert "velocity_mean" in meta_dict
    assert "convergence_sum" in meta_dict
    assert "hotspot_count" in meta_dict
    assert "continental_fraction" in meta_dict

    visuals_dir = config.run_visual_dir() / "tectonics"
    assert (visuals_dir / "plates.png").exists()


def test_tectonics_cache_hit(tmp_path: Path):
    config = _make_config(tmp_path, "tectonics-cache")
    engine1 = ExecutionEngine(config)
    engine1.run(["tectonics"])
    engine2 = ExecutionEngine(config)
    result = engine2.run(["tectonics"])["tectonics"]
    assert result.stats is not None
    assert result.stats.cache_hit


def test_tectonics_determinism_and_ratios(tmp_path: Path):
    overrides = {
        "tectonics": {
            "num_plates": 16,
            "continental_fraction": 0.4,
            "lloyd_iterations": 4,
            "time_steps": 12,
            "time_step": 0.6,
            "velocity_scale": 1.1,
        }
    }
    config = _make_config(tmp_path, "tectonics-determinism", overrides=overrides)
    engine1 = ExecutionEngine(config)
    res1 = engine1.run(["tectonics"])["tectonics"]
    engine2 = ExecutionEngine(config)
    res2 = engine2.run(["tectonics"])["tectonics"]

    plate1 = np.array(res1.artifact_records["PlateField"].value.array(), copy=True)
    plate2 = np.array(res2.artifact_records["PlateField"].value.array(), copy=True)
    assert np.allclose(plate1, plate2)

    conv1 = np.array(res1.artifact_records["BoundaryConvergence"].value.array(), copy=True)
    conv2 = np.array(res2.artifact_records["BoundaryConvergence"].value.array(), copy=True)
    assert np.allclose(conv1, conv2)

    continental_mask = plate1[..., 1]
    actual_fraction = float(continental_mask.mean())
    assert abs(actual_fraction - 0.4) <= 0.15

    plate_ids = plate1[..., 0].astype(np.int32)
    divergence = np.array(res1.artifact_records["BoundaryDivergence"].value.array(), copy=False)
    diff_mask_east = plate_ids[:, :-1] != plate_ids[:, 1:]
    if diff_mask_east.any():
        east_diff = np.abs(conv1[:, :-1][diff_mask_east] - divergence[:, 1:][diff_mask_east])
        assert east_diff.mean() < 0.08
    diff_mask_south = plate_ids[:-1, :] != plate_ids[1:, :]
    if diff_mask_south.any():
        south_diff = np.abs(conv1[:-1, :][diff_mask_south] - divergence[1:, :][diff_mask_south])
        assert south_diff.mean() < 0.12

    subduction = np.array(res1.artifact_records["BoundarySubduction"].value.array(), copy=False)
    assert np.all((subduction >= 0.0) & (subduction <= 1.0 + 1e-6))
