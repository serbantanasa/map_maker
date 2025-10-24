from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    reg = registry()
    reg.clear()
    importlib.invalidate_caches()
    tectonics = importlib.import_module("map_maker.pipeline.stages.tectonics")
    world_age = importlib.import_module("map_maker.pipeline.stages.world_age")
    erosion = importlib.import_module("map_maker.pipeline.stages.erosion")
    importlib.reload(tectonics)
    importlib.reload(world_age)
    importlib.reload(erosion)
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
        "stage_overrides": overrides or {},
    }
    return PipelineConfig.from_mapping(base)


def test_erosion_outputs(tmp_path: Path):
    overrides = {
        "erosion": {
            "erosion_steps": 6,
            "dt": 0.9,
            "stream_power_k": 0.02,
            "sediment_capacity": 0.03,
        }
    }
    config = _make_config(tmp_path, "erosion-basic", overrides=overrides)
    engine = ExecutionEngine(config)
    results = engine.run(["tectonics", "world_age", "erosion"])
    erosion_result = results["erosion"]

    elevation = np.array(erosion_result.artifact_records["ElevationRaw"].value.array(), copy=False)
    sediment = np.array(erosion_result.artifact_records["SedimentDepth"].value.array(), copy=False)
    incision = np.array(erosion_result.artifact_records["RiverIncision"].value.array(), copy=False)

    assert elevation.shape == (64, 128)
    assert sediment.shape == (64, 128)
    assert incision.shape == (64, 128)
    assert elevation.dtype == np.float32
    assert sediment.dtype == np.float32
    assert incision.dtype == np.float32
    assert np.all(np.isfinite(elevation))
    assert np.all(np.isfinite(sediment))
    assert np.all(np.isfinite(incision))
    assert np.all(sediment >= 0.0)
    assert np.all(incision >= 0.0)

    diagnostics_record = erosion_result.artifact_records["ErosionDiagnostics"].value
    assert isinstance(diagnostics_record, pa.Table)
    assert set(diagnostics_record.column_names) == {"step", "mean_elevation", "mass_removed", "mass_deposited"}

    metadata = erosion_result.artifact_records["ErosionMetadata"].value
    assert isinstance(metadata, dict)
    for key in ("total_mass_removed", "total_mass_deposited", "final_mean_elevation"):
        assert key in metadata
    assert metadata["steps"] == 6
    assert metadata["hotspot_count"] >= 0
    sediment_mass = float(metadata.get("sediment_mass", 0.0))
    total_removed = float(metadata["total_mass_removed"])
    total_deposited = float(metadata["total_mass_deposited"])
    residual = float(metadata.get("mass_residual", 0.0))
    assert sediment_mass >= 0.0
    assert total_removed >= 0.0 and total_deposited >= 0.0
    # Residual should be small compared to total removal once sediment storage is accounted for
    assert abs(residual) <= max(1e-3, 0.01 * max(1.0, total_removed))


def test_erosion_determinism(tmp_path: Path):
    overrides = {
        "erosion": {"erosion_steps": 5, "dt": 0.7, "stream_power_k": 0.018, "sediment_capacity": 0.028}
    }
    config1 = _make_config(tmp_path, "erosion-det-a", overrides=overrides)
    config2 = _make_config(tmp_path, "erosion-det-b", overrides=overrides)

    engine1 = ExecutionEngine(config1)
    engine2 = ExecutionEngine(config2)

    res1 = engine1.run(["tectonics", "world_age", "erosion"])["erosion"]
    res2 = engine2.run(["tectonics", "world_age", "erosion"])["erosion"]

    elev1 = np.array(res1.artifact_records["ElevationRaw"].value.array(), copy=False)
    elev2 = np.array(res2.artifact_records["ElevationRaw"].value.array(), copy=False)
    sed1 = np.array(res1.artifact_records["SedimentDepth"].value.array(), copy=False)
    sed2 = np.array(res2.artifact_records["SedimentDepth"].value.array(), copy=False)

    assert np.allclose(elev1, elev2)
    assert np.allclose(sed1, sed2)

    diag1 = res1.artifact_records["ErosionDiagnostics"].value
    diag2 = res2.artifact_records["ErosionDiagnostics"].value
    assert diag1.equals(diag2)
