from __future__ import annotations

from pathlib import Path

import numpy as np

from map_maker.pipeline.models import ArtifactRecord, StageResult
from map_maker.pipeline.visualization import VisualManager, _colorize_xyz


def test_colorize_xyz_produces_rgb():
    height, width = 16, 32
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    xv, yv = np.meshgrid(x, y)
    z = np.sin(np.pi * xv) * np.cos(np.pi * yv)
    arr = np.stack((xv, yv, z), axis=-1)
    rgb = _colorize_xyz(arr)
    assert rgb.shape == (height, width, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255


def test_visual_manager_emits_png(tmp_path: Path):
    vm = VisualManager(tmp_path)
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    arr[..., 0] = np.linspace(-1.0, 1.0, 8)
    arr[..., 1] = arr[..., 0].T
    arr[..., 2] = 0.5

    stage_result = StageResult(stage_name="demo", dependencies=(), raw_artifacts={})
    artifact = ArtifactRecord(
        name="coords",
        kind="ndarray",
        checksum="dummy",
        dataset_path=tmp_path / "coords.npy",
        cache_path=tmp_path / "coords.npy",
        metadata={}
    )
    artifact.value = arr
    stage_result.register_artifact(artifact)

    outputs = vm.emit(stage_result)
    assert outputs, "Visualizer should produce at least one output"
    assert outputs[0].path.exists()
    assert outputs[0].path.suffix == ".png"

