"""Visualization support for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from PIL import Image

from .models import ArtifactRecord, StageResult


@dataclass(frozen=True)
class VisualizationRequest:
    stage_name: str
    output_dir: Path
    artifacts: List[ArtifactRecord]


@dataclass(frozen=True)
class VisualizationResult:
    path: Path
    artifact_name: str
    metadata: dict[str, Any]


class VisualManager:
    """Handles rendering PNG previews for stage artifacts."""

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root
        output_root.mkdir(parents=True, exist_ok=True)

    def make_request(self, stage_result: StageResult) -> VisualizationRequest:
        stage_dir = self._output_root / stage_result.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return VisualizationRequest(
            stage_name=stage_result.stage_name,
            output_dir=stage_dir,
            artifacts=list(stage_result.artifact_records.values()),
        )

    def emit(self, stage_result: StageResult, default_artifact: Optional[str] = None) -> list[VisualizationResult]:
        request = self.make_request(stage_result)
        artifacts = request.artifacts
        if not artifacts:
            return []
        results: list[VisualizationResult] = []
        chosen = (
            [next((artifact for artifact in artifacts if artifact.name == default_artifact), artifacts[0])]
            if default_artifact
            else [artifacts[0]]
        )

        for idx, artifact in enumerate(chosen):
            value = artifact.value
            if value is None:
                continue
            image = _render_value(value)
            if image is None:
                continue
            filename = f"{artifact.name}-{idx}.png" if idx else f"{artifact.name}.png"
            path = request.output_dir / filename
            image.save(path)
            results.append(
                VisualizationResult(
                    path=path,
                    artifact_name=artifact.name,
                    metadata={"checksum": artifact.checksum},
                )
            )
        return results

    def emit_custom(
        self, stage_result: StageResult, visualizer: "StageVisualizer"
    ) -> list[VisualizationResult]:
        request = self.make_request(stage_result)
        result = visualizer(stage_result, request)
        if result is None:
            return []
        if isinstance(result, VisualizationResult):
            return [result]
        return list(result)


def _render_value(value: Any) -> Image.Image | None:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "array"):
        array = value.array()  # type: ignore[assignment]
    else:
        return None
    if array.ndim == 2:
        data = _normalize_array(array)
        return Image.fromarray(data, mode="L")
    if array.ndim == 3 and array.shape[2] in (3, 4):
        if array.dtype != np.uint8 or array.min() < 0 or array.max() > 255:
            data = _colorize_xyz(array)
        else:
            data = np.clip(array, 0, 255).astype(np.uint8)
        mode = "RGBA" if data.shape[2] == 4 else "RGB"
        return Image.fromarray(data, mode=mode)
    return None


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    low = float(np.percentile(finite, 2))
    high = float(np.percentile(finite, 98))
    if math.isclose(low, high):
        high = low + 1.0
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _colorize_xyz(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[-1] < 3:
        pad_width = ((0, 0),) * (arr.ndim - 1) + ((0, 3 - arr.shape[-1]),)
        arr = np.pad(arr, pad_width, mode="constant")

    x = arr[..., 0]
    y = arr[..., 1]
    z = arr[..., 2]

    angle = np.arctan2(y, x)
    hue = (angle + math.pi) / (2 * math.pi)

    magnitude = np.sqrt(x * x + y * y)
    if magnitude.size:
        mag_max = np.percentile(magnitude, 98)
    else:
        mag_max = 0.0
    if mag_max <= 1e-9:
        mag_max = 1.0
    sat = np.clip(magnitude / mag_max, 0.0, 1.0)

    value = np.clip((z - z.min()) / (z.max() - z.min() + 1e-9), 0.25, 1.0)

    rgb = _hsv_to_rgb(hue, sat, value)
    return (rgb * 255).astype(np.uint8)


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.asarray(h) % 1.0
    s = np.asarray(s)
    v = np.asarray(v)

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack((r, g, b), axis=-1)


__all__ = [
    "VisualManager",
    "VisualizationRequest",
    "VisualizationResult",
]
