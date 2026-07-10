"""End-to-end generation and cartographic preview output."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter

from .config import PipelineConfig
from .execution import ExecutionEngine
from .models import ArtifactRecord, StageResult


@dataclass(frozen=True)
class GenerationResult:
    """Paths and stage results produced by a complete pipeline run."""

    run_dir: Path
    image_path: Path
    manifest_path: Path
    elapsed_seconds: float
    stages: Mapping[str, StageResult]


def _array(record: ArtifactRecord | None, name: str) -> np.ndarray:
    if record is None or record.value is None:
        raise KeyError(f"Missing generated artifact '{name}'")
    value = record.value
    if hasattr(value, "array"):
        return np.asarray(value.array())
    return np.asarray(value)


def _scaled(values: np.ndarray, mask: np.ndarray, *, invert: bool = False) -> np.ndarray:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(selected, (2.0, 98.0))
    if high - low < 1e-9:
        high = low + 1.0
    result = np.clip((values - low) / (high - low), 0.0, 1.0)
    if invert:
        result = 1.0 - result
    return result.astype(np.float32, copy=False)


def _quantile_scaled(values: np.ndarray, mask: np.ndarray, *, invert: bool = False) -> np.ndarray:
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    quantiles = np.percentile(selected, (2.0, 20.0, 50.0, 80.0, 95.0, 99.5))
    quantiles = np.maximum.accumulate(quantiles)
    for idx in range(1, len(quantiles)):
        if quantiles[idx] - quantiles[idx - 1] < 1e-9:
            quantiles[idx] = quantiles[idx - 1] + 1e-9
    targets = np.array((0.0, 0.14, 0.34, 0.56, 0.8, 1.0), dtype=np.float32)
    result = np.interp(values, quantiles, targets).astype(np.float32, copy=False)
    if invert:
        result = 1.0 - result
    return result


def _smooth_for_rendering(values: np.ndarray, passes: int = 4) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for _ in range(passes):
        north = np.pad(result[:-1], ((1, 0), (0, 0)), mode="edge")
        south = np.pad(result[1:], ((0, 1), (0, 0)), mode="edge")
        east = np.roll(result, -1, axis=1)
        west = np.roll(result, 1, axis=1)
        result = (result * 4.0 + north + south + east + west) / 8.0
    return result


def _coastal_depth(ocean: np.ndarray) -> np.ndarray:
    height, width = ocean.shape
    land = (~ocean).astype(np.uint8) * 255
    tiled = np.concatenate((land, land, land), axis=1)
    radius = max(3.0, min(height, width) / 22.0)
    blurred = Image.fromarray(tiled, mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    coast_influence = np.asarray(blurred, dtype=np.float32)[:, width : 2 * width] / 255.0
    depth = np.clip(1.0 - coast_influence * 2.15, 0.0, 1.0)
    depth[~ocean] = 0.0
    return np.sqrt(depth).astype(np.float32, copy=False)


def _palette(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    positions = np.array([position for position, _ in stops], dtype=np.float32)
    colors = np.array([color for _, color in stops], dtype=np.float32)
    channels = [np.interp(values, positions, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def render_world(
    elevation: np.ndarray,
    ocean_mask: np.ndarray,
    river_incision: np.ndarray | None = None,
    sediment_depth: np.ndarray | None = None,
) -> Image.Image:
    """Render a stable physical-map preview from persisted simulation fields."""

    elevation = np.asarray(elevation, dtype=np.float32)
    ocean = np.asarray(ocean_mask) >= 0.5
    if elevation.ndim != 2 or ocean.shape != elevation.shape:
        raise ValueError("elevation and ocean_mask must be matching 2D arrays")

    land = ~ocean
    display_elevation = _smooth_for_rendering(elevation, passes=6)
    land_height = _quantile_scaled(display_elevation, land)
    water_depth = _coastal_depth(ocean)

    water_rgb = _palette(
        water_depth,
        [
            (0.0, (100, 176, 196)),
            (0.35, (42, 116, 164)),
            (0.7, (20, 66, 116)),
            (1.0, (8, 30, 67)),
        ],
    )
    land_rgb = _palette(
        land_height,
        [
            (0.0, (73, 126, 73)),
            (0.22, (115, 145, 82)),
            (0.48, (166, 151, 102)),
            (0.72, (143, 122, 105)),
            (0.9, (202, 198, 185)),
            (1.0, (245, 246, 242)),
        ],
    )
    rgb = np.where(ocean[..., None], water_rgb, land_rgb)

    horizontal = (
        np.roll(display_elevation, -1, axis=1) - np.roll(display_elevation, 1, axis=1)
    ) * 0.5
    vertical = np.gradient(display_elevation, axis=0)
    relief_scale = float(np.percentile(np.hypot(horizontal, vertical), 95))
    relief_scale = max(relief_scale, 1e-6)
    nx = -horizontal / relief_scale
    ny = vertical / relief_scale
    nz = np.ones_like(elevation)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = (nx * -0.45 + ny * -0.35 + nz * 0.82) / norm
    shade = np.clip(0.72 + 0.32 * shade, 0.55, 1.12)
    rgb[land] *= shade[land, None]

    if sediment_depth is not None:
        sediment = np.asarray(sediment_depth, dtype=np.float32)
        if sediment.shape == elevation.shape:
            sediment_strength = _scaled(sediment, land)
            sediment_weight = (sediment_strength * 0.12)[..., None]
            sediment_color = np.array([174.0, 154.0, 107.0], dtype=np.float32)
            rgb[land] = (
                rgb[land] * (1.0 - sediment_weight[land]) + sediment_color * sediment_weight[land]
            )

    if river_incision is not None:
        incision = np.asarray(river_incision, dtype=np.float32)
        positive = incision[land & (incision > 0.0)] if incision.shape == elevation.shape else []
        if len(positive):
            threshold = float(np.percentile(positive, 82.0))
            high = float(np.percentile(positive, 99.5))
            scale = max(high - threshold, 1e-6)
            river = np.clip((incision - threshold) / scale, 0.0, 1.0) * land
            river_weight = (river * 0.8)[..., None]
            river_color = np.array([24.0, 93.0, 151.0], dtype=np.float32)
            rgb = rgb * (1.0 - river_weight) + river_color * river_weight

    shoreline = land & (
        np.roll(ocean, 1, axis=1)
        | np.roll(ocean, -1, axis=1)
        | np.pad(ocean[1:], ((0, 1), (0, 0)), constant_values=False)
        | np.pad(ocean[:-1], ((1, 0), (0, 0)), constant_values=False)
    )
    rgb[shoreline] *= 0.82

    return Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _stage_manifest(stage: StageResult, run_dir: Path) -> dict[str, Any]:
    stats = stage.stats
    return {
        "cache_key": stage.cache_key,
        "cache_hit": bool(stats.cache_hit) if stats else False,
        "duration_seconds": (stats.duration_ns / 1_000_000_000.0) if stats else None,
        "artifacts": {
            name: {
                "kind": record.kind,
                "checksum": record.checksum,
                "path": _relative_path(record.dataset_path, run_dir),
                "metadata": record.metadata,
            }
            for name, record in stage.artifact_records.items()
        },
    }


def generate_world(
    config: PipelineConfig,
    *,
    generate_stage_visuals: bool = True,
) -> GenerationResult:
    """Run the implemented physical stack through erosion and render its present state."""

    from .stages import ensure_builtin_stages

    ensure_builtin_stages()
    started = time.perf_counter()
    stages = ExecutionEngine(config, generate_visuals=generate_stage_visuals).run(["erosion"])

    world_age = stages["world_age"]
    erosion = stages["erosion"]
    elevation = _array(erosion.artifact_records.get("ElevationRaw"), "ElevationRaw")
    sediment = _array(erosion.artifact_records.get("SedimentDepth"), "SedimentDepth")
    ocean = _array(world_age.artifact_records.get("BaseOceanMask"), "BaseOceanMask")

    run_dir = config.run_output_dir()
    image_path = run_dir / "world.png"
    # RiverIncision is persisted for diagnostics, but it is not a routed river
    # network yet and must not be presented as one on the physical map.
    render_world(elevation, ocean, sediment_depth=sediment).save(image_path)

    land = ocean < 0.5
    elapsed = time.perf_counter() - started
    manifest = {
        "format_version": 1,
        "status": "complete",
        "run_id": config.run_id,
        "seed": config.rng_seed,
        "topology": config.topology,
        "resolution": config.resolution_set.native.to_dict(),
        "elapsed_seconds": elapsed,
        "preview": image_path.name,
        "statistics": {
            "land_fraction": float(np.mean(land)),
            "elevation_min": float(np.min(elevation)),
            "elevation_mean": float(np.mean(elevation)),
            "elevation_max": float(np.max(elevation)),
        },
        "stages": {name: _stage_manifest(stage, run_dir) for name, stage in stages.items()},
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")

    return GenerationResult(
        run_dir=run_dir,
        image_path=image_path,
        manifest_path=manifest_path,
        elapsed_seconds=elapsed,
        stages=stages,
    )


__all__ = ["GenerationResult", "generate_world", "render_world"]
