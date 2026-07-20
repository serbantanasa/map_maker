"""Fixed-seed integration validation and visual gallery generation."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .config import PipelineConfig
from .generate import GenerationResult, generate_world


@dataclass(frozen=True)
class ValidationThresholds:
    min_land_fraction: float = 0.18
    max_land_fraction: float = 0.36
    min_land_components: int = 2
    max_largest_landmass_fraction: float = 0.98
    min_mixed_plate_fraction: float = 0.40
    max_longitude_seam_ratio: float = 2.0
    max_plate_boundary_relief_ratio: float = 3.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ValidationThresholds":
        values = values or {}
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown validation thresholds: {', '.join(sorted(unknown))}")
        return cls(**values)


@dataclass(frozen=True)
class ValidationConfig:
    base_config: Path
    seeds: tuple[int, ...]
    output_dir: Path
    thresholds: ValidationThresholds

    @classmethod
    def from_file(cls, path: Path | str, *, output_dir: Path | None = None) -> "ValidationConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("Validation config must contain a mapping")
        raw_base = data.get("base_config")
        if not raw_base:
            raise ValueError("Validation config requires base_config")
        base_config = (path.parent / str(raw_base)).resolve()
        seeds = tuple(int(seed) for seed in data.get("seeds", ()))
        if len(seeds) < 2:
            raise ValueError("Validation config requires at least two seeds")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Validation seeds must be unique")
        if output_dir is None:
            raw_output = data.get("output_dir", "../out/validation")
            resolved_output = (path.parent / str(raw_output)).resolve()
        else:
            resolved_output = output_dir.expanduser().resolve()
        thresholds = ValidationThresholds.from_mapping(data.get("provisional_thresholds"))
        return cls(
            base_config=base_config,
            seeds=seeds,
            output_dir=resolved_output,
            thresholds=thresholds,
        )


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: Any
    expectation: str


@dataclass(frozen=True)
class WorldMetrics:
    finite_fields: bool
    land_fraction: float
    land_component_count: int
    largest_landmass_fraction: float
    mixed_plate_fraction: float
    longitude_seam_ratio: float
    plate_boundary_relief_ratio: float
    elevation_min: float
    elevation_mean: float
    elevation_max: float
    elapsed_seconds: float


@dataclass(frozen=True)
class WorldValidation:
    seed: int
    image_path: Path
    image_checksum: str
    metrics: WorldMetrics
    gates: tuple[GateResult, ...]
    cache_replay_passed: bool

    @property
    def passed(self) -> bool:
        return self.cache_replay_passed and all(gate.passed for gate in self.gates)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    report_path: Path
    gallery_path: Path
    worlds: tuple[WorldValidation, ...]
    global_gates: tuple[GateResult, ...]


def _artifact_array(result: GenerationResult, stage: str, name: str) -> np.ndarray:
    record = result.stages[stage].artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing validation artifact {stage}.{name}")
    value = record.value
    if hasattr(value, "array"):
        return np.asarray(value.array())
    return np.asarray(value)


def _major_component_sizes(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    minimum_size = max(4, mask.size // 5000)
    sizes: list[int] = []

    for start in np.flatnonzero(mask & ~visited):
        row, col = divmod(int(start), width)
        if visited[row, col]:
            continue
        visited[row, col] = True
        queue: deque[tuple[int, int]] = deque([(row, col)])
        size = 0
        while queue:
            current_row, current_col = queue.popleft()
            size += 1
            neighbors = (
                (current_row, (current_col - 1) % width),
                (current_row, (current_col + 1) % width),
                (current_row - 1, current_col),
                (current_row + 1, current_col),
            )
            for next_row, next_col in neighbors:
                if next_row < 0 or next_row >= height:
                    continue
                if mask[next_row, next_col] and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        if size >= minimum_size:
            sizes.append(size)
    return sorted(sizes, reverse=True)


def _mixed_plate_fraction(plate_ids: np.ndarray, land: np.ndarray) -> float:
    mixed = 0
    plate_count = 0
    for plate_id in np.unique(plate_ids):
        cells = plate_ids == plate_id
        if not np.any(cells):
            continue
        plate_count += 1
        land_fraction = float(np.mean(land[cells]))
        if 0.05 < land_fraction < 0.95:
            mixed += 1
    return mixed / plate_count if plate_count else 0.0


def _longitude_seam_ratio(elevation: np.ndarray) -> float:
    seam = np.abs(elevation[:, 0] - elevation[:, -1])
    interior = np.abs(np.diff(elevation, axis=1))
    baseline = float(np.mean(interior))
    return float(np.mean(seam)) / max(baseline, 1e-6)


def _plate_boundary_relief_ratio(plate_ids: np.ndarray, elevation: np.ndarray) -> float:
    boundary = plate_ids != np.roll(plate_ids, -1, axis=1)
    boundary[:-1] |= plate_ids[:-1] != plate_ids[1:]
    boundary[1:] |= plate_ids[1:] != plate_ids[:-1]
    horizontal = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) * 0.5
    vertical = np.gradient(elevation, axis=0)
    relief = np.hypot(horizontal, vertical)
    boundary_mean = float(np.mean(relief[boundary])) if np.any(boundary) else 0.0
    background = relief[~boundary]
    background_mean = float(np.mean(background)) if background.size else 0.0
    return boundary_mean / max(background_mean, 1e-6)


def _field_values(result: GenerationResult) -> list[np.ndarray]:
    fields: list[np.ndarray] = []
    for stage in result.stages.values():
        for record in stage.artifact_records.values():
            value = record.value
            if hasattr(value, "array"):
                fields.append(np.asarray(value.array()))
            elif isinstance(value, np.ndarray):
                fields.append(value)
    return fields


def _world_metrics(result: GenerationResult) -> WorldMetrics:
    plate_field = _artifact_array(result, "tectonics", "PlateField")
    ocean = _artifact_array(result, "world_age", "BaseOceanMask") >= 0.5
    elevation = _artifact_array(result, "erosion", "ElevationRaw").astype(np.float32, copy=False)
    land = ~ocean
    component_sizes = _major_component_sizes(land)
    land_cells = int(np.count_nonzero(land))
    largest_fraction = component_sizes[0] / land_cells if component_sizes and land_cells else 0.0
    plate_ids = plate_field[..., 0].astype(np.int32)
    finite = all(np.all(np.isfinite(field)) for field in _field_values(result))
    return WorldMetrics(
        finite_fields=finite,
        land_fraction=float(np.mean(land)),
        land_component_count=len(component_sizes),
        largest_landmass_fraction=float(largest_fraction),
        mixed_plate_fraction=_mixed_plate_fraction(plate_ids, land),
        longitude_seam_ratio=_longitude_seam_ratio(elevation),
        plate_boundary_relief_ratio=_plate_boundary_relief_ratio(plate_ids, elevation),
        elevation_min=float(np.min(elevation)),
        elevation_mean=float(np.mean(elevation)),
        elevation_max=float(np.max(elevation)),
        elapsed_seconds=result.elapsed_seconds,
    )


def _evaluate_world(
    metrics: WorldMetrics, thresholds: ValidationThresholds
) -> tuple[GateResult, ...]:
    return (
        GateResult("finite_fields", metrics.finite_fields, metrics.finite_fields, "all true"),
        GateResult(
            "land_fraction",
            thresholds.min_land_fraction <= metrics.land_fraction <= thresholds.max_land_fraction,
            metrics.land_fraction,
            f"{thresholds.min_land_fraction} <= value <= {thresholds.max_land_fraction}",
        ),
        GateResult(
            "land_components",
            metrics.land_component_count >= thresholds.min_land_components,
            metrics.land_component_count,
            f">= {thresholds.min_land_components}",
        ),
        GateResult(
            "largest_landmass_fraction",
            metrics.largest_landmass_fraction <= thresholds.max_largest_landmass_fraction,
            metrics.largest_landmass_fraction,
            f"<= {thresholds.max_largest_landmass_fraction}",
        ),
        GateResult(
            "mixed_plate_fraction",
            metrics.mixed_plate_fraction >= thresholds.min_mixed_plate_fraction,
            metrics.mixed_plate_fraction,
            f">= {thresholds.min_mixed_plate_fraction}",
        ),
        GateResult(
            "longitude_seam_ratio",
            metrics.longitude_seam_ratio <= thresholds.max_longitude_seam_ratio,
            metrics.longitude_seam_ratio,
            f"<= {thresholds.max_longitude_seam_ratio}",
        ),
        GateResult(
            "plate_boundary_relief_ratio",
            metrics.plate_boundary_relief_ratio <= thresholds.max_plate_boundary_relief_ratio,
            metrics.plate_boundary_relief_ratio,
            f"<= {thresholds.max_plate_boundary_relief_ratio}",
        ),
    )


def _world_config(
    base: PipelineConfig,
    *,
    seed: int,
    run_id: str,
    output_dir: Path,
    cache_dir: Path,
    log_dir: Path,
) -> PipelineConfig:
    config = deepcopy(base)
    config.rng_seed = seed
    config.run_id = run_id
    config.output_dir = output_dir
    config.cache_dir = cache_dir
    config.log_dir = log_dir
    return config


def _artifact_checksums(result: GenerationResult) -> dict[str, dict[str, str]]:
    return {
        stage_name: stage.artifact_checksums for stage_name, stage in sorted(result.stages.items())
    }


def _image_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gallery(worlds: list[WorldValidation], output_path: Path) -> None:
    columns = min(3, len(worlds))
    thumb_width = 384
    thumb_height = 192
    label_height = 28
    rows = (len(worlds) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "black"
    )
    draw = ImageDraw.Draw(canvas)
    for index, world in enumerate(worlds):
        row, column = divmod(index, columns)
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        with Image.open(world.image_path) as image:
            preview = image.convert("RGB").resize(
                (thumb_width, thumb_height), Image.Resampling.LANCZOS
            )
        canvas.paste(preview, (x, y))
        status = "PASS" if world.passed else "FAIL"
        draw.text((x + 8, y + thumb_height + 7), f"Seed {world.seed}  {status}", fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _gate_to_dict(gate: GateResult) -> dict[str, Any]:
    return asdict(gate)


def validate_gallery(config: ValidationConfig) -> ValidationResult:
    """Generate, measure, and report the fixed integration seed gallery."""

    base = PipelineConfig.from_file(config.base_config)
    output_dir = config.output_dir
    runs_dir = output_dir / "runs"
    cache_dir = output_dir / "cache"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    worlds: list[WorldValidation] = []
    generated: dict[int, GenerationResult] = {}
    for seed in config.seeds:
        world_config = _world_config(
            base,
            seed=seed,
            run_id=f"seed-{seed}",
            output_dir=runs_dir,
            cache_dir=cache_dir,
            log_dir=log_dir,
        )
        result = generate_world(world_config, generate_stage_visuals=False)
        image_checksum = _image_checksum(result.image_path)
        metrics = _world_metrics(result)
        gates = _evaluate_world(metrics, config.thresholds)

        replay = generate_world(world_config, generate_stage_visuals=False)
        cache_replay_passed = all(
            stage.stats is not None and stage.stats.cache_hit for stage in replay.stages.values()
        ) and image_checksum == _image_checksum(replay.image_path)
        cache_replay_passed = cache_replay_passed and _artifact_checksums(
            result
        ) == _artifact_checksums(replay)
        generated[seed] = result
        worlds.append(
            WorldValidation(
                seed=seed,
                image_path=result.image_path,
                image_checksum=image_checksum,
                metrics=metrics,
                gates=gates,
                cache_replay_passed=cache_replay_passed,
            )
        )

    first_seed = config.seeds[0]
    reference = generated[first_seed]
    with tempfile.TemporaryDirectory(prefix="cold-determinism-", dir=output_dir) as temporary:
        temporary_path = Path(temporary)
        cold_config = _world_config(
            base,
            seed=first_seed,
            run_id=f"seed-{first_seed}-cold",
            output_dir=temporary_path / "runs",
            cache_dir=temporary_path / "cache",
            log_dir=temporary_path / "logs",
        )
        cold = generate_world(cold_config, generate_stage_visuals=False)
        cold_determinism = _artifact_checksums(reference) == _artifact_checksums(
            cold
        ) and _image_checksum(reference.image_path) == _image_checksum(cold.image_path)

    image_checksums = {world.image_checksum for world in worlds}
    unique_worlds = len(image_checksums) == len(worlds)
    cache_replay = all(world.cache_replay_passed for world in worlds)
    global_gates = (
        GateResult(
            "cold_determinism",
            cold_determinism,
            cold_determinism,
            "all artifact and preview checksums match",
        ),
        GateResult(
            "unique_seed_outputs",
            unique_worlds,
            len(image_checksums),
            f"{len(worlds)} unique previews",
        ),
        GateResult(
            "cache_replay",
            cache_replay,
            cache_replay,
            "every stage is a cache hit and preview is unchanged",
        ),
    )
    passed = all(world.passed for world in worlds) and all(gate.passed for gate in global_gates)

    gallery_path = output_dir / "gallery.png"
    _write_gallery(worlds, gallery_path)
    report_path = output_dir / "report.json"
    report = {
        "format_version": 1,
        "status": "pass" if passed else "fail",
        "human_gallery_review_required": True,
        "base_config": str(config.base_config),
        "gallery": gallery_path.name,
        "provisional_thresholds": asdict(config.thresholds),
        "global_gates": [_gate_to_dict(gate) for gate in global_gates],
        "worlds": [
            {
                "seed": world.seed,
                "status": "pass" if world.passed else "fail",
                "image": str(world.image_path),
                "image_checksum": world.image_checksum,
                "cache_replay_passed": world.cache_replay_passed,
                "metrics": asdict(world.metrics),
                "gates": [_gate_to_dict(gate) for gate in world.gates],
            }
            for world in worlds
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    return ValidationResult(
        passed=passed,
        report_path=report_path,
        gallery_path=gallery_path,
        worlds=tuple(worlds),
        global_gates=global_gates,
    )


__all__ = [
    "GateResult",
    "ValidationConfig",
    "ValidationResult",
    "ValidationThresholds",
    "WorldMetrics",
    "WorldValidation",
    "validate_gallery",
]
