"""Fixed-seed stability screen for Causal Mineral Systems V0."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from PIL import Image, ImageDraw
import yaml  # type: ignore[import-untyped]

from .config import GridInfo, PipelineConfig, ResolutionSet
from .execution import ExecutionEngine
from .stages.mineral_systems import SYSTEM_COLORS, SYSTEM_NAMES
from .stages.sea_level import _equirectangular_rgb


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class MineralEnsembleThresholds:
    minimum_seed_count: int = 5
    minimum_family_pass_fraction: float = 0.80
    maximum_family_peak_coefficient_of_variation: float = 0.75

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "MineralEnsembleThresholds":
        mapping = mapping or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown mineral ensemble tolerances: {', '.join(sorted(unknown))}")
        minimum_seed_count = _integer(
            mapping.get("minimum_seed_count", cls.minimum_seed_count),
            name="minimum_seed_count",
        )
        minimum_pass = float(
            mapping.get("minimum_family_pass_fraction", cls.minimum_family_pass_fraction)
        )
        maximum_cv = float(
            mapping.get(
                "maximum_family_peak_coefficient_of_variation",
                cls.maximum_family_peak_coefficient_of_variation,
            )
        )
        if minimum_seed_count < 2:
            raise ValueError("minimum_seed_count must be at least 2")
        for name, value in (
            ("minimum_family_pass_fraction", minimum_pass),
            ("maximum_family_peak_coefficient_of_variation", maximum_cv),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        return cls(minimum_seed_count, minimum_pass, maximum_cv)


@dataclass(frozen=True)
class MineralEnsembleConfig:
    base_config: Path
    seeds: tuple[int, ...]
    output_dir: Path
    face_resolution: int | None
    screen_parent_boundary_residual_p95_ratio: float
    screen_maximum_parent_offset_span_m: float
    screen_maximum_parent_offset_span_relief_fraction: float
    screen_maximum_tile_bubble_correlation_p50: float
    screen_maximum_tile_bubble_correlation_p95: float
    thresholds: MineralEnsembleThresholds

    @classmethod
    def from_file(
        cls, path: Path | str, *, output_dir: Path | None = None
    ) -> "MineralEnsembleConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("Mineral ensemble config must contain a mapping")
        if _integer(data.get("format_version", 1), name="format_version") != 1:
            raise ValueError("Unsupported mineral ensemble format_version")
        raw_base = data.get("base_config")
        if not raw_base:
            raise ValueError("Mineral ensemble config requires base_config")
        base_config = (path.parent / str(raw_base)).resolve()
        thresholds = MineralEnsembleThresholds.from_mapping(
            cast(Mapping[str, object] | None, data.get("ensemble_tolerances"))
        )
        seeds = tuple(
            _integer(seed, name="seed") for seed in cast(Sequence[object], data.get("seeds", ()))
        )
        if len(seeds) < thresholds.minimum_seed_count:
            raise ValueError(
                f"Mineral ensemble requires at least {thresholds.minimum_seed_count} seeds"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("Mineral ensemble seeds must be unique")
        raw_resolution = data.get("face_resolution")
        face_resolution = (
            _integer(raw_resolution, name="face_resolution") if raw_resolution is not None else None
        )
        if face_resolution is not None and face_resolution <= 0:
            raise ValueError("face_resolution must be positive")
        screen_boundary_ratio = float(data.get("screen_parent_boundary_residual_p95_ratio", 2.25))
        if not np.isfinite(screen_boundary_ratio) or screen_boundary_ratio <= 0.0:
            raise ValueError(
                "screen_parent_boundary_residual_p95_ratio must be finite and positive"
            )
        screen_offset_span_m = float(data.get("screen_maximum_parent_offset_span_m", 10_000.0))
        screen_offset_relief = float(
            data.get("screen_maximum_parent_offset_span_relief_fraction", 20.0)
        )
        for name, value in (
            ("screen_maximum_parent_offset_span_m", screen_offset_span_m),
            (
                "screen_maximum_parent_offset_span_relief_fraction",
                screen_offset_relief,
            ),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        screen_tile_p50 = float(data.get("screen_maximum_tile_bubble_correlation_p50", 0.60))
        screen_tile_p95 = float(data.get("screen_maximum_tile_bubble_correlation_p95", 0.95))
        if not 0.0 <= screen_tile_p50 <= screen_tile_p95 <= 1.0:
            raise ValueError("screen tile-bubble correlations must satisfy 0 <= p50 <= p95 <= 1")
        resolved_output = (
            output_dir.expanduser().resolve()
            if output_dir is not None
            else (path.parent / str(data.get("output_dir", "../out/mineral-systems"))).resolve()
        )
        return cls(
            base_config,
            seeds,
            resolved_output,
            face_resolution,
            screen_boundary_ratio,
            screen_offset_span_m,
            screen_offset_relief,
            screen_tile_p50,
            screen_tile_p95,
            thresholds,
        )


@dataclass(frozen=True)
class MineralSeedReport:
    seed: int
    hard_gate_pass: bool
    hard_failures: tuple[str, ...]
    family_rows: tuple[Mapping[str, object], ...]
    candidate_count: int
    system_count: int
    state_checksum: str
    dominant_rgb: np.ndarray | None = None


@dataclass(frozen=True)
class MineralEnsembleEvaluation:
    hard_gate_pass: bool
    family_catalog: pa.Table
    seed_catalog: pa.Table
    hard_failures: tuple[str, ...]


@dataclass(frozen=True)
class MineralEnsembleResult:
    passed: bool
    report_path: Path
    family_catalog_path: Path
    seed_catalog_path: Path
    gallery_path: Path
    evaluation: MineralEnsembleEvaluation


def evaluate_mineral_ensemble(
    reports: Sequence[MineralSeedReport],
    thresholds: MineralEnsembleThresholds,
) -> MineralEnsembleEvaluation:
    failures: list[str] = []
    if len(reports) < thresholds.minimum_seed_count:
        failures.append("minimum_seed_count")
    integrity_failures = [
        failure
        for report in reports
        for failure in report.hard_failures
        if not failure.endswith("_directional_or_noncollapse")
    ]
    if integrity_failures or any(
        not report.hard_gate_pass and not report.hard_failures for report in reports
    ):
        failures.append("per_seed_integrity_gate")
    state_checksums = [report.state_checksum for report in reports]
    if any(not checksum for checksum in state_checksums) or len(set(state_checksums)) != len(
        state_checksums
    ):
        failures.append("distinct_seed_state")
    seed_rows = [
        {
            "seed": report.seed,
            "hard_gate_pass": report.hard_gate_pass,
            "hard_failures": list(report.hard_failures),
            "mineral_system_count": report.system_count,
            "deposit_candidate_count": report.candidate_count,
            "state_checksum": report.state_checksum,
        }
        for report in reports
    ]
    family_rows: list[dict[str, object]] = []
    for family_index, family_name in enumerate(SYSTEM_NAMES):
        rows = [
            next(row for row in report.family_rows if int(row["system_code"]) == family_index + 1)
            for report in reports
        ]
        peaks = np.asarray([float(row["peak_potential"]) for row in rows], dtype=np.float64)
        pass_fraction = float(
            np.mean(np.asarray([bool(row["passed"]) for row in rows], dtype=np.float64))
        )
        mean_peak = float(np.mean(peaks))
        peak_cv = float(np.std(peaks) / max(mean_peak, 1e-12))
        passed = (
            pass_fraction >= thresholds.minimum_family_pass_fraction
            and peak_cv <= thresholds.maximum_family_peak_coefficient_of_variation
        )
        if not passed:
            failures.append(f"{family_name}_ensemble")
        family_rows.append(
            {
                "system_code": family_index + 1,
                "system_name": family_name,
                "seed_count": len(rows),
                "per_seed_pass_fraction": pass_fraction,
                "peak_potential_mean": mean_peak,
                "peak_potential_minimum": float(np.min(peaks)),
                "peak_potential_maximum": float(np.max(peaks)),
                "peak_potential_coefficient_of_variation": peak_cv,
                "minimum_required_pass_fraction": thresholds.minimum_family_pass_fraction,
                "maximum_allowed_peak_coefficient_of_variation": (
                    thresholds.maximum_family_peak_coefficient_of_variation
                ),
                "passed": passed,
            }
        )
    return MineralEnsembleEvaluation(
        not failures,
        pa.Table.from_pylist(family_rows),
        pa.Table.from_pylist(seed_rows),
        tuple(sorted(set(failures))),
    )


def _world_config(
    base: PipelineConfig,
    config: MineralEnsembleConfig,
    seed: int,
) -> PipelineConfig:
    world = deepcopy(base)
    if world.topology.lower() != "cubed_sphere":
        raise ValueError("Mineral ensemble requires topology: cubed_sphere")
    if config.face_resolution is not None:
        resolution = config.face_resolution
        world.resolution_set = ResolutionSet(
            (GridInfo(resolution, resolution, face_resolution=resolution),)
        )
    native_resolution = world.resolution_set.native.face_resolution
    if native_resolution is None:
        raise ValueError("Mineral ensemble requires a cubed-sphere face resolution")
    world.rng_seed = seed
    overrides = deepcopy(world.stage_overrides)
    refinement = dict(overrides.get("basin_refinement", {}))
    refinement["maximum_parent_boundary_residual_p95_ratio"] = (
        config.screen_parent_boundary_residual_p95_ratio
    )
    refinement["maximum_parent_offset_span_m"] = config.screen_maximum_parent_offset_span_m
    refinement["maximum_parent_offset_span_relief_fraction"] = (
        config.screen_maximum_parent_offset_span_relief_fraction
    )
    refinement["maximum_tile_bubble_correlation_p50"] = (
        config.screen_maximum_tile_bubble_correlation_p50
    )
    refinement["maximum_tile_bubble_correlation_p95"] = (
        config.screen_maximum_tile_bubble_correlation_p95
    )
    overrides["basin_refinement"] = refinement
    world.stage_overrides = overrides
    world.run_id = f"causal-minerals-v0-seed-{seed}-face-{native_resolution}"
    world.output_dir = config.output_dir / "runs"
    world.cache_dir = config.output_dir / "cache"
    world.log_dir = config.output_dir / "logs"
    return world


def _dominant_preview(codes: np.ndarray) -> np.ndarray:
    rgb = np.full((*codes.shape, 3), (28, 48, 61), dtype=np.uint8)
    for code, color in enumerate(SYSTEM_COLORS, start=1):
        rgb[codes == code] = color
    return _equirectangular_rgb(rgb)


def _write_gallery(reports: Sequence[MineralSeedReport], path: Path) -> None:
    previews = [
        (report.seed, report.dominant_rgb) for report in reports if report.dominant_rgb is not None
    ]
    columns = 2
    thumb_width = 512
    thumb_height = 256
    label_height = 28
    rows = (len(previews) + columns - 1) // columns
    legend_rows = (len(SYSTEM_NAMES) + 1) // 2
    legend_height = 42 + legend_rows * 24
    canvas = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height) + legend_height),
        (238, 238, 234),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (seed, preview_rgb) in enumerate(previews):
        row, column = divmod(index, columns)
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        preview = Image.fromarray(cast(np.ndarray, preview_rgb), mode="RGB")
        preview.thumbnail((thumb_width, thumb_height), Image.Resampling.NEAREST)
        canvas.paste(preview, (x, y))
        draw.text((x + 8, y + thumb_height + 7), f"Seed {seed}", fill=(25, 27, 28))
    legend_y = rows * (thumb_height + label_height) + 8
    draw.text((8, legend_y), "Dominant causal mineral system | equirectangular", fill=(25, 27, 28))
    for index, (name, color) in enumerate(zip(SYSTEM_NAMES, SYSTEM_COLORS, strict=True)):
        column = index % 2
        row = index // 2
        x = 8 + column * 512
        y = legend_y + 24 + row * 24
        draw.rectangle((x, y, x + 14, y + 14), fill=tuple(int(value) for value in color))
        draw.text((x + 20, y), name.replace("_", " "), fill=(25, 27, 28))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_mineral_ensemble(config: MineralEnsembleConfig) -> MineralEnsembleResult:
    base = PipelineConfig.from_file(config.base_config)
    reports: list[MineralSeedReport] = []
    for seed in config.seeds:
        world = _world_config(base, config, seed)
        results = ExecutionEngine(world, generate_visuals=False).run(["mineral_systems_validation"])
        mineral = results["mineral_systems"]
        validation = results["mineral_systems_validation"]
        metadata = cast(
            Mapping[str, object],
            validation.artifact_records["MineralSystemsValidationMetadata"].value,
        )
        catalog = cast(
            pa.Table,
            validation.artifact_records["MineralSystemsValidationCatalog"].value,
        )
        mineral_metadata = cast(
            Mapping[str, object],
            mineral.artifact_records["MineralSystemsMetadata"].value,
        )
        codes_value = mineral.artifact_records["DominantMineralSystemCode"].value
        codes = np.asarray(codes_value.array() if hasattr(codes_value, "array") else codes_value)
        state_fingerprint = (
            "|".join(
                mineral.artifact_records[name].checksum
                for name in (
                    "MineralSourceSupport",
                    "MineralProcessSupport",
                    "MineralTransportSupport",
                    "MineralTrapSupport",
                    "MineralTimingSupport",
                    "MineralPreservationSupport",
                    "MineralUnresolvedSupport",
                    "MineralSystemPotential",
                    "MineralSystemConfidence",
                    "CommodityProspectivity",
                    "DominantMineralSystemCode",
                    "MineralSystemCatalog",
                    "MajorDepositCandidateCatalog",
                    "MineralSystemsMetadata",
                )
            )
            + "|"
            + "|".join(
                validation.artifact_records[name].checksum
                for name in (
                    "MineralSystemsValidationCatalog",
                    "MineralSystemsValidationMetadata",
                )
            )
        ).encode("ascii")
        state_checksum = hashlib.sha256(state_fingerprint).hexdigest()
        reports.append(
            MineralSeedReport(
                seed=seed,
                hard_gate_pass=bool(metadata["hard_gate_pass"]),
                hard_failures=tuple(
                    str(failure) for failure in cast(Sequence[object], metadata["hard_failures"])
                ),
                family_rows=tuple(catalog.to_pylist()),
                candidate_count=int(mineral_metadata["major_deposit_candidate_count"]),
                system_count=int(mineral_metadata["mineral_system_catalog_count"]),
                state_checksum=state_checksum,
                dominant_rgb=_dominant_preview(codes),
            )
        )
    evaluation = evaluate_mineral_ensemble(reports, config.thresholds)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    family_catalog_path = config.output_dir / "family_ensemble_kpis.parquet"
    seed_catalog_path = config.output_dir / "seed_execution_kpis.parquet"
    report_path = config.output_dir / "report.json"
    gallery_path = config.output_dir / "dominant_system_gallery.png"
    pq.write_table(evaluation.family_catalog, family_catalog_path)
    pq.write_table(evaluation.seed_catalog, seed_catalog_path)
    _write_gallery(reports, gallery_path)
    report_path.write_text(
        json.dumps(
            {
                "model": "causal_mineral_systems_ensemble_v0",
                "passed": evaluation.hard_gate_pass,
                "seeds": list(config.seeds),
                "face_resolution": config.face_resolution
                or base.resolution_set.native.face_resolution,
                "thresholds": asdict(config.thresholds),
                "upstream_screen_gate_adjustments": {
                    "basin_refinement.maximum_parent_boundary_residual_p95_ratio": (
                        config.screen_parent_boundary_residual_p95_ratio
                    ),
                    "basin_refinement.maximum_parent_offset_span_m": (
                        config.screen_maximum_parent_offset_span_m
                    ),
                    "basin_refinement.maximum_parent_offset_span_relief_fraction": (
                        config.screen_maximum_parent_offset_span_relief_fraction
                    ),
                    "basin_refinement.maximum_tile_bubble_correlation_p50": (
                        config.screen_maximum_tile_bubble_correlation_p50
                    ),
                    "basin_refinement.maximum_tile_bubble_correlation_p95": (
                        config.screen_maximum_tile_bubble_correlation_p95
                    ),
                    "semantics": "acceptance_only_no_terrain_state_change",
                },
                "hard_failures": list(evaluation.hard_failures),
                "earth_deposit_count_quota_applied": False,
                "petroleum_supported": False,
                "family_results": evaluation.family_catalog.to_pylist(),
                "seed_results": evaluation.seed_catalog.to_pylist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    return MineralEnsembleResult(
        evaluation.hard_gate_pass,
        report_path,
        family_catalog_path,
        seed_catalog_path,
        gallery_path,
        evaluation,
    )


__all__ = [
    "MineralEnsembleConfig",
    "MineralEnsembleEvaluation",
    "MineralEnsembleResult",
    "MineralEnsembleThresholds",
    "MineralSeedReport",
    "evaluate_mineral_ensemble",
    "run_mineral_ensemble",
]
