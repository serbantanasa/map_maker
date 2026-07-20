"""Multi-seed evaluation for the versioned Earth biosphere profile."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from .config import GridInfo, PipelineConfig, ResolutionSet
from .execution import ExecutionEngine
from .models import StageResult
from .stages.biosphere_validation import CLIMATE_STRATA, REFERENCE_PROFILE_VERSION
from .stages.functional_vegetation_validation import (
    REFERENCE_PROFILE_VERSION as FUNCTIONAL_REFERENCE_PROFILE_VERSION,
)


ENSEMBLE_METRIC_SCHEMA = pa.schema(
    [
        ("metric_id", pa.string()),
        ("scope", pa.string()),
        ("seed_count", pa.int32()),
        ("mean", pa.float64()),
        ("standard_deviation", pa.float64()),
        ("coefficient_of_variation", pa.float64()),
        ("minimum", pa.float64()),
        ("maximum", pa.float64()),
        ("absolute_range", pa.float64()),
        ("tolerance_kind", pa.string()),
        ("tolerance", pa.float64()),
        ("passed", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class BiosphereEnsembleThresholds:
    minimum_seed_count: int = 5
    minimum_earth_diagnostic_pass_fraction: float = 0.80
    maximum_land_fraction_absolute_range: float = 0.08
    maximum_global_npp_coefficient_of_variation: float = 0.35
    maximum_global_biomass_coefficient_of_variation: float = 0.40
    maximum_mean_cover_coefficient_of_variation: float = 0.25
    maximum_vegetated_fraction_coefficient_of_variation: float = 0.20
    minimum_zone_presence_seed_fraction: float = 0.60
    maximum_zone_area_fraction_absolute_range: float = 0.20
    maximum_zone_mean_npp_coefficient_of_variation: float = 0.60
    maximum_functional_fraction_coefficient_of_variation: float = 0.30
    maximum_resource_p90_coefficient_of_variation: float = 0.35
    maximum_zone_functional_mean_coefficient_of_variation: float = 0.60

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "BiosphereEnsembleThresholds":
        mapping = mapping or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown biosphere ensemble tolerances: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if name == "minimum_seed_count":
                values[name] = int(raw)
            else:
                values[name] = float(raw)
        config = cls(
            minimum_seed_count=int(values["minimum_seed_count"]),
            minimum_earth_diagnostic_pass_fraction=float(
                values["minimum_earth_diagnostic_pass_fraction"]
            ),
            maximum_land_fraction_absolute_range=float(
                values["maximum_land_fraction_absolute_range"]
            ),
            maximum_global_npp_coefficient_of_variation=float(
                values["maximum_global_npp_coefficient_of_variation"]
            ),
            maximum_global_biomass_coefficient_of_variation=float(
                values["maximum_global_biomass_coefficient_of_variation"]
            ),
            maximum_mean_cover_coefficient_of_variation=float(
                values["maximum_mean_cover_coefficient_of_variation"]
            ),
            maximum_vegetated_fraction_coefficient_of_variation=float(
                values["maximum_vegetated_fraction_coefficient_of_variation"]
            ),
            minimum_zone_presence_seed_fraction=float(
                values["minimum_zone_presence_seed_fraction"]
            ),
            maximum_zone_area_fraction_absolute_range=float(
                values["maximum_zone_area_fraction_absolute_range"]
            ),
            maximum_zone_mean_npp_coefficient_of_variation=float(
                values["maximum_zone_mean_npp_coefficient_of_variation"]
            ),
            maximum_functional_fraction_coefficient_of_variation=float(
                values["maximum_functional_fraction_coefficient_of_variation"]
            ),
            maximum_resource_p90_coefficient_of_variation=float(
                values["maximum_resource_p90_coefficient_of_variation"]
            ),
            maximum_zone_functional_mean_coefficient_of_variation=float(
                values["maximum_zone_functional_mean_coefficient_of_variation"]
            ),
        )
        if config.minimum_seed_count < 2:
            raise ValueError("minimum_seed_count must be at least 2")
        fractions = (
            "minimum_earth_diagnostic_pass_fraction",
            "maximum_land_fraction_absolute_range",
            "maximum_global_npp_coefficient_of_variation",
            "maximum_global_biomass_coefficient_of_variation",
            "maximum_mean_cover_coefficient_of_variation",
            "maximum_vegetated_fraction_coefficient_of_variation",
            "minimum_zone_presence_seed_fraction",
            "maximum_zone_area_fraction_absolute_range",
            "maximum_zone_mean_npp_coefficient_of_variation",
            "maximum_functional_fraction_coefficient_of_variation",
            "maximum_resource_p90_coefficient_of_variation",
            "maximum_zone_functional_mean_coefficient_of_variation",
        )
        for name in fractions:
            value = float(getattr(config, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return config


@dataclass(frozen=True)
class BiosphereEnsembleConfig:
    base_config: Path
    seeds: tuple[int, ...]
    output_dir: Path
    face_resolution: int | None
    thresholds: BiosphereEnsembleThresholds

    @classmethod
    def from_file(
        cls, path: Path | str, *, output_dir: Path | None = None
    ) -> "BiosphereEnsembleConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        if not isinstance(data, Mapping):
            raise TypeError("Biosphere ensemble config must contain a mapping")
        if _integer(data.get("format_version", 1), name="format_version") != 1:
            raise ValueError("Unsupported biosphere ensemble format_version")
        profile = str(data.get("reference_profile", REFERENCE_PROFILE_VERSION))
        if profile != REFERENCE_PROFILE_VERSION:
            raise ValueError(f"reference_profile must be {REFERENCE_PROFILE_VERSION}")
        functional_profile = str(
            data.get("functional_reference_profile", FUNCTIONAL_REFERENCE_PROFILE_VERSION)
        )
        if functional_profile != FUNCTIONAL_REFERENCE_PROFILE_VERSION:
            raise ValueError(
                f"functional_reference_profile must be {FUNCTIONAL_REFERENCE_PROFILE_VERSION}"
            )
        raw_base = data.get("base_config")
        if not raw_base:
            raise ValueError("Biosphere ensemble config requires base_config")
        base_config = (path.parent / str(raw_base)).resolve()
        thresholds = BiosphereEnsembleThresholds.from_mapping(
            cast(Mapping[str, object] | None, data.get("ensemble_tolerances"))
        )
        seeds = tuple(
            _integer(seed, name="seed") for seed in cast(Sequence[object], data.get("seeds", ()))
        )
        if len(seeds) < thresholds.minimum_seed_count:
            raise ValueError(
                f"Biosphere ensemble requires at least {thresholds.minimum_seed_count} unique seeds"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("Biosphere ensemble seeds must be unique")
        raw_resolution = data.get("face_resolution")
        face_resolution = (
            _integer(raw_resolution, name="face_resolution") if raw_resolution is not None else None
        )
        if face_resolution is not None and face_resolution <= 0:
            raise ValueError("face_resolution must be positive")
        if output_dir is None:
            resolved_output = (
                path.parent / str(data.get("output_dir", "../out/biosphere"))
            ).resolve()
        else:
            resolved_output = output_dir.expanduser().resolve()
        return cls(base_config, seeds, resolved_output, face_resolution, thresholds)


@dataclass(frozen=True)
class FunctionalVegetationSeedReport:
    kpis: pa.Table
    climate_distributions: pa.Table
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class BiosphereSeedReport:
    seed: int
    kpis: pa.Table
    climate_distributions: pa.Table
    metadata: Mapping[str, object]
    functional_vegetation: FunctionalVegetationSeedReport | None = None


@dataclass(frozen=True)
class EnsembleGate:
    name: str
    gate_kind: str
    passed: bool
    value: float | int | bool
    expectation: str


@dataclass(frozen=True)
class BiosphereEnsembleEvaluation:
    hard_gate_pass: bool
    stability_pass: bool
    earth_profile_pass: bool
    gates: tuple[EnsembleGate, ...]
    metric_catalog: pa.Table
    functional_profile_pass: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.hard_gate_pass
            and self.stability_pass
            and self.earth_profile_pass
            and self.functional_profile_pass
        )


@dataclass(frozen=True)
class BiosphereEnsembleResult:
    passed: bool
    execution_valid: bool
    earth_profile_pass: bool
    report_path: Path
    metric_catalog_path: Path
    seed_count: int
    gates: tuple[EnsembleGate, ...]
    functional_profile_pass: bool = True


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f"{name} must be an integer")
    return integer


def _table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing table artifact {result.stage_name}.{name}")
    return record.value.combine_chunks()


def _metadata(result: StageResult) -> Mapping[str, object]:
    record = result.artifact_records.get("BiosphereValidationMetadata")
    if record is None or not isinstance(record.value, Mapping):
        raise KeyError("Missing biosphere validation metadata")
    return cast(Mapping[str, object], record.value)


def _functional_metadata(result: StageResult) -> Mapping[str, object]:
    record = result.artifact_records.get("FunctionalVegetationValidationMetadata")
    if record is None or not isinstance(record.value, Mapping):
        raise KeyError("Missing functional vegetation validation metadata")
    return cast(Mapping[str, object], record.value)


def _kpi_rows(report: BiosphereSeedReport) -> dict[str, Mapping[str, object]]:
    return {str(row["kpi_id"]): row for row in report.kpis.to_pylist()}


def _kpi_value(report: BiosphereSeedReport, kpi_id: str) -> float:
    row = _kpi_rows(report).get(kpi_id)
    if row is None:
        raise KeyError(f"Seed {report.seed} is missing KPI {kpi_id}")
    value = row["value"]
    if not isinstance(value, (int, float)):
        raise TypeError(f"KPI {kpi_id} must be numeric")
    return float(value)


def _functional_kpi_rows(report: BiosphereSeedReport) -> dict[str, Mapping[str, object]]:
    functional = report.functional_vegetation
    if functional is None:
        raise KeyError(f"Seed {report.seed} is missing functional-vegetation validation")
    return {str(row["kpi_id"]): row for row in functional.kpis.to_pylist()}


def _functional_kpi_value(report: BiosphereSeedReport, kpi_id: str) -> float:
    row = _functional_kpi_rows(report).get(kpi_id)
    if row is None:
        raise KeyError(f"Seed {report.seed} is missing functional KPI {kpi_id}")
    value = row["value"]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Functional KPI {kpi_id} must be numeric")
    return float(value)


def _zone_summary(
    report: BiosphereSeedReport, zone_id: str, metric_id: str, statistic: str
) -> tuple[float, float, bool]:
    rows = report.climate_distributions.to_pylist()
    matches = [
        row
        for row in rows
        if row["zone_id"] == zone_id
        and row["metric_id"] == metric_id
        and row["statistic"] == statistic
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Seed {report.seed} has {len(matches)} rows for {zone_id}.{metric_id}.{statistic}"
        )
    row = matches[0]
    return float(row["zone_land_area_fraction"]), float(row["value"]), bool(row["reportable"])


def _functional_zone_summary(
    report: BiosphereSeedReport, zone_id: str, metric_id: str, statistic: str
) -> tuple[float, float, bool]:
    functional = report.functional_vegetation
    if functional is None:
        raise KeyError(f"Seed {report.seed} is missing functional-vegetation validation")
    matches = [
        row
        for row in functional.climate_distributions.to_pylist()
        if row["zone_id"] == zone_id
        and row["metric_id"] == metric_id
        and row["statistic"] == statistic
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Seed {report.seed} has {len(matches)} functional rows for "
            f"{zone_id}.{metric_id}.{statistic}"
        )
    row = matches[0]
    return float(row["zone_land_area_fraction"]), float(row["value"]), bool(row["reportable"])


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=0))
    if abs(mean) <= 1e-30:
        return 0.0 if standard_deviation <= 1e-30 else float("inf")
    return standard_deviation / abs(mean)


def evaluate_biosphere_ensemble(
    reports: Sequence[BiosphereSeedReport],
    thresholds: BiosphereEnsembleThresholds,
) -> BiosphereEnsembleEvaluation:
    """Evaluate fixed per-world catalogs without rerunning or retuning a seed."""

    if len(reports) < thresholds.minimum_seed_count:
        raise ValueError(f"Expected at least {thresholds.minimum_seed_count} seed reports")
    if len({report.seed for report in reports}) != len(reports):
        raise ValueError("Biosphere seed reports must have unique seeds")

    gates: list[EnsembleGate] = []
    metric_rows: list[dict[str, object]] = []
    profile_versions = {
        str(report.metadata.get("reference_profile_version", "missing")) for report in reports
    }
    gates.append(
        EnsembleGate(
            "reference_profile_version_match",
            "hard_invariant",
            profile_versions == {REFERENCE_PROFILE_VERSION},
            len(profile_versions),
            f"all seed reports use {REFERENCE_PROFILE_VERSION}",
        )
    )
    functional_report_count = sum(report.functional_vegetation is not None for report in reports)
    functional_profile_enabled = functional_report_count == len(reports)
    if functional_report_count:
        gates.append(
            EnsembleGate(
                "all_seed_functional_reports_present",
                "hard_invariant",
                functional_profile_enabled,
                functional_report_count,
                f"{len(reports)} of {len(reports)} seeds",
            )
        )
    if functional_profile_enabled:
        functional_versions = {
            str(report.functional_vegetation.metadata.get("reference_profile_version", "missing"))
            for report in reports
            if report.functional_vegetation is not None
        }
        gates.append(
            EnsembleGate(
                "functional_reference_profile_version_match",
                "hard_invariant",
                functional_versions == {FUNCTIONAL_REFERENCE_PROFILE_VERSION},
                len(functional_versions),
                f"all seed reports use {FUNCTIONAL_REFERENCE_PROFILE_VERSION}",
            )
        )
    hard_passes = [
        _integer(report.metadata.get("hard_gate_pass", 0), name="hard_gate_pass") == 1
        for report in reports
    ]
    gates.append(
        EnsembleGate(
            "all_seed_hard_gates",
            "hard_invariant",
            all(hard_passes),
            sum(hard_passes),
            f"{len(reports)} of {len(reports)} seeds",
        )
    )
    if functional_profile_enabled:
        functional_hard_passes = [
            _integer(
                report.functional_vegetation.metadata.get("hard_gate_pass", 0),
                name="functional_hard_gate_pass",
            )
            == 1
            for report in reports
            if report.functional_vegetation is not None
        ]
        gates.append(
            EnsembleGate(
                "all_seed_functional_hard_gates",
                "hard_invariant",
                all(functional_hard_passes),
                sum(functional_hard_passes),
                f"{len(reports)} of {len(reports)} seeds",
            )
        )

    stability_specs = (
        (
            "global_land_surface_fraction",
            "global",
            "absolute_range",
            thresholds.maximum_land_fraction_absolute_range,
        ),
        (
            "global_potential_npp_pg_c_year",
            "global_land",
            "coefficient_of_variation",
            thresholds.maximum_global_npp_coefficient_of_variation,
        ),
        (
            "global_potential_biomass_pg_c",
            "global_land",
            "coefficient_of_variation",
            thresholds.maximum_global_biomass_coefficient_of_variation,
        ),
        (
            "land_mean_potential_vegetation_cover_fraction",
            "global_land",
            "coefficient_of_variation",
            thresholds.maximum_mean_cover_coefficient_of_variation,
        ),
        (
            "potentially_vegetated_land_fraction",
            "global_land",
            "coefficient_of_variation",
            thresholds.maximum_vegetated_fraction_coefficient_of_variation,
        ),
    )

    def add_metric(
        metric_id: str,
        scope: str,
        values: Sequence[float],
        tolerance_kind: str,
        tolerance: float,
    ) -> None:
        array = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(array))
        standard_deviation = float(np.std(array, ddof=0))
        coefficient = _coefficient_of_variation(array)
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        absolute_range = maximum - minimum
        measured = coefficient if tolerance_kind == "coefficient_of_variation" else absolute_range
        passed = bool(np.isfinite(measured) and measured <= tolerance)
        metric_rows.append(
            {
                "metric_id": metric_id,
                "scope": scope,
                "seed_count": len(array),
                "mean": mean,
                "standard_deviation": standard_deviation,
                "coefficient_of_variation": coefficient,
                "minimum": minimum,
                "maximum": maximum,
                "absolute_range": absolute_range,
                "tolerance_kind": tolerance_kind,
                "tolerance": tolerance,
                "passed": passed,
            }
        )
        gates.append(
            EnsembleGate(
                f"stability.{metric_id}",
                "ensemble_stability",
                passed,
                measured,
                f"{tolerance_kind} <= {tolerance}",
            )
        )

    for metric_id, scope, tolerance_kind, tolerance in stability_specs:
        add_metric(
            metric_id,
            scope,
            [_kpi_value(report, metric_id) for report in reports],
            tolerance_kind,
            tolerance,
        )

    for definition in CLIMATE_STRATA:
        zone_id = str(definition["zone_id"])
        summaries = [
            _zone_summary(report, zone_id, "annual_potential_npp", "mean") for report in reports
        ]
        area_fractions = [summary[0] for summary in summaries]
        add_metric(
            f"zone.{zone_id}.land_area_fraction",
            zone_id,
            area_fractions,
            "absolute_range",
            thresholds.maximum_zone_area_fraction_absolute_range,
        )
        reportable_npp = [summary[1] for summary in summaries if summary[2]]
        presence_fraction = len(reportable_npp) / len(reports)
        if presence_fraction >= thresholds.minimum_zone_presence_seed_fraction:
            add_metric(
                f"zone.{zone_id}.mean_annual_potential_npp",
                zone_id,
                reportable_npp,
                "coefficient_of_variation",
                thresholds.maximum_zone_mean_npp_coefficient_of_variation,
            )

    if functional_profile_enabled:
        for metric_id in (
            "land_mean_functional_vegetated_fraction",
            "land_mean_functional_woody_fraction",
            "land_mean_functional_herbaceous_fraction",
            "land_mean_functional_xeric_low_stature_fraction",
            "land_mean_functional_hydrophytic_fraction",
            "land_mean_nonvegetated_ground_fraction",
            "land_mean_inland_open_water_fraction",
        ):
            add_metric(
                f"functional.{metric_id}",
                "global_land",
                [_functional_kpi_value(report, metric_id) for report in reports],
                "coefficient_of_variation",
                thresholds.maximum_functional_fraction_coefficient_of_variation,
            )
        for metric_id in (
            "land_fire_tendency_resource_p90",
            "land_grazing_resource_p90",
            "land_forest_resource_p90",
            "land_pasture_resource_p90",
            "land_crop_resource_p90",
        ):
            add_metric(
                f"functional.{metric_id}",
                "global_land",
                [_functional_kpi_value(report, metric_id) for report in reports],
                "coefficient_of_variation",
                thresholds.maximum_resource_p90_coefficient_of_variation,
            )
        for definition in CLIMATE_STRATA:
            zone_id = str(definition["zone_id"])
            for metric_id in (
                "functional_woody_fraction",
                "functional_herbaceous_fraction",
                "functional_hydrophytic_fraction",
                "resource_fire_tendency",
                "resource_forest",
            ):
                summaries = [
                    _functional_zone_summary(report, zone_id, metric_id, "mean")
                    for report in reports
                ]
                reportable = [summary[1] for summary in summaries if summary[2]]
                presence_fraction = len(reportable) / len(reports)
                if presence_fraction >= thresholds.minimum_zone_presence_seed_fraction:
                    add_metric(
                        f"functional.zone.{zone_id}.mean_{metric_id}",
                        zone_id,
                        reportable,
                        "coefficient_of_variation",
                        thresholds.maximum_zone_functional_mean_coefficient_of_variation,
                    )

    first_rows = _kpi_rows(reports[0])
    earth_ids = sorted(
        kpi_id
        for kpi_id, row in first_rows.items()
        if row["gate_kind"] in {"earth_diagnostic", "earth_structure"}
    )
    earth_gate_passes: list[bool] = []
    for kpi_id in earth_ids:
        applicable_statuses = []
        for report in reports:
            row = _kpi_rows(report).get(kpi_id)
            if row is None:
                raise KeyError(f"Seed {report.seed} is missing Earth diagnostic {kpi_id}")
            status = str(row["comparison_status"])
            if status != "not_applicable":
                applicable_statuses.append(status)
        if not applicable_statuses:
            continue
        pass_fraction = applicable_statuses.count("within_reference") / len(applicable_statuses)
        passed = pass_fraction >= thresholds.minimum_earth_diagnostic_pass_fraction
        earth_gate_passes.append(passed)
        gates.append(
            EnsembleGate(
                f"earth_profile.{kpi_id}",
                "earth_profile",
                passed,
                pass_fraction,
                "within-reference seed fraction >= "
                f"{thresholds.minimum_earth_diagnostic_pass_fraction}",
            )
        )

    functional_gate_passes: list[bool] = []
    if functional_profile_enabled:
        first_functional_rows = _functional_kpi_rows(reports[0])
        functional_ids = sorted(
            kpi_id
            for kpi_id, row in first_functional_rows.items()
            if row["gate_kind"] in {"earth_diagnostic", "earth_structure"}
        )
        for kpi_id in functional_ids:
            applicable_statuses = []
            for report in reports:
                row = _functional_kpi_rows(report).get(kpi_id)
                if row is None:
                    raise KeyError(f"Seed {report.seed} is missing functional diagnostic {kpi_id}")
                status = str(row["comparison_status"])
                if status != "not_applicable":
                    applicable_statuses.append(status)
            if not applicable_statuses:
                continue
            pass_fraction = applicable_statuses.count("within_reference") / len(applicable_statuses)
            passed = pass_fraction >= thresholds.minimum_earth_diagnostic_pass_fraction
            functional_gate_passes.append(passed)
            gates.append(
                EnsembleGate(
                    f"functional_profile.{kpi_id}",
                    "functional_profile",
                    passed,
                    pass_fraction,
                    "within-reference seed fraction >= "
                    f"{thresholds.minimum_earth_diagnostic_pass_fraction}",
                )
            )

    metric_catalog = pa.Table.from_pylist(metric_rows, schema=ENSEMBLE_METRIC_SCHEMA)
    hard_gate_pass = all(gate.passed for gate in gates if gate.gate_kind == "hard_invariant")
    stability_pass = all(gate.passed for gate in gates if gate.gate_kind == "ensemble_stability")
    earth_profile_pass = bool(earth_gate_passes) and all(earth_gate_passes)
    functional_profile_pass = (
        bool(functional_gate_passes) and all(functional_gate_passes)
        if functional_profile_enabled
        else functional_report_count == 0
    )
    return BiosphereEnsembleEvaluation(
        hard_gate_pass,
        stability_pass,
        earth_profile_pass,
        tuple(gates),
        metric_catalog,
        functional_profile_pass,
    )


def _world_config(
    base: PipelineConfig,
    config: BiosphereEnsembleConfig,
    seed: int,
) -> PipelineConfig:
    world = deepcopy(base)
    if world.topology.lower() != "cubed_sphere":
        raise ValueError("Earth biosphere ensemble requires topology: cubed_sphere")
    if config.face_resolution is not None:
        configured_resolution = config.face_resolution
        world.resolution_set = ResolutionSet(
            (
                GridInfo(
                    configured_resolution,
                    configured_resolution,
                    face_resolution=configured_resolution,
                ),
            )
        )
    native_resolution = world.resolution_set.native.face_resolution
    if native_resolution is None:
        raise ValueError("Earth biosphere ensemble requires a cubed-sphere face resolution")
    world.rng_seed = seed
    world.run_id = f"earth-biosphere-v1-seed-{seed}-face-{native_resolution}"
    world.output_dir = config.output_dir / "runs"
    world.cache_dir = config.output_dir / "cache"
    world.log_dir = config.output_dir / "logs"
    return world


def run_biosphere_ensemble(config: BiosphereEnsembleConfig) -> BiosphereEnsembleResult:
    """Run and persist the fixed-seed Earth biosphere calibration ensemble."""

    base = PipelineConfig.from_file(config.base_config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BiosphereSeedReport] = []
    execution_failures: dict[int, str] = {}
    for seed in config.seeds:
        world = _world_config(base, config, seed)
        try:
            results = ExecutionEngine(world, generate_visuals=False).run(
                ["functional_vegetation_validation"]
            )
        except RuntimeError as exc:
            execution_failures[seed] = str(exc)
            continue
        validation = results["biosphere_validation"]
        functional_validation = results["functional_vegetation_validation"]
        reports.append(
            BiosphereSeedReport(
                seed,
                _table(validation, "BiosphereKpiCatalog"),
                _table(validation, "BiosphereClimateDistributionCatalog"),
                dict(_metadata(validation)),
                FunctionalVegetationSeedReport(
                    _table(functional_validation, "FunctionalVegetationKpiCatalog"),
                    _table(
                        functional_validation,
                        "FunctionalVegetationClimateDistributionCatalog",
                    ),
                    dict(_functional_metadata(functional_validation)),
                ),
            )
        )

    execution_gate = EnsembleGate(
        "all_seed_execution",
        "hard_invariant",
        not execution_failures,
        len(reports),
        f"{len(config.seeds)} of {len(config.seeds)} seeds reach functional vegetation validation",
    )
    if len(reports) >= config.thresholds.minimum_seed_count:
        evaluation = evaluate_biosphere_ensemble(reports, config.thresholds)
        gates = (execution_gate, *evaluation.gates)
    else:
        enough_reports_gate = EnsembleGate(
            "minimum_successful_seed_count",
            "ensemble_stability",
            False,
            len(reports),
            f">= {config.thresholds.minimum_seed_count}",
        )
        gates = (execution_gate, enough_reports_gate)
        evaluation = BiosphereEnsembleEvaluation(
            hard_gate_pass=False,
            stability_pass=False,
            earth_profile_pass=False,
            gates=gates,
            metric_catalog=pa.Table.from_pylist([], schema=ENSEMBLE_METRIC_SCHEMA),
        )
    hard_gate_pass = not execution_failures and evaluation.hard_gate_pass
    execution_valid = hard_gate_pass and evaluation.stability_pass
    passed = (
        execution_valid and evaluation.earth_profile_pass and evaluation.functional_profile_pass
    )
    metric_catalog_path = config.output_dir / "ensemble_kpis.parquet"
    pq.write_table(evaluation.metric_catalog, metric_catalog_path)
    report_path = config.output_dir / "report.json"
    report: dict[str, Any] = {
        "format_version": 1,
        "reference_profile": REFERENCE_PROFILE_VERSION,
        "functional_reference_profile": FUNCTIONAL_REFERENCE_PROFILE_VERSION,
        "status": "pass" if passed else "outside_reference" if execution_valid else "invalid",
        "execution_valid": execution_valid,
        "hard_gate_pass": hard_gate_pass,
        "ensemble_stability_pass": evaluation.stability_pass,
        "earth_profile_pass": evaluation.earth_profile_pass,
        "functional_profile_pass": evaluation.functional_profile_pass,
        "base_config": str(config.base_config),
        "face_resolution": config.face_resolution
        or PipelineConfig.from_file(config.base_config).resolution_set.native.face_resolution,
        "seeds": list(config.seeds),
        "ensemble_tolerances": asdict(config.thresholds),
        "metric_catalog": metric_catalog_path.name,
        "requested_seed_count": len(config.seeds),
        "successful_seed_count": len(reports),
        "execution_failures": [
            {"seed": seed, "error": error} for seed, error in sorted(execution_failures.items())
        ],
        "gates": [asdict(gate) for gate in gates],
        "worlds": [_world_report_row(seed, reports, execution_failures) for seed in config.seeds],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    return BiosphereEnsembleResult(
        passed,
        execution_valid,
        evaluation.earth_profile_pass,
        report_path,
        metric_catalog_path,
        len(config.seeds),
        gates,
        evaluation.functional_profile_pass,
    )


def _world_report_row(
    seed: int,
    reports: Sequence[BiosphereSeedReport],
    execution_failures: Mapping[int, str],
) -> dict[str, object]:
    report = next((candidate for candidate in reports if candidate.seed == seed), None)
    if report is None:
        return {
            "seed": seed,
            "execution_status": "failed",
            "execution_error": execution_failures.get(seed, "unknown execution failure"),
        }
    return {
        "seed": report.seed,
        "execution_status": "completed",
        "hard_gate_pass": _integer(report.metadata.get("hard_gate_pass", 0), name="hard_gate_pass"),
        "earth_profile_status": report.metadata.get("earth_profile_status"),
        "earth_diagnostics_outside_reference": report.metadata.get(
            "earth_diagnostics_outside_reference", []
        ),
        "global_land_surface_fraction": report.metadata.get("global_land_surface_fraction"),
        "global_potential_npp_pg_c_year": report.metadata.get("global_potential_npp_pg_c_year"),
        "global_potential_biomass_pg_c": report.metadata.get("global_potential_biomass_pg_c"),
        "reportable_climate_strata": report.metadata.get("reportable_climate_strata", []),
        "functional_profile_status": (
            report.functional_vegetation.metadata.get("earth_profile_status")
            if report.functional_vegetation is not None
            else "missing"
        ),
        "functional_diagnostics_outside_reference": (
            report.functional_vegetation.metadata.get("earth_diagnostics_outside_reference", [])
            if report.functional_vegetation is not None
            else []
        ),
        "functional_global_means": (
            report.functional_vegetation.metadata.get("global_means", {})
            if report.functional_vegetation is not None
            else {}
        ),
    }


__all__ = [
    "BiosphereEnsembleConfig",
    "BiosphereEnsembleEvaluation",
    "BiosphereEnsembleResult",
    "BiosphereEnsembleThresholds",
    "BiosphereSeedReport",
    "EnsembleGate",
    "FunctionalVegetationSeedReport",
    "evaluate_biosphere_ensemble",
    "run_biosphere_ensemble",
]
