"""Atmospheric composition, hydrostatic pressure, and greenhouse forcing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage

if TYPE_CHECKING:
    from ..execution import PipelineContext


VALIDATION_PROFILES = frozenset(
    {"earthlike", "snowball", "archipelago", "hothouse", "high_productivity", "custom"}
)


@dataclass(frozen=True)
class AtmosphereConfig:
    validation_profile: str = "earthlike"
    mean_surface_pressure_kpa: float = 101.325
    oxygen_dry_fraction: float = 0.20946
    carbon_dioxide_ppm: float = 280.0
    methane_ppm: float = 0.70
    mean_molar_mass_g_mol: float = 28.965
    reference_temperature_k: float = 288.15
    reference_co2_ppm: float = 280.0
    climate_sensitivity_c_per_co2_doubling: float = 3.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "AtmosphereConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown atmosphere controls: {', '.join(sorted(unknown))}")
        profile_raw = mapping.get("validation_profile", cls.validation_profile)
        if not isinstance(profile_raw, str):
            raise ValueError("validation_profile must be a string")
        profile = profile_raw.strip().lower()
        if profile not in VALIDATION_PROFILES:
            choices = ", ".join(sorted(VALIDATION_PROFILES))
            raise ValueError(f"validation_profile must be one of: {choices}")

        values: dict[str, float | str] = {"validation_profile": profile}
        for name, field in cls.__dataclass_fields__.items():
            if name == "validation_profile":
                continue
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            values[name] = value
        config = cls(**values)  # type: ignore[arg-type]

        positive = (
            "mean_surface_pressure_kpa",
            "mean_molar_mass_g_mol",
            "reference_temperature_k",
            "reference_co2_ppm",
            "climate_sensitivity_c_per_co2_doubling",
        )
        for name in positive:
            if getattr(config, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= config.oxygen_dry_fraction <= 1.0:
            raise ValueError("oxygen_dry_fraction must be in [0, 1]")
        if config.carbon_dioxide_ppm < 0.0 or config.methane_ppm < 0.0:
            raise ValueError("gas concentrations must be nonnegative")
        specified_fraction = (
            config.oxygen_dry_fraction + (config.carbon_dioxide_ppm + config.methane_ppm) * 1e-6
        )
        if specified_fraction > 1.0:
            raise ValueError("configured dry-gas fractions exceed one")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_mapping(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, Mapping):
        raise KeyError(f"Missing dependency metadata '{name}'")
    return cast(Mapping[str, object], record.value)


@stage(
    "atmosphere",
    inputs=("planet", "elevation", "sea_level"),
    outputs=(
        "SurfacePressureKPa",
        "OxygenPartialPressureKPa",
        "CO2PartialPressurePa",
        "MethanePartialPressurePa",
        "AtmosphericCompositionCatalog",
        "AtmosphereMetadata",
    ),
    version="v2",
)
def atmosphere_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = AtmosphereConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical atmosphere requires topology: cubed_sphere")

    planet_metadata = _artifact_mapping(deps["planet"], "PlanetMetadata")
    gravity_raw = planet_metadata["surface_gravity_g"]
    if isinstance(gravity_raw, bool) or not isinstance(gravity_raw, (int, float)):
        raise TypeError("planet surface_gravity_g must be numeric")
    gravity_m_s2 = float(gravity_raw) * 9.80665
    molar_mass_kg_mol = config.mean_molar_mass_g_mol / 1_000.0
    scale_height_m = (
        8.314462618 * config.reference_temperature_k / (molar_mass_kg_mol * gravity_m_s2)
    )
    if not np.isfinite(scale_height_m) or scale_height_m <= 0.0:
        raise RuntimeError("atmospheric scale height is invalid")

    elevation = np.asarray(
        _artifact_array(deps["sea_level"], "SurfaceElevationM"), dtype=np.float64
    )
    atmospheric_height = np.maximum(elevation, 0.0)
    pressure = config.mean_surface_pressure_kpa * np.exp(-atmospheric_height / scale_height_m)
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise RuntimeError("hydrostatic surface pressure must be finite and positive")
    co2_fraction = config.carbon_dioxide_ppm * 1e-6
    methane_fraction = config.methane_ppm * 1e-6
    background_fraction = 1.0 - config.oxygen_dry_fraction - co2_fraction - methane_fraction
    fields = {
        "SurfacePressureKPa": pressure,
        "OxygenPartialPressureKPa": pressure * config.oxygen_dry_fraction,
        "CO2PartialPressurePa": pressure * 1_000.0 * co2_fraction,
        "MethanePartialPressurePa": pressure * 1_000.0 * methane_fraction,
    }
    handles = {}
    for name, values in fields.items():
        handle = context.arena.allocate_array(
            f"atmosphere_{name.lower()}", context.topology.face_shape, np.dtype(np.float32)
        )
        handle.mutable_view()[...] = np.asarray(values, dtype=np.float32)
        handle.seal()
        handles[name] = handle

    composition = pa.table(
        {
            "gas": ["oxygen", "carbon_dioxide", "methane", "background"],
            "dry_mole_fraction": np.asarray(
                [
                    config.oxygen_dry_fraction,
                    co2_fraction,
                    methane_fraction,
                    background_fraction,
                ],
                dtype=np.float64,
            ),
            "reference_partial_pressure_pa": np.asarray(
                [
                    config.mean_surface_pressure_kpa * 1_000.0 * config.oxygen_dry_fraction,
                    config.mean_surface_pressure_kpa * 1_000.0 * co2_fraction,
                    config.mean_surface_pressure_kpa * 1_000.0 * methane_fraction,
                    config.mean_surface_pressure_kpa * 1_000.0 * background_fraction,
                ],
                dtype=np.float64,
            ),
            "role": [
                "aerobic_support",
                "carbon_substrate_and_greenhouse",
                "greenhouse_trace_gas",
                "bulk_background",
            ],
        }
    )

    co2_ratio = config.carbon_dioxide_ppm / config.reference_co2_ppm
    co2_forcing_w_m2 = 5.35 * math.log(co2_ratio) if co2_ratio > 0.0 else None
    if co2_ratio > 0.0:
        greenhouse_temperature_offset_c = config.climate_sensitivity_c_per_co2_doubling * math.log2(
            co2_ratio
        )
    else:
        # A zero-CO2 atmosphere is representable, but this logarithmic
        # approximation cannot predict its climate response.
        greenhouse_temperature_offset_c = 0.0
    areas = np.asarray(context.topology.cell_areas, dtype=np.float64)
    total_area = float(np.sum(areas))
    area_mean_pressure = float(np.sum(pressure * areas) / total_area)
    earth_diagnostics = {
        "pressure_in_earth_diagnostic_range": int(60.0 <= area_mean_pressure <= 130.0),
        "oxygen_in_earth_diagnostic_range": int(
            15.0 <= area_mean_pressure * config.oxygen_dry_fraction <= 25.0
        ),
        "co2_in_earth_diagnostic_range": int(100.0 <= config.carbon_dioxide_ppm <= 1_000.0),
    }
    metadata = {
        **asdict(config),
        "model": "hydrostatic_dry_atmosphere_v1",
        "topology": "cubed_sphere",
        "validation_profile_version": "v1",
        "scale_height_m": scale_height_m,
        "background_dry_fraction": background_fraction,
        "area_mean_surface_pressure_kpa": area_mean_pressure,
        "minimum_surface_pressure_kpa": float(np.min(pressure)),
        "maximum_surface_pressure_kpa": float(np.max(pressure)),
        "co2_radiative_forcing_w_m2": co2_forcing_w_m2,
        "co2_greenhouse_temperature_offset_c": greenhouse_temperature_offset_c,
        "greenhouse_approximation_supported": int(co2_ratio > 0.0),
        "greenhouse_approximation_in_calibration_range": int(0.25 <= co2_ratio <= 4.0),
        "earth_diagnostics": earth_diagnostics,
        "earth_diagnostic_pass": int(all(earth_diagnostics.values())),
        "hard_gate_pass": 1,
        "profile_has_reference_diagnostics": int(config.validation_profile == "earthlike"),
        "profile_calibration_status": (
            "provisional_reference" if config.validation_profile == "earthlike" else "uncalibrated"
        ),
        "pressure_semantics": "hydrostatic_dry_surface_pressure_at_land_elevation_or_sea_level",
        "composition_semantics": "configured_dry_mole_fractions_with_background_remainder",
        "methane_climate_coupling_implemented": 0,
        "vertical_temperature_structure_implemented": 0,
    }
    context.logger.log_event({"type": "atmosphere_summary", "stage": "atmosphere", **metadata})
    return {
        **handles,
        "AtmosphericCompositionCatalog": composition,
        "AtmosphereMetadata": metadata,
    }


__all__ = ["AtmosphereConfig", "VALIDATION_PROFILES", "atmosphere_stage"]
