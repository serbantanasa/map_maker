"""Raw environmental resources for later trait-first biosphere modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
from PIL import Image

from .._biosphere_envelope_native import run_biosphere_envelope
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .climate import _cube_net_rgb, _palette

if TYPE_CHECKING:
    from ..execution import PipelineContext


MONTHLY_ENVELOPE_OUTPUTS = (
    "MonthlySurfacePARMJm2",
    "MonthlyLiquidWaterOpportunity",
    "MonthlyThermalOpportunity",
    "MonthlyTerrestrialPrimaryEnergyPotentialMJm2",
)

ANNUAL_ENVELOPE_OUTPUTS = (
    "AnnualSurfacePARMJm2",
    "AnnualTerrestrialPrimaryEnergyPotentialMJm2",
    "CarbonSubstrateRelativeToReference",
    "AerobicOxygenRelativeToReference",
    "TerrestrialSurfaceSupportFraction",
    "NutrientSupportIndex",
    "EnvironmentalStressIndex",
    "BiosphereEnvelopeConfidence",
)

PRIMARY_ENERGY_PALETTE = (
    (0.0, (44, 44, 46)),
    (1.0, (104, 75, 53)),
    (3.0, (155, 133, 72)),
    (6.0, (102, 151, 94)),
    (12.0, (49, 128, 112)),
    (25.0, (47, 100, 137)),
)

STRESS_PALETTE = (
    (0.0, (54, 119, 100)),
    (0.25, (123, 157, 92)),
    (0.50, (197, 177, 93)),
    (0.75, (174, 105, 70)),
    (1.0, (91, 54, 65)),
)


@dataclass(frozen=True)
class BiosphereEnvelopeConfig:
    par_fraction: float = 0.43
    shortwave_transmission: float = 0.65
    thermal_minimum_c: float = -10.0
    thermal_optimum_low_c: float = 15.0
    thermal_optimum_high_c: float = 30.0
    thermal_maximum_c: float = 50.0
    water_input_half_saturation_mm: float = 50.0
    nutrient_half_saturation_index: float = 0.5
    co2_half_saturation_pa: float = 20.0
    reference_oxygen_partial_pressure_kpa: float = 21.22
    photosynthetic_conversion_efficiency: float = 0.0295
    minimum_productive_energy_mj_m2_year: float = 5.0
    nonreference_profile_confidence_multiplier: float = 0.75
    maximum_annual_aggregation_relative_error: float = 1e-5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "BiosphereEnvelopeConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown biosphere-envelope controls: {', '.join(sorted(unknown))}")
        values: dict[str, float] = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            values[name] = value
        config = cls(**values)
        fractions = (
            "par_fraction",
            "shortwave_transmission",
            "photosynthetic_conversion_efficiency",
            "nonreference_profile_confidence_multiplier",
        )
        for name in fractions:
            if not 0.0 <= getattr(config, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not (
            config.thermal_minimum_c
            < config.thermal_optimum_low_c
            <= config.thermal_optimum_high_c
            < config.thermal_maximum_c
        ):
            raise ValueError("thermal response temperatures must be ordered")
        positive = (
            "water_input_half_saturation_mm",
            "nutrient_half_saturation_index",
            "co2_half_saturation_pa",
            "reference_oxygen_partial_pressure_kpa",
            "maximum_annual_aggregation_relative_error",
        )
        for name in positive:
            if getattr(config, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if config.minimum_productive_energy_mj_m2_year < 0.0:
            raise ValueError("minimum_productive_energy_mj_m2_year must be nonnegative")
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


def _metadata_float(metadata: Mapping[str, object], name: str) -> float:
    raw = metadata[name]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{name} metadata must be numeric")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{name} metadata must be finite")
    return value


def _result_array(result: StageResult, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _visualizer(
    result: StageResult, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    energy = _result_array(result, "AnnualTerrestrialPrimaryEnergyPotentialMJm2")
    stress = _result_array(result, "EnvironmentalStressIndex")
    surface = _result_array(result, "TerrestrialSurfaceSupportFraction")
    if energy is None or stress is None or surface is None:
        return None
    energy_rgb = _palette(energy, PRIMARY_ENERGY_PALETTE)
    stress_rgb = _palette(stress, STRESS_PALETTE)
    land = surface > 0.0
    energy_rgb[~land] = 0
    stress_rgb[~land] = 0
    outputs = []
    for filename, image, artifact in (
        (
            "annual_primary_energy_potential.png",
            energy_rgb,
            "AnnualTerrestrialPrimaryEnergyPotentialMJm2",
        ),
        ("environmental_stress.png", stress_rgb, "EnvironmentalStressIndex"),
    ):
        output = request.output_dir / filename
        Image.fromarray(_cube_net_rgb(image), mode="RGB").save(output)
        outputs.append(
            VisualizationResult(output, artifact, {"model": "biosphere_resource_envelope_v2"})
        )
    return outputs


@stage(
    "biosphere_envelope",
    inputs=(
        "surface_materials",
        "atmosphere",
        "planet",
        "climate",
        "sea_level",
    ),
    outputs=(
        *MONTHLY_ENVELOPE_OUTPUTS,
        *ANNUAL_ENVELOPE_OUTPUTS,
        "BiosphereEnvelopeMetadata",
    ),
    version="v4",
    native_libraries=("biosphere_envelope_native",),
    visualizer=_visualizer,
)
def biosphere_envelope_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = BiosphereEnvelopeConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("biosphere envelope requires topology: cubed_sphere")

    shape = context.topology.face_shape
    monthly_shape = (12, *shape)
    output_shapes = {
        **{name: monthly_shape for name in MONTHLY_ENVELOPE_OUTPUTS},
        **{name: shape for name in ANNUAL_ENVELOPE_OUTPUTS},
    }
    handles = {
        name: context.arena.allocate_array(
            f"biosphere_envelope_{name.lower()}", output_shape, np.dtype(np.float32)
        )
        for name, output_shape in output_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}

    planet_metadata = _artifact_mapping(deps["planet"], "PlanetMetadata")
    atmosphere_metadata = _artifact_mapping(deps["atmosphere"], "AtmosphereMetadata")
    orbital_period_days = _metadata_float(planet_metadata, "orbital_period_days")
    seconds_per_month = orbital_period_days * 86_400.0 / 12.0
    reference_co2_partial_pressure_pa = (
        _metadata_float(atmosphere_metadata, "mean_surface_pressure_kpa")
        * 1_000.0
        * _metadata_float(atmosphere_metadata, "reference_co2_ppm")
        * 1e-6
    )
    profile_has_reference_diagnostics = int(
        _metadata_float(atmosphere_metadata, "profile_has_reference_diagnostics")
    )
    confidence_multiplier = (
        1.0
        if profile_has_reference_diagnostics
        else config.nonreference_profile_confidence_multiplier
    )
    output_names = {
        "monthly_par_out": "MonthlySurfacePARMJm2",
        "monthly_liquid_opportunity_out": "MonthlyLiquidWaterOpportunity",
        "monthly_thermal_opportunity_out": "MonthlyThermalOpportunity",
        "monthly_primary_energy_out": "MonthlyTerrestrialPrimaryEnergyPotentialMJm2",
        "annual_par_out": "AnnualSurfacePARMJm2",
        "annual_primary_energy_out": "AnnualTerrestrialPrimaryEnergyPotentialMJm2",
        "carbon_substrate_relative_out": "CarbonSubstrateRelativeToReference",
        "aerobic_oxygen_relative_out": "AerobicOxygenRelativeToReference",
        "terrestrial_surface_support_out": "TerrestrialSurfaceSupportFraction",
        "nutrient_support_out": "NutrientSupportIndex",
        "environmental_stress_out": "EnvironmentalStressIndex",
        "confidence_out": "BiosphereEnvelopeConfidence",
    }
    surface_materials = deps["surface_materials"]
    with context.timed("biosphere_resource_envelope_kernel"):
        metadata = run_biosphere_envelope(
            seconds_per_month=seconds_per_month,
            par_fraction=config.par_fraction,
            shortwave_transmission=config.shortwave_transmission,
            thermal_minimum_c=config.thermal_minimum_c,
            thermal_optimum_low_c=config.thermal_optimum_low_c,
            thermal_optimum_high_c=config.thermal_optimum_high_c,
            thermal_maximum_c=config.thermal_maximum_c,
            water_input_half_saturation_mm=config.water_input_half_saturation_mm,
            nutrient_half_saturation_index=config.nutrient_half_saturation_index,
            co2_half_saturation_pa=config.co2_half_saturation_pa,
            reference_co2_partial_pressure_pa=reference_co2_partial_pressure_pa,
            reference_oxygen_partial_pressure_kpa=(config.reference_oxygen_partial_pressure_kpa),
            photosynthetic_conversion_efficiency=(config.photosynthetic_conversion_efficiency),
            minimum_productive_energy_mj_m2_year=(config.minimum_productive_energy_mj_m2_year),
            confidence_multiplier=confidence_multiplier,
            areas=np.ascontiguousarray(context.topology.cell_areas, dtype=np.float64),
            ocean=np.ascontiguousarray(
                _artifact_array(deps["sea_level"], "SurfaceOceanMask"), dtype=np.float32
            ),
            monthly_insolation=np.ascontiguousarray(
                _artifact_array(deps["planet"], "MonthlyInsolationWm2"), dtype=np.float32
            ),
            monthly_temperature=np.ascontiguousarray(
                _artifact_array(deps["climate"], "MonthlySurfaceTemperatureC"),
                dtype=np.float32,
            ),
            monthly_liquid_input=np.ascontiguousarray(
                _artifact_array(surface_materials, "MonthlySoilLiquidInputMm"),
                dtype=np.float32,
            ),
            monthly_soil_saturation=np.ascontiguousarray(
                _artifact_array(surface_materials, "MonthlySoilSaturationFraction"),
                dtype=np.float32,
            ),
            soil_bearing=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilBearingFraction"), dtype=np.float32
            ),
            nutrient_potential=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilNutrientPotential"),
                dtype=np.float32,
            ),
            fertility_potential=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilFertilityPotential"),
                dtype=np.float32,
            ),
            salinity=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilSalinityIndex"), dtype=np.float32
            ),
            soil_confidence=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilConfidence"), dtype=np.float32
            ),
            co2_partial_pressure=np.ascontiguousarray(
                _artifact_array(deps["atmosphere"], "CO2PartialPressurePa"),
                dtype=np.float32,
            ),
            oxygen_partial_pressure=np.ascontiguousarray(
                _artifact_array(deps["atmosphere"], "OxygenPartialPressureKPa"),
                dtype=np.float32,
            ),
            **{native: views[artifact] for native, artifact in output_names.items()},
        )

    monthly_par = np.asarray(views["MonthlySurfacePARMJm2"], dtype=np.float64)
    monthly_primary = np.asarray(
        views["MonthlyTerrestrialPrimaryEnergyPotentialMJm2"], dtype=np.float64
    )
    annual_par = np.asarray(views["AnnualSurfacePARMJm2"], dtype=np.float64)
    annual_primary = np.asarray(
        views["AnnualTerrestrialPrimaryEnergyPotentialMJm2"], dtype=np.float64
    )
    par_aggregation_error = float(
        np.max(np.abs(np.sum(monthly_par, axis=0) - annual_par))
        / max(float(np.max(annual_par)), 1e-12)
    )
    primary_aggregation_error = float(
        np.max(np.abs(np.sum(monthly_primary, axis=0) - annual_primary))
        / max(float(np.max(annual_primary)), 1e-12)
    )
    maximum_aggregation_error = max(par_aggregation_error, primary_aggregation_error)
    if maximum_aggregation_error > config.maximum_annual_aggregation_relative_error:
        raise RuntimeError("biosphere-envelope annual aggregation audit failed")
    for name in ("MonthlyLiquidWaterOpportunity", "MonthlyThermalOpportunity"):
        values = np.asarray(views[name])
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"{name} falls outside [0, 1]")
    if np.any(monthly_primary > monthly_par * config.photosynthetic_conversion_efficiency + 1e-5):
        raise RuntimeError("primary-energy potential exceeds its PAR energy bound")
    ocean = _artifact_array(deps["sea_level"], "SurfaceOceanMask") >= 0.5
    if np.any(annual_primary[ocean] != 0.0):
        raise RuntimeError("terrestrial primary energy is nonzero over ocean")

    for handle in handles.values():
        handle.seal()
    metadata.update(
        {
            **asdict(config),
            "model": "biosphere_resource_envelope_v2",
            "topology": "cubed_sphere",
            "validation_profile": atmosphere_metadata["validation_profile"],
            "validation_profile_version": atmosphere_metadata["validation_profile_version"],
            "profile_has_reference_diagnostics": profile_has_reference_diagnostics,
            "profile_calibration_status": atmosphere_metadata["profile_calibration_status"],
            "confidence_multiplier": confidence_multiplier,
            "seconds_per_equal_time_month": seconds_per_month,
            "reference_co2_partial_pressure_pa": reference_co2_partial_pressure_pa,
            "annual_aggregation_relative_error": maximum_aggregation_error,
            "hard_gate_pass": 1,
            "biosphere_envelope_ready_for_traits": 1,
            "raw_resource_fields_are_canonical": 1,
            "combined_energy_is_universal_habitability_score": 0,
            "oxygen_limited_primary_production_implemented": 0,
            "ocean_productivity_implemented": 0,
            "biome_labels_implemented": 0,
            "vegetation_feedback_implemented": 0,
            "par_semantics": "surface_shortwave_PAR_energy_after_configured_mean_transmission",
            "primary_energy_semantics": (
                "bounded_provisional_terrestrial_photosynthetic_chemical_energy_potential"
            ),
            "environmental_stress_semantics": (
                "one_minus_PAR_weighted_thermal_water_and_carbon_response_on_supported_land"
            ),
        }
    )
    context.logger.log_event(
        {"type": "biosphere_envelope_summary", "stage": "biosphere_envelope", **metadata}
    )
    return {**handles, "BiosphereEnvelopeMetadata": metadata}


__all__ = [
    "ANNUAL_ENVELOPE_OUTPUTS",
    "BiosphereEnvelopeConfig",
    "MONTHLY_ENVELOPE_OUTPUTS",
    "biosphere_envelope_stage",
]
