"""Continuous terrestrial producer-community potentials and adaptation pressures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
from PIL import Image

from .._potential_biosphere_native import run_potential_biosphere
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .climate import _cube_net_rgb, _palette

if TYPE_CHECKING:
    from ..execution import PipelineContext


MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS = ("MonthlyPotentialNPPKgCM2",)

NORMALIZED_POTENTIAL_BIOSPHERE_OUTPUTS = (
    "PotentialVegetationCoverFraction",
    "GrowingSeasonFraction",
    "ProductivitySeasonalityIndex",
    "DroughtAdaptationPressure",
    "ColdAdaptationPressure",
    "HeatAdaptationPressure",
    "WaterloggingAdaptationPressure",
    "SalinityAdaptationPressure",
    "PotentialWoodyAllocationTrait",
    "PotentialResourceConservativeTrait",
    "PotentialFuelContinuityIndex",
    "PotentialBiosphereConfidence",
)

SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS = (
    "AnnualPotentialNPPKgCM2",
    "PotentialStandingBiomassKgCM2",
    "PotentialRootingDepthM",
    "PotentialCanopyHeightM",
    "PotentialLeafAreaIndex",
    *NORMALIZED_POTENTIAL_BIOSPHERE_OUTPUTS,
)

NPP_PALETTE = (
    (0.0, (48, 47, 48)),
    (0.02, (113, 80, 54)),
    (0.05, (164, 143, 75)),
    (0.10, (111, 157, 91)),
    (0.25, (50, 132, 105)),
    (0.50, (45, 104, 137)),
)

COVER_PALETTE = (
    (0.0, (54, 52, 50)),
    (0.10, (127, 91, 58)),
    (0.25, (164, 142, 76)),
    (0.50, (115, 153, 83)),
    (0.75, (58, 128, 92)),
    (1.0, (38, 91, 77)),
)


@dataclass(frozen=True)
class PotentialBiosphereConfig:
    energy_per_kg_carbon_mj: float = 39.9
    cover_half_saturation_npp_kg_c_m2_year: float = 0.30
    active_month_thermal_threshold: float = 0.15
    active_month_water_threshold: float = 0.10
    cold_pressure_reference_c: float = -15.0
    cold_pressure_release_c: float = 10.0
    heat_pressure_onset_c: float = 30.0
    heat_pressure_reference_c: float = 50.0
    minimum_biomass_residence_years: float = 0.5
    maximum_biomass_residence_years: float = 45.0
    biomass_residence_baseline_fraction: float = 0.10
    woody_biomass_residence_weight: float = 0.60
    resource_conservative_biomass_residence_weight: float = 0.40
    low_productivity_biomass_residence_weight: float = 2.50
    maximum_rooting_depth_m: float = 6.0
    maximum_canopy_height_m: float = 45.0
    maximum_leaf_area_index: float = 8.0
    maximum_standing_biomass_kg_c_m2: float = 40.0
    maximum_annual_aggregation_relative_error: float = 1e-5
    maximum_energy_conversion_relative_error: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "PotentialBiosphereConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown potential-biosphere controls: {', '.join(sorted(unknown))}")
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
        positive = (
            "energy_per_kg_carbon_mj",
            "cover_half_saturation_npp_kg_c_m2_year",
            "maximum_rooting_depth_m",
            "maximum_canopy_height_m",
            "maximum_leaf_area_index",
            "maximum_standing_biomass_kg_c_m2",
            "maximum_annual_aggregation_relative_error",
            "maximum_energy_conversion_relative_error",
        )
        for name in positive:
            if getattr(config, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "active_month_thermal_threshold",
            "active_month_water_threshold",
            "biomass_residence_baseline_fraction",
        ):
            if not 0.0 <= getattr(config, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        residence_weights = (
            config.woody_biomass_residence_weight,
            config.resource_conservative_biomass_residence_weight,
            config.low_productivity_biomass_residence_weight,
        )
        if any(weight < 0.0 for weight in residence_weights) or sum(residence_weights) <= 0.0:
            raise ValueError("biomass residence weights must be nonnegative with a positive sum")
        if config.cold_pressure_reference_c >= config.cold_pressure_release_c:
            raise ValueError("cold pressure reference must be below its release temperature")
        if config.heat_pressure_onset_c >= config.heat_pressure_reference_c:
            raise ValueError("heat pressure onset must be below its reference temperature")
        if not (
            0.0 <= config.minimum_biomass_residence_years <= config.maximum_biomass_residence_years
        ):
            raise ValueError("biomass residence-year bounds are invalid")
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


def _result_array(result: StageResult, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _visualizer(
    result: StageResult, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    npp = _result_array(result, "AnnualPotentialNPPKgCM2")
    cover = _result_array(result, "PotentialVegetationCoverFraction")
    woody = _result_array(result, "PotentialWoodyAllocationTrait")
    conservative = _result_array(result, "PotentialResourceConservativeTrait")
    waterlogging = _result_array(result, "WaterloggingAdaptationPressure")
    if any(value is None for value in (npp, cover, woody, conservative, waterlogging)):
        return None
    assert npp is not None
    assert cover is not None
    assert woody is not None
    assert conservative is not None
    assert waterlogging is not None
    land = cover > 0.0
    npp_rgb = _palette(npp, NPP_PALETTE)
    cover_rgb = _palette(cover, COVER_PALETTE)
    trait_rgb = (
        np.stack((conservative, woody, waterlogging), axis=-1).clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    npp_rgb[~land] = 0
    cover_rgb[~land] = 0
    trait_rgb[~land] = 0
    outputs = []
    for filename, image, artifact, metadata in (
        ("annual_potential_npp.png", npp_rgb, "AnnualPotentialNPPKgCM2", {}),
        ("potential_vegetation_cover.png", cover_rgb, "PotentialVegetationCoverFraction", {}),
        (
            "potential_trait_composite.png",
            trait_rgb,
            "PotentialBiosphereTraits",
            {"red": "resource_conservative", "green": "woody", "blue": "waterlogging"},
        ),
    ):
        output = request.output_dir / filename
        Image.fromarray(_cube_net_rgb(image), mode="RGB").save(output)
        outputs.append(
            VisualizationResult(
                output,
                artifact,
                {"model": "trait_first_potential_biosphere_v2", **metadata},
            )
        )
    return outputs


@stage(
    "potential_biosphere",
    inputs=("biosphere_envelope", "surface_materials", "climate", "sea_level"),
    outputs=(
        *MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS,
        *SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS,
        "PotentialBiosphereMetadata",
    ),
    version="v4",
    native_libraries=("potential_biosphere_native",),
    visualizer=_visualizer,
)
def potential_biosphere_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = PotentialBiosphereConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("potential biosphere requires topology: cubed_sphere")

    shape = context.topology.face_shape
    monthly_shape = (12, *shape)
    output_shapes = {
        "MonthlyPotentialNPPKgCM2": monthly_shape,
        **{name: shape for name in SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS},
    }
    handles = {
        name: context.arena.allocate_array(
            f"potential_biosphere_{name.lower()}", output_shape, np.dtype(np.float32)
        )
        for name, output_shape in output_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    output_names = {
        "monthly_npp_out": "MonthlyPotentialNPPKgCM2",
        "annual_npp_out": "AnnualPotentialNPPKgCM2",
        "vegetation_cover_out": "PotentialVegetationCoverFraction",
        "standing_biomass_out": "PotentialStandingBiomassKgCM2",
        "growing_season_out": "GrowingSeasonFraction",
        "productivity_seasonality_out": "ProductivitySeasonalityIndex",
        "drought_pressure_out": "DroughtAdaptationPressure",
        "cold_pressure_out": "ColdAdaptationPressure",
        "heat_pressure_out": "HeatAdaptationPressure",
        "waterlogging_pressure_out": "WaterloggingAdaptationPressure",
        "salinity_pressure_out": "SalinityAdaptationPressure",
        "woody_trait_out": "PotentialWoodyAllocationTrait",
        "resource_conservative_trait_out": "PotentialResourceConservativeTrait",
        "rooting_depth_out": "PotentialRootingDepthM",
        "canopy_height_out": "PotentialCanopyHeightM",
        "leaf_area_index_out": "PotentialLeafAreaIndex",
        "fuel_continuity_out": "PotentialFuelContinuityIndex",
        "confidence_out": "PotentialBiosphereConfidence",
    }
    envelope = deps["biosphere_envelope"]
    surface_materials = deps["surface_materials"]
    climate = deps["climate"]
    sea_level = deps["sea_level"]
    with context.timed("trait_first_potential_biosphere_kernel"):
        metadata = run_potential_biosphere(
            energy_per_kg_carbon_mj=config.energy_per_kg_carbon_mj,
            cover_half_saturation_npp_kg_c_m2_year=(config.cover_half_saturation_npp_kg_c_m2_year),
            active_month_thermal_threshold=config.active_month_thermal_threshold,
            active_month_water_threshold=config.active_month_water_threshold,
            cold_pressure_reference_c=config.cold_pressure_reference_c,
            cold_pressure_release_c=config.cold_pressure_release_c,
            heat_pressure_onset_c=config.heat_pressure_onset_c,
            heat_pressure_reference_c=config.heat_pressure_reference_c,
            minimum_biomass_residence_years=config.minimum_biomass_residence_years,
            maximum_biomass_residence_years=config.maximum_biomass_residence_years,
            biomass_residence_baseline_fraction=(config.biomass_residence_baseline_fraction),
            woody_biomass_residence_weight=config.woody_biomass_residence_weight,
            resource_conservative_biomass_residence_weight=(
                config.resource_conservative_biomass_residence_weight
            ),
            low_productivity_biomass_residence_weight=(
                config.low_productivity_biomass_residence_weight
            ),
            maximum_rooting_depth_m=config.maximum_rooting_depth_m,
            maximum_canopy_height_m=config.maximum_canopy_height_m,
            maximum_leaf_area_index=config.maximum_leaf_area_index,
            maximum_standing_biomass_kg_c_m2=(config.maximum_standing_biomass_kg_c_m2),
            areas=np.ascontiguousarray(context.topology.cell_areas, dtype=np.float64),
            ocean=np.ascontiguousarray(
                _artifact_array(sea_level, "SurfaceOceanMask"), dtype=np.float32
            ),
            monthly_primary_energy=np.ascontiguousarray(
                _artifact_array(envelope, "MonthlyTerrestrialPrimaryEnergyPotentialMJm2"),
                dtype=np.float32,
            ),
            monthly_thermal_opportunity=np.ascontiguousarray(
                _artifact_array(envelope, "MonthlyThermalOpportunity"), dtype=np.float32
            ),
            monthly_water_opportunity=np.ascontiguousarray(
                _artifact_array(envelope, "MonthlyLiquidWaterOpportunity"),
                dtype=np.float32,
            ),
            monthly_temperature=np.ascontiguousarray(
                _artifact_array(climate, "MonthlySurfaceTemperatureC"), dtype=np.float32
            ),
            monthly_soil_saturation=np.ascontiguousarray(
                _artifact_array(surface_materials, "MonthlySoilSaturationFraction"),
                dtype=np.float32,
            ),
            surface_support=np.ascontiguousarray(
                _artifact_array(envelope, "TerrestrialSurfaceSupportFraction"),
                dtype=np.float32,
            ),
            nutrient_support=np.ascontiguousarray(
                _artifact_array(envelope, "NutrientSupportIndex"), dtype=np.float32
            ),
            environmental_stress=np.ascontiguousarray(
                _artifact_array(envelope, "EnvironmentalStressIndex"), dtype=np.float32
            ),
            soil_depth=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilDepthM"), dtype=np.float32
            ),
            regolith_depth=np.ascontiguousarray(
                _artifact_array(surface_materials, "RegolithDepthM"), dtype=np.float32
            ),
            salinity=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilSalinityIndex"), dtype=np.float32
            ),
            hydric_fraction=np.ascontiguousarray(
                _artifact_array(surface_materials, "HydricSoilFraction"), dtype=np.float32
            ),
            soil_confidence=np.ascontiguousarray(
                _artifact_array(surface_materials, "SoilConfidence"), dtype=np.float32
            ),
            envelope_confidence=np.ascontiguousarray(
                _artifact_array(envelope, "BiosphereEnvelopeConfidence"),
                dtype=np.float32,
            ),
            **{native: views[artifact] for native, artifact in output_names.items()},
        )

    monthly_energy = np.asarray(
        _artifact_array(envelope, "MonthlyTerrestrialPrimaryEnergyPotentialMJm2"),
        dtype=np.float64,
    )
    monthly_npp = np.asarray(views["MonthlyPotentialNPPKgCM2"], dtype=np.float64)
    annual_npp = np.asarray(views["AnnualPotentialNPPKgCM2"], dtype=np.float64)
    expected_npp = monthly_energy / config.energy_per_kg_carbon_mj
    conversion_error = float(
        np.max(np.abs(monthly_npp - expected_npp)) / max(float(np.max(expected_npp)), 1e-12)
    )
    aggregation_error = float(
        np.max(np.abs(np.sum(monthly_npp, axis=0) - annual_npp))
        / max(float(np.max(annual_npp)), 1e-12)
    )
    if conversion_error > config.maximum_energy_conversion_relative_error:
        raise RuntimeError("potential-biosphere energy conversion audit failed")
    if aggregation_error > config.maximum_annual_aggregation_relative_error:
        raise RuntimeError("potential-biosphere annual aggregation audit failed")
    for name in NORMALIZED_POTENTIAL_BIOSPHERE_OUTPUTS:
        values = np.asarray(views[name])
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"{name} falls outside [0, 1]")
    regolith = _artifact_array(surface_materials, "RegolithDepthM")
    roots = np.asarray(views["PotentialRootingDepthM"])
    if np.any(roots > np.minimum(regolith, config.maximum_rooting_depth_m) + 1e-5):
        raise RuntimeError("potential rooting depth exceeds accessible regolith")
    bounded_outputs = {
        "PotentialCanopyHeightM": config.maximum_canopy_height_m,
        "PotentialLeafAreaIndex": config.maximum_leaf_area_index,
        "PotentialStandingBiomassKgCM2": config.maximum_standing_biomass_kg_c_m2,
    }
    for name, maximum in bounded_outputs.items():
        values = np.asarray(views[name])
        if np.any(values < 0.0) or np.any(values > maximum + 1e-5):
            raise RuntimeError(f"{name} exceeds its configured physical bound")
    ocean = _artifact_array(sea_level, "SurfaceOceanMask") >= 0.5
    terrestrial_outputs = (
        "AnnualPotentialNPPKgCM2",
        "PotentialVegetationCoverFraction",
        "PotentialStandingBiomassKgCM2",
        "PotentialRootingDepthM",
        "PotentialCanopyHeightM",
        "PotentialLeafAreaIndex",
        "PotentialFuelContinuityIndex",
    )
    if np.any(monthly_npp[:, ocean] != 0.0) or any(
        np.any(np.asarray(views[name])[ocean] != 0.0) for name in terrestrial_outputs
    ):
        raise RuntimeError("terrestrial potential-biosphere state is nonzero over ocean")

    for handle in handles.values():
        handle.seal()
    envelope_metadata = _artifact_mapping(envelope, "BiosphereEnvelopeMetadata")
    actual_maximum_rooting_depth_m = metadata["maximum_rooting_depth_m"]
    actual_maximum_canopy_height_m = metadata["maximum_canopy_height_m"]
    metadata.update(
        {
            **asdict(config),
            "model": "trait_first_potential_biosphere_v2",
            "topology": "cubed_sphere",
            "validation_profile": envelope_metadata["validation_profile"],
            "profile_calibration_status": envelope_metadata["profile_calibration_status"],
            "energy_conversion_relative_error": conversion_error,
            "annual_aggregation_relative_error": aggregation_error,
            "actual_maximum_rooting_depth_m": actual_maximum_rooting_depth_m,
            "actual_maximum_canopy_height_m": actual_maximum_canopy_height_m,
            "hard_gate_pass": 1,
            "potential_biosphere_ready_for_functional_types": 1,
            "colonized_photosynthetic_producer_assumption": 1,
            "actual_vegetation_state_implemented": 0,
            "evolutionary_history_implemented": 0,
            "species_implemented": 0,
            "functional_type_fractions_implemented": 0,
            "biome_labels_implemented": 0,
            "disturbance_events_implemented": 0,
            "vegetation_feedback_implemented": 0,
            "npp_semantics": "potential_carbon_fixation_from_bounded_15b0_chemical_energy",
            "trait_semantics": "continuous_equilibrium_producer_community_potential",
            "pressure_semantics": "environmental_adaptation_requirement_not_organism_trait",
        }
    )
    context.logger.log_event(
        {"type": "potential_biosphere_summary", "stage": "potential_biosphere", **metadata}
    )
    return {**handles, "PotentialBiosphereMetadata": metadata}


__all__ = [
    "MONTHLY_POTENTIAL_BIOSPHERE_OUTPUTS",
    "NORMALIZED_POTENTIAL_BIOSPHERE_OUTPUTS",
    "PotentialBiosphereConfig",
    "SCALAR_POTENTIAL_BIOSPHERE_OUTPUTS",
    "potential_biosphere_stage",
]
