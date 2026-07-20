"""Fractional L2 surface materials and property-first initial soils."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from PIL import Image

from .._surface_materials_native import run_surface_materials
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .climate import _cube_net_rgb, _palette

if TYPE_CHECKING:
    from ..execution import PipelineContext

MATERIAL_OUTPUTS = (
    "BedrockSurfaceFraction",
    "ResidualRegolithFraction",
    "ColluviumFraction",
    "AlluviumFraction",
    "LacustrineSedimentFraction",
    "GlacialDepositFraction",
    "VolcaniclasticFraction",
)

SOIL_OUTPUTS = (
    "SoilBearingFraction",
    "RegolithDepthM",
    "SoilDepthM",
    "SandFraction",
    "SiltFraction",
    "ClayFraction",
    "CoarseFragmentFraction",
    "BulkDensityKgM3",
    "PotentialSoilOrganicCarbonKgM2",
    "SoilPH",
    "SoilCarbonateFraction",
    "SoilSalinityIndex",
    "SoilDrainageIndex",
    "SoilAvailableWaterCapacityMm",
    "SoilNutrientPotential",
    "SoilFertilityPotential",
    "SoilErodibility",
    "SurfaceResetAgeKa",
    "HydricSoilFraction",
    "SoilConfidence",
    "AnnualSoilWaterStorageChangeMm",
)

MONTHLY_OUTPUTS = (
    "MonthlySoilLiquidInputMm",
    "MonthlySoilWaterMm",
    "MonthlySoilSaturationFraction",
    "MonthlyActualEvapotranspirationMm",
    "MonthlySoilRunoffMm",
    "MonthlyDeepDrainageMm",
)

EFFECTIVE_SURFACE_WATER_OUTPUTS = (
    "EffectiveLakeFraction",
    "EffectiveWetlandFraction",
    "EffectiveSurfaceWaterHydroperiod",
    "RefinedSurfaceWaterMask",
)

MATERIAL_CODES = {
    0: "ocean_or_unclassified",
    1: "exposed_bedrock",
    2: "residual_regolith",
    3: "colluvium",
    4: "alluvium",
    5: "lacustrine_sediment",
    6: "glacial_deposit",
    7: "volcaniclastic",
}

MATERIAL_COLORS: np.ndarray = np.asarray(
    [
        (112, 108, 102),
        (151, 105, 62),
        (117, 96, 78),
        (180, 159, 100),
        (112, 136, 132),
        (195, 205, 207),
        (78, 69, 65),
    ],
    dtype=np.float64,
)

DEPTH_PALETTE = (
    (0.0, (60, 62, 60)),
    (0.2, (118, 94, 65)),
    (0.6, (157, 126, 77)),
    (1.2, (170, 154, 93)),
    (2.0, (124, 145, 93)),
    (3.0, (78, 116, 91)),
)

HYDRIC_PALETTE = (
    (0.0, (57, 60, 56)),
    (0.02, (91, 104, 88)),
    (0.15, (96, 139, 129)),
    (0.40, (75, 135, 156)),
    (1.0, (187, 216, 213)),
)


@dataclass(frozen=True)
class SurfaceMaterialsConfig:
    spinup_years: int = 24
    maximum_regolith_depth_m: float = 20.0
    maximum_soil_depth_m: float = 3.0
    maximum_alluvial_fraction: float = 0.65
    maximum_lacustrine_fraction: float = 0.85
    maximum_glacial_fraction: float = 0.80
    weathering_temperature_scale_c: float = 22.0
    weathering_precipitation_scale_mm: float = 1_600.0
    soil_evaporation_factor: float = 1.0
    monthly_deep_drainage_fraction: float = 0.06
    maximum_component_balance_error: float = 1e-5
    maximum_water_balance_relative_error: float = 1e-5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "SurfaceMaterialsConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown surface-material controls: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            values[name] = int(raw) if name == "spinup_years" else float(raw)
        config = cls(**values)  # type: ignore[arg-type]
        if not 2 <= config.spinup_years <= 2_000:
            raise ValueError("spinup_years must be in [2, 2000]")
        if not 0.0 < config.maximum_soil_depth_m <= config.maximum_regolith_depth_m:
            raise ValueError(
                "maximum_soil_depth_m must be positive and no greater than maximum_regolith_depth_m"
            )
        bounds = {
            "maximum_regolith_depth_m": (0.01, 1_000.0),
            "maximum_alluvial_fraction": (0.0, 1.0),
            "maximum_lacustrine_fraction": (0.0, 1.0),
            "maximum_glacial_fraction": (0.0, 1.0),
            "weathering_temperature_scale_c": (0.1, 100.0),
            "weathering_precipitation_scale_mm": (1.0, 20_000.0),
            "soil_evaporation_factor": (0.0, 10.0),
            "monthly_deep_drainage_fraction": (0.0, 1.0),
            "maximum_component_balance_error": (1e-12, 0.01),
            "maximum_water_balance_relative_error": (1e-12, 0.01),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(config, name)
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return cast(pa.Table, record.value)


def _artifact_mapping(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, Mapping):
        raise KeyError(f"Missing dependency metadata '{name}'")
    return cast(Mapping[str, object], record.value)


def _fixed_list(table: pa.Table, name: str, size: int) -> np.ndarray:
    values = table[name].combine_chunks().values
    return np.asarray(values, dtype=np.float64).reshape(table.num_rows, size)


def _refined_surface_fields(
    topology: CubedSphereGrid,
    final_surface_water: StageResult,
) -> dict[str, np.ndarray]:
    shape = topology.face_shape
    total = int(np.prod(shape))
    fields: dict[str, np.ndarray] = {
        "refined_mask": np.zeros(total, dtype=np.float32),
        "refined_lake_fraction": np.zeros(total, dtype=np.float32),
        "refined_wetland_fraction": np.zeros(total, dtype=np.float32),
        "refined_hydroperiod": np.zeros(total, dtype=np.float32),
        "refined_salinity": np.zeros(total, dtype=np.float32),
        "recent_erosion_depth": np.zeros(total, dtype=np.float32),
        "recent_deposition_depth": np.zeros(total, dtype=np.float32),
    }
    cells = _artifact_table(final_surface_water, "FinalOutletCorrectedBasinCellCatalog")
    if cells.num_rows == 0:
        return {name: value.reshape(shape) for name, value in fields.items()}
    fine_ids = np.asarray(cells["fine_cell_id"], dtype=np.int64)
    parent_ids = np.asarray(cells["parent_cell_id"], dtype=np.int64)
    child_area_km2 = np.asarray(cells["area_km2"], dtype=np.float64)
    if np.any(parent_ids < 0) or np.any(parent_ids >= total):
        raise RuntimeError("refined surface-water parent ID is outside the L2 grid")
    parent_unique = np.unique(parent_ids)
    fields["refined_mask"][parent_unique] = 1.0

    parent_area_km2 = np.zeros(total, dtype=np.float64)
    np.add.at(parent_area_km2, parent_ids, child_area_km2)
    if np.any(~np.isfinite(parent_area_km2[parent_unique])) or np.any(
        parent_area_km2[parent_unique] <= 0.0
    ):
        raise RuntimeError("refined surface-water parent has invalid physical area")

    eroded_volume = np.asarray(cells["subgrid_eroded_volume_m3"], dtype=np.float64)
    deposited_volume = np.asarray(cells["floodplain_deposited_volume_m3"], dtype=np.float64)
    eroded_parent: np.ndarray = np.zeros(total, dtype=np.float64)
    deposited_parent: np.ndarray = np.zeros(total, dtype=np.float64)
    np.add.at(eroded_parent, parent_ids, eroded_volume)
    np.add.at(deposited_parent, parent_ids, deposited_volume)
    parent_area_m2 = parent_area_km2 * 1e6
    fields["recent_erosion_depth"] = np.asarray(
        np.divide(
            eroded_parent,
            parent_area_m2,
            out=np.zeros_like(eroded_parent),
            where=parent_area_m2 > 0.0,
        ),
        dtype=np.float32,
    )
    fields["recent_deposition_depth"] = np.asarray(
        np.divide(
            deposited_parent,
            parent_area_m2,
            out=np.zeros_like(deposited_parent),
            where=parent_area_m2 > 0.0,
        ),
        dtype=np.float32,
    )

    water_cells = _artifact_table(final_surface_water, "SeasonalSurfaceWaterCellCatalog")
    if water_cells.num_rows == 0:
        return {name: value.reshape(shape) for name, value in fields.items()}
    order = np.argsort(fine_ids, kind="stable")
    sorted_ids = fine_ids[order]
    water_ids = np.asarray(water_cells["fine_cell_id"], dtype=np.int64)
    positions = np.searchsorted(sorted_ids, water_ids)
    if np.any(positions >= sorted_ids.size) or np.any(sorted_ids[positions] != water_ids):
        raise RuntimeError("surface-water cell is missing from the final refined basin")
    cell_rows = order[positions]
    water_parents = parent_ids[cell_rows]
    water_child_area = child_area_km2[cell_rows]
    mean_fraction = np.asarray(water_cells["mean_inundation_fraction"], dtype=np.float64)
    monthly_fraction = _fixed_list(water_cells, "monthly_inundation_fraction", 12)
    hydroperiod = np.mean(monthly_fraction > 1e-6, axis=1)
    class_code = np.asarray(water_cells["class_code"], dtype=np.int32)
    wet_area = water_child_area * mean_fraction
    lake_area: np.ndarray = np.zeros(total, dtype=np.float64)
    wetland_area: np.ndarray = np.zeros(total, dtype=np.float64)
    wet_area_total: np.ndarray = np.zeros(total, dtype=np.float64)
    hydroperiod_area: np.ndarray = np.zeros(total, dtype=np.float64)
    lake_rows = np.isin(class_code, (2, 3))
    wetland_rows = np.isin(class_code, (1, 4))
    np.add.at(lake_area, water_parents[lake_rows], wet_area[lake_rows])
    np.add.at(wetland_area, water_parents[wetland_rows], wet_area[wetland_rows])
    np.add.at(wet_area_total, water_parents, wet_area)
    np.add.at(hydroperiod_area, water_parents, wet_area * hydroperiod)

    candidate_table = _artifact_table(final_surface_water, "SurfaceWaterCandidateCatalog")
    candidate_ids = np.asarray(candidate_table["depression_id"], dtype=np.int64)
    candidate_salinity = np.asarray(candidate_table["salinity_index"], dtype=np.float64)
    candidate_order = np.argsort(candidate_ids, kind="stable")
    sorted_candidates = candidate_ids[candidate_order]
    depression_ids = np.asarray(water_cells["depression_id"], dtype=np.int64)
    candidate_positions = np.searchsorted(sorted_candidates, depression_ids)
    if np.any(candidate_positions >= sorted_candidates.size) or np.any(
        sorted_candidates[candidate_positions] != depression_ids
    ):
        raise RuntimeError("surface-water cell references an unknown depression")
    salinity_area: np.ndarray = np.zeros(total, dtype=np.float64)
    salinity = candidate_salinity[candidate_order[candidate_positions]]
    np.add.at(salinity_area, water_parents, wet_area * salinity)

    represented_water_area = lake_area + wetland_area
    if np.any(
        represented_water_area[parent_unique] > parent_area_km2[parent_unique] * (1.0 + 1e-8)
    ):
        raise RuntimeError("refined surface-water area exceeds its physical parent area")

    fields["refined_lake_fraction"] = np.asarray(
        np.clip(
            np.divide(
                lake_area,
                parent_area_km2,
                out=np.zeros_like(lake_area),
                where=parent_area_km2 > 0.0,
            ),
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    fields["refined_wetland_fraction"] = np.asarray(
        np.clip(
            np.divide(
                wetland_area,
                parent_area_km2,
                out=np.zeros_like(wetland_area),
                where=parent_area_km2 > 0.0,
            ),
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    fields["refined_hydroperiod"] = np.asarray(
        np.divide(
            hydroperiod_area,
            wet_area_total,
            out=np.zeros_like(hydroperiod_area),
            where=wet_area_total > 0.0,
        ),
        dtype=np.float32,
    )
    fields["refined_salinity"] = np.asarray(
        np.divide(
            salinity_area,
            wet_area_total,
            out=np.zeros_like(salinity_area),
            where=wet_area_total > 0.0,
        ),
        dtype=np.float32,
    )

    projection_audits = (
        (
            "lake area",
            float(np.sum(fields["refined_lake_fraction"] * parent_area_km2)),
            float(np.sum(lake_area)),
        ),
        (
            "wetland area",
            float(np.sum(fields["refined_wetland_fraction"] * parent_area_km2)),
            float(np.sum(wetland_area)),
        ),
        (
            "erosion volume",
            float(np.sum(fields["recent_erosion_depth"] * parent_area_m2)),
            float(np.sum(eroded_volume)),
        ),
        (
            "deposition volume",
            float(np.sum(fields["recent_deposition_depth"] * parent_area_m2)),
            float(np.sum(deposited_volume)),
        ),
    )
    for name, represented, expected in projection_audits:
        relative_error = abs(represented - expected) / max(abs(expected), 1.0)
        if relative_error > 1e-6:
            raise RuntimeError(f"refined surface-water {name} projection is not conservative")
    return {name: np.ascontiguousarray(value.reshape(shape)) for name, value in fields.items()}


def _result_array(result: StageResult, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _visualizer(
    result: StageResult, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    fractions = [_result_array(result, name) for name in MATERIAL_OUTPUTS]
    soil_depth = _result_array(result, "SoilDepthM")
    hydric = _result_array(result, "HydricSoilFraction")
    if any(value is None for value in fractions) or soil_depth is None or hydric is None:
        return None
    material_stack = np.stack([value for value in fractions if value is not None], axis=-1)
    material_rgb = np.clip(material_stack @ MATERIAL_COLORS, 0.0, 255.0).astype(np.uint8)
    land = np.sum(material_stack, axis=-1) > 0.0
    depth_rgb = _palette(soil_depth, DEPTH_PALETTE)
    hydric_rgb = _palette(hydric, HYDRIC_PALETTE)
    depth_rgb[~land] = 0
    hydric_rgb[~land] = 0
    metadata = _artifact_mapping(result, "SurfaceMaterialsMetadata")
    outputs = (
        (
            "surface_material_mix.png",
            _cube_net_rgb(material_rgb),
            "SurfaceMaterialFractions",
        ),
        ("soil_depth.png", _cube_net_rgb(depth_rgb), "SoilDepthM"),
        (
            "hydric_soil_fraction.png",
            _cube_net_rgb(hydric_rgb),
            "HydricSoilFraction",
        ),
    )
    visualizations: list[VisualizationResult] = []
    for filename, image, artifact_name in outputs:
        output = request.output_dir / filename
        Image.fromarray(image, mode="RGB").save(output)
        visualizations.append(
            VisualizationResult(
                output,
                artifact_name,
                {
                    "model": metadata["model"],
                    "soil_readiness": metadata["surface_materials_ready_for_biomes"],
                },
            )
        )
    return visualizations


@stage(
    "surface_materials",
    inputs=(
        "hydrology_validation",
        "surface_water_final",
        "hydrology",
        "cryosphere",
        "climate",
        "elevation",
        "geology",
        "world_age",
    ),
    outputs=(
        *MATERIAL_OUTPUTS,
        "DominantSurfaceMaterialCode",
        *SOIL_OUTPUTS,
        *MONTHLY_OUTPUTS,
        *EFFECTIVE_SURFACE_WATER_OUTPUTS,
        "SurfaceMaterialsMetadata",
    ),
    version="v3",
    native_libraries=("surface_materials_native",),
    visualizer=_visualizer,
)
def surface_materials_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = SurfaceMaterialsConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("surface materials require topology: cubed_sphere")
    validation_metadata = _artifact_mapping(
        deps["hydrology_validation"], "HydrologyValidationMetadata"
    )
    water_metadata = _artifact_mapping(deps["surface_water_final"], "SurfaceWaterMetadata")
    if validation_metadata.get("hard_gate_pass") != 1:
        raise RuntimeError("surface materials require passing hydrology hard gates")
    if water_metadata.get("surface_water_ready_for_soils") != 1:
        raise RuntimeError("surface materials require converged final surface water")

    shape = context.topology.face_shape
    monthly_shape = (12, *shape)
    output_shapes = {
        **{name: shape for name in MATERIAL_OUTPUTS},
        **{name: shape for name in SOIL_OUTPUTS},
        **{name: monthly_shape for name in MONTHLY_OUTPUTS},
        **{name: shape for name in EFFECTIVE_SURFACE_WATER_OUTPUTS},
        "DominantSurfaceMaterialCode": shape,
    }
    handles = {
        name: context.arena.allocate_array(
            f"surface_materials_{name.lower()}",
            output_shape,
            np.dtype(np.uint8 if name == "DominantSurfaceMaterialCode" else np.float32),
        )
        for name, output_shape in output_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    refined = _refined_surface_fields(context.topology, deps["surface_water_final"])
    maximum_component_error = config.maximum_component_balance_error
    maximum_water_error = config.maximum_water_balance_relative_error
    geology = deps["geology"]
    elevation = deps["elevation"]
    hydrology = deps["hydrology"]
    climate = deps["climate"]
    cryosphere = deps["cryosphere"]
    world_age = deps["world_age"]
    coarse_lake = np.ascontiguousarray(_artifact_array(hydrology, "LakeFraction"), dtype=np.float32)
    coarse_wetland = np.ascontiguousarray(
        _artifact_array(hydrology, "WetlandFraction"), dtype=np.float32
    )
    refined_mask = refined["refined_mask"] >= 0.5
    views["EffectiveLakeFraction"][:] = np.where(
        refined_mask, refined["refined_lake_fraction"], coarse_lake
    )
    views["EffectiveWetlandFraction"][:] = np.where(
        refined_mask, refined["refined_wetland_fraction"], coarse_wetland
    )
    views["EffectiveSurfaceWaterHydroperiod"][:] = np.where(
        refined_mask,
        refined["refined_hydroperiod"],
        np.clip(coarse_lake + 0.65 * coarse_wetland, 0.0, 1.0),
    )
    views["RefinedSurfaceWaterMask"][:] = refined["refined_mask"]
    output_names = {
        "bedrock_out": "BedrockSurfaceFraction",
        "residual_out": "ResidualRegolithFraction",
        "colluvium_out": "ColluviumFraction",
        "alluvium_out": "AlluviumFraction",
        "lacustrine_out": "LacustrineSedimentFraction",
        "glacial_out": "GlacialDepositFraction",
        "volcaniclastic_out": "VolcaniclasticFraction",
        "dominant_material_out": "DominantSurfaceMaterialCode",
        "soil_bearing_out": "SoilBearingFraction",
        "regolith_depth_out": "RegolithDepthM",
        "soil_depth_out": "SoilDepthM",
        "sand_out": "SandFraction",
        "silt_out": "SiltFraction",
        "clay_out": "ClayFraction",
        "coarse_fragments_out": "CoarseFragmentFraction",
        "bulk_density_out": "BulkDensityKgM3",
        "organic_carbon_out": "PotentialSoilOrganicCarbonKgM2",
        "soil_ph_out": "SoilPH",
        "carbonate_out": "SoilCarbonateFraction",
        "salinity_out": "SoilSalinityIndex",
        "drainage_out": "SoilDrainageIndex",
        "available_water_capacity_out": "SoilAvailableWaterCapacityMm",
        "nutrient_potential_out": "SoilNutrientPotential",
        "fertility_potential_out": "SoilFertilityPotential",
        "erodibility_out": "SoilErodibility",
        "reset_age_out": "SurfaceResetAgeKa",
        "hydric_fraction_out": "HydricSoilFraction",
        "soil_confidence_out": "SoilConfidence",
        "monthly_liquid_input_out": "MonthlySoilLiquidInputMm",
        "monthly_soil_water_out": "MonthlySoilWaterMm",
        "monthly_saturation_out": "MonthlySoilSaturationFraction",
        "monthly_evapotranspiration_out": "MonthlyActualEvapotranspirationMm",
        "monthly_runoff_out": "MonthlySoilRunoffMm",
        "monthly_deep_drainage_out": "MonthlyDeepDrainageMm",
        "annual_storage_change_out": "AnnualSoilWaterStorageChangeMm",
    }
    with context.timed("surface_materials_and_initial_soils_kernel"):
        metadata = run_surface_materials(
            spinup_years=config.spinup_years,
            maximum_regolith_depth_m=config.maximum_regolith_depth_m,
            maximum_soil_depth_m=config.maximum_soil_depth_m,
            maximum_alluvial_fraction=config.maximum_alluvial_fraction,
            maximum_lacustrine_fraction=config.maximum_lacustrine_fraction,
            maximum_glacial_fraction=config.maximum_glacial_fraction,
            weathering_temperature_scale_c=config.weathering_temperature_scale_c,
            weathering_precipitation_scale_mm=config.weathering_precipitation_scale_mm,
            soil_evaporation_factor=config.soil_evaporation_factor,
            monthly_deep_drainage_fraction=config.monthly_deep_drainage_fraction,
            areas=np.ascontiguousarray(context.topology.cell_areas, dtype=np.float64),
            ocean=np.ascontiguousarray(
                _artifact_array(world_age, "BaseOceanMask"), dtype=np.float32
            ),
            province_class=np.ascontiguousarray(
                _artifact_array(geology, "GeologicalProvinceClass"), dtype=np.uint8
            ),
            crust_age=np.ascontiguousarray(
                _artifact_array(geology, "CrustAgeGa"), dtype=np.float32
            ),
            rock_strength=np.ascontiguousarray(
                _artifact_array(geology, "RockStrength"), dtype=np.float32
            ),
            accommodation=np.ascontiguousarray(
                _artifact_array(geology, "SedimentAccommodation"), dtype=np.float32
            ),
            province_confidence=np.ascontiguousarray(
                _artifact_array(geology, "ProvinceConfidence"), dtype=np.float32
            ),
            elevation_confidence=np.ascontiguousarray(
                _artifact_array(elevation, "ElevationConfidence"), dtype=np.float32
            ),
            relief=np.ascontiguousarray(
                _artifact_array(elevation, "TerrainReliefM"), dtype=np.float32
            ),
            flow_slope=np.ascontiguousarray(
                _artifact_array(hydrology, "FlowSlope"), dtype=np.float32
            ),
            river_corridor=np.ascontiguousarray(
                _artifact_array(hydrology, "RiverCorridor"), dtype=np.float32
            ),
            floodplain=np.ascontiguousarray(
                _artifact_array(hydrology, "FloodplainPotential"), dtype=np.float32
            ),
            lake_fraction=coarse_lake,
            wetland_fraction=coarse_wetland,
            depression_fill_depth=np.ascontiguousarray(
                _artifact_array(hydrology, "DepressionFillDepthM"), dtype=np.float32
            ),
            refined_mask=refined["refined_mask"],
            refined_lake_fraction=refined["refined_lake_fraction"],
            refined_wetland_fraction=refined["refined_wetland_fraction"],
            refined_hydroperiod=refined["refined_hydroperiod"],
            refined_salinity=refined["refined_salinity"],
            recent_erosion_depth=refined["recent_erosion_depth"],
            recent_deposition_depth=refined["recent_deposition_depth"],
            glacier_fraction=np.ascontiguousarray(
                _artifact_array(cryosphere, "GlacierIceFraction"), dtype=np.float32
            ),
            annual_temperature=np.ascontiguousarray(
                _artifact_array(climate, "AnnualMeanTemperatureC"), dtype=np.float32
            ),
            annual_precipitation=np.ascontiguousarray(
                _artifact_array(climate, "AnnualPrecipitationMm"), dtype=np.float32
            ),
            monthly_temperature=np.ascontiguousarray(
                _artifact_array(climate, "MonthlySurfaceTemperatureC"), dtype=np.float32
            ),
            monthly_precipitation=np.ascontiguousarray(
                _artifact_array(climate, "MonthlyPrecipitationMm"), dtype=np.float32
            ),
            monthly_evaporation=np.ascontiguousarray(
                _artifact_array(climate, "MonthlyEvaporationMm"), dtype=np.float32
            ),
            monthly_snowfall=np.ascontiguousarray(
                _artifact_array(cryosphere, "MonthlySnowfallMm"), dtype=np.float32
            ),
            monthly_snowmelt=np.ascontiguousarray(
                _artifact_array(cryosphere, "MonthlySnowmeltMm"), dtype=np.float32
            ),
            monthly_glacier_melt=np.ascontiguousarray(
                _artifact_array(cryosphere, "MonthlyGlacierMeltMm"), dtype=np.float32
            ),
            **{
                native_name: views[artifact_name]
                for native_name, artifact_name in output_names.items()
            },
        )

    ocean = _artifact_array(world_age, "BaseOceanMask") >= 0.5
    land = ~ocean
    material_sum = np.sum(
        np.stack([np.asarray(views[name], dtype=np.float64) for name in MATERIAL_OUTPUTS]),
        axis=0,
    )
    texture_sum = (
        np.asarray(views["SandFraction"], dtype=np.float64)
        + np.asarray(views["SiltFraction"], dtype=np.float64)
        + np.asarray(views["ClayFraction"], dtype=np.float64)
    )
    material_error = float(np.max(np.abs(material_sum[land] - 1.0)))
    texture_error = float(np.max(np.abs(texture_sum[land] - 1.0)))
    monthly_input = np.asarray(views["MonthlySoilLiquidInputMm"], dtype=np.float64)
    monthly_outputs = sum(
        np.asarray(views[name], dtype=np.float64)
        for name in (
            "MonthlyActualEvapotranspirationMm",
            "MonthlySoilRunoffMm",
            "MonthlyDeepDrainageMm",
        )
    )
    residual = np.sum(monthly_input - monthly_outputs, axis=0) - np.asarray(
        views["AnnualSoilWaterStorageChangeMm"], dtype=np.float64
    )
    areas = np.asarray(context.topology.cell_areas, dtype=np.float64)
    water_balance_error = float(
        np.sum(np.abs(residual[land]) * areas[land])
        / max(float(np.sum(monthly_input[:, land] * areas[land])), 1e-12)
    )
    if material_error > maximum_component_error or texture_error > maximum_component_error:
        raise RuntimeError("surface-material component balance audit failed")
    if water_balance_error > maximum_water_error:
        raise RuntimeError("initial-soil water-balance audit failed")
    if np.any(np.asarray(views["SoilDepthM"])[land] > np.asarray(views["RegolithDepthM"])[land]):
        raise RuntimeError("soil depth exceeds regolith depth")
    effective_lake = np.asarray(views["EffectiveLakeFraction"], dtype=np.float64)
    effective_wetland = np.asarray(views["EffectiveWetlandFraction"], dtype=np.float64)
    glacier = _artifact_array(cryosphere, "GlacierIceFraction")
    soil_bearing = np.asarray(views["SoilBearingFraction"], dtype=np.float64)
    if np.any(effective_lake < 0.0) or np.any(effective_lake > 1.0):
        raise RuntimeError("effective lake fractions are outside [0, 1]")
    if np.any(effective_wetland < 0.0) or np.any(effective_wetland > 1.0):
        raise RuntimeError("effective wetland fractions are outside [0, 1]")
    available_ground = np.maximum(1.0 - effective_lake - glacier, 0.0)
    if np.any(soil_bearing[land] > available_ground[land] + 1e-6):
        raise RuntimeError("soil-bearing support exceeds effective non-open land")
    land_area = float(np.sum(areas[land]))

    def land_mean(name: str) -> float:
        values = np.asarray(views[name], dtype=np.float64)
        return float(np.sum(values[land] * areas[land]) / land_area)

    def annual_land_mean(name: str) -> float:
        values = np.sum(np.asarray(views[name], dtype=np.float64), axis=0)
        return float(np.sum(values[land] * areas[land]) / land_area)

    annual_input_mean = annual_land_mean("MonthlySoilLiquidInputMm")
    annual_runoff_mean = annual_land_mean("MonthlySoilRunoffMm")
    for handle in handles.values():
        handle.seal()
    metadata.update(
        {
            **asdict(config),
            "model": "fractional_surface_materials_and_initial_soils_v2",
            "topology": "cubed_sphere",
            "material_codes": {str(code): name for code, name in MATERIAL_CODES.items()},
            "material_fraction_semantics": "mutually_exclusive_L2_component_area_fractions",
            "texture_semantics": "fine_earth_sand_silt_clay_sum_to_one",
            "coarse_fragment_semantics": "separate_fraction_of_total_soil_material",
            "monthly_water_semantics": "whole_L2_cell_equivalent_depth_over_non_open_land",
            "material_balance_max_error": material_error,
            "texture_balance_max_error": texture_error,
            "water_balance_relative_error": water_balance_error,
            "land_mean_available_water_capacity_mm": land_mean("SoilAvailableWaterCapacityMm"),
            "land_mean_soil_ph": land_mean("SoilPH"),
            "land_mean_soil_salinity_index": land_mean("SoilSalinityIndex"),
            "land_mean_soil_fertility_potential": land_mean("SoilFertilityPotential"),
            "land_mean_annual_soil_liquid_input_mm": annual_input_mean,
            "land_mean_annual_actual_evapotranspiration_mm": annual_land_mean(
                "MonthlyActualEvapotranspirationMm"
            ),
            "land_mean_annual_soil_runoff_mm": annual_runoff_mean,
            "land_mean_annual_deep_drainage_mm": annual_land_mean("MonthlyDeepDrainageMm"),
            "soil_runoff_input_fraction": annual_runoff_mean / max(annual_input_mean, 1e-12),
            "refined_surface_water_parent_count": int(
                np.count_nonzero(refined["refined_mask"] >= 0.5)
            ),
            "refined_surface_projection": "conservative_child_area_km2_v2",
            "effective_lake_land_area_fraction": land_mean("EffectiveLakeFraction"),
            "effective_wetland_land_area_fraction": land_mean("EffectiveWetlandFraction"),
            "hydrology_hard_gate_pass": 1,
            "surface_water_ready_for_soils": 1,
            "surface_materials_ready_for_biomes": 1,
            "hydrology_feedback_applied": 0,
            "groundwater_routing_implemented": 0,
            "ecological_wetland_confirmation_implemented": 0,
            "vegetation_feedback_implemented": 0,
            "soil_taxonomy_labels_implemented": 0,
        }
    )
    context.logger.log_event(
        {"type": "surface_materials_summary", "stage": "surface_materials", **metadata}
    )
    return {**handles, "SurfaceMaterialsMetadata": metadata}


__all__ = [
    "MATERIAL_CODES",
    "MATERIAL_OUTPUTS",
    "MONTHLY_OUTPUTS",
    "SOIL_OUTPUTS",
    "SurfaceMaterialsConfig",
    "surface_materials_stage",
]
