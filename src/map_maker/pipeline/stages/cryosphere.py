"""Bounded seasonal snow, firn, and glacier mass-balance stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
from PIL import Image

from .._cryosphere_native import run_cryosphere
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .climate import _cube_net_rgb, _palette

if TYPE_CHECKING:
    from ..execution import PipelineContext

MONTHLY_OUTPUTS = (
    "MonthlySnowfallMm",
    "MonthlySnowmeltMm",
    "MonthlySnowWaterEquivalentMm",
    "MonthlyFirnToIceMm",
    "MonthlyGlacierMeltMm",
    "MonthlyGlacierIceWaterEquivalentMm",
    "MonthlySeaIceFraction",
    "MonthlySeaIceThicknessM",
    "MonthlyRunoffPotentialMm",
)

ANNUAL_OUTPUTS = (
    "AnnualGlacierMassBalanceMm",
    "AnnualGlacierFlowExportMm",
    "AnnualGlacierFlowImportMm",
    "AnnualGlacierCalvingMm",
    "AnnualGlacierSublimationMm",
    "GlacierIceFraction",
)

GLACIER_PALETTE = (
    (0.0, (45, 56, 51)),
    (0.001, (104, 126, 125)),
    (0.02, (135, 179, 190)),
    (0.10, (181, 218, 224)),
    (0.40, (226, 240, 239)),
    (1.0, (255, 255, 255)),
)


@dataclass(frozen=True)
class CryosphereConfig:
    spinup_years: int = 120
    lapse_rate_c_per_km: float = 6.5
    relief_elevation_multiplier: float = 3.0
    maximum_highland_fraction: float = 0.40
    snow_degree_day_melt_mm_c_month: float = 12.0
    glacier_degree_day_melt_mm_c_month: float = 16.0
    firn_conversion_fraction_year: float = 0.65
    snow_sublimation_fraction_month: float = 0.004
    glacier_sublimation_fraction_month: float = 0.0002
    glacier_flow_activation_mm: float = 1_000.0
    glacier_flow_fraction_year: float = 0.08
    glacier_reference_thickness_mm: float = 100_000.0
    sea_ice_freezing_temperature_c: float = -1.8
    sea_ice_melt_temperature_c: float = -0.2
    sea_ice_freeze_rate_mm_c_month: float = 50.0
    sea_ice_melt_rate_mm_c_month: float = 320.0
    sea_ice_reference_thickness_mm: float = 1_800.0
    sea_ice_maximum_thickness_mm: float = 3_500.0
    runoff_base_fraction: float = 0.35
    maximum_mass_balance_relative_error: float = 1e-5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CryosphereConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown cryosphere controls: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            values[name] = int(raw) if name == "spinup_years" else float(raw)
        config = cls(**values)  # type: ignore[arg-type]
        if not 2 <= config.spinup_years <= 2_000:
            raise ValueError("spinup_years must be in [2, 2000]")
        bounds = {
            "lapse_rate_c_per_km": (0.0, 12.0),
            "relief_elevation_multiplier": (0.0, 6.0),
            "maximum_highland_fraction": (0.0, 1.0),
            "snow_degree_day_melt_mm_c_month": (0.1, 200.0),
            "glacier_degree_day_melt_mm_c_month": (0.1, 200.0),
            "firn_conversion_fraction_year": (0.0, 1.0),
            "snow_sublimation_fraction_month": (0.0, 1.0),
            "glacier_sublimation_fraction_month": (0.0, 1.0),
            "glacier_flow_activation_mm": (0.0, 1_000_000.0),
            "glacier_flow_fraction_year": (0.0, 1.0),
            "glacier_reference_thickness_mm": (1.0, 10_000_000.0),
            "sea_ice_freezing_temperature_c": (-10.0, 2.0),
            "sea_ice_melt_temperature_c": (-10.0, 10.0),
            "sea_ice_freeze_rate_mm_c_month": (0.1, 2_000.0),
            "sea_ice_melt_rate_mm_c_month": (0.1, 2_000.0),
            "sea_ice_reference_thickness_mm": (1.0, 10_000.0),
            "sea_ice_maximum_thickness_mm": (1.0, 50_000.0),
            "runoff_base_fraction": (0.0, 1.0),
            "maximum_mass_balance_relative_error": (1e-12, 0.01),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(config, name)
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
        if config.sea_ice_melt_temperature_c < config.sea_ice_freezing_temperature_c:
            raise ValueError(
                "sea_ice_melt_temperature_c must be greater than or equal to "
                "sea_ice_freezing_temperature_c"
            )
        if config.sea_ice_maximum_thickness_mm < config.sea_ice_reference_thickness_mm:
            raise ValueError(
                "sea_ice_maximum_thickness_mm must be greater than or equal to "
                "sea_ice_reference_thickness_mm"
            )
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _result_array(result: StageResult, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _visualizer(result: StageResult, request: VisualizationRequest) -> VisualizationResult | None:
    fraction = _result_array(result, "GlacierIceFraction")
    if fraction is None:
        return None
    output = request.output_dir / "glacier_fraction.png"
    Image.fromarray(_cube_net_rgb(_palette(fraction, GLACIER_PALETTE)), mode="RGB").save(output)
    sea_ice = _result_array(result, "MonthlySeaIceFraction")
    if sea_ice is not None:
        maximum_sea_ice = np.max(sea_ice, axis=0)
        sea_ice_output = request.output_dir / "sea_ice_maximum_fraction.png"
        Image.fromarray(
            _cube_net_rgb(_palette(maximum_sea_ice, GLACIER_PALETTE)), mode="RGB"
        ).save(sea_ice_output)
    metadata_value = result.artifact_records["CryosphereMetadata"].value
    if not isinstance(metadata_value, Mapping):
        raise TypeError("CryosphereMetadata must be a mapping")
    metadata = cast(Mapping[str, object], metadata_value)
    return VisualizationResult(
        output,
        "GlacierIceFraction",
        {
            "glacierized_land_area_fraction": metadata["glacierized_land_area_fraction"],
            "glacier_ice_land_area_fraction": metadata["glacier_ice_land_area_fraction"],
            "maximum_sea_ice_ocean_area_fraction": metadata[
                "maximum_sea_ice_ocean_area_fraction"
            ],
        },
    )


@stage(
    "cryosphere",
    inputs=("climate", "elevation", "sea_level"),
    outputs=(*MONTHLY_OUTPUTS, *ANNUAL_OUTPUTS, "CryosphereMetadata"),
    version="v5",
    native_libraries=("cryosphere_native",),
    visualizer=_visualizer,
)
def cryosphere_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = CryosphereConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical cryosphere requires topology: cubed_sphere")
    shape = context.topology.face_shape
    monthly_shape = (12, *shape)
    artifact_shapes = {
        **{name: monthly_shape for name in MONTHLY_OUTPUTS},
        **{name: shape for name in ANNUAL_OUTPUTS},
    }
    handles = {
        name: context.arena.allocate_array(
            f"cryosphere_{name.lower()}", field_shape, np.dtype(np.float32)
        )
        for name, field_shape in artifact_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    climate = deps["climate"]
    controls = asdict(config)
    maximum_balance_error = controls.pop("maximum_mass_balance_relative_error")
    with context.timed("seasonal_snow_firn_glacier_sea_ice_kernel"):
        metadata = run_cryosphere(
            **controls,
            areas=context.topology.cell_areas,
            neighbors=context.topology.neighbor_indices,
            ocean=np.ascontiguousarray(
                _artifact_array(deps["sea_level"], "SurfaceOceanMask"), dtype=np.float32
            ),
            elevation=np.ascontiguousarray(
                _artifact_array(deps["sea_level"], "SurfaceElevationM"), dtype=np.float32
            ),
            relief=np.ascontiguousarray(
                _artifact_array(deps["elevation"], "TerrainReliefM"), dtype=np.float32
            ),
            temperature=np.ascontiguousarray(
                _artifact_array(climate, "MonthlySurfaceTemperatureC"), dtype=np.float32
            ),
            precipitation=np.ascontiguousarray(
                _artifact_array(climate, "MonthlyPrecipitationMm"), dtype=np.float32
            ),
            evaporation=np.ascontiguousarray(
                _artifact_array(climate, "MonthlyEvaporationMm"), dtype=np.float32
            ),
            snowfall_out=views["MonthlySnowfallMm"],
            snowmelt_out=views["MonthlySnowmeltMm"],
            snowpack_out=views["MonthlySnowWaterEquivalentMm"],
            firn_to_ice_out=views["MonthlyFirnToIceMm"],
            glacier_melt_out=views["MonthlyGlacierMeltMm"],
            glacier_ice_out=views["MonthlyGlacierIceWaterEquivalentMm"],
            sea_ice_fraction_out=views["MonthlySeaIceFraction"],
            sea_ice_thickness_m_out=views["MonthlySeaIceThicknessM"],
            runoff_out=views["MonthlyRunoffPotentialMm"],
            annual_mass_balance_out=views["AnnualGlacierMassBalanceMm"],
            annual_flow_export_out=views["AnnualGlacierFlowExportMm"],
            annual_flow_import_out=views["AnnualGlacierFlowImportMm"],
            annual_calving_out=views["AnnualGlacierCalvingMm"],
            annual_sublimation_out=views["AnnualGlacierSublimationMm"],
            glacier_fraction_out=views["GlacierIceFraction"],
        )

    areas = np.asarray(context.topology.cell_areas, dtype=np.float64)
    ocean = _artifact_array(deps["sea_level"], "SurfaceOceanMask") >= 0.5
    land = ~ocean
    firn = np.sum(np.asarray(views["MonthlyFirnToIceMm"], dtype=np.float64), axis=0)
    melt = np.sum(np.asarray(views["MonthlyGlacierMeltMm"], dtype=np.float64), axis=0)
    snowfall = np.sum(np.asarray(views["MonthlySnowfallMm"], dtype=np.float64), axis=0)
    mass_balance = np.asarray(views["AnnualGlacierMassBalanceMm"], dtype=np.float64)
    flow_export = np.asarray(views["AnnualGlacierFlowExportMm"], dtype=np.float64)
    flow_import = np.asarray(views["AnnualGlacierFlowImportMm"], dtype=np.float64)
    sublimation = np.asarray(views["AnnualGlacierSublimationMm"], dtype=np.float64)
    # Ice reservoir balance: storage change equals firn/direct-ice input from snow
    # minus melt, sublimation, and net flow (all mm water equivalent).
    residual = mass_balance - (firn - melt - sublimation - flow_export + flow_import)
    absolute_residual = float(abs(np.sum(residual * areas)))
    reference_volume = float(np.sum((firn + melt + sublimation + flow_export) * areas))
    relative_residual = absolute_residual / max(reference_volume, 1e-12)
    if relative_residual > maximum_balance_error:
        raise RuntimeError("cryosphere mass-balance audit failed")
    # Hard conservation: ice inputs cannot exceed snowfall on land (no free ice).
    ice_input_excess = float(np.sum(np.maximum(firn - snowfall, 0.0)[land] * areas[land]))
    snowfall_volume = float(np.sum(snowfall[land] * areas[land]))
    ice_input_excess_fraction = ice_input_excess / max(snowfall_volume, 1e-12)
    if ice_input_excess_fraction > maximum_balance_error:
        raise RuntimeError(
            "cryosphere invented ice mass: firn/direct-ice exceeds snowfall "
            f"({ice_input_excess_fraction:.3e} of land snowfall volume)"
        )
    for handle in handles.values():
        handle.seal()
    metadata.update(
        {
            **asdict(config),
            "model": "conserved_land_ice_plus_thermodynamic_sea_ice_v4",
            "topology": "cubed_sphere",
            "mass_balance_relative_error": relative_residual,
            "ice_input_excess_of_snowfall_fraction": ice_input_excess_fraction,
            "glacier_mass_balance_implemented": 1,
            "parameterized_glacier_flow_implemented": 1,
            "dynamic_ice_stress_flow_implemented": 0,
            "glacial_erosion_implemented": 0,
            "polar_ice_cap_mode": "conserved_snowfall_cold_climate_routing",
            "mountain_glacier_mode": "relief_peak_cooled_highland_fraction",
            "synthetic_precipitation_floor": 0,
            "synthetic_ice_source": 0,
            "sea_ice_implemented": 1,
            "sea_ice_semantics": "temperature_driven_freeze_melt_thickness_reservoir",
            "melt_provenance": "seasonal_snow_and_glacier_ice_are_separate",
            "runoff_semantics": "pre_soil_routing_potential_including_glacier_melt",
        }
    )
    context.logger.log_event({"type": "cryosphere_summary", "stage": "cryosphere", **metadata})
    return {**handles, "CryosphereMetadata": metadata}


__all__ = ["CryosphereConfig", "cryosphere_stage"]
