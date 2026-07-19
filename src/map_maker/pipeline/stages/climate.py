"""Seasonal energy-balance climate and orographic moisture transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from PIL import Image

from .._climate_native import run_cubed_sphere_climate
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult

if TYPE_CHECKING:
    from ..execution import PipelineContext


@dataclass(frozen=True)
class ClimateConfig:
    spinup_years: int = 24
    moisture_spinup_years: int = 3
    moisture_steps_per_month_at_face_128: int = 16
    moisture_diffusion_substeps_at_face_128: int = 2
    synoptic_mixing_passes_at_face_128: int = 16
    greenhouse_offset_c: float = 0.0
    land_albedo: float = 0.30
    ocean_albedo: float = 0.28
    olr_intercept_w_m2: float = 203.0
    olr_slope_w_m2_c: float = 2.09
    heat_transport_w_m2: float = 35.0
    land_thermal_response: float = 0.22
    ocean_thermal_response: float = 0.10
    atmospheric_exchange: float = 0.16
    lapse_rate_c_per_km: float = 6.5
    wind_scale: float = 1.0
    moisture_advection_fraction: float = 0.38
    moisture_diffusion_fraction: float = 0.50
    orographic_factor: float = 0.18
    rain_shadow_factor: float = 0.70
    runoff_base_fraction: float = 0.35

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ClimateConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown climate controls: {', '.join(sorted(unknown))}")
        integer_names = {
            "spinup_years",
            "moisture_spinup_years",
            "moisture_steps_per_month_at_face_128",
            "moisture_diffusion_substeps_at_face_128",
            "synoptic_mixing_passes_at_face_128",
        }
        values: dict[str, Any] = {}
        for name in known:
            if name not in mapping:
                continue
            raw = mapping[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if name in integer_names:
                value = int(raw)
                if float(raw) != value:
                    raise ValueError(f"{name} must be an integer")
                values[name] = value
            else:
                values[name] = float(raw)
        config = cls(**values)
        for name, value in asdict(config).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        integer_bounds = {
            "spinup_years": (2, 200),
            "moisture_spinup_years": (1, 20),
            "moisture_steps_per_month_at_face_128": (2, 64),
            "moisture_diffusion_substeps_at_face_128": (1, 8),
            "synoptic_mixing_passes_at_face_128": (1, 64),
        }
        float_bounds = {
            "greenhouse_offset_c": (-20.0, 20.0),
            "land_albedo": (0.05, 0.80),
            "ocean_albedo": (0.02, 0.60),
            "olr_intercept_w_m2": (100.0, 300.0),
            "olr_slope_w_m2_c": (0.5, 5.0),
            "heat_transport_w_m2": (0.0, 120.0),
            "land_thermal_response": (0.01, 0.80),
            "ocean_thermal_response": (0.01, 0.50),
            "atmospheric_exchange": (0.0, 0.50),
            "lapse_rate_c_per_km": (3.0, 10.0),
            "wind_scale": (0.25, 3.0),
            "moisture_advection_fraction": (0.0, 0.95),
            "moisture_diffusion_fraction": (0.0, 0.50),
            "orographic_factor": (0.0, 2.0),
            "rain_shadow_factor": (0.0, 3.0),
            "runoff_base_fraction": (0.0, 0.80),
        }
        config_values = asdict(config)
        for name, (minimum, maximum) in {**integer_bounds, **float_bounds}.items():
            value = config_values[name]
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if config.ocean_thermal_response >= config.land_thermal_response:
            raise ValueError("ocean_thermal_response must be lower than land_thermal_response")
        if config.land_thermal_response + config.atmospheric_exchange > 1.0:
            raise ValueError("land thermal response plus atmospheric exchange must not exceed 1")
        if config.ocean_thermal_response + config.atmospheric_exchange > 1.0:
            raise ValueError("ocean thermal response plus atmospheric exchange must not exceed 1")
        if config.moisture_advection_fraction + config.moisture_diffusion_fraction > 1.0:
            raise ValueError("moisture advection plus diffusion must not exceed 1")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_mapping(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    if not isinstance(record.value, Mapping):
        raise TypeError(f"Dependency artifact '{name}' must be a mapping")
    return record.value


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


def _cube_net_rgb(faces: np.ndarray) -> np.ndarray:
    resolution = faces.shape[1]
    net = np.zeros((resolution * 3, resolution * 4, 3), dtype=np.uint8)
    placements = {3: (1, 0), 0: (1, 1), 2: (1, 2), 1: (1, 3), 4: (0, 1), 5: (2, 1)}
    for face, (net_row, net_col) in placements.items():
        row = net_row * resolution
        col = net_col * resolution
        net[row : row + resolution, col : col + resolution] = faces[face]
    return net


def _palette(
    values: np.ndarray, stops: tuple[tuple[float, tuple[int, int, int]], ...]
) -> np.ndarray:
    positions = np.asarray([position for position, _ in stops], dtype=np.float32)
    colors = np.asarray([color for _, color in stops], dtype=np.float32)
    channels = [np.interp(values, positions, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)


def _montage(
    monthly: np.ndarray,
    palette_stops: tuple[tuple[float, tuple[int, int, int]], ...],
) -> np.ndarray:
    month_nets = [_cube_net_rgb(_palette(monthly[month], palette_stops)) for month in range(12)]
    panel_height, panel_width = month_nets[0].shape[:2]
    result = np.zeros((3 * panel_height, 4 * panel_width, 3), dtype=np.uint8)
    for month, month_net in enumerate(month_nets):
        row, col = divmod(month, 4)
        result[
            row * panel_height : (row + 1) * panel_height,
            col * panel_width : (col + 1) * panel_width,
        ] = month_net
    return result


TEMPERATURE_PALETTE = (
    (-60.0, (17, 28, 74)),
    (-30.0, (39, 87, 145)),
    (-10.0, (112, 166, 190)),
    (0.0, (210, 230, 216)),
    (12.0, (115, 164, 91)),
    (24.0, (224, 191, 83)),
    (35.0, (193, 89, 53)),
    (50.0, (104, 30, 39)),
)

PRECIPITATION_PALETTE = (
    (0.0, (94, 62, 45)),
    (100.0, (161, 118, 68)),
    (300.0, (210, 184, 108)),
    (600.0, (111, 154, 87)),
    (1200.0, (48, 130, 115)),
    (2200.0, (40, 102, 151)),
    (4000.0, (205, 230, 235)),
)


def _climate_visualizer(
    result: StageResult, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    monthly_temperature = _result_array(result, "MonthlySurfaceTemperatureC")
    monthly_precipitation = _result_array(result, "MonthlyPrecipitationMm")
    annual_temperature = _result_array(result, "AnnualMeanTemperatureC")
    annual_precipitation = _result_array(result, "AnnualPrecipitationMm")
    wind = _result_array(result, "MonthlyWindVectorXYZMps")
    speed = _result_array(result, "MonthlyWindSpeedMps")
    if any(
        value is None
        for value in (
            monthly_temperature,
            monthly_precipitation,
            annual_temperature,
            annual_precipitation,
            wind,
            speed,
        )
    ):
        return None
    assert monthly_temperature is not None
    assert monthly_precipitation is not None
    assert annual_temperature is not None
    assert annual_precipitation is not None
    assert wind is not None
    assert speed is not None

    outputs: list[VisualizationResult] = []
    annual_temperature_path = request.output_dir / "annual_temperature.png"
    Image.fromarray(
        _cube_net_rgb(_palette(annual_temperature, TEMPERATURE_PALETTE)), mode="RGB"
    ).save(annual_temperature_path)
    outputs.append(
        VisualizationResult(
            annual_temperature_path, "AnnualMeanTemperatureC", {"scale_c": [-60, 50]}
        )
    )

    annual_precipitation_path = request.output_dir / "annual_precipitation.png"
    Image.fromarray(
        _cube_net_rgb(_palette(annual_precipitation, PRECIPITATION_PALETTE)), mode="RGB"
    ).save(annual_precipitation_path)
    outputs.append(
        VisualizationResult(
            annual_precipitation_path,
            "AnnualPrecipitationMm",
            {"scale_mm_year": [0, 4000]},
        )
    )

    temperature_months_path = request.output_dir / "monthly_temperature.png"
    Image.fromarray(_montage(monthly_temperature, TEMPERATURE_PALETTE), mode="RGB").save(
        temperature_months_path
    )
    outputs.append(
        VisualizationResult(temperature_months_path, "MonthlySurfaceTemperatureC", {"months": 12})
    )

    monthly_precipitation_scale = tuple(
        (position / 12.0, color) for position, color in PRECIPITATION_PALETTE
    )
    precipitation_months_path = request.output_dir / "monthly_precipitation.png"
    Image.fromarray(_montage(monthly_precipitation, monthly_precipitation_scale), mode="RGB").save(
        precipitation_months_path
    )
    outputs.append(
        VisualizationResult(precipitation_months_path, "MonthlyPrecipitationMm", {"months": 12})
    )

    mean_wind = np.mean(wind, axis=0)
    mean_speed = np.mean(speed, axis=0)
    wind_direction = 0.5 + 0.5 * mean_wind / np.maximum(mean_speed[..., None], 1e-6)
    brightness = np.clip(0.35 + mean_speed / 18.0, 0.35, 1.0)
    wind_rgb = (wind_direction * brightness[..., None] * 255.0).clip(0, 255).astype(np.uint8)
    wind_path = request.output_dir / "prevailing_wind.png"
    Image.fromarray(_cube_net_rgb(wind_rgb), mode="RGB").save(wind_path)
    outputs.append(
        VisualizationResult(
            wind_path,
            "MonthlyWindVectorXYZMps",
            {"rgb": "global_xyz_direction", "brightness": "mean_speed"},
        )
    )
    return outputs


MONTHLY_SCALAR_OUTPUTS = (
    "MonthlySurfaceTemperatureC",
    "MonthlyWindSpeedMps",
    "MonthlyPrecipitationMm",
    "MonthlyRelativeHumidity",
    "MonthlySnowfallMm",
    "MonthlySnowmeltMm",
    "MonthlySnowWaterEquivalentMm",
    "MonthlyEvaporationMm",
    "MonthlyRunoffPotentialMm",
)


@stage(
    "climate",
    inputs=("planet", "atmosphere", "elevation", "world_age"),
    outputs=(
        *MONTHLY_SCALAR_OUTPUTS,
        "MonthlyWindVectorXYZMps",
        "ClimateOrographyM",
        "AnnualMeanTemperatureC",
        "AnnualPrecipitationMm",
        "AnnualAridityIndex",
        "ClimateMetadata",
    ),
    version="v5",
    native_libraries=("climate_native",),
    visualizer=_climate_visualizer,
)
def climate_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = ClimateConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical climate requires topology: cubed_sphere")

    shape = context.topology.face_shape
    monthly_shape = (12, *shape)
    artifact_shapes = {
        **{name: monthly_shape for name in MONTHLY_SCALAR_OUTPUTS},
        "MonthlyWindVectorXYZMps": (*monthly_shape, 3),
        "ClimateOrographyM": shape,
        "AnnualMeanTemperatureC": shape,
        "AnnualPrecipitationMm": shape,
        "AnnualAridityIndex": shape,
    }
    handles = {
        name: context.arena.allocate_array(
            f"climate_{name.lower()}", artifact_shape, np.dtype(np.float32)
        )
        for name, artifact_shape in artifact_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    planet = deps["planet"]
    atmosphere = deps["atmosphere"]
    elevation = deps["elevation"]
    world_age = deps["world_age"]
    effective_transport_steps = max(
        2,
        round(config.moisture_steps_per_month_at_face_128 * context.topology.face_resolution / 128),
    )
    effective_diffusion_substeps = max(
        1,
        round(
            config.moisture_diffusion_substeps_at_face_128
            * context.topology.face_resolution
            / 128
        ),
    )
    effective_moisture_steps = effective_transport_steps * effective_diffusion_substeps
    effective_moisture_advection_fraction = (
        config.moisture_advection_fraction / effective_diffusion_substeps
    )
    effective_synoptic_mixing_passes = max(
        1,
        round(
            config.synoptic_mixing_passes_at_face_128
            * (context.topology.face_resolution / 128) ** 2
        ),
    )
    kernel_controls = asdict(config)
    kernel_controls.pop("moisture_steps_per_month_at_face_128")
    kernel_controls.pop("moisture_diffusion_substeps_at_face_128")
    kernel_controls.pop("synoptic_mixing_passes_at_face_128")
    kernel_controls["moisture_advection_fraction"] = effective_moisture_advection_fraction
    atmosphere_metadata = _artifact_mapping(atmosphere, "AtmosphereMetadata")
    composition_greenhouse_offset = _metadata_float(
        atmosphere_metadata, "co2_greenhouse_temperature_offset_c"
    )
    effective_greenhouse_offset = config.greenhouse_offset_c + composition_greenhouse_offset
    if not np.isfinite(effective_greenhouse_offset):
        raise RuntimeError("effective greenhouse offset is not finite")
    kernel_controls["greenhouse_offset_c"] = effective_greenhouse_offset

    with context.timed("seasonal_climate_kernel"):
        metadata = run_cubed_sphere_climate(
            **kernel_controls,
            moisture_steps_per_month=effective_moisture_steps,
            synoptic_mixing_passes=effective_synoptic_mixing_passes,
            areas=context.topology.cell_areas,
            neighbors=context.topology.neighbor_indices,
            xyz=context.topology.xyz,
            latitudes=context.topology.latitude,
            elevation=_artifact_array(elevation, "BedrockElevationM"),
            relief=_artifact_array(elevation, "TerrainReliefM"),
            ocean=_artifact_array(world_age, "BaseOceanMask"),
            insolation=_artifact_array(planet, "MonthlyInsolationWm2"),
            declination=_artifact_array(planet, "SolarDeclinationRad"),
            climate_orography_out=views["ClimateOrographyM"],
            temperature_out=views["MonthlySurfaceTemperatureC"],
            wind_xyz_out=views["MonthlyWindVectorXYZMps"],
            wind_speed_out=views["MonthlyWindSpeedMps"],
            precipitation_out=views["MonthlyPrecipitationMm"],
            humidity_out=views["MonthlyRelativeHumidity"],
            snowfall_out=views["MonthlySnowfallMm"],
            snowmelt_out=views["MonthlySnowmeltMm"],
            snowpack_out=views["MonthlySnowWaterEquivalentMm"],
            evaporation_out=views["MonthlyEvaporationMm"],
            runoff_out=views["MonthlyRunoffPotentialMm"],
            annual_temperature_out=views["AnnualMeanTemperatureC"],
            annual_precipitation_out=views["AnnualPrecipitationMm"],
            aridity_out=views["AnnualAridityIndex"],
        )

    for handle in handles.values():
        handle.seal()
    planet_metadata = _artifact_mapping(planet, "PlanetMetadata")
    metadata.update(
        {
            **asdict(config),
            "effective_transport_steps_per_month": effective_transport_steps,
            "effective_moisture_diffusion_substeps": effective_diffusion_substeps,
            "effective_moisture_steps_per_month": effective_moisture_steps,
            "effective_moisture_advection_fraction": (
                effective_moisture_advection_fraction
            ),
            "effective_synoptic_mixing_passes": effective_synoptic_mixing_passes,
            "transport_reference_face_resolution": 128,
            "topology": "cubed_sphere",
            "model": "seasonal_energy_moisture_climate_v5",
            "month_semantics": planet_metadata["forcing_semantics"],
            "temperature_semantics": "surface_air_energy_balance_with_elevation_lapse",
            "wind_semantics": "global_xyz_tangent_vector",
            "precipitation_semantics": "advected_moisture_with_orographic_condensation",
            "evaporation_semantics": "ocean_actual_and_provisional_land_actual",
            "runoff_semantics": "pre_soil_routing_potential",
            "configured_climate_greenhouse_offset_c": config.greenhouse_offset_c,
            "composition_greenhouse_offset_c": composition_greenhouse_offset,
            "effective_greenhouse_offset_c": effective_greenhouse_offset,
            "atmosphere_validation_profile": atmosphere_metadata["validation_profile"],
            "composition_dependent_radiative_transfer_implemented": 0,
        }
    )
    context.logger.log_event({"type": "climate_summary", "stage": "climate", **metadata})
    return {**handles, "ClimateMetadata": metadata}


__all__ = ["ClimateConfig", "climate_stage"]
