"""Planetary boundary conditions and monthly orbital forcing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from PIL import Image

from .._planet_native import run_cubed_sphere_planet
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


@dataclass(frozen=True)
class PlanetConfig:
    planet_radius_earth: float = 1.0
    surface_gravity_g: float = 1.0
    mean_density_earth: float = 1.0
    star_luminosity_solar: float = 1.0
    star_effective_temperature_k: float = 5772.0
    semi_major_axis_au: float = 1.0
    eccentricity: float = 0.0167
    obliquity_deg: float = 23.44
    rotation_period_hours: float = 24.0
    orbital_period_days: float = 365.2422
    perihelion_day: float = 3.0
    northern_vernal_equinox_day: float = 79.0
    moon_mass_lunar: float = 1.0
    moon_distance_km: float = 384_400.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "PlanetConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown planet controls: {', '.join(sorted(unknown))}")
        values = {name: float(mapping[name]) for name in known if name in mapping}
        config = cls(**values)
        for name, value in asdict(config).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        bounds = {
            "planet_radius_earth": (0.7, 1.5),
            "surface_gravity_g": (0.7, 1.5),
            "mean_density_earth": (0.7, 1.5),
            "star_luminosity_solar": (0.1, 3.0),
            "star_effective_temperature_k": (3500.0, 7500.0),
            "semi_major_axis_au": (0.3, 3.0),
            "eccentricity": (0.0, 0.3),
            "obliquity_deg": (0.0, 60.0),
            "rotation_period_hours": (8.0, 72.0),
            "orbital_period_days": (60.0, 1500.0),
            "moon_mass_lunar": (0.0, 81.3),
            "moon_distance_km": (20_000.0, 2_000_000.0),
        }
        values_dict = asdict(config)
        for name, (minimum, maximum) in bounds.items():
            value = values_dict[name]
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if not 0.0 <= config.perihelion_day < config.orbital_period_days:
            raise ValueError("perihelion_day must fall within the orbital year")
        if not 0.0 <= config.northern_vernal_equinox_day < config.orbital_period_days:
            raise ValueError("northern_vernal_equinox_day must fall within the orbital year")
        mean_flux_ratio = config.star_luminosity_solar / config.semi_major_axis_au**2
        if not 0.65 <= mean_flux_ratio <= 1.5:
            raise ValueError("star_luminosity_solar / semi_major_axis_au^2 must be in [0.65, 1.5]")
        return config


def _result_array(result, name: str) -> np.ndarray | None:
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


def _solar_palette(values: np.ndarray, maximum: float) -> np.ndarray:
    stops = np.array([0.0, 0.12, 0.3, 0.5, 0.72, 1.0], dtype=np.float32) * maximum
    colors = np.array(
        [
            (4, 9, 25),
            (31, 50, 103),
            (42, 120, 142),
            (132, 175, 96),
            (232, 173, 68),
            (255, 245, 198),
        ],
        dtype=np.float32,
    )
    channels = [np.interp(values, stops, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)


def _planet_visualizer(result, request: VisualizationRequest) -> list[VisualizationResult] | None:
    monthly = _result_array(result, "MonthlyInsolationWm2")
    annual = _result_array(result, "AnnualMeanInsolationWm2")
    seasonality = _result_array(result, "InsolationSeasonalityWm2")
    if monthly is None or annual is None or seasonality is None or monthly.ndim != 4:
        return None

    annual_path = request.output_dir / "annual_mean_insolation.png"
    Image.fromarray(_cube_net_rgb(_solar_palette(annual, 500.0)), mode="RGB").save(annual_path)
    seasonality_path = request.output_dir / "insolation_seasonality.png"
    Image.fromarray(_cube_net_rgb(_solar_palette(seasonality, 900.0)), mode="RGB").save(
        seasonality_path
    )

    month_nets = [_cube_net_rgb(_solar_palette(monthly[month], 600.0)) for month in range(12)]
    panel_height, panel_width = month_nets[0].shape[:2]
    montage = np.zeros((3 * panel_height, 4 * panel_width, 3), dtype=np.uint8)
    for month, month_net in enumerate(month_nets):
        row, col = divmod(month, 4)
        montage[
            row * panel_height : (row + 1) * panel_height,
            col * panel_width : (col + 1) * panel_width,
        ] = month_net
    monthly_path = request.output_dir / "monthly_insolation.png"
    Image.fromarray(montage, mode="RGB").save(monthly_path)
    return [
        VisualizationResult(annual_path, "AnnualMeanInsolationWm2", {"scale_w_m2": [0, 500]}),
        VisualizationResult(seasonality_path, "InsolationSeasonalityWm2", {"scale_w_m2": [0, 900]}),
        VisualizationResult(
            monthly_path, "MonthlyInsolationWm2", {"months": 12, "scale_w_m2": [0, 600]}
        ),
    ]


@stage(
    "planet",
    inputs=("geometry",),
    outputs=(
        "MonthlyInsolationWm2",
        "MonthlyDaylightHours",
        "AnnualMeanInsolationWm2",
        "InsolationSeasonalityWm2",
        "PolarLightExtremeFraction",
        "OrbitalDistanceAU",
        "SolarDeclinationRad",
        "PlanetMetadata",
    ),
    version="v1",
    native_libraries=("planet_native",),
    visualizer=_planet_visualizer,
)
def planet_stage(context, deps, config_mapping: Mapping[str, object]):
    del deps
    config = PlanetConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical planetary forcing requires topology: cubed_sphere")

    shape = context.topology.face_shape
    artifact_shapes = {
        "MonthlyInsolationWm2": (12, *shape),
        "MonthlyDaylightHours": (12, *shape),
        "AnnualMeanInsolationWm2": shape,
        "InsolationSeasonalityWm2": shape,
        "PolarLightExtremeFraction": shape,
        "OrbitalDistanceAU": (12,),
        "SolarDeclinationRad": (12,),
    }
    handles = {
        name: context.arena.allocate_array(f"planet_{name.lower()}", artifact_shape, np.float32)
        for name, artifact_shape in artifact_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}

    with context.timed("planet_forcing_kernel"):
        metadata = run_cubed_sphere_planet(
            star_luminosity_solar=config.star_luminosity_solar,
            semi_major_axis_au=config.semi_major_axis_au,
            eccentricity=config.eccentricity,
            obliquity_radians=np.deg2rad(config.obliquity_deg),
            rotation_period_hours=config.rotation_period_hours,
            orbital_period_days=config.orbital_period_days,
            perihelion_day=config.perihelion_day,
            northern_vernal_equinox_day=config.northern_vernal_equinox_day,
            moon_mass_lunar=config.moon_mass_lunar,
            moon_distance_km=config.moon_distance_km,
            areas=context.topology.cell_areas,
            latitudes=context.topology.latitude,
            monthly_insolation_out=views["MonthlyInsolationWm2"],
            monthly_daylight_out=views["MonthlyDaylightHours"],
            annual_mean_out=views["AnnualMeanInsolationWm2"],
            seasonality_out=views["InsolationSeasonalityWm2"],
            polar_extreme_fraction_out=views["PolarLightExtremeFraction"],
            orbital_distance_out=views["OrbitalDistanceAU"],
            solar_declination_out=views["SolarDeclinationRad"],
        )

    for handle in handles.values():
        handle.seal()
    metadata.update(
        {
            **asdict(config),
            "topology": "cubed_sphere",
            "forcing_semantics": "twelve_equal_time_monthly_daily_means",
            "solar_constant_w_m2": 1361.0,
            "model": "keplerian_orbital_forcing_v1",
        }
    )
    context.logger.log_event({"type": "planet_summary", "stage": "planet", **metadata})
    return {**handles, "PlanetMetadata": metadata}


__all__ = ["PlanetConfig", "planet_stage"]
