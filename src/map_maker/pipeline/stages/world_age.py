"""Age-conditioned crust-state initialization for rectangular and spherical worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image

from .._world_age_native import run_cubed_sphere_world_age, run_world_age_kernels
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


@dataclass(frozen=True)
class WorldAgeConfig:
    world_age: float = 4.1
    thermal_decay_half_life: float = 1.8
    hotspot_scale: float = 0.9
    isostasy_factor: float = 0.6
    radiogenic_heat_scale: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "WorldAgeConfig":
        if not mapping:
            return cls()
        return cls(
            world_age=float(mapping.get("world_age", cls.world_age)),
            thermal_decay_half_life=float(
                mapping.get("thermal_decay_half_life", cls.thermal_decay_half_life)
            ),
            hotspot_scale=float(mapping.get("hotspot_scale", cls.hotspot_scale)),
            isostasy_factor=float(mapping.get("isostasy_factor", cls.isostasy_factor)),
            radiogenic_heat_scale=float(
                mapping.get("radiogenic_heat_scale", cls.radiogenic_heat_scale)
            ),
        )


def _artifact_view(result, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    if hasattr(value, "array"):
        array = np.array(value.array(), copy=False)
    else:
        array = np.array(value, copy=False)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    return array


def _log_world_age(
    context, metadata: Mapping[str, object], *, seed: int, config: WorldAgeConfig
) -> None:
    event = {
        "type": "world_age_summary",
        "stage": "world_age",
        "seed": seed,
        "world_age": config.world_age,
        "thermal_decay_half_life": config.thermal_decay_half_life,
        "radiogenic_heat_scale": config.radiogenic_heat_scale,
        "convective_vigor": metadata.get("convective_vigor"),
        "hotspot_count": metadata.get("hotspot_count"),
        "uplift_mean": metadata.get("uplift_mean"),
        "subsidence_mean": metadata.get("subsidence_mean"),
        "uplift_sigma_gt1": metadata.get("uplift_sigma_gt1"),
        "uplift_sigma_gt2": metadata.get("uplift_sigma_gt2"),
        "subsidence_sigma_gt1": metadata.get("subsidence_sigma_gt1"),
        "subsidence_sigma_gt2": metadata.get("subsidence_sigma_gt2"),
        "hotspot_density": metadata.get("hotspot_density"),
    }
    if "proto_ocean_area_fraction" in metadata:
        event["proto_ocean_area_fraction"] = metadata["proto_ocean_area_fraction"]
    else:
        event["water_fraction"] = metadata.get("water_fraction")
    context.logger.log_event(event)


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


def _scaled(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, (2.0, 98.0))
    return np.clip((values - low) / max(float(high - low), 1e-9), 0.0, 1.0).astype(np.float32)


def _world_age_visualizer(
    result, request: VisualizationRequest
) -> list[VisualizationResult] | None:
    isostasy = _result_array(result, "IsostaticOffset")
    uplift = _result_array(result, "UpliftRate")
    subsidence = _result_array(result, "SubsidenceRate")
    shear = _result_array(result, "ShearMagnitude")
    proto_ocean = _result_array(result, "BaseOceanMask")
    if (
        isostasy is None
        or uplift is None
        or subsidence is None
        or shear is None
        or proto_ocean is None
        or isostasy.ndim != 3
        or isostasy.shape[0] != 6
    ):
        return None

    results = []
    scale = max(float(np.percentile(np.abs(isostasy), 98)), 1e-6)
    signed = np.clip(isostasy / scale, -1.0, 1.0)
    isostasy_rgb = np.empty((*isostasy.shape, 3), dtype=np.uint8)
    isostasy_rgb[..., 0] = np.where(signed > 0, 110 + signed * 145, 40).astype(np.uint8)
    isostasy_rgb[..., 1] = (95 + (1.0 - np.abs(signed)) * 95).astype(np.uint8)
    isostasy_rgb[..., 2] = np.where(signed < 0, 120 - signed * 135, 45).astype(np.uint8)
    isostasy_path = request.output_dir / "isostatic_potential.png"
    Image.fromarray(_cube_net_rgb(isostasy_rgb), mode="RGB").save(isostasy_path)
    results.append(
        VisualizationResult(path=isostasy_path, artifact_name="IsostaticOffset", metadata={})
    )

    rates = np.zeros((*uplift.shape, 3), dtype=np.uint8)
    rates[..., 0] = (_scaled(uplift) * 255).astype(np.uint8)
    rates[..., 1] = (_scaled(shear) * 190).astype(np.uint8)
    rates[..., 2] = (_scaled(subsidence) * 255).astype(np.uint8)
    rates_path = request.output_dir / "tectonic_rates.png"
    Image.fromarray(_cube_net_rgb(rates), mode="RGB").save(rates_path)
    results.append(VisualizationResult(path=rates_path, artifact_name="TectonicRates", metadata={}))

    crust = np.zeros((*proto_ocean.shape, 3), dtype=np.uint8)
    ocean = proto_ocean >= 0.5
    crust[ocean] = np.array([38, 102, 153], dtype=np.uint8)
    crust[~ocean] = np.array([177, 151, 99], dtype=np.uint8)
    crust_path = request.output_dir / "proto_crust.png"
    Image.fromarray(_cube_net_rgb(crust), mode="RGB").save(crust_path)
    results.append(VisualizationResult(path=crust_path, artifact_name="BaseOceanMask", metadata={}))
    return results


@stage(
    "world_age",
    inputs=("tectonics",),
    outputs=(
        "CrustThickness",
        "IsostaticOffset",
        "UpliftRate",
        "SubsidenceRate",
        "TectonicCompression",
        "TectonicExtension",
        "ShearMagnitude",
        "CoastalExposure",
        "LithosphereStiffness",
        "BaseOceanMask",
        "HotspotEvents",
        "WorldAgeMetadata",
    ),
    version="v2",
    visualizer=_world_age_visualizer,
)
def world_age_stage(context, deps, config_mapping):
    config = WorldAgeConfig.from_mapping(config_mapping)
    is_cubed_sphere = isinstance(context.topology, CubedSphereGrid)
    shape = context.topology.face_shape if is_cubed_sphere else context.topology.shape

    tectonics_result = deps["tectonics"]
    plate_arr = _artifact_view(tectonics_result, "PlateField")
    convergence_arr = _artifact_view(tectonics_result, "BoundaryConvergence")
    divergence_arr = _artifact_view(tectonics_result, "BoundaryDivergence")
    subduction_arr = _artifact_view(tectonics_result, "BoundarySubduction")
    shear_arr = _artifact_view(tectonics_result, "BoundaryShear")
    hotspot_arr = _artifact_view(tectonics_result, "HotspotMap")

    def allocate(name: str):
        if is_cubed_sphere:
            return context.arena.allocate_array(name, shape, dtype=np.float32)
        return context.arena.allocate_grid(name, shape, dtype=np.float32)

    handles = {
        "CrustThickness": allocate("world_age_crust_thickness"),
        "IsostaticOffset": allocate("world_age_isostatic_offset"),
        "UpliftRate": allocate("world_age_uplift_rate"),
        "SubsidenceRate": allocate("world_age_subsidence_rate"),
        "TectonicCompression": allocate("world_age_tectonic_compression"),
        "TectonicExtension": allocate("world_age_tectonic_extension"),
        "ShearMagnitude": allocate("world_age_shear_magnitude"),
        "CoastalExposure": allocate("world_age_coastal_exposure"),
        "LithosphereStiffness": allocate("world_age_lithosphere_stiffness"),
        "BaseOceanMask": allocate("world_age_base_ocean_mask"),
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}

    rng = context.rng("world_age")
    seed = int(rng.integers(0, 2**63 - 1))

    with context.timed("world_age_kernel"):
        common = {
            "seed": seed,
            "world_age": config.world_age,
            "thermal_decay_half_life": config.thermal_decay_half_life,
            "hotspot_scale": config.hotspot_scale,
            "isostasy_factor": config.isostasy_factor,
            "radiogenic_heat_scale": config.radiogenic_heat_scale,
            "plate_field": plate_arr,
            "convergence_field": convergence_arr,
            "divergence_field": divergence_arr,
            "subduction_field": subduction_arr,
            "shear_field": shear_arr,
            "hotspot_field": hotspot_arr,
            "crust_thickness_out": views["CrustThickness"],
            "isostatic_offset_out": views["IsostaticOffset"],
            "uplift_out": views["UpliftRate"],
            "subsidence_out": views["SubsidenceRate"],
            "compression_out": views["TectonicCompression"],
            "extension_out": views["TectonicExtension"],
            "shear_out": views["ShearMagnitude"],
            "lithosphere_stiffness_out": views["LithosphereStiffness"],
        }
        if is_cubed_sphere:
            events_table, metadata = run_cubed_sphere_world_age(
                **common,
                areas=context.topology.cell_areas,
                neighbors=context.topology.neighbor_indices,
                margin_proximity_out=views["CoastalExposure"],
                proto_ocean_mask_out=views["BaseOceanMask"],
            )
        else:
            events_table, metadata = run_world_age_kernels(
                **common,
                height=shape[0],
                width=shape[1],
                coastal_exposure_out=views["CoastalExposure"],
                base_ocean_mask_out=views["BaseOceanMask"],
            )

    for handle in handles.values():
        handle.seal()

    metadata = dict(metadata)
    metadata.update(
        {
            "world_age": config.world_age,
            "thermal_decay_half_life": config.thermal_decay_half_life,
            "hotspot_scale": config.hotspot_scale,
            "isostasy_factor": config.isostasy_factor,
            "radiogenic_heat_scale": config.radiogenic_heat_scale,
            "crust_state_model": (
                "cubed_sphere_area_weighted_v1" if is_cubed_sphere else "provisional_rectangular_v1"
            ),
        }
    )

    _log_world_age(context, metadata, seed=seed, config=config)

    return {
        **handles,
        "HotspotEvents": events_table,
        "WorldAgeMetadata": metadata,
    }


__all__ = ["WorldAgeConfig", "world_age_stage"]
