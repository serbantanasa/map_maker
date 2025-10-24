"""Implementation scaffold for Stage 4 – World Age & Thermal Adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .._world_age_native import run_world_age_kernels
from ..registry import stage


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
            thermal_decay_half_life=float(mapping.get("thermal_decay_half_life", cls.thermal_decay_half_life)),
            hotspot_scale=float(mapping.get("hotspot_scale", cls.hotspot_scale)),
            isostasy_factor=float(mapping.get("isostasy_factor", cls.isostasy_factor)),
            radiogenic_heat_scale=float(mapping.get("radiogenic_heat_scale", cls.radiogenic_heat_scale)),
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


def _log_world_age(context, metadata: Mapping[str, object], *, seed: int, config: WorldAgeConfig) -> None:
    context.logger.log_event(
        {
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
            "water_fraction": metadata.get("water_fraction"),
            "uplift_sigma_gt1": metadata.get("uplift_sigma_gt1"),
            "uplift_sigma_gt2": metadata.get("uplift_sigma_gt2"),
            "subsidence_sigma_gt1": metadata.get("subsidence_sigma_gt1"),
            "subsidence_sigma_gt2": metadata.get("subsidence_sigma_gt2"),
            "hotspot_density": metadata.get("hotspot_density"),
        }
    )


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
)
def world_age_stage(context, deps, config_mapping):
    config = WorldAgeConfig.from_mapping(config_mapping)
    height, width = context.topology.shape

    tectonics_result = deps["tectonics"]
    plate_arr = _artifact_view(tectonics_result, "PlateField")
    convergence_arr = _artifact_view(tectonics_result, "BoundaryConvergence")
    divergence_arr = _artifact_view(tectonics_result, "BoundaryDivergence")
    subduction_arr = _artifact_view(tectonics_result, "BoundarySubduction")
    shear_arr = _artifact_view(tectonics_result, "BoundaryShear")
    hotspot_arr = _artifact_view(tectonics_result, "HotspotMap")

    crust_handle = context.arena.allocate_grid("world_age_crust_thickness", (height, width), dtype=np.float32)
    isostasy_handle = context.arena.allocate_grid("world_age_isostatic_offset", (height, width), dtype=np.float32)
    uplift_handle = context.arena.allocate_grid("world_age_uplift_rate", (height, width), dtype=np.float32)
    subsidence_handle = context.arena.allocate_grid("world_age_subsidence_rate", (height, width), dtype=np.float32)
    compression_handle = context.arena.allocate_grid("world_age_tectonic_compression", (height, width), dtype=np.float32)
    extension_handle = context.arena.allocate_grid("world_age_tectonic_extension", (height, width), dtype=np.float32)
    shear_handle = context.arena.allocate_grid("world_age_shear_magnitude", (height, width), dtype=np.float32)
    coastal_handle = context.arena.allocate_grid("world_age_coastal_exposure", (height, width), dtype=np.float32)
    lithosphere_handle = context.arena.allocate_grid("world_age_lithosphere_stiffness", (height, width), dtype=np.float32)
    ocean_mask_handle = context.arena.allocate_grid("world_age_base_ocean_mask", (height, width), dtype=np.float32)

    crust_view = crust_handle.mutable_view()
    isostasy_view = isostasy_handle.mutable_view()
    uplift_view = uplift_handle.mutable_view()
    subsidence_view = subsidence_handle.mutable_view()
    compression_view = compression_handle.mutable_view()
    extension_view = extension_handle.mutable_view()
    shear_view = shear_handle.mutable_view()
    coastal_view = coastal_handle.mutable_view()
    lithosphere_view = lithosphere_handle.mutable_view()
    ocean_mask_view = ocean_mask_handle.mutable_view()

    rng = context.rng("world_age")
    seed = int(rng.integers(0, 2**63 - 1))

    with context.timed("world_age_kernel"):
        events_table, metadata = run_world_age_kernels(
            height=height,
            width=width,
            seed=seed,
            world_age=config.world_age,
            thermal_decay_half_life=config.thermal_decay_half_life,
            hotspot_scale=config.hotspot_scale,
            isostasy_factor=config.isostasy_factor,
            radiogenic_heat_scale=config.radiogenic_heat_scale,
            plate_field=plate_arr,
            convergence_field=convergence_arr,
            divergence_field=divergence_arr,
            subduction_field=subduction_arr,
            shear_field=shear_arr,
            hotspot_field=hotspot_arr,
            crust_thickness_out=crust_view,
            isostatic_offset_out=isostasy_view,
            uplift_out=uplift_view,
            subsidence_out=subsidence_view,
            compression_out=compression_view,
            extension_out=extension_view,
            shear_out=shear_view,
            coastal_exposure_out=coastal_view,
            lithosphere_stiffness_out=lithosphere_view,
            base_ocean_mask_out=ocean_mask_view,
        )

    crust_handle.seal()
    isostasy_handle.seal()
    uplift_handle.seal()
    subsidence_handle.seal()
    compression_handle.seal()
    extension_handle.seal()
    shear_handle.seal()
    coastal_handle.seal()
    lithosphere_handle.seal()
    ocean_mask_handle.seal()

    metadata = dict(metadata)
    metadata.update(
        {
            "world_age": config.world_age,
            "thermal_decay_half_life": config.thermal_decay_half_life,
            "hotspot_scale": config.hotspot_scale,
            "isostasy_factor": config.isostasy_factor,
            "radiogenic_heat_scale": config.radiogenic_heat_scale,
        }
    )

    _log_world_age(context, metadata, seed=seed, config=config)

    return {
        "CrustThickness": crust_handle,
        "IsostaticOffset": isostasy_handle,
        "UpliftRate": uplift_handle,
        "SubsidenceRate": subsidence_handle,
        "TectonicCompression": compression_handle,
        "TectonicExtension": extension_handle,
        "ShearMagnitude": shear_handle,
        "CoastalExposure": coastal_handle,
        "LithosphereStiffness": lithosphere_handle,
        "BaseOceanMask": ocean_mask_handle,
        "HotspotEvents": events_table,
        "WorldAgeMetadata": metadata,
    }


__all__ = ["WorldAgeConfig", "world_age_stage"]
