"""Stage 5 – Erosion & Sedimentation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pyarrow as pa

from .._erosion_native import run_erosion_kernels
from ..registry import stage


@dataclass(frozen=True)
class ErosionConfig:
    steps: int = 12
    dt: float = 0.8
    stream_power_k: float = 0.015
    sediment_capacity: float = 0.02
    coastal_wave_energy: float = 0.5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ErosionConfig":
        if not mapping:
            return cls()
        return cls(
            steps=int(mapping.get("erosion_steps", mapping.get("steps", cls.steps))),
            dt=float(mapping.get("dt", cls.dt)),
            stream_power_k=float(mapping.get("stream_power_k", cls.stream_power_k)),
            sediment_capacity=float(mapping.get("sediment_capacity", cls.sediment_capacity)),
            coastal_wave_energy=float(mapping.get("coastal_wave_energy", cls.coastal_wave_energy)),
        )


def _artifact_array(record, *, name: str) -> np.ndarray:
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


def _ensure_hotspot_table(table: pa.Table | None) -> pa.Table:
    if isinstance(table, pa.Table):
        return table
    if table is None:
        return pa.table(
            {
                "row": pa.array([], type=pa.int32()),
                "col": pa.array([], type=pa.int32()),
                "strength": pa.array([], type=pa.float32()),
                "plume_factor": pa.array([], type=pa.float32()),
            }
        )
    raise TypeError(f"Expected pyarrow.Table for hotspots, got {type(table)!r}")


def _hotspot_grid(table: pa.Table, height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.float32)
    if table.num_rows == 0:
        return grid
    names = set(table.column_names)
    required = {"row", "col", "strength", "plume_factor"}
    if not required.issubset(names):
        return grid

    rows = table.column("row").to_numpy(zero_copy_only=False)
    cols = table.column("col").to_numpy(zero_copy_only=False)
    strengths = table.column("strength").to_numpy(zero_copy_only=False)
    plume = table.column("plume_factor").to_numpy(zero_copy_only=False)

    for r, c, strength, plume_factor in zip(rows, cols, strengths, plume, strict=False):
        rr = int(np.clip(r, 0, height - 1))
        cc = int(np.clip(c, 0, width - 1))
        sigma = 1.2 + float(plume_factor) * 1.6
        radius = int(max(2, round(sigma * 3.0)))
        r0 = max(0, rr - radius)
        r1 = min(height - 1, rr + radius)
        c0 = max(0, cc - radius)
        c1 = min(width - 1, cc + radius)
        if r1 < r0 or c1 < c0:
            continue
        yy = np.arange(r0, r1 + 1, dtype=np.float32)[:, None]
        xx = np.arange(c0, c1 + 1, dtype=np.float32)[None, :]
        dy = yy - rr
        dx = xx - cc
        exponent = -((dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32, copy=False)
        kernel = np.exp(exponent).astype(np.float32, copy=False)
        kernel *= float(strength) * (0.6 + 0.8 * float(plume_factor))
        grid[r0 : r1 + 1, c0 : c1 + 1] += kernel

    max_val = float(grid.max())
    if max_val > 1e-6:
        grid /= max_val
    return grid


def _log_erosion(context, metadata: Mapping[str, object]) -> None:
    context.logger.log_event(
        {
            "type": "erosion_summary",
            "stage": "erosion",
            "steps": metadata.get("steps"),
            "dt": metadata.get("dt"),
            "stream_power_k": metadata.get("stream_power_k"),
            "sediment_capacity": metadata.get("sediment_capacity"),
            "coastal_wave_energy": metadata.get("coastal_wave_energy"),
            "total_mass_removed": metadata.get("total_mass_removed"),
            "total_mass_deposited": metadata.get("total_mass_deposited"),
            "mass_residual": metadata.get("mass_residual"),
            "final_mean_elevation": metadata.get("final_mean_elevation"),
        }
    )


@stage(
    "erosion",
    inputs=("tectonics", "world_age"),
    outputs=(
        "ElevationRaw",
        "SedimentDepth",
        "RiverIncision",
        "ErosionDiagnostics",
        "ErosionMetadata",
    ),
    version="v3",
)
def erosion_stage(context, deps, config_mapping):
    config = ErosionConfig.from_mapping(config_mapping)
    height, width = context.topology.shape

    tectonics = deps["tectonics"]
    world_age = deps["world_age"]

    plate_field = _artifact_array(tectonics.artifact_records.get("PlateField"), name="PlateField")

    crust = _artifact_array(world_age.artifact_records.get("CrustThickness"), name="CrustThickness")
    isostasy = _artifact_array(world_age.artifact_records.get("IsostaticOffset"), name="IsostaticOffset")
    uplift = _artifact_array(world_age.artifact_records.get("UpliftRate"), name="UpliftRate")
    subsidence = _artifact_array(world_age.artifact_records.get("SubsidenceRate"), name="SubsidenceRate")
    compression = _artifact_array(world_age.artifact_records.get("TectonicCompression"), name="TectonicCompression")
    extension = _artifact_array(world_age.artifact_records.get("TectonicExtension"), name="TectonicExtension")
    shear = _artifact_array(world_age.artifact_records.get("ShearMagnitude"), name="ShearMagnitude")
    coastal_exposure = _artifact_array(world_age.artifact_records.get("CoastalExposure"), name="CoastalExposure")
    lithosphere = _artifact_array(world_age.artifact_records.get("LithosphereStiffness"), name="LithosphereStiffness")
    base_ocean = _artifact_array(world_age.artifact_records.get("BaseOceanMask"), name="BaseOceanMask")

    hotspot_record = world_age.artifact_records.get("HotspotEvents")
    hotspot_table = _ensure_hotspot_table(hotspot_record.value if hotspot_record else None)
    hotspot_grid = _hotspot_grid(hotspot_table, height, width)

    elevation_handle = context.arena.allocate_grid("erosion_elevation_raw", (height, width), dtype=np.float32)
    sediment_handle = context.arena.allocate_grid("erosion_sediment_depth", (height, width), dtype=np.float32)
    incision_handle = context.arena.allocate_grid("erosion_river_incision", (height, width), dtype=np.float32)

    elevation_view = elevation_handle.mutable_view()
    sediment_view = sediment_handle.mutable_view()
    incision_view = incision_handle.mutable_view()

    rng = context.rng("erosion")
    seed = int(rng.integers(0, 2**63 - 1))

    with context.timed("erosion_kernel"):
        diagnostics_table, stats = run_erosion_kernels(
            height=height,
            width=width,
            steps=config.steps,
            dt=config.dt,
            stream_power_k=config.stream_power_k,
            sediment_capacity=config.sediment_capacity,
            coastal_wave_energy=config.coastal_wave_energy,
            plate_field=plate_field,
            crust_thickness=crust,
            isostatic_offset=isostasy,
            uplift_rate=uplift,
            subsidence_rate=subsidence,
            compression=compression,
            extension=extension,
            shear=shear,
            coastal_exposure=coastal_exposure,
            lithosphere_stiffness=lithosphere,
            base_ocean_mask=base_ocean,
            hotspot_influence=hotspot_grid,
            elevation_out=elevation_view,
            sediment_out=sediment_view,
            incision_out=incision_view,
        )

    elevation_handle.seal()
    sediment_handle.seal()
    incision_handle.seal()

    metadata = {
        **stats,
        "steps": config.steps,
        "dt": config.dt,
        "stream_power_k": config.stream_power_k,
        "sediment_capacity": config.sediment_capacity,
        "coastal_wave_energy": config.coastal_wave_energy,
        "hotspot_count": int(hotspot_table.num_rows),
        "rng_seed": seed,
    }

    _log_erosion(context, metadata)

    return {
        "ElevationRaw": elevation_handle,
        "SedimentDepth": sediment_handle,
        "RiverIncision": incision_handle,
        "ErosionDiagnostics": diagnostics_table,
        "ErosionMetadata": metadata,
    }


__all__ = ["ErosionConfig", "erosion_stage"]
