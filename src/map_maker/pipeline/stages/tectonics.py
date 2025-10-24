"""Implementation scaffold for Stage 3 – Tectonics & Plate Modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .._tectonics_native import PLATE_FIELD_COMPONENTS, run_tectonics_kernels
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


@dataclass(frozen=True)
class TectonicsConfig:
    num_plates: int = 24
    continental_fraction: float = 0.35
    lloyd_iterations: int = 3
    velocity_scale: float = 1.0
    drift_bias: float = 0.15
    wrap_x: bool = True
    wrap_y: bool = False
    hotspot_density: float = 0.02
    subduction_bias: float = 0.5
    time_steps: int = 32
    time_step: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "TectonicsConfig":
        if not mapping:
            return cls()
        return cls(
            num_plates=int(mapping.get("num_plates", cls.num_plates)),
            continental_fraction=float(mapping.get("continental_fraction", cls.continental_fraction)),
            lloyd_iterations=int(mapping.get("lloyd_iterations", cls.lloyd_iterations)),
            velocity_scale=float(mapping.get("velocity_scale", cls.velocity_scale)),
            drift_bias=float(mapping.get("drift_bias", cls.drift_bias)),
            wrap_x=bool(mapping.get("wrap_x", cls.wrap_x)),
            wrap_y=bool(mapping.get("wrap_y", cls.wrap_y)),
            hotspot_density=float(mapping.get("hotspot_density", cls.hotspot_density)),
            subduction_bias=float(mapping.get("subduction_bias", cls.subduction_bias)),
            time_steps=int(mapping.get("time_steps", cls.time_steps)),
            time_step=float(mapping.get("time_step", cls.time_step)),
        )


def _artifact_array(record) -> Optional[np.ndarray]:
    if not record or record.value is None:
        return None
    if hasattr(record.value, "array"):
        return np.array(record.value.array(), copy=False)
    return np.asarray(record.value)


def _tectonics_visualizer(result, request: VisualizationRequest) -> Optional[list[VisualizationResult]]:
    plate_record = result.artifact_records.get("PlateField")
    if not plate_record or plate_record.value is None:
        return None
    plate_data = _artifact_array(plate_record)
    if plate_data is None or plate_data.ndim != 3 or plate_data.shape[2] < PLATE_FIELD_COMPONENTS:
        return None

    height, width, _ = plate_data.shape
    plate_ids = plate_data[..., 0].astype(np.int32)
    continental_mask = plate_data[..., 1] >= 0.5
    unique_ids = np.unique(plate_ids)
    rng = np.random.default_rng(42)

    classified = np.zeros((height, width, 3), dtype=np.uint8)
    for pid in unique_ids:
        mask = plate_ids == pid
        if not np.any(mask):
            continue
        is_continental = bool(np.mean(continental_mask[mask]) >= 0.5)
        t = rng.random()
        if is_continental:
            base_a = np.array([210, 60, 40], dtype=np.float32)
            base_b = np.array([245, 200, 70], dtype=np.float32)
        else:
            base_a = np.array([40, 90, 185], dtype=np.float32)
            base_b = np.array([20, 145, 120], dtype=np.float32)
        color = (base_a + t * (base_b - base_a)).clip(0.0, 255.0).astype(np.uint8)
        classified[mask] = color

    from PIL import Image

    results: list[VisualizationResult] = []
    plates_path = request.output_dir / "plates.png"
    plates_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(classified, mode="RGB").save(plates_path)
    results.append(VisualizationResult(path=plates_path, artifact_name="PlateField", metadata={}))

    convergence = _artifact_array(result.artifact_records.get("BoundaryConvergence"))
    divergence = _artifact_array(result.artifact_records.get("BoundaryDivergence"))
    subduction = _artifact_array(result.artifact_records.get("BoundarySubduction"))

    if convergence is not None and divergence is not None and subduction is not None:
        conv_norm = (convergence.clip(min=0.0) / (float(convergence.max()) + 1e-6)).astype(np.float32)
        div_norm = (divergence.clip(min=0.0) / (float(divergence.max()) + 1e-6)).astype(np.float32)
        sub_norm = (subduction.clip(min=0.0) / (float(subduction.max()) + 1e-6)).astype(np.float32)
        boundary = np.zeros((height, width, 3), dtype=np.float32)
        boundary[..., 0] = sub_norm * 255.0  # subduction → red
        boundary[..., 2] = div_norm * 255.0  # abduction/divergence → blue
        shear = _artifact_array(result.artifact_records.get("BoundaryShear"))
        if shear is not None and shear.max() > 0:
            shear_norm = (shear / float(shear.max())).astype(np.float32)
            boundary += (shear_norm * 60.0)[..., None]
        boundary_img = boundary.clip(0.0, 255.0).astype(np.uint8)
        boundary_path = request.output_dir / "boundaries.png"
        Image.fromarray(boundary_img, mode="RGB").save(boundary_path)
        results.append(VisualizationResult(path=boundary_path, artifact_name="BoundaryVisualization", metadata={}))

    hotspot = _artifact_array(result.artifact_records.get("HotspotMap"))
    if hotspot is not None and hotspot.max() > 0:
        base = np.empty((height, width, 3), dtype=np.float32)
        base[continental_mask] = np.array([232, 210, 170], dtype=np.float32)
        base[~continental_mask] = np.array([175, 210, 235], dtype=np.float32)
        hot_norm = (hotspot / float(hotspot.max()))[..., None].astype(np.float32)
        weight = np.clip(hot_norm * 0.75, 0.0, 0.75)
        overlay = np.array([255.0, 120.0, 0.0], dtype=np.float32)
        hot_img = base * (1.0 - weight) + overlay * weight
        hot_img = hot_img.clip(0.0, 255.0).astype(np.uint8)
        hotspot_path = request.output_dir / "hotspots.png"
        Image.fromarray(hot_img, mode="RGB").save(hotspot_path)
        results.append(VisualizationResult(path=hotspot_path, artifact_name="HotspotMap", metadata={}))

    return results


@stage(
    "tectonics",
    outputs=(
        "PlateField",
        "BoundaryConvergence",
        "BoundaryDivergence",
        "BoundaryShear",
        "BoundarySubduction",
        "HotspotMap",
        "TectonicsMetadata",
    ),
    visualizer=_tectonics_visualizer,
)
def tectonics_stage(context, deps, config_mapping):
    config = TectonicsConfig.from_mapping(config_mapping)
    height, width = context.topology.shape

    rng = context.rng("tectonics")
    seed = int(rng.integers(0, 2**63 - 1))

    plate_handle = context.arena.allocate_array(
        "tectonics_plate_field",
        (height, width, PLATE_FIELD_COMPONENTS),
        dtype=np.float32,
    )
    convergence_handle = context.arena.allocate_grid("tectonics_convergence", (height, width), dtype=np.float32)
    divergence_handle = context.arena.allocate_grid("tectonics_divergence", (height, width), dtype=np.float32)
    shear_handle = context.arena.allocate_grid("tectonics_shear", (height, width), dtype=np.float32)
    subduction_handle = context.arena.allocate_grid("tectonics_subduction", (height, width), dtype=np.float32)
    hotspot_handle = context.arena.allocate_grid("tectonics_hotspot", (height, width), dtype=np.float32)

    plate_view = plate_handle.mutable_view()
    convergence_view = convergence_handle.mutable_view()
    divergence_view = divergence_handle.mutable_view()
    shear_view = shear_handle.mutable_view()
    subduction_view = subduction_handle.mutable_view()
    hotspot_view = hotspot_handle.mutable_view()

    with context.timed("tectonics_kernel"):
        metadata = run_tectonics_kernels(
            height=height,
            width=width,
            seed=seed,
            num_plates=config.num_plates,
            continental_fraction=config.continental_fraction,
            velocity_scale=config.velocity_scale,
            drift_bias=config.drift_bias,
            hotspot_density=config.hotspot_density,
            subduction_bias=config.subduction_bias,
            lloyd_iterations=config.lloyd_iterations,
            time_steps=config.time_steps,
            time_step=config.time_step,
            wrap_x=config.wrap_x,
            wrap_y=config.wrap_y,
            plate_field=plate_view,
            convergence_field=convergence_view,
            divergence_field=divergence_view,
            shear_field=shear_view,
            subduction_field=subduction_view,
            hotspot_field=hotspot_view,
        )

    continental_fraction_actual = float(np.mean(plate_view[..., 1]))

    plate_handle.seal()
    convergence_handle.seal()
    divergence_handle.seal()
    shear_handle.seal()
    subduction_handle.seal()
    hotspot_handle.seal()

    metadata = dict(metadata)
    metadata.update(
        {
            "num_plates": config.num_plates,
            "continental_fraction_target": config.continental_fraction,
            "lloyd_iterations": config.lloyd_iterations,
            "time_steps": config.time_steps,
            "time_step": config.time_step,
            "continental_fraction_actual": continental_fraction_actual,
            "hotspot_density": config.hotspot_density,
            "subduction_bias": config.subduction_bias,
        }
    )

    context.logger.log_event(
        {
            "type": "tectonics_summary",
            "stage": "tectonics",
            "seed": seed,
            "num_plates": config.num_plates,
            "velocity_scale": config.velocity_scale,
            "hotspot_mean": metadata.get("hotspot_mean"),
            "continental_fraction": metadata.get("continental_fraction"),
        }
    )

    return {
        "PlateField": plate_handle,
        "BoundaryConvergence": convergence_handle,
        "BoundaryDivergence": divergence_handle,
        "BoundaryShear": shear_handle,
        "BoundarySubduction": subduction_handle,
        "HotspotMap": hotspot_handle,
        "TectonicsMetadata": metadata,
    }


__all__ = [
    "TectonicsConfig",
    "tectonics_stage",
]
