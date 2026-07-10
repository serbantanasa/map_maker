"""Connected geological process provinces on the canonical cubed sphere."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from PIL import Image

from .._geology_native import BOUNDARY_REGIMES, PROVINCE_CLASSES, run_cubed_sphere_geology
from ..cubed_sphere import CubedSphereGrid
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult


def _artifact_array(result, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
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


def _result_array(result, name: str) -> np.ndarray | None:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        return None
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _geology_visualizer(result, request: VisualizationRequest) -> list[VisualizationResult] | None:
    classes = _result_array(result, "GeologicalProvinceClass")
    regimes = _result_array(result, "BoundaryRegime")
    crust_age = _result_array(result, "CrustAgeGa")
    if classes is None or regimes is None or crust_age is None or classes.shape[0] != 6:
        return None

    province_palette = np.array(
        [
            [0, 0, 0],
            [164, 139, 91],
            [187, 174, 126],
            [116, 150, 104],
            [143, 74, 62],
            [176, 86, 96],
            [207, 124, 55],
            [211, 190, 136],
            [42, 88, 132],
            [47, 143, 155],
            [116, 74, 145],
            [194, 66, 48],
        ],
        dtype=np.uint8,
    )
    province_rgb = province_palette[np.clip(classes.astype(np.intp), 0, len(province_palette) - 1)]
    province_path = request.output_dir / "geological_provinces.png"
    Image.fromarray(_cube_net_rgb(province_rgb), mode="RGB").save(province_path)

    regime_palette = np.array(
        [
            [0, 0, 0],
            [90, 90, 90],
            [238, 64, 53],
            [237, 139, 45],
            [172, 80, 167],
            [214, 77, 139],
            [39, 176, 191],
            [241, 209, 70],
        ],
        dtype=np.uint8,
    )
    boundary_confidence = _result_array(result, "BoundaryConfidence")
    if boundary_confidence is None:
        return None
    strongest_edge = np.argmax(boundary_confidence, axis=-1)[..., None]
    cell_regimes = np.take_along_axis(regimes, strongest_edge, axis=-1)[..., 0]
    regime_rgb = regime_palette[np.clip(cell_regimes.astype(np.intp), 0, len(regime_palette) - 1)]
    regime_path = request.output_dir / "boundary_regimes.png"
    Image.fromarray(_cube_net_rgb(regime_rgb), mode="RGB").save(regime_path)

    age_scale = max(float(np.percentile(crust_age, 99)), 1e-6)
    normalized = np.clip(crust_age / age_scale, 0.0, 1.0)
    age_rgb = np.empty((*crust_age.shape, 3), dtype=np.uint8)
    age_rgb[..., 0] = (55 + normalized * 180).astype(np.uint8)
    age_rgb[..., 1] = (125 + normalized * 90).astype(np.uint8)
    age_rgb[..., 2] = (170 - normalized * 120).astype(np.uint8)
    age_path = request.output_dir / "crust_age.png"
    Image.fromarray(_cube_net_rgb(age_rgb), mode="RGB").save(age_path)

    return [
        VisualizationResult(province_path, "GeologicalProvinceClass", {}),
        VisualizationResult(regime_path, "BoundaryRegime", {}),
        VisualizationResult(age_path, "CrustAgeGa", {}),
    ]


@stage(
    "geology",
    inputs=("tectonics", "world_age"),
    outputs=(
        "GeologicalProvinceID",
        "GeologicalProvinceClass",
        "CrustAgeGa",
        "RockStrength",
        "SedimentAccommodation",
        "ProvinceConfidence",
        "BoundarySegmentID",
        "BoundaryRegime",
        "BoundaryConfidence",
        "GeologicalProvinceCatalog",
        "BoundarySegmentCatalog",
        "GeologyMetadata",
    ),
    version="v1",
    native_libraries=("geology_native",),
    visualizer=_geology_visualizer,
)
def geology_stage(context, deps, config_mapping: Mapping[str, object]):
    if config_mapping:
        unknown = ", ".join(sorted(config_mapping))
        raise ValueError(f"geology v1 has no configurable controls; received: {unknown}")
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("canonical geology requires topology: cubed_sphere")

    shape = context.topology.face_shape
    edge_shape = (*shape, 4)
    tectonics = deps["tectonics"]
    world_age = deps["world_age"]
    world_metadata = world_age.artifact_records["WorldAgeMetadata"].value
    world_age_ga = float(world_metadata["world_age"])

    handles = {
        "GeologicalProvinceID": context.arena.allocate_array(
            "geology_province_id", shape, dtype=np.int32
        ),
        "GeologicalProvinceClass": context.arena.allocate_array(
            "geology_province_class", shape, dtype=np.uint8
        ),
        "CrustAgeGa": context.arena.allocate_array("geology_crust_age", shape, dtype=np.float32),
        "RockStrength": context.arena.allocate_array(
            "geology_rock_strength", shape, dtype=np.float32
        ),
        "SedimentAccommodation": context.arena.allocate_array(
            "geology_sediment_accommodation", shape, dtype=np.float32
        ),
        "ProvinceConfidence": context.arena.allocate_array(
            "geology_province_confidence", shape, dtype=np.float32
        ),
        "BoundarySegmentID": context.arena.allocate_array(
            "geology_boundary_segment_id", edge_shape, dtype=np.int32
        ),
        "BoundaryRegime": context.arena.allocate_array(
            "geology_boundary_regime", edge_shape, dtype=np.uint8
        ),
        "BoundaryConfidence": context.arena.allocate_array(
            "geology_boundary_confidence", edge_shape, dtype=np.float32
        ),
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}
    with context.timed("geology_kernel"):
        province_catalog, boundary_catalog, metadata = run_cubed_sphere_geology(
            world_age_ga=world_age_ga,
            areas=context.topology.cell_areas,
            neighbors=context.topology.neighbor_indices,
            plate_field=_artifact_array(tectonics, "PlateField"),
            subduction=_artifact_array(tectonics, "BoundarySubduction"),
            isostasy=_artifact_array(world_age, "IsostaticOffset"),
            uplift=_artifact_array(world_age, "UpliftRate"),
            subsidence=_artifact_array(world_age, "SubsidenceRate"),
            compression=_artifact_array(world_age, "TectonicCompression"),
            extension=_artifact_array(world_age, "TectonicExtension"),
            shear=_artifact_array(world_age, "ShearMagnitude"),
            margin=_artifact_array(world_age, "CoastalExposure"),
            stiffness=_artifact_array(world_age, "LithosphereStiffness"),
            proto_ocean=_artifact_array(world_age, "BaseOceanMask"),
            province_id_out=views["GeologicalProvinceID"],
            province_class_out=views["GeologicalProvinceClass"],
            crust_age_out=views["CrustAgeGa"],
            rock_strength_out=views["RockStrength"],
            accommodation_out=views["SedimentAccommodation"],
            province_confidence_out=views["ProvinceConfidence"],
            boundary_segment_id_out=views["BoundarySegmentID"],
            boundary_regime_out=views["BoundaryRegime"],
            boundary_confidence_out=views["BoundaryConfidence"],
        )
    for handle in handles.values():
        handle.seal()

    metadata.update(
        {
            "world_age_ga": world_age_ga,
            "province_classes": {str(code): name for code, name in PROVINCE_CLASSES.items()},
            "boundary_regimes": {str(code): name for code, name in BOUNDARY_REGIMES.items()},
            "topology": "cubed_sphere",
        }
    )
    context.logger.log_event({"type": "geology_summary", "stage": "geology", **metadata})
    return {
        **handles,
        "GeologicalProvinceCatalog": province_catalog,
        "BoundarySegmentCatalog": boundary_catalog,
        "GeologyMetadata": metadata,
    }


__all__ = ["geology_stage"]
