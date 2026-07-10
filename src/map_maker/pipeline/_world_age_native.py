"""Rust-backed native bindings for the world age stage."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pyarrow as pa
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_WORLD_AGE_ABI_VERSION = 1

HOTSPOT_EVENT_DTYPE = np.dtype(
    [
        ("row", "<i4"),
        ("col", "<i4"),
        ("strength", "<f4"),
        ("plume_factor", "<f4"),
    ]
)

SPHERICAL_HOTSPOT_EVENT_DTYPE = np.dtype(
    [
        ("global_cell_id", "<i4"),
        ("strength", "<f4"),
        ("plume_factor", "<f4"),
    ]
)

_CDEF = """
typedef struct {
    int32_t row;
    int32_t col;
    float strength;
    float plume_factor;
} HotspotEvent;

typedef struct {
    HotspotEvent* data;
    size_t len;
} HotspotEventArray;

typedef struct {
    int32_t global_cell_id;
    float strength;
    float plume_factor;
} SphericalHotspotEvent;

typedef struct {
    SphericalHotspotEvent* data;
    size_t len;
} SphericalHotspotEventArray;

typedef struct {
    float convective_vigor;
    float mean_crust_thickness;
    float std_crust_thickness;
    float mean_isostatic_offset;
    int32_t hotspot_count;
    float uplift_mean;
    float subsidence_mean;
    float thermal_decay_factor;
    float water_fraction;
    float uplift_sigma_gt1;
    float uplift_sigma_gt2;
    float uplift_sigma_gt3;
    float subsidence_sigma_gt1;
    float subsidence_sigma_gt2;
    float subsidence_sigma_gt3;
    float hotspot_density;
} WorldAgeStats;

int32_t world_age_run(
    int32_t height,
    int32_t width,
    uint64_t seed,
    float world_age,
    float thermal_decay_half_life,
    float hotspot_scale,
    float isostasy_factor,
    float radiogenic_heat_scale,
    int32_t plate_components,
    const float* plate_field,
    const float* convergence_field,
    const float* divergence_field,
    const float* subduction_field,
    const float* shear_field,
    const float* hotspot_field,
    float* crust_thickness_out,
    float* isostasy_out,
    float* uplift_out,
    float* subsidence_out,
    float* compression_out,
    float* extension_out,
    float* shear_out,
    float* coastal_exposure_out,
    float* lithosphere_stiffness_out,
    float* base_ocean_mask_out,
    HotspotEventArray* events_out,
    WorldAgeStats* stats_out
);

void world_age_free_events(HotspotEventArray array);
uint32_t cubed_sphere_world_age_abi_version(void);
int32_t world_age_run_cubed_sphere(
    int32_t cell_count,
    uint64_t seed,
    float world_age,
    float thermal_decay_half_life,
    float hotspot_scale,
    float isostasy_factor,
    float radiogenic_heat_scale,
    int32_t plate_components,
    const double* area,
    const int32_t* neighbors,
    const float* plate_field,
    const float* convergence_field,
    const float* divergence_field,
    const float* subduction_field,
    const float* shear_field,
    const float* hotspot_field,
    float* crust_thickness_out,
    float* isostasy_out,
    float* uplift_out,
    float* subsidence_out,
    float* compression_out,
    float* extension_out,
    float* shear_out,
    float* margin_proximity_out,
    float* lithosphere_stiffness_out,
    float* proto_ocean_mask_out,
    SphericalHotspotEventArray* events_out,
    WorldAgeStats* stats_out
);
void world_age_free_spherical_events(SphericalHotspotEventArray array);
"""


def _as_read_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.array(array, copy=False)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not arr.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    return arr


def _as_write_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.array(array, copy=False)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be contiguous")
    if not arr.flags["ALIGNED"]:
        raise ValueError(f"{name} must be aligned")
    if not arr.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return arr


def _require_disjoint(buffers: dict[str, np.ndarray]) -> None:
    items = list(buffers.items())
    for index, (first_name, first) in enumerate(items):
        for second_name, second in items[index + 1 :]:
            if np.shares_memory(first, second):
                raise ValueError(f"{first_name} and {second_name} buffers must not overlap")


_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("world_age_native")
_lib = _ffi.dlopen(str(native_library_path("world_age_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_world_age_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "world_age_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_WORLD_AGE_ABI_VERSION:
    raise NativeLibraryAbiError(
        "world_age_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_WORLD_AGE_ABI_VERSION}"
    )


def _events_to_arrow(event_struct: Any, *, free_after: bool = True) -> pa.Table:
    length = int(event_struct.len)
    if length <= 0 or int(_ffi.cast("uintptr_t", event_struct.data)) == 0:
        if free_after:
            _lib.world_age_free_events(event_struct)
        return pa.table(
            {
                "row": pa.array([], type=pa.int32()),
                "col": pa.array([], type=pa.int32()),
                "strength": pa.array([], type=pa.float32()),
                "plume_factor": pa.array([], type=pa.float32()),
            }
        )

    buffer = _ffi.buffer(event_struct.data, length * HOTSPOT_EVENT_DTYPE.itemsize)
    events_np = np.frombuffer(buffer, dtype=HOTSPOT_EVENT_DTYPE, count=length).copy()
    if free_after:
        _lib.world_age_free_events(event_struct)
    return pa.table(
        {
            "row": pa.array(events_np["row"], type=pa.int32()),
            "col": pa.array(events_np["col"], type=pa.int32()),
            "strength": pa.array(events_np["strength"], type=pa.float32()),
            "plume_factor": pa.array(events_np["plume_factor"], type=pa.float32()),
        }
    )


def _spherical_events_to_arrow(event_struct: Any, *, face_resolution: int) -> pa.Table:
    length = int(event_struct.len)
    if length <= 0 or int(_ffi.cast("uintptr_t", event_struct.data)) == 0:
        _lib.world_age_free_spherical_events(event_struct)
        return pa.table(
            {
                "global_cell_id": pa.array([], type=pa.int32()),
                "face": pa.array([], type=pa.int8()),
                "row": pa.array([], type=pa.int32()),
                "col": pa.array([], type=pa.int32()),
                "strength": pa.array([], type=pa.float32()),
                "plume_factor": pa.array([], type=pa.float32()),
            }
        )
    buffer = _ffi.buffer(event_struct.data, length * SPHERICAL_HOTSPOT_EVENT_DTYPE.itemsize)
    events = np.frombuffer(buffer, dtype=SPHERICAL_HOTSPOT_EVENT_DTYPE, count=length).copy()
    _lib.world_age_free_spherical_events(event_struct)
    global_ids = events["global_cell_id"]
    face_size = face_resolution * face_resolution
    faces = (global_ids // face_size).astype(np.int8)
    within_face = global_ids % face_size
    rows = (within_face // face_resolution).astype(np.int32)
    cols = (within_face % face_resolution).astype(np.int32)
    return pa.table(
        {
            "global_cell_id": pa.array(global_ids, type=pa.int32()),
            "face": pa.array(faces, type=pa.int8()),
            "row": pa.array(rows, type=pa.int32()),
            "col": pa.array(cols, type=pa.int32()),
            "strength": pa.array(events["strength"], type=pa.float32()),
            "plume_factor": pa.array(events["plume_factor"], type=pa.float32()),
        }
    )


def _stats_metadata(stats: Any) -> Dict[str, Any]:
    return {
        "convective_vigor": float(stats.convective_vigor),
        "mean_crust_thickness": float(stats.mean_crust_thickness),
        "std_crust_thickness": float(stats.std_crust_thickness),
        "mean_isostatic_offset": float(stats.mean_isostatic_offset),
        "hotspot_count": int(stats.hotspot_count),
        "uplift_mean": float(stats.uplift_mean),
        "subsidence_mean": float(stats.subsidence_mean),
        "thermal_decay_factor": float(stats.thermal_decay_factor),
        "water_fraction": float(stats.water_fraction),
        "uplift_sigma_gt1": float(stats.uplift_sigma_gt1),
        "uplift_sigma_gt2": float(stats.uplift_sigma_gt2),
        "uplift_sigma_gt3": float(stats.uplift_sigma_gt3),
        "subsidence_sigma_gt1": float(stats.subsidence_sigma_gt1),
        "subsidence_sigma_gt2": float(stats.subsidence_sigma_gt2),
        "subsidence_sigma_gt3": float(stats.subsidence_sigma_gt3),
        "hotspot_density": float(stats.hotspot_density),
    }


def run_world_age_kernels(
    *,
    height: int,
    width: int,
    seed: int,
    world_age: float,
    thermal_decay_half_life: float,
    hotspot_scale: float,
    isostasy_factor: float,
    radiogenic_heat_scale: float,
    plate_field: np.ndarray,
    convergence_field: np.ndarray,
    divergence_field: np.ndarray,
    subduction_field: np.ndarray,
    shear_field: np.ndarray,
    hotspot_field: np.ndarray,
    crust_thickness_out: np.ndarray,
    isostatic_offset_out: np.ndarray,
    uplift_out: np.ndarray,
    subsidence_out: np.ndarray,
    compression_out: np.ndarray,
    extension_out: np.ndarray,
    shear_out: np.ndarray,
    coastal_exposure_out: np.ndarray,
    lithosphere_stiffness_out: np.ndarray,
    base_ocean_mask_out: np.ndarray,
) -> Tuple[pa.Table, Dict[str, Any]]:
    plate_arr = _as_read_array(plate_field, name="plate_field")
    if plate_arr.ndim != 3:
        raise ValueError(f"plate_field must be 3D (H, W, C); got shape {plate_arr.shape}")
    expected_grid = (int(height), int(width))
    if tuple(int(dim) for dim in plate_arr.shape[:2]) != expected_grid:
        raise ValueError(
            f"plate_field expected leading shape {expected_grid}, got {plate_arr.shape}"
        )
    convergence_arr = _as_read_array(convergence_field, name="convergence_field")
    divergence_arr = _as_read_array(divergence_field, name="divergence_field")
    subduction_arr = _as_read_array(subduction_field, name="subduction_field")
    shear_arr = _as_read_array(shear_field, name="shear_field")
    hotspot_arr = _as_read_array(hotspot_field, name="hotspot_field")
    for arr, name in (
        (convergence_arr, "convergence_field"),
        (divergence_arr, "divergence_field"),
        (subduction_arr, "subduction_field"),
        (shear_arr, "shear_field"),
        (hotspot_arr, "hotspot_field"),
    ):
        if arr.shape != expected_grid:
            raise ValueError(f"{name} expected shape {expected_grid}, got {arr.shape}")

    crust_arr = _as_write_array(crust_thickness_out, name="crust_thickness_out")
    isostasy_arr = _as_write_array(isostatic_offset_out, name="isostatic_offset_out")
    uplift_arr = _as_write_array(uplift_out, name="uplift_out")
    subsidence_arr = _as_write_array(subsidence_out, name="subsidence_out")
    compression_arr = _as_write_array(compression_out, name="compression_out")
    extension_arr = _as_write_array(extension_out, name="extension_out")
    shear_out_arr = _as_write_array(shear_out, name="shear_out")
    coastal_exposure_arr = _as_write_array(coastal_exposure_out, name="coastal_exposure_out")
    lithosphere_arr = _as_write_array(lithosphere_stiffness_out, name="lithosphere_stiffness_out")
    base_ocean_mask_arr = _as_write_array(base_ocean_mask_out, name="base_ocean_mask_out")

    for arr, name in (
        (crust_arr, "crust_thickness_out"),
        (isostasy_arr, "isostatic_offset_out"),
        (uplift_arr, "uplift_out"),
        (subsidence_arr, "subsidence_out"),
        (compression_arr, "compression_out"),
        (extension_arr, "extension_out"),
        (shear_out_arr, "shear_out"),
        (coastal_exposure_arr, "coastal_exposure_out"),
        (lithosphere_arr, "lithosphere_stiffness_out"),
        (base_ocean_mask_arr, "base_ocean_mask_out"),
    ):
        if arr.shape != expected_grid:
            raise ValueError(f"{name} expected shape {expected_grid}, got {arr.shape}")

    _require_disjoint(
        {
            "plate_field": plate_arr,
            "convergence_field": convergence_arr,
            "divergence_field": divergence_arr,
            "subduction_field": subduction_arr,
            "shear_field": shear_arr,
            "hotspot_field": hotspot_arr,
            "crust_thickness_out": crust_arr,
            "isostatic_offset_out": isostasy_arr,
            "uplift_out": uplift_arr,
            "subsidence_out": subsidence_arr,
            "compression_out": compression_arr,
            "extension_out": extension_arr,
            "shear_out": shear_out_arr,
            "coastal_exposure_out": coastal_exposure_arr,
            "lithosphere_stiffness_out": lithosphere_arr,
            "base_ocean_mask_out": base_ocean_mask_arr,
        }
    )

    events_ptr = _ffi.new("HotspotEventArray*")
    stats_ptr = _ffi.new("WorldAgeStats*")

    result = _lib.world_age_run(
        int(height),
        int(width),
        int(seed),
        float(world_age),
        float(thermal_decay_half_life),
        float(hotspot_scale),
        float(isostasy_factor),
        float(radiogenic_heat_scale),
        int(plate_arr.shape[2] if plate_arr.ndim == 3 else 1),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", plate_arr, require_writable=False)),
        _ffi.cast(
            "const float*", _ffi.from_buffer("float[]", convergence_arr, require_writable=False)
        ),
        _ffi.cast(
            "const float*", _ffi.from_buffer("float[]", divergence_arr, require_writable=False)
        ),
        _ffi.cast(
            "const float*", _ffi.from_buffer("float[]", subduction_arr, require_writable=False)
        ),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", shear_arr, require_writable=False)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", hotspot_arr, require_writable=False)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", crust_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", isostasy_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", uplift_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", subsidence_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", compression_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", extension_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", shear_out_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", coastal_exposure_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", lithosphere_arr)),
        _ffi.cast("float*", _ffi.from_buffer("float[]", base_ocean_mask_arr)),
        events_ptr,
        stats_ptr,
    )
    if result != 0:
        raise RuntimeError(f"world_age_run failed with code {result}")

    events_table = _events_to_arrow(events_ptr[0])
    stats = stats_ptr[0]
    return events_table, _stats_metadata(stats)


def run_cubed_sphere_world_age(
    *,
    seed: int,
    world_age: float,
    thermal_decay_half_life: float,
    hotspot_scale: float,
    isostasy_factor: float,
    radiogenic_heat_scale: float,
    areas: np.ndarray,
    neighbors: np.ndarray,
    plate_field: np.ndarray,
    convergence_field: np.ndarray,
    divergence_field: np.ndarray,
    subduction_field: np.ndarray,
    shear_field: np.ndarray,
    hotspot_field: np.ndarray,
    crust_thickness_out: np.ndarray,
    isostatic_offset_out: np.ndarray,
    uplift_out: np.ndarray,
    subsidence_out: np.ndarray,
    compression_out: np.ndarray,
    extension_out: np.ndarray,
    shear_out: np.ndarray,
    margin_proximity_out: np.ndarray,
    lithosphere_stiffness_out: np.ndarray,
    proto_ocean_mask_out: np.ndarray,
) -> Tuple[pa.Table, Dict[str, Any]]:
    area_array = np.require(areas, dtype=np.float64, requirements=["C", "A"])
    neighbor_array = np.require(neighbors, dtype=np.int32, requirements=["C", "A"])
    if (
        area_array.ndim != 3
        or area_array.shape[0] != 6
        or area_array.shape[1] != area_array.shape[2]
    ):
        raise ValueError("areas must have shape (6, n, n)")
    face_shape = area_array.shape
    face_resolution = face_shape[1]
    if neighbor_array.shape != (*face_shape, 4):
        raise ValueError(f"neighbors must have shape {(*face_shape, 4)}")

    plate_array = _as_read_array(plate_field, name="plate_field")
    if plate_array.ndim != 4 or plate_array.shape[:3] != face_shape or plate_array.shape[3] < 7:
        raise ValueError(f"plate_field must have shape {(*face_shape, 7)} or more components")
    scalar_inputs = {
        "convergence_field": _as_read_array(convergence_field, name="convergence_field"),
        "divergence_field": _as_read_array(divergence_field, name="divergence_field"),
        "subduction_field": _as_read_array(subduction_field, name="subduction_field"),
        "shear_field": _as_read_array(shear_field, name="shear_field"),
        "hotspot_field": _as_read_array(hotspot_field, name="hotspot_field"),
    }
    scalar_outputs = {
        "crust_thickness_out": _as_write_array(crust_thickness_out, name="crust_thickness_out"),
        "isostatic_offset_out": _as_write_array(isostatic_offset_out, name="isostatic_offset_out"),
        "uplift_out": _as_write_array(uplift_out, name="uplift_out"),
        "subsidence_out": _as_write_array(subsidence_out, name="subsidence_out"),
        "compression_out": _as_write_array(compression_out, name="compression_out"),
        "extension_out": _as_write_array(extension_out, name="extension_out"),
        "shear_out": _as_write_array(shear_out, name="shear_out"),
        "margin_proximity_out": _as_write_array(margin_proximity_out, name="margin_proximity_out"),
        "lithosphere_stiffness_out": _as_write_array(
            lithosphere_stiffness_out, name="lithosphere_stiffness_out"
        ),
        "proto_ocean_mask_out": _as_write_array(proto_ocean_mask_out, name="proto_ocean_mask_out"),
    }
    for name, array in {**scalar_inputs, **scalar_outputs}.items():
        if array.shape != face_shape:
            raise ValueError(f"{name} must have shape {face_shape}")
    buffers = {
        "areas": area_array,
        "neighbors": neighbor_array,
        "plate_field": plate_array,
        **scalar_inputs,
        **scalar_outputs,
    }
    _require_disjoint(buffers)

    events_ptr = _ffi.new("SphericalHotspotEventArray*")
    stats_ptr = _ffi.new("WorldAgeStats*")
    result = _lib.world_age_run_cubed_sphere(
        int(np.prod(face_shape, dtype=np.int64)),
        int(seed),
        float(world_age),
        float(thermal_decay_half_life),
        float(hotspot_scale),
        float(isostasy_factor),
        float(radiogenic_heat_scale),
        int(plate_array.shape[3]),
        _ffi.cast("const double*", _ffi.from_buffer("double[]", area_array)),
        _ffi.cast("const int32_t*", _ffi.from_buffer("int32_t[]", neighbor_array)),
        _ffi.cast("const float*", _ffi.from_buffer("float[]", plate_array, require_writable=False)),
        _ffi.cast(
            "const float*",
            _ffi.from_buffer("float[]", scalar_inputs["convergence_field"], require_writable=False),
        ),
        _ffi.cast(
            "const float*",
            _ffi.from_buffer("float[]", scalar_inputs["divergence_field"], require_writable=False),
        ),
        _ffi.cast(
            "const float*",
            _ffi.from_buffer("float[]", scalar_inputs["subduction_field"], require_writable=False),
        ),
        _ffi.cast(
            "const float*",
            _ffi.from_buffer("float[]", scalar_inputs["shear_field"], require_writable=False),
        ),
        _ffi.cast(
            "const float*",
            _ffi.from_buffer("float[]", scalar_inputs["hotspot_field"], require_writable=False),
        ),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["crust_thickness_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["isostatic_offset_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["uplift_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["subsidence_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["compression_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["extension_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["shear_out"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["margin_proximity_out"])),
        _ffi.cast(
            "float*",
            _ffi.from_buffer("float[]", scalar_outputs["lithosphere_stiffness_out"]),
        ),
        _ffi.cast("float*", _ffi.from_buffer("float[]", scalar_outputs["proto_ocean_mask_out"])),
        events_ptr,
        stats_ptr,
    )
    if result != 0:
        raise RuntimeError(f"world_age_run_cubed_sphere failed with code {result}")
    events = _spherical_events_to_arrow(events_ptr[0], face_resolution=face_resolution)
    metadata = _stats_metadata(stats_ptr[0])
    proto_ocean_area_fraction = metadata.pop("water_fraction")
    metadata.update(
        {
            "event_coordinate_basis": "cubed_sphere_global_cell_id",
            "event_semantics": "tectonic_thermal_anomaly_proxy_not_mantle_plume",
            "plume_factor_semantics": "thermal_anomaly_intensity_compatibility_name",
            "isostasy_weighting": "spherical_cell_area",
            "ocean_mask_semantics": "oceanic_crust_candidate_not_final_water",
            "coastal_exposure_semantics": "continental_margin_proximity",
            "proto_ocean_area_fraction": proto_ocean_area_fraction,
            "spatial_scale_basis": "approximate_angular_radians",
        }
    )
    return events, metadata


__all__ = [
    "CUBED_SPHERE_WORLD_AGE_ABI_VERSION",
    "HOTSPOT_EVENT_DTYPE",
    "SPHERICAL_HOTSPOT_EVENT_DTYPE",
    "run_cubed_sphere_world_age",
    "run_world_age_kernels",
]
