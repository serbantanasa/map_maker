"""Rust-backed depression-aware global hydrology bindings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from cffi import FFI

from .._native import NativeLibraryAbiError, native_library_info, native_library_path

CUBED_SPHERE_HYDROLOGY_ABI_VERSION = 3

WATER_BODY_CLASSES = {
    1: "wetland",
    2: "endorheic_lake",
    3: "stable_lake",
    4: "overflow_lake",
    5: "breached_depression",
}

RIVER_MORPHOLOGY_CLASSES = {
    1: "mountain_torrent",
    2: "upland_river",
    3: "lowland_meander",
    4: "braided_river",
    5: "delta_distributary",
    6: "endorheic_inflow",
    7: "hydrologic_connector",
}

BED_MATERIAL_CLASSES = {
    0: "not_applicable",
    1: "bedrock",
    2: "gravel",
    3: "sand",
    4: "silt",
}

HYDROLOGY_CONTROL_NAMES = {
    "planet_radius_m",
    "minimum_depression_depth_m",
    "wetland_mean_depth_m",
    "endorheic_aridity_threshold",
    "maximum_fill_time_years",
    "lake_seepage_mm_year",
    "subgrid_relief_scale",
    "subgrid_connected_basin_fraction",
    "breach_score_threshold",
    "maximum_breach_incision_m",
    "breach_length_cells",
    "river_discharge_threshold_m3s",
    "river_contributing_area_threshold_km2",
    "river_minimum_discharge_m3s",
}

LAKE_RECORD_DTYPE = np.dtype(
    [
        ("depression_id", "<i4"),
        ("lake_id", "<i4"),
        ("class_code", "<i4"),
        ("sink_cell", "<i4"),
        ("outlet_cell", "<i4"),
        ("outlet_receiver", "<i4"),
        ("cell_count", "<i4"),
        ("water_cell_count", "<i4"),
        ("open_outlet", "<i4"),
        ("area_km2", "<f8"),
        ("water_area_km2", "<f8"),
        ("volume_km3", "<f8"),
        ("surface_elevation_m", "<f4"),
        ("maximum_depth_m", "<f4"),
        ("mean_depth_m", "<f4"),
        ("catchment_area_km2", "<f8"),
        ("mean_inflow_m3s", "<f4"),
        ("annual_inflow_km3", "<f8"),
        ("annual_evaporation_km3", "<f8"),
        ("annual_seepage_km3", "<f8"),
        ("annual_balance_km3", "<f8"),
        ("fill_time_years", "<f4"),
        ("salinity_index", "<f4"),
        ("mean_aridity_index", "<f4"),
        ("breach_incision_m", "<f4"),
    ],
    align=True,
)

BREACH_RECORD_DTYPE = np.dtype(
    [
        ("breach_id", "<i4"),
        ("depression_id", "<i4"),
        ("outlet_cell", "<i4"),
        ("downstream_cell", "<i4"),
        ("pre_breach_spill_elevation_m", "<f4"),
        ("post_breach_outlet_elevation_m", "<f4"),
        ("incision_m", "<f4"),
        ("gorge_length_km", "<f4"),
        ("sediment_pulse_km3", "<f8"),
        ("trigger_score", "<f4"),
    ],
    align=True,
)

REACH_RECORD_DTYPE = np.dtype(
    [
        ("reach_id", "<i4"),
        ("from_node", "<i4"),
        ("to_node", "<i4"),
        ("downstream_reach_id", "<i4"),
        ("basin_id", "<i4"),
        ("vertex_offset", "<i4"),
        ("vertex_count", "<i4"),
        ("strahler_order", "<i4"),
        ("morphology_code", "<i4"),
        ("bed_material_code", "<i4"),
        ("flow_direction_vector", "<f4", (3,)),
        ("slope", "<f4"),
        ("discharge_mean", "<f4"),
        ("discharge_seasonal", "<f4", (12,)),
        ("velocity_mean", "<f4"),
        ("velocity_seasonal", "<f4", (12,)),
        ("stream_power", "<f4"),
        ("channel_width_m", "<f4"),
        ("channel_depth_m", "<f4"),
        ("valley_width_m", "<f4"),
        ("floodplain_width_m", "<f4"),
        ("meander_index", "<f4"),
        ("braiding_index", "<f4"),
        ("incision_m", "<f4"),
        ("sediment_load_kg_s", "<f4"),
    ],
    align=True,
)

_CDEF = """
typedef struct {
    double planet_radius_m;
    float minimum_depression_depth_m;
    float wetland_mean_depth_m;
    float endorheic_aridity_threshold;
    float maximum_fill_time_years;
    float lake_seepage_mm_year;
    float subgrid_relief_scale;
    float subgrid_connected_basin_fraction;
    float breach_score_threshold;
    float maximum_breach_incision_m;
    int32_t breach_length_cells;
    float river_discharge_threshold_m3s;
    double river_contributing_area_threshold_km2;
    float river_minimum_discharge_m3s;
} HydrologyConfig;

typedef struct {
    int32_t depression_id;
    int32_t lake_id;
    int32_t class_code;
    int32_t sink_cell;
    int32_t outlet_cell;
    int32_t outlet_receiver;
    int32_t cell_count;
    int32_t water_cell_count;
    int32_t open_outlet;
    double area_km2;
    double water_area_km2;
    double volume_km3;
    float surface_elevation_m;
    float maximum_depth_m;
    float mean_depth_m;
    double catchment_area_km2;
    float mean_inflow_m3s;
    double annual_inflow_km3;
    double annual_evaporation_km3;
    double annual_seepage_km3;
    double annual_balance_km3;
    float fill_time_years;
    float salinity_index;
    float mean_aridity_index;
    float breach_incision_m;
} LakeRecord;
typedef struct { LakeRecord* data; size_t len; } LakeRecordArray;

typedef struct {
    int32_t breach_id;
    int32_t depression_id;
    int32_t outlet_cell;
    int32_t downstream_cell;
    float pre_breach_spill_elevation_m;
    float post_breach_outlet_elevation_m;
    float incision_m;
    float gorge_length_km;
    double sediment_pulse_km3;
    float trigger_score;
} BreachRecord;
typedef struct { BreachRecord* data; size_t len; } BreachRecordArray;

typedef struct {
    int32_t reach_id;
    int32_t from_node;
    int32_t to_node;
    int32_t downstream_reach_id;
    int32_t basin_id;
    int32_t vertex_offset;
    int32_t vertex_count;
    int32_t strahler_order;
    int32_t morphology_code;
    int32_t bed_material_code;
    float flow_direction_vector[3];
    float slope;
    float discharge_mean;
    float discharge_seasonal[12];
    float velocity_mean;
    float velocity_seasonal[12];
    float stream_power;
    float channel_width_m;
    float channel_depth_m;
    float valley_width_m;
    float floodplain_width_m;
    float meander_index;
    float braiding_index;
    float incision_m;
    float sediment_load_kg_s;
} RiverReachRecord;
typedef struct { RiverReachRecord* data; size_t len; } RiverReachRecordArray;
typedef struct { int32_t* data; size_t len; } Int32Array;

typedef struct {
    int32_t depression_count;
    int32_t lake_count;
    int32_t breach_count;
    int32_t basin_count;
    int32_t reach_count;
    int32_t wetland_count;
    int32_t endorheic_count;
    int32_t stable_lake_count;
    int32_t overflow_lake_count;
    int32_t land_cell_count;
    int32_t river_cell_count;
    int32_t closed_sink_count;
    int32_t topology_valid;
    double maximum_contributing_area_km2;
    float maximum_discharge_m3s;
    double annual_runoff_km3;
    double lake_volume_km3;
    double breach_sediment_pulse_km3;
    double annual_open_water_loss_km3;
    double conservation_relative_error;
} HydrologyStats;

uint32_t cubed_sphere_hydrology_abi_version(void);
int32_t hydrology_run_cubed_sphere(
    int32_t cell_count,
    const HydrologyConfig* config,
    const double* area_steradians,
    const int32_t* neighbors,
    const float* xyz,
    const float* elevation,
    const float* relief,
    const float* rock_strength,
    const float* accommodation,
    const uint8_t* ocean,
    const float* runoff,
    const float* evaporation,
    const float* aridity,
    int32_t* depression_id_out,
    int32_t* lake_id_out,
    uint8_t* water_class_out,
    float* lake_fraction_out,
    float* wetland_fraction_out,
    float* fill_depth_out,
    float* hydrologic_elevation_out,
    float* breach_incision_out,
    int32_t* receiver_out,
    float* flow_direction_out,
    float* flow_slope_out,
    double* contributing_area_out,
    float* monthly_discharge_out,
    float* mean_discharge_out,
    float* velocity_out,
    float* stream_power_out,
    int32_t* basin_id_out,
    uint8_t* sink_type_out,
    float* river_corridor_out,
    float* floodplain_out,
    LakeRecordArray* lake_records_out,
    BreachRecordArray* breach_records_out,
    RiverReachRecordArray* reach_records_out,
    Int32Array* reach_vertices_out,
    HydrologyStats* stats_out
);
void hydrology_free_lakes(LakeRecordArray array);
void hydrology_free_breaches(BreachRecordArray array);
void hydrology_free_reaches(RiverReachRecordArray array);
void hydrology_free_i32(Int32Array array);
"""


def _read(array: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f"{name} must be contiguous and aligned")
    return value


def _write(array: np.ndarray, *, name: str, dtype: np.dtype) -> np.ndarray:
    value = _read(array, name=name, dtype=dtype)
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_disjoint(inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> None:
    output_items = list(outputs.items())
    for index, (first_name, first) in enumerate(output_items):
        for second_name, second in output_items[index + 1 :]:
            if np.shares_memory(first, second):
                raise ValueError(f"{first_name} and {second_name} buffers must not overlap")
        for input_name, source in inputs.items():
            if np.shares_memory(first, source):
                raise ValueError(f"{first_name} and {input_name} buffers must not overlap")


_ffi = FFI()
_ffi.cdef(_CDEF)
native_library_info("hydrology_native")
_lib = _ffi.dlopen(str(native_library_path("hydrology_native")))
try:
    _cubed_sphere_abi = int(_lib.cubed_sphere_hydrology_abi_version())
except AttributeError as exc:
    raise NativeLibraryAbiError(
        "hydrology_native lacks the cubed-sphere API; rebuild native libraries"
    ) from exc
if _cubed_sphere_abi != CUBED_SPHERE_HYDROLOGY_ABI_VERSION:
    raise NativeLibraryAbiError(
        "hydrology_native cubed-sphere ABI "
        f"{_cubed_sphere_abi} does not match expected {CUBED_SPHERE_HYDROLOGY_ABI_VERSION}"
    )


def _copy_records(record_array: Any, *, dtype: np.dtype, free: Any) -> np.ndarray:
    length = int(record_array.len)
    if length <= 0:
        free(record_array)
        return np.empty(0, dtype=dtype)
    try:
        buffer = _ffi.buffer(record_array.data, length * dtype.itemsize)
        return np.frombuffer(buffer, dtype=dtype, count=length).copy()
    finally:
        free(record_array)


def _copy_i32(record_array: Any) -> np.ndarray:
    length = int(record_array.len)
    if length <= 0:
        _lib.hydrology_free_i32(record_array)
        return np.empty(0, dtype=np.int32)
    try:
        buffer = _ffi.buffer(record_array.data, length * np.dtype(np.int32).itemsize)
        return np.frombuffer(buffer, dtype=np.int32, count=length).copy()
    finally:
        _lib.hydrology_free_i32(record_array)


def _depression_table(records: np.ndarray) -> pa.Table:
    names = [WATER_BODY_CLASSES.get(int(code), "unknown") for code in records["class_code"]]
    return pa.table(
        {
            "depression_id": pa.array(records["depression_id"], type=pa.int32()),
            "lake_id": pa.array(records["lake_id"], type=pa.int32()),
            "class_code": pa.array(records["class_code"], type=pa.int32()),
            "class_name": pa.array(names, type=pa.string()),
            "sink_cell": pa.array(records["sink_cell"], type=pa.int32()),
            "outlet_cell": pa.array(records["outlet_cell"], type=pa.int32()),
            "outlet_receiver": pa.array(records["outlet_receiver"], type=pa.int32()),
            "open_outlet": pa.array(records["open_outlet"].astype(bool), type=pa.bool_()),
            "cell_count": pa.array(records["cell_count"], type=pa.int32()),
            "water_cell_count": pa.array(records["water_cell_count"], type=pa.int32()),
            "area_km2": pa.array(records["area_km2"], type=pa.float64()),
            "water_area_km2": pa.array(records["water_area_km2"], type=pa.float64()),
            "volume_km3": pa.array(records["volume_km3"], type=pa.float64()),
            "surface_elevation_m": pa.array(records["surface_elevation_m"], type=pa.float32()),
            "maximum_depth_m": pa.array(records["maximum_depth_m"], type=pa.float32()),
            "mean_depth_m": pa.array(records["mean_depth_m"], type=pa.float32()),
            "catchment_area_km2": pa.array(records["catchment_area_km2"], type=pa.float64()),
            "mean_inflow_m3s": pa.array(records["mean_inflow_m3s"], type=pa.float32()),
            "annual_inflow_km3": pa.array(records["annual_inflow_km3"], type=pa.float64()),
            "annual_evaporation_km3": pa.array(
                records["annual_evaporation_km3"], type=pa.float64()
            ),
            "annual_seepage_km3": pa.array(records["annual_seepage_km3"], type=pa.float64()),
            "annual_balance_km3": pa.array(records["annual_balance_km3"], type=pa.float64()),
            "fill_time_years": pa.array(records["fill_time_years"], type=pa.float32()),
            "salinity_index": pa.array(records["salinity_index"], type=pa.float32()),
            "mean_aridity_index": pa.array(records["mean_aridity_index"], type=pa.float32()),
            "breach_incision_m": pa.array(records["breach_incision_m"], type=pa.float32()),
        }
    )


def _breach_table(records: np.ndarray) -> pa.Table:
    return pa.table({name: pa.array(records[name]) for name in BREACH_RECORD_DTYPE.names or ()})


def _fixed_list(values: np.ndarray, size: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), size
    )


def _reach_table(records: np.ndarray, vertices: np.ndarray) -> pa.Table:
    downstream = records["downstream_reach_id"]
    upstream: list[list[int]] = [[] for _ in range(len(records))]
    for reach_id, target in enumerate(downstream):
        if target >= 0:
            upstream[int(target)].append(reach_id)
    polylines = [
        vertices[int(offset) : int(offset + count)].tolist()
        for offset, count in zip(records["vertex_offset"], records["vertex_count"], strict=True)
    ]
    morphology = [
        RIVER_MORPHOLOGY_CLASSES.get(int(code), "unknown") for code in records["morphology_code"]
    ]
    bed_material = [
        BED_MATERIAL_CLASSES.get(int(code), "unknown") for code in records["bed_material_code"]
    ]
    return pa.table(
        {
            "reach_id": pa.array(records["reach_id"], type=pa.int32()),
            "from_node": pa.array(records["from_node"], type=pa.int32()),
            "to_node": pa.array(records["to_node"], type=pa.int32()),
            "upstream_reach_ids": pa.array(upstream, type=pa.list_(pa.int32())),
            "downstream_reach_id": pa.array(downstream, type=pa.int32()),
            "basin_id": pa.array(records["basin_id"], type=pa.int32()),
            "cell_path": pa.array(polylines, type=pa.list_(pa.int32())),
            "reach_kind": pa.array(
                np.where(records["morphology_code"] == 7, "connector", "channel"),
                type=pa.string(),
            ),
            "flow_direction_vector": _fixed_list(records["flow_direction_vector"], 3),
            "slope": pa.array(records["slope"], type=pa.float32()),
            "strahler_order": pa.array(records["strahler_order"], type=pa.int32()),
            "discharge_mean": pa.array(records["discharge_mean"], type=pa.float32()),
            "discharge_seasonal": _fixed_list(records["discharge_seasonal"], 12),
            "velocity_mean": pa.array(records["velocity_mean"], type=pa.float32()),
            "velocity_seasonal": _fixed_list(records["velocity_seasonal"], 12),
            "stream_power": pa.array(records["stream_power"], type=pa.float32()),
            "channel_width_m": pa.array(records["channel_width_m"], type=pa.float32()),
            "channel_depth_m": pa.array(records["channel_depth_m"], type=pa.float32()),
            "valley_width_m": pa.array(records["valley_width_m"], type=pa.float32()),
            "floodplain_width_m": pa.array(records["floodplain_width_m"], type=pa.float32()),
            "meander_index": pa.array(records["meander_index"], type=pa.float32()),
            "braiding_index": pa.array(records["braiding_index"], type=pa.float32()),
            "incision_m": pa.array(records["incision_m"], type=pa.float32()),
            "sediment_load": pa.array(records["sediment_load_kg_s"], type=pa.float32()),
            "bed_material": pa.array(bed_material, type=pa.string()),
            "morphology_class": pa.array(morphology, type=pa.string()),
        }
    )


def run_cubed_sphere_hydrology(
    *,
    controls: dict[str, int | float],
    areas: np.ndarray,
    neighbors: np.ndarray,
    xyz: np.ndarray,
    elevation: np.ndarray,
    relief: np.ndarray,
    rock_strength: np.ndarray,
    accommodation: np.ndarray,
    ocean: np.ndarray,
    runoff: np.ndarray,
    evaporation: np.ndarray,
    aridity: np.ndarray,
    outputs: dict[str, np.ndarray],
) -> tuple[pa.Table, pa.Table, pa.Table, dict[str, Any]]:
    if set(controls) != HYDROLOGY_CONTROL_NAMES:
        missing = sorted(HYDROLOGY_CONTROL_NAMES - set(controls))
        extra = sorted(set(controls) - HYDROLOGY_CONTROL_NAMES)
        raise ValueError(f"hydrology controls mismatch; missing={missing}, extra={extra}")
    if any(not np.isfinite(value) for value in controls.values()):
        raise ValueError("hydrology controls must be finite")
    area_array = _read(areas, name="areas", dtype=np.dtype(np.float64))
    if (
        area_array.ndim != 3
        or area_array.shape[0] != 6
        or area_array.shape[1] != area_array.shape[2]
    ):
        raise ValueError("areas must have shape (6, n, n)")
    shape = area_array.shape
    monthly_shape = (12, *shape)
    input_arrays = {
        "areas": area_array,
        "neighbors": _read(neighbors, name="neighbors", dtype=np.dtype(np.int32)),
        "xyz": _read(xyz, name="xyz", dtype=np.dtype(np.float32)),
        "elevation": _read(elevation, name="elevation", dtype=np.dtype(np.float32)),
        "relief": _read(relief, name="relief", dtype=np.dtype(np.float32)),
        "rock_strength": _read(rock_strength, name="rock_strength", dtype=np.dtype(np.float32)),
        "accommodation": _read(accommodation, name="accommodation", dtype=np.dtype(np.float32)),
        "ocean": _read(ocean, name="ocean", dtype=np.dtype(np.uint8)),
        "runoff": _read(runoff, name="runoff", dtype=np.dtype(np.float32)),
        "evaporation": _read(evaporation, name="evaporation", dtype=np.dtype(np.float32)),
        "aridity": _read(aridity, name="aridity", dtype=np.dtype(np.float32)),
    }
    expected_inputs = {
        "neighbors": (*shape, 4),
        "xyz": (*shape, 3),
        "elevation": shape,
        "relief": shape,
        "rock_strength": shape,
        "accommodation": shape,
        "ocean": shape,
        "runoff": monthly_shape,
        "evaporation": monthly_shape,
        "aridity": shape,
    }
    for name, expected_shape in expected_inputs.items():
        if input_arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    dtypes: dict[str, np.dtype[Any]] = {
        "DepressionID": np.dtype(np.int32),
        "LakeID": np.dtype(np.int32),
        "WaterBodyClass": np.dtype(np.uint8),
        "LakeFraction": np.dtype(np.float32),
        "WetlandFraction": np.dtype(np.float32),
        "DepressionFillDepthM": np.dtype(np.float32),
        "HydrologicElevationM": np.dtype(np.float32),
        "BreachIncisionM": np.dtype(np.float32),
        "FlowReceiverID": np.dtype(np.int32),
        "FlowDirectionXYZ": np.dtype(np.float32),
        "FlowSlope": np.dtype(np.float32),
        "ContributingAreaKm2": np.dtype(np.float64),
        "MonthlyDischargeM3s": np.dtype(np.float32),
        "MeanDischargeM3s": np.dtype(np.float32),
        "MeanFlowVelocityMps": np.dtype(np.float32),
        "StreamPowerW": np.dtype(np.float32),
        "BasinID": np.dtype(np.int32),
        "FlowSinkType": np.dtype(np.uint8),
        "RiverCorridor": np.dtype(np.float32),
        "FloodplainPotential": np.dtype(np.float32),
    }
    expected_outputs = {
        **{name: shape for name in dtypes},
        "FlowDirectionXYZ": (*shape, 3),
        "MonthlyDischargeM3s": monthly_shape,
    }
    if set(outputs) != set(dtypes):
        missing = sorted(set(dtypes) - set(outputs))
        extra = sorted(set(outputs) - set(dtypes))
        raise ValueError(f"output buffers mismatch; missing={missing}, extra={extra}")
    output_arrays = {
        name: _write(outputs[name], name=name, dtype=dtype) for name, dtype in dtypes.items()
    }
    for name, expected_shape in expected_outputs.items():
        if output_arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    _require_disjoint(input_arrays, output_arrays)

    config_ptr = _ffi.new("HydrologyConfig*")
    for name, value in controls.items():
        setattr(config_ptr[0], name, value)
    lakes_ptr = _ffi.new("LakeRecordArray*")
    breaches_ptr = _ffi.new("BreachRecordArray*")
    reaches_ptr = _ffi.new("RiverReachRecordArray*")
    vertices_ptr = _ffi.new("Int32Array*")
    stats_ptr = _ffi.new("HydrologyStats*")
    result = _lib.hydrology_run_cubed_sphere(
        int(np.prod(shape, dtype=np.int64)),
        config_ptr,
        _ffi.cast(
            "const double*",
            _ffi.from_buffer("double[]", input_arrays["areas"], require_writable=False),
        ),
        _ffi.cast(
            "const int32_t*",
            _ffi.from_buffer("int32_t[]", input_arrays["neighbors"], require_writable=False),
        ),
        *[
            _ffi.cast(
                "const float*",
                _ffi.from_buffer("float[]", input_arrays[name], require_writable=False),
            )
            for name in ("xyz", "elevation", "relief", "rock_strength", "accommodation")
        ],
        _ffi.cast(
            "const uint8_t*",
            _ffi.from_buffer("uint8_t[]", input_arrays["ocean"], require_writable=False),
        ),
        *[
            _ffi.cast(
                "const float*",
                _ffi.from_buffer("float[]", input_arrays[name], require_writable=False),
            )
            for name in ("runoff", "evaporation", "aridity")
        ],
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", output_arrays["DepressionID"])),
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", output_arrays["LakeID"])),
        _ffi.cast("uint8_t*", _ffi.from_buffer("uint8_t[]", output_arrays["WaterBodyClass"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["LakeFraction"])),
        _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays["WetlandFraction"])),
        *[
            _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
            for name in ("DepressionFillDepthM", "HydrologicElevationM", "BreachIncisionM")
        ],
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", output_arrays["FlowReceiverID"])),
        *[
            _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
            for name in ("FlowDirectionXYZ", "FlowSlope")
        ],
        _ffi.cast("double*", _ffi.from_buffer("double[]", output_arrays["ContributingAreaKm2"])),
        *[
            _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
            for name in (
                "MonthlyDischargeM3s",
                "MeanDischargeM3s",
                "MeanFlowVelocityMps",
                "StreamPowerW",
            )
        ],
        _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", output_arrays["BasinID"])),
        _ffi.cast("uint8_t*", _ffi.from_buffer("uint8_t[]", output_arrays["FlowSinkType"])),
        *[
            _ffi.cast("float*", _ffi.from_buffer("float[]", output_arrays[name]))
            for name in ("RiverCorridor", "FloodplainPotential")
        ],
        lakes_ptr,
        breaches_ptr,
        reaches_ptr,
        vertices_ptr,
        stats_ptr,
    )
    if result != 0:
        messages = {
            1: "null or invalid FFI argument",
            2: "invalid numeric input or topology",
            3: "world must contain both land and ocean",
            4: "priority flood could not cover the global graph",
            5: "drainage graph contains a cycle",
        }
        raise RuntimeError(
            f"hydrology_run_cubed_sphere failed: {messages.get(result, f'status {result}')}"
        )

    lake_records = _copy_records(
        lakes_ptr[0], dtype=LAKE_RECORD_DTYPE, free=_lib.hydrology_free_lakes
    )
    breach_records = _copy_records(
        breaches_ptr[0], dtype=BREACH_RECORD_DTYPE, free=_lib.hydrology_free_breaches
    )
    reach_records = _copy_records(
        reaches_ptr[0], dtype=REACH_RECORD_DTYPE, free=_lib.hydrology_free_reaches
    )
    reach_vertices = _copy_i32(vertices_ptr[0])
    stats = stats_ptr[0]
    metadata: dict[str, Any] = {
        name: int(getattr(stats, name))
        for name in (
            "depression_count",
            "lake_count",
            "breach_count",
            "basin_count",
            "reach_count",
            "wetland_count",
            "endorheic_count",
            "stable_lake_count",
            "overflow_lake_count",
            "land_cell_count",
            "river_cell_count",
            "closed_sink_count",
            "topology_valid",
        )
    }
    metadata.update(
        {
            name: float(getattr(stats, name))
            for name in (
                "maximum_contributing_area_km2",
                "maximum_discharge_m3s",
                "annual_runoff_km3",
                "lake_volume_km3",
                "breach_sediment_pulse_km3",
                "annual_open_water_loss_km3",
                "conservation_relative_error",
            )
        }
    )
    return (
        _depression_table(lake_records),
        _breach_table(breach_records),
        _reach_table(reach_records, reach_vertices),
        metadata,
    )


__all__ = [
    "BED_MATERIAL_CLASSES",
    "CUBED_SPHERE_HYDROLOGY_ABI_VERSION",
    "RIVER_MORPHOLOGY_CLASSES",
    "WATER_BODY_CLASSES",
    "run_cubed_sphere_hydrology",
]
