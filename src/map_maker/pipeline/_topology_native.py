"""Native C++ accelerated routines for topology precomputation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal

import numpy as np
from cffi import FFI


CDEF_SRC = """
    void build_sphere(int height, int width, double* areas);
    void build_cylinder(int height, int width, double* areas);
    void build_torus(int height, int width, double* areas);
    void build_neighbors(int height, int width, int wrap_rows, int wrap_cols,
                         int32_t* neighbors, float* weights);
"""

_FFI = FFI()
_FFI.cdef(CDEF_SRC)

_C_SRC = r"""
#include <cmath>
#include <cstdint>

namespace {

constexpr double PI = 3.14159265358979323846;

}

extern "C" {

void build_sphere(int height, int width, double* areas) {
    const double delta_lambda = 2.0 * PI / static_cast<double>(width);
    for (int i = 0; i < height; ++i) {
        const double phi1 = PI / 2.0 - (static_cast<double>(i + 1) / height) * PI;
        const double phi2 = PI / 2.0 - (static_cast<double>(i) / height) * PI;
        const double strip_area = std::abs(std::sin(phi2) - std::sin(phi1)) * delta_lambda;
        for (int j = 0; j < width; ++j) {
            areas[i * width + j] = strip_area;
        }
    }
}

void build_cylinder(int height, int width, double* areas) {
    const double cell_area = 1.0 / static_cast<double>(height * width);
    for (int idx = 0; idx < height * width; ++idx) {
        areas[idx] = cell_area;
    }
}

void build_torus(int height, int width, double* areas) {
    const double cell_area = 1.0 / static_cast<double>(height * width);
    for (int idx = 0; idx < height * width; ++idx) {
        areas[idx] = cell_area;
    }
}

void build_neighbors(int height, int width, int wrap_rows, int wrap_cols,
                     int32_t* neighbors, float* weights) {
    static const int OFFSETS[8][2] = {
        {-1,  0},
        { 1,  0},
        { 0, -1},
        { 0,  1},
        {-1, -1},
        {-1,  1},
        { 1, -1},
        { 1,  1},
    };

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const int cell_index = i * width + j;
            for (int k = 0; k < 8; ++k) {
                int ni = i + OFFSETS[k][0];
                int nj = j + OFFSETS[k][1];

                if (wrap_rows) {
                    if (ni < 0) {
                        ni += height;
                    } else if (ni >= height) {
                        ni -= height;
                    }
                }
                if (wrap_cols) {
                    if (nj < 0) {
                        nj += width;
                    } else if (nj >= width) {
                        nj -= width;
                    }
                }

                bool valid = true;
                if (!wrap_rows && (ni < 0 || ni >= height)) {
                    valid = false;
                }
                if (!wrap_cols && (nj < 0 || nj >= width)) {
                    valid = false;
                }

                const int slot = cell_index * 8 + k;
                if (!valid) {
                    neighbors[slot] = -1;
                    weights[slot] = 0.0f;
                } else {
                    neighbors[slot] = ni * width + nj;
                    const float di = static_cast<float>(OFFSETS[k][0]);
                    const float dj = static_cast<float>(OFFSETS[k][1]);
                    const float dist = std::sqrt(di * di + dj * dj);
                    weights[slot] = dist;
                }
            }
        }
    }
}

}
"""


def _ensure_native_module():
    module_name = "map_maker.pipeline._topology_native_cffi"
    try:
        return importlib.import_module(module_name)
    except ImportError:
        ffibuilder = FFI()
        ffibuilder.cdef(CDEF_SRC)
        ffibuilder.set_source(
            module_name,
            _C_SRC,
            source_extension=".cc",
            extra_compile_args=["-std=c++17"],
            language="c++",
        )
        build_dir = Path(__file__).resolve().parent
        output_path = Path(ffibuilder.compile(tmpdir=str(build_dir), verbose=False))
        prefix = "map_maker.pipeline."
        target_name = output_path.name
        if target_name.startswith(prefix):
            target_name = target_name[len(prefix) :]
        target_path = build_dir / target_name
        if output_path != target_path:
            output_path.replace(target_path)
        return importlib.import_module(module_name)


_native = _ensure_native_module()
_lib = _native.lib
_ffi = _native.ffi


TopologyKind = Literal["sphere", "cylinder", "torus"]


def compute_cell_areas(kind: TopologyKind, height: int, width: int) -> np.ndarray:
    areas = np.empty((height, width), dtype=np.float64)
    ptr = _ffi.cast("double*", _ffi.from_buffer("double[]", areas))
    if kind == "sphere":
        _lib.build_sphere(height, width, ptr)
    elif kind == "cylinder":
        _lib.build_cylinder(height, width, ptr)
    elif kind == "torus":
        _lib.build_torus(height, width, ptr)
    else:
        raise ValueError(f"Unsupported topology kind '{kind}'")
    return areas


def compute_neighbors(kind: TopologyKind, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    neighbors = np.full((height, width, 8), -1, dtype=np.int32)
    weights = np.zeros((height, width, 8), dtype=np.float32)
    wrap_rows = 1 if kind == "torus" else 0
    wrap_cols = 1
    if kind == "sphere":
        wrap_cols = 1
    elif kind == "cylinder":
        wrap_cols = 1
    elif kind == "torus":
        wrap_cols = 1
    else:
        raise ValueError(f"Unsupported topology kind '{kind}'")
    ptr_neighbors = _ffi.cast("int32_t*", _ffi.from_buffer("int32_t[]", neighbors))
    ptr_weights = _ffi.cast("float*", _ffi.from_buffer("float[]", weights))
    _lib.build_neighbors(height, width, wrap_rows, wrap_cols, ptr_neighbors, ptr_weights)
    return neighbors, weights


__all__ = ["compute_cell_areas", "compute_neighbors", "TopologyKind"]
