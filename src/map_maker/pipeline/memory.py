"""Shared memory arena and data handle implementations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import uuid
from typing import Any, Iterable, Optional, Tuple

import numpy as np

DEFAULT_ALIGNMENT = 64


def _aligned_empty(shape: Tuple[int, ...], dtype: np.dtype, alignment: int) -> tuple[np.ndarray, np.ndarray]:
    """Allocate an aligned ndarray and return both the view and owning buffer."""
    dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    raw = np.empty(size + alignment, dtype=np.uint8)
    offset = (-raw.ctypes.data) % alignment
    buffer = raw[offset : offset + size]
    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    return array, raw


@dataclass
class ArenaAllocation:
    key: str
    name: str
    array: np.ndarray
    base_buffer: np.ndarray
    dtype: np.dtype
    shape: Tuple[int, ...]
    sealed: bool = False

    def bytes(self) -> int:
        return int(self.array.nbytes)


class MemoryArena:
    """Process-wide arena that owns aligned numpy buffers."""

    def __init__(self, alignment: int = DEFAULT_ALIGNMENT) -> None:
        self._alignment = alignment
        self._allocations: dict[str, ArenaAllocation] = {}
        self._lock = threading.Lock()
        self._bytes_allocated = 0

    def allocate_grid(self, name: str, shape: Tuple[int, int], dtype: np.dtype = np.float32) -> "GridHandle":
        if len(shape) != 2:
            raise ValueError("Grid allocations must be 2D")
        return self._allocate(name=name, shape=shape, dtype=dtype, handle_cls=GridHandle)

    def allocate_array(self, name: str, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> "ArrayHandle":
        if len(shape) == 0:
            raise ValueError("Array allocations must have positive rank")
        return self._allocate(name=name, shape=shape, dtype=dtype, handle_cls=ArrayHandle)

    def allocate_vector(self, name: str, length: int, dtype: np.dtype = np.float32) -> "VectorHandle":
        return self._allocate(name=name, shape=(length,), dtype=dtype, handle_cls=VectorHandle)

    def _allocate(
        self, name: str, shape: Tuple[int, ...], dtype: np.dtype, handle_cls: type["ArrayHandle"]
    ) -> "ArrayHandle":
        shape_tuple = tuple(int(dim) for dim in shape)
        dtype = np.dtype(dtype)
        array, base = _aligned_empty(shape_tuple, dtype, self._alignment)
        key = uuid.uuid4().hex
        allocation = ArenaAllocation(
            key=key,
            name=name,
            array=array,
            base_buffer=base,
            dtype=dtype,
            shape=shape_tuple,
        )
        with self._lock:
            self._allocations[key] = allocation
            self._bytes_allocated += allocation.bytes()
        return handle_cls(self, key)

    def _get_allocation(self, key: str) -> ArenaAllocation:
        try:
            return self._allocations[key]
        except KeyError as exc:
            raise KeyError(f"Unknown allocation key {key}") from exc

    def seal(self, key: str) -> None:
        with self._lock:
            allocation = self._get_allocation(key)
            if allocation.sealed:
                return
            allocation.array.setflags(write=False)
            allocation.sealed = True

    def release(self, key: str) -> None:
        with self._lock:
            allocation = self._allocations.pop(key, None)
            if allocation:
                self._bytes_allocated -= allocation.bytes()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "allocations": len(self._allocations),
                "bytes_allocated": self._bytes_allocated,
                "alignment": self._alignment,
            }


class ArrayHandle:
    """Base handle for arrays owned by the arena."""

    __slots__ = ("_arena", "_key")

    def __init__(self, arena: MemoryArena, key: str) -> None:
        self._arena = arena
        self._key = key

    def mutable_view(self) -> np.ndarray:
        allocation = self._arena._get_allocation(self._key)
        if allocation.sealed:
            raise RuntimeError("Allocation already sealed; cannot request mutable view")
        allocation.array.setflags(write=True)
        return allocation.array

    def array(self) -> np.ndarray:
        allocation = self._arena._get_allocation(self._key)
        allocation.array.setflags(write=False)
        return allocation.array

    def seal(self) -> None:
        self._arena.seal(self._key)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._arena._get_allocation(self._key).shape

    @property
    def dtype(self) -> np.dtype:
        return self._arena._get_allocation(self._key).dtype

    def checksum(self) -> str:
        array = self.array()
        hasher = hashlib.blake2b()
        hasher.update(memoryview(array))
        return hasher.hexdigest()

    def to_serializable(self) -> dict[str, Any]:
        allocation = self._arena._get_allocation(self._key)
        return {
            "shape": allocation.shape,
            "dtype": str(allocation.dtype),
            "sealed": allocation.sealed,
            "checksum": self.checksum(),
        }

    def __array__(self) -> np.ndarray:  # pragma: no cover - implicit numpy bridge
        return self.array()

    def __repr__(self) -> str:
        allocation = self._arena._get_allocation(self._key)
        state = "sealed" if allocation.sealed else "mutable"
        return f"{self.__class__.__name__}(shape={allocation.shape}, dtype={allocation.dtype}, {state})"


class GridHandle(ArrayHandle):
    """Handle referencing a 2D grid."""


class VectorHandle(ArrayHandle):
    """Handle referencing a 1D vector."""
