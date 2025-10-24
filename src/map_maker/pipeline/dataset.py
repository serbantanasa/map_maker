"""Dataset writing utilities for pipeline runs."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from .memory import ArrayHandle, GridHandle, MemoryArena, VectorHandle
from .models import ArtifactRecord, StageResult


def _hash_bytes(data: bytes) -> str:
    hasher = hashlib.blake2b()
    hasher.update(data)
    return hasher.hexdigest()


class DatasetWriter:
    """Persists artifacts to disk and updates stage results."""

    def __init__(self, dataset_root: Path) -> None:
        self._dataset_root = dataset_root
        dataset_root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage_name: str) -> Path:
        path = self._dataset_root / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist(self, stage_result: StageResult, cache_dir: Path) -> None:
        stage_dir = self.stage_dir(stage_result.stage_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, value in stage_result.raw_artifacts.items():
            record = self._write_artifact(stage_dir, cache_dir, name, value)
            stage_result.register_artifact(record)
        stage_result.raw_artifacts.clear()

    def hydrate_from_cache(self, stage_result: StageResult, arena: MemoryArena) -> None:
        stage_dir = self.stage_dir(stage_result.stage_name)
        for record in stage_result.artifact_records.values():
            filename = record.metadata.get("filename", record.cache_path.name)
            dataset_path = stage_dir / filename
            if record.cache_path != dataset_path:
                shutil.copy2(record.cache_path, dataset_path)
            record.dataset_path = dataset_path
            record.value = self._load_value(record, arena)

    def _write_artifact(
        self, stage_dir: Path, cache_dir: Path, name: str, value: Any
    ) -> ArtifactRecord:
        if isinstance(value, GridHandle):
            return self._write_grid(stage_dir, cache_dir, name, value)
        if isinstance(value, VectorHandle):
            return self._write_vector(stage_dir, cache_dir, name, value)
        if isinstance(value, ArrayHandle):
            return self._write_array(stage_dir, cache_dir, name, value)
        if isinstance(value, np.ndarray):
            return self._write_ndarray(stage_dir, cache_dir, name, value)
        if isinstance(value, pa.Table):
            return self._write_arrow_table(stage_dir, cache_dir, name, value)
        if isinstance(value, (dict, list, int, float, str, bool)) or value is None:
            return self._write_json(stage_dir, cache_dir, name, value)
        raise TypeError(f"Unsupported artifact type for '{name}': {type(value)!r}")

    def _write_grid(self, stage_dir: Path, cache_dir: Path, name: str, handle: GridHandle) -> ArtifactRecord:
        handle.seal()
        array = np.array(handle.array(), copy=False)
        filename = f"{name}.npy"
        cache_path = cache_dir / filename
        np.save(cache_path, array, allow_pickle=False)
        dataset_path = stage_dir / filename
        if cache_path != dataset_path:
            shutil.copy2(cache_path, dataset_path)
        return ArtifactRecord(
            name=name,
            kind="grid",
            checksum=handle.checksum(),
            dataset_path=dataset_path,
            cache_path=cache_path,
            metadata={"filename": filename, "shape": handle.shape, "dtype": str(handle.dtype)},
            value=handle,
        )

    def _write_vector(self, stage_dir: Path, cache_dir: Path, name: str, handle: VectorHandle) -> ArtifactRecord:
        handle.seal()
        array = np.array(handle.array(), copy=False)
        filename = f"{name}.npy"
        cache_path = cache_dir / filename
        np.save(cache_path, array, allow_pickle=False)
        dataset_path = stage_dir / filename
        if cache_path != dataset_path:
            shutil.copy2(cache_path, dataset_path)
        return ArtifactRecord(
            name=name,
            kind="vector",
            checksum=handle.checksum(),
            dataset_path=dataset_path,
            cache_path=cache_path,
            metadata={"filename": filename, "length": array.shape[0], "dtype": str(handle.dtype)},
            value=handle,
        )

    def _write_array(self, stage_dir: Path, cache_dir: Path, name: str, handle: ArrayHandle) -> ArtifactRecord:
        handle.seal()
        array = np.array(handle.array(), copy=False)
        filename = f"{name}.npy"
        cache_path = cache_dir / filename
        np.save(cache_path, array, allow_pickle=False)
        dataset_path = stage_dir / filename
        if cache_path != dataset_path:
            shutil.copy2(cache_path, dataset_path)
        return ArtifactRecord(
            name=name,
            kind="array",
            checksum=handle.checksum(),
            dataset_path=dataset_path,
            cache_path=cache_path,
            metadata={"filename": filename, "shape": array.shape, "dtype": str(handle.dtype)},
            value=handle,
        )

    def _write_ndarray(self, stage_dir: Path, cache_dir: Path, name: str, array: np.ndarray) -> ArtifactRecord:
        filename = f"{name}.npy"
        cache_path = cache_dir / filename
        np.save(cache_path, array, allow_pickle=False)
        dataset_path = stage_dir / filename
        if cache_path != dataset_path:
            shutil.copy2(cache_path, dataset_path)
        checksum = _hash_bytes(array.tobytes())
        return ArtifactRecord(
            name=name,
            kind="ndarray",
            checksum=checksum,
            dataset_path=dataset_path,
            cache_path=cache_path,
            metadata={"filename": filename, "shape": array.shape, "dtype": str(array.dtype)},
            value=array,
        )

    def _write_arrow_table(self, stage_dir: Path, cache_dir: Path, name: str, table: pa.Table) -> ArtifactRecord:
        filename = f"{name}.arrow"
        cache_path = cache_dir / filename
        feather.write_feather(table, cache_path)
        dataset_path = stage_dir / filename
        if cache_path != dataset_path:
            shutil.copy2(cache_path, dataset_path)
        data_bytes = cache_path.read_bytes()
        checksum = _hash_bytes(data_bytes)
        return ArtifactRecord(
            name=name,
            kind="arrow",
            checksum=checksum,
            dataset_path=dataset_path,
            cache_path=cache_path,
            metadata={"filename": filename, "schema": table.schema.to_string()},
            value=table,
        )

    def _write_json(self, stage_dir: Path, cache_dir: Path, name: str, value: Any) -> ArtifactRecord:
        filename = f"{name}.json"
        cache_path = cache_dir / filename
        encoded = json.dumps(value, sort_keys=True).encode("utf8")
        cache_path.write_bytes(encoded)
        dataset_path = stage_dir / filename
        if cache_path != dataset_path:
            shutil.copy2(cache_path, dataset_path)
        return ArtifactRecord(
            name=name,
            kind="json",
            checksum=_hash_bytes(encoded),
            dataset_path=dataset_path,
            cache_path=cache_path,
            metadata={"filename": filename},
            value=value,
        )

    def _load_value(self, record: ArtifactRecord, arena: MemoryArena) -> Any:
        if record.kind in {"grid", "vector", "array", "ndarray"}:
            array = np.load(record.dataset_path, allow_pickle=False)
            shape = tuple(record.metadata.get("shape", array.shape))
            if record.kind == "grid" or (record.kind == "ndarray" and len(shape) == 2):
                grid_shape = tuple(int(dim) for dim in shape)[:2]
                handle = arena.allocate_grid(record.name, grid_shape)  # type: ignore[arg-type]
                handle.mutable_view()[...] = array.reshape(shape)
                handle.seal()
                return handle
            if record.kind == "vector" or (record.kind == "ndarray" and len(shape) == 1):
                length = int(np.prod(shape, dtype=np.int64))
                handle = arena.allocate_vector(record.name, length, dtype=array.dtype)
                handle.mutable_view()[...] = array.reshape(-1)
                handle.seal()
                return handle
            if record.kind == "array":
                shape_tuple = tuple(int(dim) for dim in shape)
                if shape_tuple and len(shape_tuple) > 1:
                    handle = arena.allocate_array(record.name, shape_tuple, dtype=array.dtype)
                    handle.mutable_view()[...] = array.reshape(shape_tuple)
                else:
                    length = int(np.prod(shape, dtype=np.int64))
                    handle = arena.allocate_vector(record.name, length, dtype=array.dtype)
                    handle.mutable_view()[...] = array.reshape(-1)
                handle.seal()
                return handle
            return array.reshape(shape)
        if record.kind == "json":
            return json.loads(record.dataset_path.read_text(encoding="utf8"))
        if record.kind == "arrow":
            table = feather.read_table(record.dataset_path)
            return table
        return np.load(record.dataset_path, allow_pickle=False)
