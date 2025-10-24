"""Core data models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


@dataclass
class StageStats:
    """Timing and resource metrics for a stage execution."""

    start_ns: int
    end_ns: int
    duration_ns: int
    cpu_time_ns: int
    memory_bytes: int
    cache_hit: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "cpu_time_ns": self.cpu_time_ns,
            "memory_bytes": self.memory_bytes,
            "cache_hit": self.cache_hit,
        }


@dataclass
class ArtifactRecord:
    """Represents a persisted artifact on disk."""

    name: str
    kind: str
    checksum: str
    dataset_path: Path
    cache_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    value: Any | None = None

    def to_manifest(self, base_dir: Path) -> Dict[str, Any]:
        relative = self.cache_path
        if self.cache_path.is_absolute():
            try:
                relative = self.cache_path.relative_to(base_dir)
            except ValueError:
                relative = self.cache_path
        return {
            "name": self.name,
            "kind": self.kind,
            "checksum": self.checksum,
            "path": str(relative),
            "metadata": self.metadata,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], base_dir: Path) -> "ArtifactRecord":
        raw_path = Path(manifest["path"])
        if not raw_path.is_absolute():
            raw_path = base_dir / raw_path
        return cls(
            name=str(manifest["name"]),
            kind=str(manifest["kind"]),
            checksum=str(manifest["checksum"]),
            dataset_path=raw_path,
            cache_path=raw_path,
            metadata=dict(manifest.get("metadata", {})),
        )


@dataclass
class StageResult:
    """Captured outcome of a pipeline stage."""

    stage_name: str
    dependencies: tuple[str, ...]
    raw_artifacts: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: StageStats | None = None
    artifact_records: Dict[str, ArtifactRecord] = field(default_factory=dict)
    cache_key: str | None = None

    @classmethod
    def from_output(
        cls,
        stage_name: str,
        dependencies: Iterable[str],
        output: "StageOutput",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StageResult":
        if isinstance(output, StageResult):
            result = replace(output, stage_name=stage_name, dependencies=tuple(dependencies))
            if metadata:
                result.metadata.update(metadata)
            return result
        raw: Dict[str, Any]
        extra_meta: Dict[str, Any] = {}
        if isinstance(output, Mapping):
            raw = dict(output)
        elif isinstance(output, Iterable):
            raw = {f"artifact_{idx}": value for idx, value in enumerate(output)}
        else:
            raw = {"value": output}
        if metadata:
            extra_meta.update(metadata)
        return cls(
            stage_name=stage_name,
            dependencies=tuple(dependencies),
            raw_artifacts=raw,
            metadata=extra_meta,
        )

    def register_artifact(self, record: ArtifactRecord) -> None:
        self.artifact_records[record.name] = record

    def record_stats(self, stats: StageStats) -> None:
        self.stats = stats

    def set_cache_key(self, cache_key: str) -> None:
        self.cache_key = cache_key

    @property
    def artifact_checksums(self) -> Dict[str, str]:
        return {name: record.checksum for name, record in self.artifact_records.items()}

    def to_manifest(self, base_dir: Path) -> Dict[str, Any]:
        if self.cache_key is None:
            raise ValueError("Cannot serialize StageResult without cache_key")
        if self.stats is None:
            raise ValueError("Cannot serialize StageResult without stats")
        return {
            "stage_name": self.stage_name,
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
            "cache_key": self.cache_key,
            "stats": self.stats.to_dict(),
            "artifacts": [record.to_manifest(base_dir) for record in self.artifact_records.values()],
        }


StageOutput = Mapping[str, Any] | Iterable[Any] | Any
