"""Stage output caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import ArtifactRecord, StageResult, StageStats


class CacheManager:
    """Persist stage outputs to cache directories keyed by signature."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

    def cache_dir(self, stage_name: str, cache_key: str) -> Path:
        return self._base_dir / stage_name / cache_key

    def load(self, stage_name: str, cache_key: str) -> Optional[StageResult]:
        directory = self.cache_dir(stage_name, cache_key)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return None
        data = json.loads(manifest_path.read_text(encoding="utf8"))
        stats_dict = data["stats"]
        stats = StageStats(
            start_ns=int(stats_dict["start_ns"]),
            end_ns=int(stats_dict["end_ns"]),
            duration_ns=int(stats_dict["duration_ns"]),
            cpu_time_ns=int(stats_dict["cpu_time_ns"]),
            memory_bytes=int(stats_dict["memory_bytes"]),
            cache_hit=True,
        )
        result = StageResult(
            stage_name=str(data["stage_name"]),
            dependencies=tuple(data.get("dependencies", ())),
            raw_artifacts={},
            metadata=dict(data.get("metadata", {})),
        )
        result.record_stats(stats)
        result.set_cache_key(str(data["cache_key"]))
        artifacts = []
        for manifest in data.get("artifacts", []):
            artifacts.append(ArtifactRecord.from_manifest(manifest, directory))
        for artifact in artifacts:
            result.register_artifact(artifact)
        return result

    def store(self, stage_result: StageResult) -> None:
        if stage_result.cache_key is None:
            raise ValueError("Cannot store StageResult without cache_key")
        directory = self.cache_dir(stage_result.stage_name, stage_result.cache_key)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        manifest = stage_result.to_manifest(directory)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8")

