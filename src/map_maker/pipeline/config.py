"""Configuration models for the next-generation pipeline framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional
import uuid

import yaml


@dataclass(frozen=True)
class GridInfo:
    """Metadata describing a single resolution level."""

    height: int
    width: int
    scale_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"height": self.height, "width": self.width, "scale_factor": self.scale_factor}


@dataclass(frozen=True)
class ResolutionSet:
    """Collection of resolution levels used by the pipeline."""

    levels: tuple[GridInfo, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"levels": [level.to_dict() for level in self.levels]}

    @property
    def native(self) -> GridInfo:
        return self.levels[0]


def _expand_dir(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _default_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"


@dataclass
class PipelineConfig:
    """Top-level configuration for a pipeline run."""

    topology: str
    resolution_set: ResolutionSet
    run_id: str = field(default_factory=_default_run_id)
    rng_seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("out"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    stage_overrides: Dict[str, Mapping[str, Any]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PipelineConfig":
        topology = mapping.get("topology", "sphere")
        resolutions = mapping.get("resolutions")
        if not resolutions:
            raise ValueError("Pipeline config requires at least one resolution entry")
        levels = []
        for entry in resolutions:
            height = entry.get("height")
            width = entry.get("width")
            if height is None or width is None:
                raise ValueError("Resolution entries must include 'height' and 'width'")
            level = GridInfo(
                height=int(height),
                width=int(width),
                scale_factor=float(entry.get("scale_factor", 1.0)),
            )
            levels.append(level)

        resolution_set = ResolutionSet(tuple(levels))

        run_id = mapping.get("run_id") or _default_run_id()
        rng_seed = int(mapping.get("rng_seed", 0))
        output_dir = _expand_dir(Path(mapping.get("output_dir", "out")))
        cache_dir = _expand_dir(Path(mapping.get("cache_dir", output_dir / "cache")))
        log_dir = _expand_dir(Path(mapping.get("log_dir", output_dir / "logs")))

        overrides = mapping.get("stage_overrides", {})
        if not isinstance(overrides, MutableMapping):
            raise TypeError("stage_overrides must be a mapping of stage names to configuration dicts")
        stage_overrides: Dict[str, Mapping[str, Any]] = {}
        for key, value in overrides.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"Stage override '{key}' must be a mapping, got {type(value)!r}")
            stage_overrides[str(key)] = dict(value)

        extra = dict(mapping)
        for consumed in ("topology", "resolutions", "run_id", "rng_seed", "output_dir", "cache_dir", "log_dir", "stage_overrides"):
            extra.pop(consumed, None)

        return cls(
            topology=str(topology),
            resolution_set=resolution_set,
            run_id=str(run_id),
            rng_seed=rng_seed,
            output_dir=output_dir,
            cache_dir=cache_dir,
            log_dir=log_dir,
            stage_overrides=stage_overrides,
            extra=extra,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "PipelineConfig":
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf8") as fh:
            if path.suffix.lower() in {".yml", ".yaml"}:
                data = yaml.safe_load(fh)
            else:
                data = json.load(fh)
        if not isinstance(data, Mapping):
            raise TypeError(f"Configuration file must contain a mapping, got {type(data)!r}")
        return cls.from_mapping(data)

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        for directory in (self.output_dir, self.cache_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def stage_config(self, stage_name: str) -> Mapping[str, Any]:
        """Return the merged configuration for a stage."""
        return self.stage_overrides.get(stage_name, {})

    def run_output_dir(self) -> Path:
        """Directory where this run stores datasets."""
        return _expand_dir(self.output_dir / self.run_id)

    def run_visual_dir(self) -> Path:
        return _expand_dir(self.run_output_dir() / "visuals")

    def run_dataset_dir(self) -> Path:
        return _expand_dir(self.run_output_dir() / "datasets")

    def run_log_path(self) -> Path:
        return _expand_dir(self.log_dir / f"{self.run_id}.jsonl")


def load_config(source: Path | str) -> PipelineConfig:
    """Convenience helper for CLI consumers."""
    return PipelineConfig.from_file(source)

