"""Pipeline execution engine."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import numpy as np

from .cache import CacheManager
from .config import PipelineConfig, ResolutionSet
from .dataset import DatasetWriter
from .logging import RunLogger
from .memory import MemoryArena
from .models import StageResult, StageStats
from .registry import StageDescriptor, registry
from .topology import Topology, load_topology
from .visualization import VisualManager


def _serialize_for_hash(data: Any) -> Any:
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if isinstance(data, Mapping):
        return {str(k): _serialize_for_hash(v) for k, v in sorted(data.items(), key=lambda item: str(item[0]))}
    if isinstance(data, Iterable) and not isinstance(data, (bytes, bytearray)):
        return [_serialize_for_hash(item) for item in data]
    return repr(data)


class RngPool:
    """Deterministic RNG factory keyed by stage name."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)

    def for_stage(self, stage_name: str) -> np.random.Generator:
        payload = f"{stage_name}:{self._seed}".encode("utf8")
        digest = hashlib.blake2b(payload, digest_size=16)
        seed = int.from_bytes(digest.digest()[:8], "little", signed=False)
        return np.random.default_rng(seed)


class PipelineContext:
    """Shared state passed to stages during execution."""

    def __init__(
        self,
        config: PipelineConfig,
        topology: Topology,
        resolution_set: ResolutionSet,
        arena: MemoryArena,
        rng_pool: RngPool,
        dataset_writer: DatasetWriter,
        cache_manager: CacheManager,
        logger: RunLogger,
    ) -> None:
        self.config = config
        self.topology = topology
        self.resolution_set = resolution_set
        self.arena = arena
        self._rng_pool = rng_pool
        self.dataset_writer = dataset_writer
        self.cache_manager = cache_manager
        self.logger = logger
        self.run_dir = config.run_output_dir()
        self._current_stage: str | None = None

    def rng(self, stage_name: str) -> np.random.Generator:
        return self._rng_pool.for_stage(stage_name)

    def set_current_stage(self, stage_name: str | None) -> None:
        self._current_stage = stage_name

    @contextmanager
    def timed(self, label: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.logger.log_event(
                {
                    "type": "timed_scope",
                    "stage": self._current_stage,
                    "label": label,
                    "duration_ns": end - start,
                }
            )


class ExecutionEngine:
    """Coordinates stage execution with caching, logging, and visualization."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        generate_visuals: bool = False,
    ) -> None:
        config.ensure_directories()
        run_dir = config.run_output_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = config.run_dataset_dir()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._logger = RunLogger(config.run_log_path())
        self._arena = MemoryArena()
        self._rng_pool = RngPool(config.rng_seed)
        self._cache = CacheManager(config.cache_dir)
        self._dataset_writer = DatasetWriter(dataset_dir)
        self._visuals = VisualManager(config.run_visual_dir()) if generate_visuals else None
        topology = load_topology(config.topology, config.resolution_set.native)
        self._context = PipelineContext(
            config=config,
            topology=topology,
            resolution_set=config.resolution_set,
            arena=self._arena,
            rng_pool=self._rng_pool,
            dataset_writer=self._dataset_writer,
            cache_manager=self._cache,
            logger=self._logger,
        )

    @property
    def context(self) -> PipelineContext:
        return self._context

    def _dependency_graph(self, stages: Iterable[str]) -> Dict[str, set[str]]:
        descriptors = registry().descriptors()
        graph: Dict[str, set[str]] = {}
        for stage_name in stages:
            if stage_name not in descriptors:
                raise KeyError(f"Stage '{stage_name}' not registered")
            descriptor = descriptors[stage_name]
            graph[stage_name] = set(descriptor.inputs)
        return graph

    def _topological_order(self, stages: Iterable[str]) -> list[str]:
        graph = self._dependency_graph(stages)
        sorter = TopologicalSorter(graph)
        try:
            order = list(sorter.static_order())
        except Exception as exc:  # pragma: no cover - TopologicalSorter raises CycleError
            raise ValueError(f"Invalid stage graph: {exc}") from exc
        return [stage for stage in order if stage in graph]

    def _compute_cache_key(
        self,
        descriptor: StageDescriptor,
        stage_config: Mapping[str, Any],
        dependency_results: Mapping[str, StageResult],
    ) -> str:
        payload = {
            "stage": descriptor.name,
            "version": descriptor.version,
            "config": _serialize_for_hash(stage_config),
            "topology": {
                "type": self._context.config.topology,
                "shape": self._context.topology.shape,
                "resolutions": [level.to_dict() for level in self._context.config.resolution_set.levels],
            },
            "deps": {
                name: {
                    "cache_key": result.cache_key,
                    "artifacts": result.artifact_checksums,
                }
                for name, result in sorted(dependency_results.items(), key=lambda item: item[0])
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
        return hashlib.blake2b(encoded, digest_size=16).hexdigest()

    def run(self, stages: Iterable[str] | None = None) -> Dict[str, StageResult]:
        descriptors = registry().descriptors()
        selected = list(stages) if stages else list(descriptors.keys())
        order = self._topological_order(selected)
        results: Dict[str, StageResult] = {}

        for stage_name in order:
            descriptor = descriptors[stage_name]
            dependencies = {name: results[name] for name in descriptor.inputs}
            stage_config = self._context.config.stage_config(stage_name)
            cache_key = self._compute_cache_key(descriptor, stage_config, dependencies)
            cache_dir = self._context.cache_manager.cache_dir(stage_name, cache_key)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._context.set_current_stage(stage_name)
            self._logger.log_stage_start(stage_name, cache_key)

            cached = self._context.cache_manager.load(stage_name, cache_key)
            if cached:
                cached.set_cache_key(cache_key)
                self._dataset_writer.hydrate_from_cache(cached, self._arena)
                if self._visuals:
                    if descriptor.visualizer:
                        self._visuals.emit_custom(cached, descriptor.visualizer)
                    else:
                        self._visuals.emit(cached)
                self._logger.log_stage_end(cached)
                results[stage_name] = cached
                self._context.set_current_stage(None)
                continue

            start_ns = time.perf_counter_ns()
            start_cpu = time.process_time_ns()
            output = descriptor.callable(self._context, dependencies, stage_config)
            result = StageResult.from_output(stage_name, descriptor.inputs, output)
            result.set_cache_key(cache_key)
            self._dataset_writer.persist(result, cache_dir)
            end_ns = time.perf_counter_ns()
            end_cpu = time.process_time_ns()
            memory_bytes = self._arena.stats()["bytes_allocated"]
            stats = StageStats(
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                cpu_time_ns=end_cpu - start_cpu,
                memory_bytes=memory_bytes,
                cache_hit=False,
            )
            result.record_stats(stats)
            self._context.cache_manager.store(result)
            if self._visuals:
                if descriptor.visualizer:
                    self._visuals.emit_custom(result, descriptor.visualizer)
                else:
                    self._visuals.emit(result)
            self._logger.log_stage_end(result)
            results[stage_name] = result
            self._context.set_current_stage(None)

        self._logger.close()
        return results


__all__ = ["ExecutionEngine", "PipelineContext", "RngPool"]
