"""Stage registration and metadata management."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .models import StageOutput, StageResult

StageCallable = Callable[["PipelineContext", Mapping[str, StageResult], Mapping[str, Any]], StageOutput]
StageVisualizer = Callable[
    [StageResult, "VisualizationRequest"],
    Optional["VisualizationResult" | Iterable["VisualizationResult"]],
]


@dataclass(frozen=True)
class StageDescriptor:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    callable: StageCallable
    version: str = "v1"
    visualizer: Optional[StageVisualizer] = None
    description: Optional[str] = None


class StageRegistry:
    """Global stage registry supporting decorator-based registration."""

    def __init__(self) -> None:
        self._stages: Dict[str, StageDescriptor] = {}

    def register(self, descriptor: StageDescriptor) -> None:
        if descriptor.name in self._stages:
            raise ValueError(f"Stage '{descriptor.name}' already registered")
        self._stages[descriptor.name] = descriptor

    def get(self, name: str) -> StageDescriptor:
        try:
            return self._stages[name]
        except KeyError as exc:
            raise KeyError(f"Unknown stage '{name}'") from exc

    def clear(self) -> None:
        self._stages.clear()

    def descriptors(self) -> Dict[str, StageDescriptor]:
        return dict(self._stages)

    def __contains__(self, name: str) -> bool:
        return name in self._stages


_REGISTRY = StageRegistry()


def stage(
    name: str,
    inputs: Iterable[str] | None = None,
    outputs: Iterable[str] | None = None,
    *,
    version: str = "v1",
    visualizer: StageVisualizer | None = None,
    description: str | None = None,
) -> Callable[[StageCallable], StageCallable]:
    """Decorator registering a pipeline stage."""

    def decorator(func: StageCallable) -> StageCallable:
        descriptor = StageDescriptor(
            name=name,
            inputs=tuple(inputs or ()),
            outputs=tuple(outputs or ()),
            callable=func,
            version=version,
            visualizer=visualizer or getattr(func, "visualizer", None),
            description=description or getattr(func, "__doc__", None),
        )
        _REGISTRY.register(descriptor)

        @wraps(func)
        def wrapper(
            context: "PipelineContext",
            dependencies: Mapping[str, StageResult],
            config: Mapping[str, Any],
        ) -> StageOutput:
            return func(context, dependencies, config)

        return wrapper

    return decorator


def registry() -> StageRegistry:
    return _REGISTRY


# Lazy imports placed at end to avoid circular dependencies
from .visualization import VisualizationRequest, VisualizationResult  # noqa: E402
