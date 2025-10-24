"""Next-generation pipeline framework exports."""

from .cache import CacheManager
from .config import PipelineConfig, ResolutionSet, GridInfo
from .dataset import DatasetWriter
from .execution import ExecutionEngine, PipelineContext, RngPool
from .logging import RunLogger
from .memory import MemoryArena, GridHandle, VectorHandle
from .models import ArtifactRecord, StageResult, StageStats
from .registry import StageDescriptor, StageRegistry, stage, registry
from .topology import Topology, load_topology
from .visualization import VisualManager

# Ensure built-in stages are registered on import.
from . import stages  # noqa: F401,E402

__all__ = [
    "CacheManager",
    "PipelineConfig",
    "ResolutionSet",
    "GridInfo",
    "DatasetWriter",
    "ExecutionEngine",
    "PipelineContext",
    "RngPool",
    "RunLogger",
    "MemoryArena",
    "GridHandle",
    "VectorHandle",
    "ArtifactRecord",
    "StageResult",
    "StageStats",
    "StageDescriptor",
    "StageRegistry",
    "stage",
    "registry",
    "Topology",
    "load_topology",
    "VisualManager",
]
