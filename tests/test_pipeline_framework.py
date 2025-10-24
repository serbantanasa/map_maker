from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from map_maker.pipeline import (
    ExecutionEngine,
    GridInfo,
    PipelineConfig,
    RngPool,
    load_topology,
    registry,
    stage,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    reg = registry()
    reg.clear()
    yield
    reg.clear()


def test_topology_loader_returns_expected_class():
    from map_maker.pipeline.topology import CylinderTopology, SphereTopology, TorusTopology

    grid = GridInfo(height=4, width=8)
    assert isinstance(load_topology("sphere", grid), SphereTopology)
    assert isinstance(load_topology("cylinder", grid), CylinderTopology)
    assert isinstance(load_topology("torus", grid), TorusTopology)


def test_rng_pool_stage_seed_deterministic():
    pool = RngPool(1234)
    sample_a_1 = pool.for_stage("alpha").integers(0, 100, size=5)
    sample_a_2 = pool.for_stage("alpha").integers(0, 100, size=5)
    sample_b = pool.for_stage("beta").integers(0, 100, size=5)
    np.testing.assert_array_equal(sample_a_1, sample_a_2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(sample_a_1, sample_b)


def _make_config(tmp_path: Path, run_id: str) -> PipelineConfig:
    base = {
        "topology": "sphere",
        "resolutions": [{"height": 6, "width": 6}],
        "output_dir": str(tmp_path / "output"),
        "cache_dir": str(tmp_path / "cache"),
        "log_dir": str(tmp_path / "logs"),
        "rng_seed": 1337,
        "stage_overrides": {
            "stage_a": {"value": 2.0},
            "stage_b": {"scale": 0.5},
        },
    }
    base["run_id"] = run_id
    return PipelineConfig.from_mapping(base)


def test_execution_engine_caches_and_logs(tmp_path: Path):
    calls: Dict[str, int] = {"stage_a": 0, "stage_b": 0, "stage_c": 0}

    @stage("stage_a", outputs=("a_grid",))
    def stage_a(context, deps, config):
        calls["stage_a"] += 1
        grid = context.arena.allocate_grid("a_grid", context.topology.shape)
        with context.timed("initialize_grid"):
            grid.mutable_view().fill(float(config.get("value", 1.0)))
        return {"a_grid": grid}

    @stage("stage_b", inputs=("stage_a",), outputs=("b_value",))
    def stage_b(context, deps, config):
        calls["stage_b"] += 1
        handle = deps["stage_a"].artifact_records["a_grid"].value
        assert handle is not None
        array = np.array(handle.array(), copy=False)
        total = float(array.sum()) * float(config.get("scale", 1.0))
        return {"b_value": total}

    @stage("stage_c", inputs=("stage_b",), outputs=("summary",))
    def stage_c(context, deps, config):
        calls["stage_c"] += 1
        value = deps["stage_b"].artifact_records["b_value"].value
        rng = context.rng("stage_c")
        noise = float(rng.normal())
        return {"summary": {"value": value, "noise": noise}}

    config1 = _make_config(tmp_path, "run-one")
    engine1 = ExecutionEngine(config1, generate_visuals=True)
    results1 = engine1.run()

    assert list(results1.keys()) == ["stage_a", "stage_b", "stage_c"]
    assert calls == {"stage_a": 1, "stage_b": 1, "stage_c": 1}
    assert not results1["stage_a"].stats.cache_hit

    log_path1 = config1.run_log_path()
    assert log_path1.exists()
    events = [json.loads(line) for line in log_path1.read_text().splitlines()]
    stages_started = [event["stage"] for event in events if event.get("type") == "stage_start"]
    assert stages_started == ["stage_a", "stage_b", "stage_c"]
    timed = [event for event in events if event.get("type") == "timed_scope"]
    assert any(event.get("label") == "initialize_grid" for event in timed)

    visuals_dir = config1.run_visual_dir() / "stage_a"
    assert any(path.suffix == ".png" for path in visuals_dir.glob("*.png"))

    config2 = _make_config(tmp_path, "run-two")
    engine2 = ExecutionEngine(config2, generate_visuals=True)
    results2 = engine2.run()

    assert calls == {"stage_a": 1, "stage_b": 1, "stage_c": 1}, "Stages should be served from cache"
    assert results2["stage_a"].stats.cache_hit
    assert results2["stage_b"].stats.cache_hit
    assert results2["stage_c"].stats.cache_hit

    summary1 = results1["stage_c"].artifact_records["summary"].value
    summary2 = results2["stage_c"].artifact_records["summary"].value
    assert summary1 == summary2


def test_cycle_detection(tmp_path: Path):
    reg = registry()
    reg.clear()

    @stage("alpha", inputs=("beta",))
    def stage_alpha(context, deps, config):
        return {"value": 1}

    @stage("beta", inputs=("alpha",))
    def stage_beta(context, deps, config):
        return {"value": 2}

    config = _make_config(tmp_path, "cycle")
    engine = ExecutionEngine(config)
    with pytest.raises(ValueError):
        engine.run()
