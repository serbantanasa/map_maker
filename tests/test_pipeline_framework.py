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


def test_requesting_terminal_stage_runs_dependency_closure(tmp_path: Path):
    calls: list[str] = []

    @stage("foundation")
    def foundation(context, deps, config):
        calls.append("foundation")
        return {"value": 2}

    @stage("middle", inputs=("foundation",))
    def middle(context, deps, config):
        calls.append("middle")
        value = deps["foundation"].artifact_records["value"].value
        return {"value": value + 3}

    @stage("terminal", inputs=("middle",))
    def terminal(context, deps, config):
        calls.append("terminal")
        value = deps["middle"].artifact_records["value"].value
        return {"value": value * 2}

    config = _make_config(tmp_path, "dependency-closure")
    results = ExecutionEngine(config).run(["terminal"])

    assert calls == ["foundation", "middle", "terminal"]
    assert list(results) == ["foundation", "middle", "terminal"]
    assert results["terminal"].artifact_records["value"].value == 10


def test_native_fingerprint_change_invalidates_stage_cache(tmp_path: Path, monkeypatch):
    import map_maker.pipeline.execution as execution

    calls = 0

    @stage("native_sensitive")
    def native_sensitive(context, deps, config):
        nonlocal calls
        calls += 1
        return {"value": calls}

    monkeypatch.setattr(
        execution,
        "simulation_native_fingerprints",
        lambda: {"kernel": {"abi_version": 1, "sha256": "a" * 64}},
    )
    first = ExecutionEngine(_make_config(tmp_path, "fingerprint-a")).run(["native_sensitive"])
    monkeypatch.setattr(
        execution,
        "simulation_native_fingerprints",
        lambda: {"kernel": {"abi_version": 1, "sha256": "b" * 64}},
    )
    second = ExecutionEngine(_make_config(tmp_path, "fingerprint-b")).run(["native_sensitive"])

    assert calls == 2
    assert first["native_sensitive"].cache_key != second["native_sensitive"].cache_key
    assert not second["native_sensitive"].stats.cache_hit


def test_corrupt_cache_artifact_is_recomputed(tmp_path: Path):
    calls = 0

    @stage("corruptible")
    def corruptible(context, deps, config):
        nonlocal calls
        calls += 1
        grid = context.arena.allocate_grid("corruptible_grid", context.topology.shape)
        grid.mutable_view().fill(7.0)
        return {"grid": grid}

    first_config = _make_config(tmp_path, "corrupt-first")
    first = ExecutionEngine(first_config).run(["corruptible"])["corruptible"]
    cache_path = first.artifact_records["grid"].cache_path
    native_grid = first_config.resolution_set.native
    np.save(cache_path, np.zeros((native_grid.height, native_grid.width), dtype=np.float32))

    second_config = _make_config(tmp_path, "corrupt-second")
    second = ExecutionEngine(second_config).run(["corruptible"])["corruptible"]
    restored = np.asarray(second.artifact_records["grid"].value.array())

    assert calls == 2
    assert second.stats is not None and not second.stats.cache_hit
    assert np.all(restored == 7.0)
    events = [json.loads(line) for line in second_config.run_log_path().read_text().splitlines()]
    assert any(event.get("type") == "cache_corruption" for event in events)


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
