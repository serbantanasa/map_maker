"""earth_relief_v1 multi-seed morphology and elevation-contract KPIs."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.cubed_sphere import CubedSphereGrid


# Canonical Earthlike face-128 contract after relief-first elevation (v4).
EARTH_RELIEF_V1 = {
    "continental_mean_cap_m": 3_500.0,
    "orogenic_mean_cap_m": 3_000.0,
    "continental_bedrock_floor_m": -650.0,
    "peak_proxy_relief_coefficient": 0.45,
    "max_land_mean_elevation_m": 3_500.0,
    "max_land_mean_above_4000_area_fraction": 0.001,
    "min_land_relief_p95_m": 800.0,
    "min_land_relief_max_m": 1_800.0,
    "min_land_relief_above_1000_area_fraction": 0.02,
    "min_ocean_depth_spread_p95_minus_p50_m": 150.0,
    "min_ocean_depth_max_m": 5_500.0,
    "max_proto_continental_land_below_floor_area_fraction": 0.0,
}

# Fixed face-64 screen seeds (same family as biosphere ensemble).
FACE64_SEEDS = (42, 101, 202, 303, 404, 505)


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "sea_level",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(tmp_path: Path, seed: int, face_resolution: int) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": face_resolution}],
            "rng_seed": seed,
            "run_id": f"earth-relief-v1-{seed}-f{face_resolution}",
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 24,
                    "continental_fraction": 0.42,
                    "lloyd_iterations": 4,
                },
                "world_age": {"world_age": 4.1},
                "elevation": {
                    "collision_height_m": 5200.0,
                    "arc_height_m": 3000.0,
                    "ridge_height_m": 2200.0,
                    "trench_depth_m": 6200.0,
                    "rift_depth_m": 1100.0,
                },
                "sea_level": {"target_ocean_area_fraction": 0.65},
            },
        }
    )


def _array(result, name: str) -> np.ndarray:
    return np.asarray(result.artifact_records[name].value.array())


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = q * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(values) - 1)
    return float(values[index])


def _evaluate_seed(tmp_path: Path, seed: int, face_resolution: int) -> dict[str, float]:
    engine = ExecutionEngine(
        _config(tmp_path / f"s{seed}", seed, face_resolution), generate_visuals=False
    )
    results = engine.run(["sea_level"])
    elevation = results["elevation"]
    sea = results["sea_level"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)
    areas = grid.cell_areas

    surface = _array(sea, "SurfaceElevationM")
    ocean_mask = _array(sea, "SurfaceOceanMask") >= 0.5
    depth = _array(sea, "OceanDepthM")
    relief = _array(elevation, "TerrainReliefM")
    orogen = _array(elevation, "OrogenicElevationM")
    bedrock = _array(elevation, "BedrockElevationM")
    proto_ocean = _array(results["world_age"], "BaseOceanMask") >= 0.5
    land = ~ocean_mask
    land_area = float(np.sum(areas[land]))
    metadata = elevation.artifact_records["ElevationMetadata"].value
    peak_coeff = float(metadata["peak_proxy_relief_coefficient"])
    peak = surface + peak_coeff * relief

    ocean_depth = depth[ocean_mask].astype(np.float64)
    ocean_weights = areas[ocean_mask]
    land_surface = surface[land].astype(np.float64)
    land_relief = relief[land].astype(np.float64)
    land_weights = areas[land]
    continental_land = land & ~proto_ocean

    return {
        "seed": float(seed),
        "mean_orogeny_scale": float(metadata["mean_orogeny_scale"]),
        "orogen_max_m": float(np.max(orogen)),
        "land_mean_max_m": float(np.max(land_surface)),
        "land_mean_p95_m": _weighted_percentile(land_surface, land_weights, 0.95),
        "land_mean_above_4000_area_fraction": float(
            np.sum(areas[land & (surface > 4_000.0)]) / max(land_area, 1e-30)
        ),
        "land_relief_p95_m": _weighted_percentile(land_relief, land_weights, 0.95),
        "land_relief_max_m": float(np.max(land_relief)),
        "land_relief_above_1000_area_fraction": float(
            np.sum(areas[land & (relief > 1_000.0)]) / max(land_area, 1e-30)
        ),
        "peak_proxy_max_m": float(np.max(peak[land])),
        "peak_proxy_p95_m": _weighted_percentile(peak[land].astype(np.float64), land_weights, 0.95),
        "continental_bedrock_min_m": float(np.min(bedrock[continental_land]))
        if np.any(continental_land)
        else 0.0,
        "ocean_depth_p50_m": _weighted_percentile(ocean_depth, ocean_weights, 0.50),
        "ocean_depth_p95_m": _weighted_percentile(ocean_depth, ocean_weights, 0.95),
        "ocean_depth_max_m": float(np.max(ocean_depth)),
        "ocean_depth_spread_m": _weighted_percentile(ocean_depth, ocean_weights, 0.95)
        - _weighted_percentile(ocean_depth, ocean_weights, 0.50),
        "metadata_mean_scale": float(metadata["mean_orogeny_scale"]),
        "metadata_model": 1.0
        if metadata["model"] == "causal_pre_erosion_components_v4_max_process_relief_peaks"
        else 0.0,
    }


def _assert_contract(metrics: dict[str, float]) -> None:
    c = EARTH_RELIEF_V1
    assert metrics["land_mean_max_m"] <= c["max_land_mean_elevation_m"] + 1e-3
    assert metrics["land_mean_above_4000_area_fraction"] <= c["max_land_mean_above_4000_area_fraction"]
    assert metrics["orogen_max_m"] <= c["orogenic_mean_cap_m"] + 1e-3
    assert metrics["land_relief_p95_m"] >= c["min_land_relief_p95_m"]
    assert metrics["land_relief_max_m"] >= c["min_land_relief_max_m"]
    assert (
        metrics["land_relief_above_1000_area_fraction"]
        >= c["min_land_relief_above_1000_area_fraction"]
    )
    assert metrics["continental_bedrock_min_m"] >= c["continental_bedrock_floor_m"] - 1e-2
    assert metrics["ocean_depth_spread_m"] >= c["min_ocean_depth_spread_p95_minus_p50_m"]
    assert metrics["ocean_depth_max_m"] >= c["min_ocean_depth_max_m"]
    assert metrics["metadata_mean_scale"] == pytest.approx(0.62)
    assert metrics["metadata_model"] == 1.0
    # Peak proxy must exceed mean (relief is doing real work) without sky means.
    assert metrics["peak_proxy_max_m"] > metrics["land_mean_max_m"]
    assert metrics["peak_proxy_max_m"] < 6_000.0


def test_earth_relief_v1_canonical_face128_seed42(tmp_path: Path):
    metrics = _evaluate_seed(tmp_path, seed=42, face_resolution=128)
    _assert_contract(metrics)


def test_earth_relief_v1_face64_ensemble_screen(tmp_path: Path):
    """Multi-seed morphology screen at face 64 (cheaper than full biosphere)."""
    rows = [_evaluate_seed(tmp_path, seed=seed, face_resolution=64) for seed in FACE64_SEEDS]
    for metrics in rows:
        _assert_contract(metrics)
    relief_p95 = [row["land_relief_p95_m"] for row in rows]
    assert min(relief_p95) >= EARTH_RELIEF_V1["min_land_relief_p95_m"] * 0.85
    assert max(row["land_mean_max_m"] for row in rows) <= EARTH_RELIEF_V1["max_land_mean_elevation_m"]
