from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._elevation_native import run_cubed_sphere_elevation
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.elevation import ElevationConfig
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    reg = registry()
    reg.clear()
    for module_name in ("geometry", "tectonics", "world_age", "geology", "elevation"):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield
    reg.clear()


def _config(
    tmp_path: Path, run_id: str, *, seed: int = 42, face_resolution: int = 32
) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": face_resolution}],
            "rng_seed": seed,
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 16,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 4,
                },
                "world_age": {"world_age": 4.1},
            },
        }
    )


def _array(result, name: str) -> np.ndarray:
    return np.asarray(result.artifact_records[name].value.array())


def _cross_face_gradient_ratio(values: np.ndarray, neighbors: np.ndarray) -> float:
    flat = values.reshape(-1)
    flat_neighbors = neighbors.reshape(-1, 4)
    face_size = values.shape[1] * values.shape[2]
    interior: list[float] = []
    cross_face: list[float] = []
    for source in range(flat.size):
        source_face = source // face_size
        for target_value in flat_neighbors[source]:
            target = int(target_value)
            if source >= target:
                continue
            delta = abs(float(flat[source] - flat[target]))
            if target // face_size == source_face:
                interior.append(delta)
            else:
                cross_face.append(delta)
    return float(np.mean(cross_face)) / max(float(np.mean(interior)), 1e-6)


def test_elevation_outputs_causal_components_and_visuals(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "elevation-basic"), generate_visuals=True)
    results = engine.run(["elevation"])
    elevation = results["elevation"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)

    field_names = (
        "CrustalElevationM",
        "OrogenicElevationM",
        "BasinDepressionM",
        "BedrockElevationM",
        "TerrainReliefM",
        "ElevationConfidence",
    )
    fields = {name: _array(elevation, name) for name in field_names}
    for field in fields.values():
        assert field.shape == grid.face_shape
        assert field.dtype == np.float32
        assert np.all(np.isfinite(field))

    assert np.all(fields["OrogenicElevationM"] >= 0.0)
    assert np.all(fields["BasinDepressionM"] >= 0.0)
    assert np.all(fields["TerrainReliefM"] > 0.0)
    assert np.all((fields["ElevationConfidence"] >= 0.0) & (fields["ElevationConfidence"] <= 1.0))

    ocean = _array(results["world_age"], "BaseOceanMask") >= 0.5
    bedrock = fields["BedrockElevationM"]
    assert float(np.mean(bedrock[~ocean])) > float(np.mean(bedrock[ocean])) + 2500.0
    assert float(np.percentile(bedrock[ocean], 90)) < 0.0
    assert float(np.std(bedrock[~ocean])) > 200.0
    assert float(np.max(fields["OrogenicElevationM"])) > 500.0
    assert float(np.max(fields["BasinDepressionM"])) > 500.0
    assert _cross_face_gradient_ratio(bedrock, grid.neighbor_indices) < 2.5

    regimes = _array(results["geology"], "BoundaryRegime").reshape(-1, 4)
    segment_ids = _array(results["geology"], "BoundarySegmentID").reshape(-1, 4)
    neighbors = grid.neighbor_indices.reshape(-1, 4)
    active_spine = np.any(np.isin(regimes, [2, 3, 4, 5, 6]), axis=1)
    corridor = active_spine.copy()
    for _ in range(2):
        corridor |= np.any(corridor[neighbors], axis=1)
    corridor_flanks = corridor & ~active_spine
    orogenic_flat = fields["OrogenicElevationM"].reshape(-1)
    basin_flat = fields["BasinDepressionM"].reshape(-1)
    assert float(np.mean(orogenic_flat[active_spine])) < 2.5 * float(
        np.mean(orogenic_flat[corridor_flanks])
    )
    assert float(np.mean(basin_flat[active_spine])) < 3.0 * float(
        np.mean(basin_flat[corridor_flanks])
    )

    segment_variation: list[float] = []
    combined_relief = orogenic_flat + basin_flat
    for segment_id in np.unique(segment_ids[segment_ids >= 0]):
        cells = np.flatnonzero(np.any(segment_ids == segment_id, axis=1))
        values = combined_relief[cells]
        if cells.size >= 6 and float(np.mean(values)) > 100.0:
            segment_variation.append(float(np.std(values) / np.mean(values)))
    assert segment_variation
    assert float(np.median(segment_variation)) > 0.20

    metadata = elevation.artifact_records["ElevationMetadata"].value
    assert metadata["model"] == "causal_pre_erosion_components_v3_mean_cap_relief_peaks"
    assert metadata["continental_mean_cap_m"] == 3500.0
    assert metadata["orogenic_mean_cap_m"] == 3000.0
    assert metadata["history_semantics"] == "initial_morphology_not_eroded_present_day"
    assert metadata["continental_mean_m"] > metadata["oceanic_mean_m"]
    assert metadata["elevation_min_m"] == pytest.approx(float(np.min(bedrock)), abs=1e-3)
    assert metadata["elevation_max_m"] == pytest.approx(float(np.max(bedrock)), abs=1e-3)
    hotspot_events = results["world_age"].artifact_records["HotspotEvents"].value
    assert metadata["hotspot_event_count"] == hotspot_events.num_rows

    visual_dir = engine.context.config.run_visual_dir() / "elevation"
    assert (visual_dir / "bedrock_elevation.png").is_file()
    assert (visual_dir / "orogenic_morphology.png").is_file()
    assert (visual_dir / "orogenic_elevation.png").is_file()
    assert (visual_dir / "basin_depression.png").is_file()


def test_elevation_is_deterministic_and_seed_sensitive(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "elevation-first", seed=17)).run(["elevation"])[
        "elevation"
    ]
    second = ExecutionEngine(_config(tmp_path, "elevation-second", seed=17)).run(["elevation"])[
        "elevation"
    ]
    alternate = ExecutionEngine(_config(tmp_path, "elevation-alternate", seed=18)).run(
        ["elevation"]
    )["elevation"]
    for name in (
        "CrustalElevationM",
        "OrogenicElevationM",
        "BasinDepressionM",
        "BedrockElevationM",
        "TerrainReliefM",
        "ElevationConfidence",
    ):
        np.testing.assert_array_equal(_array(first, name), _array(second, name))
    assert not np.array_equal(
        _array(first, "BedrockElevationM"), _array(alternate, "BedrockElevationM")
    )


def _ffi_arguments(face_resolution: int = 8) -> tuple[dict[str, object], int, int]:
    grid = CubedSphereGrid.create(face_resolution)
    shape = grid.face_shape

    def scalar(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    plate = np.zeros((*shape, 7), dtype=np.float32)
    plate[..., 1] = 1.0
    plate[..., 2] = 36.0
    plate[..., 3] = 2.75

    source = 0
    target = int(grid.neighbor_indices.reshape(-1, 4)[source, 0])
    plate.reshape(-1, 7)[source, 0] = 0.0
    plate.reshape(-1, 7)[source, 1] = 0.0
    plate.reshape(-1, 7)[source, 2] = 7.0
    plate.reshape(-1, 7)[source, 3] = 3.2
    plate.reshape(-1, 7)[target, 0] = 1.0

    proto_ocean = scalar(0.0)
    proto_ocean.reshape(-1)[source] = 1.0
    crust_age = scalar(2.0)
    crust_age.reshape(-1)[source] = 0.2
    crust_age.reshape(-1)[target] = 0.05
    regimes = np.zeros((*shape, 4), dtype=np.uint8)
    boundary_confidence = np.zeros((*shape, 4), dtype=np.float32)
    flat_neighbors = grid.neighbor_indices.reshape(-1, 4)
    reverse_slot = int(np.flatnonzero(flat_neighbors[target] == source)[0])
    regimes.reshape(-1, 4)[source, 0] = 3
    regimes.reshape(-1, 4)[target, reverse_slot] = 3
    boundary_confidence.reshape(-1, 4)[source, 0] = 1.0
    boundary_confidence.reshape(-1, 4)[target, reverse_slot] = 1.0

    arguments: dict[str, object] = {
        "seed": 42,
        "collision_height_m": 5200.0,
        "arc_height_m": 2800.0,
        "ridge_height_m": 1800.0,
        "trench_depth_m": 3600.0,
        "rift_depth_m": 950.0,
        "areas": grid.cell_areas,
        "neighbors": grid.neighbor_indices,
        "plate_field": plate,
        "crust_thickness": np.where(proto_ocean > 0.5, 7.0, 36.0).astype(np.float32),
        "isostasy": np.where(proto_ocean > 0.5, -1.4, 2.5).astype(np.float32),
        "uplift": scalar(0.4),
        "subsidence": scalar(0.1),
        "compression": scalar(0.7),
        "extension": scalar(0.0),
        "shear": scalar(0.0),
        "stiffness": scalar(0.6),
        "proto_ocean": proto_ocean,
        "hotspot": scalar(0.0),
        "crust_age": crust_age,
        "rock_strength": scalar(0.7),
        "accommodation": scalar(0.1),
        "province_confidence": scalar(0.8),
        "boundary_regime": regimes,
        "boundary_confidence": boundary_confidence,
        "crustal_out": np.empty(shape, dtype=np.float32),
        "orogenic_out": np.empty(shape, dtype=np.float32),
        "basin_out": np.empty(shape, dtype=np.float32),
        "bedrock_out": np.empty(shape, dtype=np.float32),
        "relief_out": np.empty(shape, dtype=np.float32),
        "confidence_out": np.empty(shape, dtype=np.float32),
    }
    return arguments, source, target


def test_subduction_polarity_places_trench_and_arc_on_opposite_sides():
    arguments, source, target = _ffi_arguments()
    run_cubed_sphere_elevation(**arguments)
    basin = arguments["basin_out"].reshape(-1)
    orogenic = arguments["orogenic_out"].reshape(-1)
    assert basin[source] > basin[target] + 500.0
    assert orogenic[target] > orogenic[source] + 100.0


def test_elevation_ffi_rejects_overlap_and_shape():
    arguments, _, _ = _ffi_arguments()
    arguments["bedrock_out"] = arguments["crustal_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_cubed_sphere_elevation(**arguments)

    arguments, _, _ = _ffi_arguments()
    arguments["crust_age"] = np.empty((6, 8, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        run_cubed_sphere_elevation(**arguments)


def test_elevation_config_and_cli_require_explicit_valid_controls(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown elevation controls"):
        ElevationConfig.from_mapping({"plateau_height": 9000})
    with pytest.raises(ValueError, match=r"in \[0, 20000\]"):
        ElevationConfig.from_mapping({"collision_height_m": -1})

    rectangular = PipelineConfig.from_mapping(
        {
            "topology": "sphere",
            "resolutions": [{"height": 16, "width": 32}],
            "run_id": "rectangular-elevation",
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
        }
    )
    with pytest.raises(NotImplementedError, match="requires topology: cubed_sphere"):
        ExecutionEngine(rectangular).run(["elevation"])
    with pytest.raises(SystemExit) as error:
        pipeline_tools_main(["--stage", "elevation"])
    assert error.value.code == 2
