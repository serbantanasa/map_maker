from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._tectonics_native import (
    PLATE_FIELD_COMPONENTS,
    SPHERICAL_PLATE_FIELD_COMPONENTS,
    run_cubed_sphere_tectonics,
    run_tectonics_kernels,
)
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    reg = registry()
    reg.clear()
    geometry = importlib.import_module("map_maker.pipeline.stages.geometry")
    module = importlib.import_module("map_maker.pipeline.stages.tectonics")
    importlib.reload(geometry)
    importlib.reload(module)
    yield
    reg.clear()


def _make_config(tmp_path: Path, run_id: str, overrides: dict | None = None) -> PipelineConfig:
    base = {
        "topology": "sphere",
        "resolutions": [{"height": 64, "width": 128}],
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "log_dir": str(tmp_path / "logs"),
        "run_id": run_id,
    }
    if overrides:
        base["stage_overrides"] = overrides
    return PipelineConfig.from_mapping(base)


def test_tectonics_outputs_and_shapes(tmp_path: Path):
    config = _make_config(
        tmp_path,
        "tectonics-test",
        overrides={
            "tectonics": {
                "num_plates": 12,
                "velocity_scale": 0.8,
                "hotspot_density": 0.05,
            },
        },
    )
    engine = ExecutionEngine(config, generate_visuals=True)
    results = engine.run(["tectonics"])
    tectonics_result = results["tectonics"]

    plate_handle = tectonics_result.artifact_records["PlateField"].value
    assert plate_handle is not None
    plate_array = np.array(plate_handle.array(), copy=False)
    assert plate_array.shape == (64, 128, PLATE_FIELD_COMPONENTS)
    assert plate_array.dtype == np.float32

    conv = tectonics_result.artifact_records["BoundaryConvergence"].value
    assert conv is not None
    conv_array = np.array(conv.array(), copy=False)
    assert conv_array.shape == (64, 128)
    assert conv_array.dtype == np.float32

    hotspot = tectonics_result.artifact_records["HotspotMap"].value
    assert hotspot is not None
    hotspot_array = np.array(hotspot.array(), copy=False)
    assert hotspot_array.min() >= 0.0
    assert hotspot_array.max() <= 1.0 + 1e-6

    subduction = tectonics_result.artifact_records["BoundarySubduction"].value
    assert subduction is not None
    subduction_array = np.array(subduction.array(), copy=False)
    assert subduction_array.shape == (64, 128)
    assert np.all((subduction_array >= 0.0) & (subduction_array <= 1.0 + 1e-6))

    metadata_record = tectonics_result.artifact_records["TectonicsMetadata"]
    meta_dict = metadata_record.value
    assert isinstance(meta_dict, dict)
    assert meta_dict.get("num_plates") == 12
    assert "velocity_mean" in meta_dict
    assert "convergence_sum" in meta_dict
    assert "hotspot_count" in meta_dict
    assert "continental_fraction" in meta_dict

    visuals_dir = config.run_visual_dir() / "tectonics"
    assert (visuals_dir / "plates.png").exists()


def test_tectonics_cache_hit(tmp_path: Path):
    config = _make_config(tmp_path, "tectonics-cache")
    engine1 = ExecutionEngine(config)
    engine1.run(["tectonics"])
    engine2 = ExecutionEngine(config)
    result = engine2.run(["tectonics"])["tectonics"]
    assert result.stats is not None
    assert result.stats.cache_hit


def test_tectonics_determinism_and_ratios(tmp_path: Path):
    overrides = {
        "tectonics": {
            "num_plates": 16,
            "continental_fraction": 0.4,
            "lloyd_iterations": 4,
            "time_steps": 12,
            "time_step": 0.6,
            "velocity_scale": 1.1,
        }
    }
    config = _make_config(tmp_path, "tectonics-determinism", overrides=overrides)
    engine1 = ExecutionEngine(config)
    res1 = engine1.run(["tectonics"])["tectonics"]
    engine2 = ExecutionEngine(config)
    res2 = engine2.run(["tectonics"])["tectonics"]

    plate1 = np.array(res1.artifact_records["PlateField"].value.array(), copy=True)
    plate2 = np.array(res2.artifact_records["PlateField"].value.array(), copy=True)
    assert np.allclose(plate1, plate2)

    conv1 = np.array(res1.artifact_records["BoundaryConvergence"].value.array(), copy=True)
    conv2 = np.array(res2.artifact_records["BoundaryConvergence"].value.array(), copy=True)
    assert np.allclose(conv1, conv2)

    continental_mask = plate1[..., 1]
    actual_fraction = float(continental_mask.mean())
    assert abs(actual_fraction - 0.4) <= 0.15

    plate_ids = plate1[..., 0].astype(np.int32)
    divergence = np.array(res1.artifact_records["BoundaryDivergence"].value.array(), copy=False)
    diff_mask_east = plate_ids[:, :-1] != plate_ids[:, 1:]
    if diff_mask_east.any():
        east_diff = np.abs(conv1[:, :-1][diff_mask_east] - divergence[:, 1:][diff_mask_east])
        assert east_diff.mean() < 0.08
    diff_mask_south = plate_ids[:-1, :] != plate_ids[1:, :]
    if diff_mask_south.any():
        south_diff = np.abs(conv1[:-1, :][diff_mask_south] - divergence[1:, :][diff_mask_south])
        assert south_diff.mean() < 0.12

    subduction = np.array(res1.artifact_records["BoundarySubduction"].value.array(), copy=False)
    assert np.all((subduction >= 0.0) & (subduction <= 1.0 + 1e-6))


@pytest.mark.parametrize(("rng_seed", "plate_count"), [(42, 16), (40, 24), (7, 20)])
def test_cubed_sphere_tectonics_is_seam_free_tangent_and_connected(
    tmp_path: Path, rng_seed: int, plate_count: int
):
    config = PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 32}],
            "rng_seed": rng_seed,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "run_id": "cubed-tectonics",
            "stage_overrides": {
                "tectonics": {
                    "num_plates": plate_count,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 4,
                }
            },
        }
    )
    engine = ExecutionEngine(config, generate_visuals=True)
    result = engine.run(["tectonics"])["tectonics"]
    assert isinstance(engine.context.topology, CubedSphereGrid)
    grid = engine.context.topology

    plate = np.asarray(result.artifact_records["PlateField"].value.array())
    convergence = np.asarray(result.artifact_records["BoundaryConvergence"].value.array())
    metadata = result.artifact_records["TectonicsMetadata"].value
    assert plate.shape == (*grid.face_shape, SPHERICAL_PLATE_FIELD_COMPONENTS)
    assert convergence.shape == grid.face_shape
    assert np.unique(plate[..., 0]).size == plate_count
    assert metadata["velocity_basis"] == "global_xyz_tangent"
    assert metadata["kinematic_model"] == "spherical_angular_velocity_v2"

    velocity = plate[..., 4:7]
    tangent_error = np.abs(np.sum(velocity * grid.xyz, axis=-1))
    assert float(np.max(tangent_error)) < 1e-6
    continental_fraction = float(np.sum(plate[..., 1] * grid.cell_areas) / np.sum(grid.cell_areas))
    assert abs(continental_fraction - 0.35) < 0.002

    plate_ids = plate[..., 0].astype(np.int32).reshape(-1)
    neighbors = grid.neighbor_indices.reshape(-1, 4)
    for plate_id in range(plate_count):
        members = np.flatnonzero(plate_ids == plate_id)
        assert members.size > 0
        visited = {int(members[0])}
        pending = [int(members[0])]
        while pending:
            cell = pending.pop()
            for neighbor in neighbors[cell]:
                neighbor = int(neighbor)
                if plate_ids[neighbor] == plate_id and neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        assert len(visited) == members.size

    continental = plate[..., 1].astype(bool).reshape(-1)
    unvisited_land = set(np.flatnonzero(continental).tolist())
    continent_areas = []
    flat_areas = grid.cell_areas.reshape(-1)
    while unvisited_land:
        start = unvisited_land.pop()
        pending = [start]
        component_area = 0.0
        while pending:
            cell = pending.pop()
            component_area += float(flat_areas[cell])
            for neighbor in neighbors[cell]:
                neighbor = int(neighbor)
                if neighbor in unvisited_land:
                    unvisited_land.remove(neighbor)
                    pending.append(neighbor)
        continent_areas.append(component_area)
    total_area = float(np.sum(flat_areas))
    significant_continents = [area for area in continent_areas if area / total_area >= 0.001]
    assert 3 <= len(significant_continents) <= 16
    assert max(continent_areas) / sum(continent_areas) < 0.55

    face_edges = np.zeros(grid.face_shape, dtype=bool)
    face_edges[:, (0, -1), :] = True
    face_edges[:, :, (0, -1)] = True
    edge_mean = float(np.mean(convergence[face_edges]))
    interior_mean = float(np.mean(convergence[~face_edges]))
    assert 0.35 < edge_mean / interior_mean < 2.5

    flat_xyz = grid.xyz.reshape(-1, 3).astype(np.float64)
    flat_velocity = velocity.reshape(-1, 3).astype(np.float64)
    flat_convergence = convergence.reshape(-1)
    flat_divergence = np.asarray(
        result.artifact_records["BoundaryDivergence"].value.array()
    ).reshape(-1)
    source = np.repeat(np.arange(grid.cell_count), 4)
    target = neighbors.reshape(-1)
    unlike = (plate_ids[source] != plate_ids[target]) & (source < target)
    source = source[unlike]
    target = target[unlike]
    cosine = np.sum(flat_xyz[source] * flat_xyz[target], axis=1)
    first_tangent = flat_xyz[target] - cosine[:, None] * flat_xyz[source]
    second_tangent = flat_xyz[source] - cosine[:, None] * flat_xyz[target]
    first_tangent /= np.linalg.norm(first_tangent, axis=1)[:, None]
    second_tangent /= np.linalg.norm(second_tangent, axis=1)[:, None]
    closing = np.sum(flat_velocity[source] * first_tangent, axis=1) + np.sum(
        flat_velocity[target] * second_tangent, axis=1
    )
    convergence_on_edge = (flat_convergence[source] + flat_convergence[target]) * 0.5
    divergence_on_edge = (flat_divergence[source] + flat_divergence[target]) * 0.5
    strongly_closing = closing > np.percentile(closing, 70)
    strongly_opening = closing < np.percentile(closing, 30)
    assert float(np.mean(convergence_on_edge[strongly_closing])) > float(
        np.mean(divergence_on_edge[strongly_closing])
    )
    assert float(np.mean(divergence_on_edge[strongly_opening])) > float(
        np.mean(convergence_on_edge[strongly_opening])
    )
    face_size = grid.face_resolution * grid.face_resolution
    cross_face = source // face_size != target // face_size
    assert int(np.count_nonzero(cross_face)) > 0
    cross_closing = closing[cross_face]
    cross_response = convergence_on_edge[cross_face] - divergence_on_edge[cross_face]
    nonzero_cross = np.abs(cross_closing) > np.percentile(np.abs(cross_closing), 35)
    sign_agreement = np.mean(
        np.sign(cross_response[nonzero_cross]) == np.sign(cross_closing[nonzero_cross])
    )
    assert float(sign_agreement) > 0.55
    assert (config.run_visual_dir() / "tectonics" / "plates.png").is_file()


def test_cubed_sphere_tectonics_cli(tmp_path: Path):
    output_dir = tmp_path / "out"
    assert (
        pipeline_tools_main(
            [
                "--stage",
                "tectonics",
                "--topology",
                "cubed_sphere",
                "--face-resolution",
                "16",
                "--tectonics-plates",
                "8",
                "--run-id",
                "cubed-cli",
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    assert (output_dir / "cubed-cli" / "visuals" / "tectonics" / "plates.png").is_file()
    assert (output_dir / "cubed-cli" / "visuals" / "tectonics" / "crust_provinces.png").is_file()
    with pytest.raises(SystemExit) as error:
        pipeline_tools_main(
            [
                "--stage",
                "tectonics",
                "--topology",
                "cubed_sphere",
                "--face-resolution",
                "16",
                "--tectonics-steps",
                "4",
            ]
        )
    assert error.value.code == 2


def test_cubed_sphere_ffi_rejects_overlapping_and_unaligned_outputs():
    grid = CubedSphereGrid.create(4)
    shape = grid.face_shape
    plate = np.empty((*shape, SPHERICAL_PLATE_FIELD_COMPONENTS), dtype=np.float32)
    shared = np.empty(shape, dtype=np.float32)
    scalar = [np.empty(shape, dtype=np.float32) for _ in range(4)]
    arguments = {
        "xyz": grid.xyz,
        "areas": grid.cell_areas,
        "neighbors": grid.neighbor_indices,
        "seed": 3,
        "num_plates": 8,
        "continental_fraction": 0.35,
        "velocity_scale": 1.0,
        "drift_bias": 0.1,
        "hotspot_density": 0.02,
        "subduction_bias": 0.5,
        "lloyd_iterations": 2,
        "plate_field": plate,
        "convergence_field": shared,
        "divergence_field": shared,
        "shear_field": scalar[0],
        "subduction_field": scalar[1],
        "hotspot_field": scalar[2],
    }
    with pytest.raises(ValueError, match="must not overlap"):
        run_cubed_sphere_tectonics(**arguments)

    arguments["num_plates"] = grid.cell_count // 4 + 1
    arguments["divergence_field"] = scalar[3]
    with pytest.raises(ValueError, match="cell_count // 4"):
        run_cubed_sphere_tectonics(**arguments)
    arguments["num_plates"] = 8

    raw = bytearray(int(np.prod(shape)) * np.dtype(np.float32).itemsize + 1)
    unaligned = np.frombuffer(raw, dtype=np.float32, offset=1).reshape(shape)
    arguments["divergence_field"] = scalar[3]
    arguments["convergence_field"] = unaligned
    with pytest.raises(ValueError, match="must be aligned"):
        run_cubed_sphere_tectonics(**arguments)


def test_cubed_sphere_rejects_ineffective_rectangular_controls(tmp_path: Path):
    config = PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 8}],
            "run_id": "invalid-controls",
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {"tectonics": {"num_plates": 6, "time_steps": 12}},
        }
    )
    with pytest.raises(ValueError, match="does not use: time_steps"):
        ExecutionEngine(config).run(["tectonics"])


def test_rectangular_ffi_rejects_undersized_outputs():
    height, width = 8, 16
    scalar = np.empty((height, width), dtype=np.float32)
    arguments = {
        "height": height,
        "width": width,
        "seed": 1,
        "num_plates": 4,
        "continental_fraction": 0.35,
        "velocity_scale": 1.0,
        "drift_bias": 0.1,
        "hotspot_density": 0.02,
        "subduction_bias": 0.5,
        "lloyd_iterations": 2,
        "time_steps": 2,
        "time_step": 0.5,
        "wrap_x": True,
        "wrap_y": False,
        "plate_field": np.empty((height, width - 1, PLATE_FIELD_COMPONENTS), dtype=np.float32),
        "convergence_field": scalar.copy(),
        "divergence_field": scalar.copy(),
        "shear_field": scalar.copy(),
        "subduction_field": scalar.copy(),
        "hotspot_field": scalar.copy(),
    }
    with pytest.raises(ValueError, match="plate_field must have shape"):
        run_tectonics_kernels(**arguments)
