from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._geology_native import (
    BOUNDARY_REGIMES,
    PROVINCE_CLASSES,
    run_cubed_sphere_geology,
)
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    reg = registry()
    reg.clear()
    for module_name in ("geometry", "tectonics", "world_age", "geology"):
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


def _assert_labels_connected(labels: np.ndarray, neighbors: np.ndarray, valid: np.ndarray) -> None:
    flat_labels = labels.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_neighbors = neighbors.reshape(-1, 4)
    for label in np.unique(flat_labels[flat_valid]):
        members = np.flatnonzero(flat_valid & (flat_labels == label))
        seen = {int(members[0])}
        stack = [int(members[0])]
        while stack:
            cell = stack.pop()
            for adjacent in flat_neighbors[cell]:
                neighbor = int(adjacent)
                if flat_valid[neighbor] and flat_labels[neighbor] == label and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        assert len(seen) == len(members), f"label {label} is disconnected"


def _assert_edge_segments_connected(segment_ids: np.ndarray, neighbors: np.ndarray) -> None:
    flat_segments = segment_ids.reshape(-1, 4)
    flat_neighbors = neighbors.reshape(-1, 4)
    edges: list[tuple[int, int, int]] = []
    incident: dict[int, list[int]] = {}
    for source in range(flat_segments.shape[0]):
        for slot, target_value in enumerate(flat_neighbors[source]):
            target = int(target_value)
            segment_id = int(flat_segments[source, slot])
            if segment_id < 0 or source >= target:
                continue
            edge_index = len(edges)
            edges.append((source, target, segment_id))
            incident.setdefault(source, []).append(edge_index)
            incident.setdefault(target, []).append(edge_index)
    edge_segments = np.array([edge[2] for edge in edges], dtype=np.int32)
    for segment_id in np.unique(edge_segments):
        members = np.flatnonzero(edge_segments == segment_id)
        seen = {int(members[0])}
        stack = [int(members[0])]
        while stack:
            edge_index = stack.pop()
            source, target, _ = edges[edge_index]
            for cell in (source, target):
                nearby_cells = [cell, *(int(value) for value in flat_neighbors[cell])]
                for nearby in nearby_cells:
                    for adjacent in incident.get(nearby, []):
                        if edge_segments[adjacent] == segment_id and adjacent not in seen:
                            seen.add(adjacent)
                            stack.append(adjacent)
        assert len(seen) == len(members), f"edge segment {segment_id} is disconnected"


def test_geology_outputs_catalogs_and_global_connectivity(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "geology-basic"), generate_visuals=True)
    results = engine.run(["geology"])
    geology = results["geology"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)

    expected_dtypes = {
        "GeologicalProvinceID": np.int32,
        "GeologicalProvinceClass": np.uint8,
        "CrustAgeGa": np.float32,
        "RockStrength": np.float32,
        "SedimentAccommodation": np.float32,
        "ProvinceConfidence": np.float32,
        "BoundarySegmentID": np.int32,
        "BoundaryRegime": np.uint8,
        "BoundaryConfidence": np.float32,
    }
    arrays = {name: _array(geology, name) for name in expected_dtypes}
    for name, array in arrays.items():
        expected_shape = (*grid.face_shape, 4) if name.startswith("Boundary") else grid.face_shape
        assert array.shape == expected_shape
        assert array.dtype == expected_dtypes[name]
        assert np.all(np.isfinite(array))

    province_ids = arrays["GeologicalProvinceID"]
    province_classes = arrays["GeologicalProvinceClass"]
    assert set(np.unique(province_classes)).issubset(PROVINCE_CLASSES)
    _assert_labels_connected(
        province_ids, grid.neighbor_indices, np.ones(grid.face_shape, dtype=bool)
    )

    province_catalog = geology.artifact_records["GeologicalProvinceCatalog"].value
    assert isinstance(province_catalog, pa.Table)
    np.testing.assert_array_equal(
        np.sort(province_catalog["province_id"].to_numpy()), np.arange(province_catalog.num_rows)
    )
    assert int(np.sum(province_catalog["cell_count"].to_numpy())) == grid.cell_count
    assert float(np.sum(province_catalog["area_steradians"].to_numpy())) == pytest.approx(
        4.0 * math.pi, abs=1e-8
    )
    assert set(province_catalog["class_name"].to_pylist()).issubset(PROVINCE_CLASSES.values())
    assert np.any(province_catalog["parent_plate_id"].to_numpy() < 0)

    faces = np.indices(grid.face_shape)[0].reshape(-1)
    flat_provinces = province_ids.reshape(-1)
    assert any(
        np.unique(faces[flat_provinces == province_id]).size > 1
        for province_id in np.unique(flat_provinces)
    )

    plate = _array(results["tectonics"], "PlateField")
    flat_plate_ids = plate[..., 0].astype(np.int32).reshape(-1)
    flat_neighbors = grid.neighbor_indices.reshape(-1, 4)
    expected_boundary = flat_plate_ids[:, None] != flat_plate_ids[flat_neighbors]
    segment_ids = arrays["BoundarySegmentID"]
    regimes = arrays["BoundaryRegime"]
    boundary = segment_ids >= 0
    np.testing.assert_array_equal(boundary.reshape(-1, 4), expected_boundary)
    assert np.all(regimes[~boundary] == 0)
    assert set(np.unique(regimes[boundary])).issubset(BOUNDARY_REGIMES)
    _assert_edge_segments_connected(segment_ids, grid.neighbor_indices)
    flat_segment_ids = segment_ids.reshape(-1, 4)
    flat_regimes = regimes.reshape(-1, 4)
    flat_confidence = arrays["BoundaryConfidence"].reshape(-1, 4)
    for source in range(grid.cell_count):
        for slot, target_value in enumerate(flat_neighbors[source]):
            target = int(target_value)
            reverse_slot = int(np.flatnonzero(flat_neighbors[target] == source)[0])
            assert flat_segment_ids[source, slot] == flat_segment_ids[target, reverse_slot]
            assert flat_regimes[source, slot] == flat_regimes[target, reverse_slot]
            assert flat_confidence[source, slot] == flat_confidence[target, reverse_slot]

    segment_catalog = geology.artifact_records["BoundarySegmentCatalog"].value
    assert isinstance(segment_catalog, pa.Table)
    np.testing.assert_array_equal(
        np.sort(segment_catalog["segment_id"].to_numpy()), np.arange(segment_catalog.num_rows)
    )
    assert int(np.sum(segment_catalog["edge_count"].to_numpy())) * 2 == int(
        np.count_nonzero(boundary)
    )
    assert np.all(segment_catalog["plate_a"].to_numpy() < segment_catalog["plate_b"].to_numpy())
    assert set(segment_catalog["regime_name"].to_pylist()).issubset(BOUNDARY_REGIMES.values())

    oceanic = plate[..., 1] < 0.5
    crust_age = arrays["CrustAgeGa"]
    assert float(np.max(crust_age[oceanic])) <= 0.25 + 1e-6
    assert float(np.mean(crust_age[~oceanic])) > float(np.mean(crust_age[oceanic]))
    for name in ("RockStrength", "SedimentAccommodation", "ProvinceConfidence"):
        assert np.all((arrays[name] >= 0.0) & (arrays[name] <= 1.0))

    metadata = geology.artifact_records["GeologyMetadata"].value
    assert metadata["history_semantics"] == "initialization_not_simulated_deep_time"
    assert metadata["province_count"] == province_catalog.num_rows
    assert metadata["boundary_segment_count"] == segment_catalog.num_rows
    visual_dir = engine.context.config.run_visual_dir() / "geology"
    assert (visual_dir / "geological_provinces.png").is_file()
    assert (visual_dir / "boundary_regimes.png").is_file()
    assert (visual_dir / "crust_age.png").is_file()


def test_geology_is_deterministic(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "geology-first")).run(["geology"])["geology"]
    second = ExecutionEngine(_config(tmp_path, "geology-second")).run(["geology"])["geology"]
    for name in (
        "GeologicalProvinceID",
        "GeologicalProvinceClass",
        "CrustAgeGa",
        "RockStrength",
        "SedimentAccommodation",
        "ProvinceConfidence",
        "BoundarySegmentID",
        "BoundaryRegime",
        "BoundaryConfidence",
    ):
        np.testing.assert_array_equal(_array(first, name), _array(second, name))
    assert first.artifact_records["GeologicalProvinceCatalog"].value.equals(
        second.artifact_records["GeologicalProvinceCatalog"].value
    )
    assert first.artifact_records["BoundarySegmentCatalog"].value.equals(
        second.artifact_records["BoundarySegmentCatalog"].value
    )


def test_geology_merges_sub_resolution_fragments_and_defers_volcanic_class(tmp_path: Path):
    geology = ExecutionEngine(
        _config(tmp_path, "geology-fragments", seed=11, face_resolution=64)
    ).run(["geology"])["geology"]
    catalog = geology.artifact_records["GeologicalProvinceCatalog"].value
    stable = np.isin(catalog["class_code"].to_numpy(), [1, 2, 3, 11])
    minimum_area = np.where(stable, 4.0 * math.pi / 8192.0, 4.0 * math.pi / 16384.0)
    undersized = np.flatnonzero(catalog["area_steradians"].to_numpy() < minimum_area)
    province_ids = _array(geology, "GeologicalProvinceID").reshape(-1)
    confidence = _array(geology, "ProvinceConfidence").reshape(-1)
    dataset_root = Path(geology.artifact_records["GeologicalProvinceID"].dataset_path).parents[1]
    crust = np.load(dataset_root / "world_age" / "BaseOceanMask.npy").reshape(-1) >= 0.5
    neighbors = np.load(dataset_root / "geometry" / "NeighborsD4.npy").reshape(-1, 4)
    for row in undersized:
        province_id = int(catalog["province_id"][int(row)].as_py())
        cells = np.flatnonzero(province_ids == province_id)
        outside_neighbors = neighbors[cells].reshape(-1)
        same_crust_outside = (crust[outside_neighbors] == crust[cells[0]]) & (
            province_ids[outside_neighbors] != province_id
        )
        assert not np.any(same_crust_outside)
        assert np.all(confidence[cells] <= 0.5)
    assert "volcanic_province" not in catalog["class_name"].to_pylist()


def _ffi_arguments(face_resolution: int = 4) -> dict[str, object]:
    grid = CubedSphereGrid.create(face_resolution)
    shape = grid.face_shape
    plate = np.zeros((*shape, 7), dtype=np.float32)
    plate[..., 0] = np.indices(shape)[0]
    plate[..., 1] = np.indices(shape)[0] < 3
    plate[..., 2] = np.where(plate[..., 1] > 0.5, 35.0, 7.0)
    plate[..., 3] = np.where(plate[..., 1] > 0.5, 2.75, 3.0)
    scalar_inputs = [np.full(shape, 0.1, dtype=np.float32) for _ in range(10)]
    scalar_inputs[-1][...] = plate[..., 1] < 0.5
    return {
        "world_age_ga": 4.1,
        "areas": grid.cell_areas,
        "neighbors": grid.neighbor_indices,
        "plate_field": plate,
        **dict(
            zip(
                (
                    "subduction",
                    "isostasy",
                    "uplift",
                    "subsidence",
                    "compression",
                    "extension",
                    "shear",
                    "margin",
                    "stiffness",
                    "proto_ocean",
                ),
                scalar_inputs,
                strict=True,
            )
        ),
        "province_id_out": np.empty(shape, dtype=np.int32),
        "province_class_out": np.empty(shape, dtype=np.uint8),
        "crust_age_out": np.empty(shape, dtype=np.float32),
        "rock_strength_out": np.empty(shape, dtype=np.float32),
        "accommodation_out": np.empty(shape, dtype=np.float32),
        "province_confidence_out": np.empty(shape, dtype=np.float32),
        "boundary_segment_id_out": np.empty((*shape, 4), dtype=np.int32),
        "boundary_regime_out": np.empty((*shape, 4), dtype=np.uint8),
        "boundary_confidence_out": np.empty((*shape, 4), dtype=np.float32),
    }


def test_geology_ffi_rejects_overlap_alignment_and_shape():
    arguments = _ffi_arguments()
    arguments["rock_strength_out"] = arguments["crust_age_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_cubed_sphere_geology(**arguments)

    arguments = _ffi_arguments()
    shape = arguments["crust_age_out"].shape
    raw = bytearray(int(np.prod(shape)) * np.dtype(np.float32).itemsize + 1)
    arguments["crust_age_out"] = np.frombuffer(raw, dtype=np.float32, offset=1).reshape(shape)
    with pytest.raises(ValueError, match="must be aligned"):
        run_cubed_sphere_geology(**arguments)

    arguments = _ffi_arguments()
    arguments["margin"] = np.empty((6, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        run_cubed_sphere_geology(**arguments)


def test_geology_requires_cubed_sphere_and_cli_config(tmp_path: Path):
    rectangular = PipelineConfig.from_mapping(
        {
            "topology": "sphere",
            "resolutions": [{"height": 16, "width": 32}],
            "run_id": "rectangular-geology",
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
        }
    )
    with pytest.raises(NotImplementedError, match="requires topology: cubed_sphere"):
        ExecutionEngine(rectangular).run(["geology"])
    with pytest.raises(SystemExit) as error:
        pipeline_tools_main(["--stage", "geology"])
    assert error.value.code == 2
