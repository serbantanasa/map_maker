from __future__ import annotations

from collections import deque
import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.hydrology import HydrologyConfig
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_hydrology_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "planet",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "climate",
        "hydrology",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(tmp_path: Path, run_id: str, *, seed: int = 23) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 20}],
            "rng_seed": seed,
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 14,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 3,
                },
                "world_age": {"world_age": 4.1},
                "climate": {
                    "spinup_years": 10,
                    "moisture_spinup_years": 2,
                    "moisture_steps_per_month_at_face_128": 16,
                },
            },
        }
    )


def _array(result, name: str) -> np.ndarray:
    return np.asarray(result.artifact_records[name].value.array())


def _assert_acyclic(receiver: np.ndarray, ocean: np.ndarray) -> None:
    flat_receiver = receiver.reshape(-1)
    flat_ocean = ocean.reshape(-1)
    indegree = np.zeros(len(flat_receiver), dtype=np.int32)
    land_cells = np.flatnonzero(~flat_ocean)
    for cell in land_cells:
        target = int(flat_receiver[cell])
        if target >= 0 and not flat_ocean[target]:
            indegree[target] += 1
    queue = deque(int(cell) for cell in land_cells if indegree[cell] == 0)
    visited = 0
    while queue:
        cell = queue.popleft()
        visited += 1
        target = int(flat_receiver[cell])
        if target >= 0 and not flat_ocean[target]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    assert visited == len(land_cells)


def test_hydrology_outputs_depression_aware_global_graph_and_catalogs(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "hydrology-basic"), generate_visuals=True)
    results = engine.run(["hydrology"])
    hydrology = results["hydrology"]
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)
    shape = grid.face_shape

    expected_dtypes = {
        "DepressionID": np.int32,
        "LakeID": np.int32,
        "WaterBodyClass": np.uint8,
        "DepressionFillDepthM": np.float32,
        "HydrologicElevationM": np.float32,
        "BreachIncisionM": np.float32,
        "FlowReceiverID": np.int32,
        "FlowDirectionXYZ": np.float32,
        "FlowSlope": np.float32,
        "ContributingAreaKm2": np.float64,
        "MonthlyDischargeM3s": np.float32,
        "MeanDischargeM3s": np.float32,
        "MeanFlowVelocityMps": np.float32,
        "StreamPowerW": np.float32,
        "BasinID": np.int32,
        "FlowSinkType": np.uint8,
        "RiverCorridor": np.float32,
        "FloodplainPotential": np.float32,
    }
    arrays = {name: _array(hydrology, name) for name in expected_dtypes}
    for name, dtype in expected_dtypes.items():
        expected_shape = shape
        if name == "FlowDirectionXYZ":
            expected_shape = (*shape, 3)
        elif name == "MonthlyDischargeM3s":
            expected_shape = (12, *shape)
        assert arrays[name].shape == expected_shape
        assert arrays[name].dtype == dtype
        assert np.all(np.isfinite(arrays[name]))

    ocean = _array(results["world_age"], "BaseOceanMask") >= 0.5
    land = ~ocean
    receiver = arrays["FlowReceiverID"]
    _assert_acyclic(receiver, ocean)
    flat_receiver = receiver.reshape(-1)
    flat_ocean = ocean.reshape(-1)
    flat_sink = arrays["FlowSinkType"].reshape(-1)
    for cell in np.flatnonzero(land.reshape(-1)):
        target = int(flat_receiver[cell])
        if target < 0:
            assert flat_sink[cell] in {2, 3, 4}
        elif flat_ocean[target]:
            assert flat_sink[cell] == 1

    contributing_area = arrays["ContributingAreaKm2"].reshape(-1)
    monthly_discharge = arrays["MonthlyDischargeM3s"].reshape(12, -1)
    for cell in np.flatnonzero(land.reshape(-1)):
        target = int(flat_receiver[cell])
        if target >= 0 and not flat_ocean[target]:
            assert contributing_area[target] + 1e-8 >= contributing_area[cell]
            if arrays["LakeID"].reshape(-1)[target] < 0:
                assert np.all(monthly_discharge[:, target] + 1e-3 >= monthly_discharge[:, cell])

    routed = land & (receiver >= 0)
    direction = arrays["FlowDirectionXYZ"]
    np.testing.assert_allclose(np.linalg.norm(direction[routed], axis=-1), 1.0, atol=2e-5)
    radial_leak = np.sum(direction[routed] * grid.xyz[routed], axis=-1)
    assert float(np.max(np.abs(radial_leak))) < 2e-5
    assert np.all(arrays["FlowSlope"] >= 0.0)
    assert np.all(arrays["MeanDischargeM3s"] >= 0.0)
    assert np.all(arrays["MeanFlowVelocityMps"] >= 0.0)
    assert np.all(arrays["StreamPowerW"] >= 0.0)
    assert np.all(arrays["RiverCorridor"][arrays["LakeID"] >= 0] == 0.0)

    depression_catalog = hydrology.artifact_records["DepressionCatalog"].value
    lake_catalog = hydrology.artifact_records["LakeCatalog"].value
    breach_catalog = hydrology.artifact_records["BreachCatalog"].value
    basin_catalog = hydrology.artifact_records["BasinCatalog"].value
    drainage_graph = hydrology.artifact_records["DrainageGraph"].value
    reaches = hydrology.artifact_records["RiverReachCatalog"].value
    for table in (
        depression_catalog,
        lake_catalog,
        breach_catalog,
        basin_catalog,
        drainage_graph,
        reaches,
    ):
        assert isinstance(table, pa.Table)
    assert depression_catalog.num_rows > 0
    assert lake_catalog.num_rows > 0
    assert basin_catalog.num_rows > 0
    assert drainage_graph.num_rows == int(np.count_nonzero(land))
    assert reaches.num_rows > 0
    assert {
        "cell_id",
        "receiver_id",
        "basin_id",
        "sink_type",
        "depression_id",
        "lake_id",
        "contributing_area_km2",
        "mean_discharge_m3s",
    } == set(drainage_graph.column_names)
    assert {"open_outlet", "outlet_cell", "annual_balance_km3"} <= set(
        depression_catalog.column_names
    )
    required_reach_fields = {
        "reach_id",
        "from_node",
        "to_node",
        "upstream_reach_ids",
        "downstream_reach_id",
        "basin_id",
        "cell_path",
        "polyline_on_cubed_sphere",
        "flow_direction_vector",
        "slope",
        "strahler_order",
        "discharge_mean",
        "discharge_seasonal",
        "velocity_mean",
        "velocity_seasonal",
        "stream_power",
        "channel_width_m",
        "channel_depth_m",
        "valley_width_m",
        "floodplain_width_m",
        "meander_index",
        "braiding_index",
        "incision_m",
        "sediment_load",
        "bed_material",
        "morphology_class",
    }
    assert required_reach_fields <= set(reaches.column_names)
    reach_ids = np.asarray(reaches["reach_id"].combine_chunks())
    downstream = np.asarray(reaches["downstream_reach_id"].combine_chunks())
    np.testing.assert_array_equal(reach_ids, np.arange(len(reach_ids), dtype=np.int32))
    assert np.all((downstream == -1) | ((downstream >= 0) & (downstream < len(reach_ids))))
    for reach_id in reach_ids:
        seen: set[int] = set()
        current = int(reach_id)
        while current >= 0:
            assert current not in seen
            seen.add(current)
            current = int(downstream[current])

    flat_xyz = grid.xyz.reshape(-1, 3)
    for index in range(min(100, reaches.num_rows)):
        cell_path = reaches["cell_path"][index].as_py()
        geometry = np.asarray(reaches["polyline_on_cubed_sphere"][index].as_py())
        assert len(cell_path) >= 2
        assert len(geometry) >= len(cell_path)
        np.testing.assert_allclose(np.linalg.norm(geometry, axis=1), 1.0, atol=2e-5)
        np.testing.assert_allclose(geometry[0], flat_xyz[cell_path[0]], atol=2e-6)
        np.testing.assert_allclose(geometry[-1], flat_xyz[cell_path[-1]], atol=2e-6)

    metadata = hydrology.artifact_records["HydrologyMetadata"].value
    assert metadata["topology_valid"] == 1
    assert metadata["conservation_relative_error"] < 1e-10
    assert metadata["annual_open_water_loss_km3"] >= 0.0
    assert metadata["depression_count"] == depression_catalog.num_rows
    assert metadata["lake_count"] == lake_catalog.num_rows
    assert metadata["basin_count"] == basin_catalog.num_rows
    assert metadata["reach_count"] == reaches.num_rows
    assert metadata["open_lake_count"] > 0
    assert 0.0 < metadata["lake_land_cell_fraction"] < 0.25
    assert 0.0 <= metadata["closed_drainage_land_fraction"] < 1.0

    visual_dir = engine.context.config.run_visual_dir() / "hydrology"
    for filename in (
        "lakes_and_rivers.png",
        "drainage_basins.png",
        "discharge_and_floodplains.png",
        "breach_incision.png",
    ):
        assert (visual_dir / filename).is_file()


def test_hydrology_is_deterministic_and_cacheable(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "hydrology-first")).run(["hydrology"])["hydrology"]
    second = ExecutionEngine(_config(tmp_path, "hydrology-second")).run(["hydrology"])["hydrology"]
    for name in (
        "DepressionID",
        "LakeID",
        "WaterBodyClass",
        "HydrologicElevationM",
        "FlowReceiverID",
        "ContributingAreaKm2",
        "MonthlyDischargeM3s",
        "BasinID",
        "RiverCorridor",
    ):
        np.testing.assert_array_equal(_array(first, name), _array(second, name))
    for name in (
        "DepressionCatalog",
        "LakeCatalog",
        "BreachCatalog",
        "BasinCatalog",
        "DrainageGraph",
        "RiverReachCatalog",
    ):
        assert first.artifact_records[name].value.equals(second.artifact_records[name].value)
    assert second.stats is not None and second.stats.cache_hit


def test_hydrology_config_and_cli_reject_invalid_controls():
    with pytest.raises(ValueError, match="Unknown hydrology controls"):
        HydrologyConfig.from_mapping({"draw_rivers": True})
    with pytest.raises(ValueError, match="wetland_maximum_depth_m"):
        HydrologyConfig.from_mapping(
            {"minimum_depression_depth_m": 50.0, "wetland_maximum_depth_m": 20.0}
        )
    with pytest.raises(ValueError, match="river_minimum_discharge_m3s"):
        HydrologyConfig.from_mapping(
            {"river_minimum_discharge_m3s": 500.0, "river_discharge_threshold_m3s": 100.0}
        )
    with pytest.raises(SystemExit):
        pipeline_tools_main(["--stage", "hydrology"])
