from __future__ import annotations

from collections import deque
import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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
        "atmosphere",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "climate",
        "cryosphere",
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
        "LakeFraction": np.float32,
        "WetlandFraction": np.float32,
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
    depression_catalog = hydrology.artifact_records["DepressionCatalog"].value
    open_outlet_cells = set(
        depression_catalog.filter(depression_catalog["open_outlet"])["outlet_cell"].to_pylist()
    )
    for cell in np.flatnonzero(land.reshape(-1)):
        target = int(flat_receiver[cell])
        if target >= 0 and not flat_ocean[target]:
            assert contributing_area[target] + 1e-8 >= contributing_area[cell]
            if target not in open_outlet_cells:
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
    water_fraction = arrays["LakeFraction"] + arrays["WetlandFraction"]
    assert np.all((arrays["LakeFraction"] >= 0.0) & (arrays["LakeFraction"] <= 1.0))
    assert np.all((arrays["WetlandFraction"] >= 0.0) & (arrays["WetlandFraction"] <= 1.0))
    assert np.all(water_fraction <= 1.0 + 1e-6)
    assert np.all(arrays["LakeFraction"][arrays["WaterBodyClass"] == 1] == 0.0)
    assert np.all(arrays["WetlandFraction"][arrays["WaterBodyClass"] != 1] == 0.0)
    assert np.all(arrays["LakeID"][arrays["WetlandFraction"] > 0.0] == -1)
    assert np.any((water_fraction > 0.0) & (water_fraction < 1.0))
    assert np.all(arrays["RiverCorridor"][water_fraction >= 0.5] == 0.0)

    lake_catalog = hydrology.artifact_records["LakeCatalog"].value
    wetland_catalog = hydrology.artifact_records["WetlandCatalog"].value
    breach_catalog = hydrology.artifact_records["BreachCatalog"].value
    basin_catalog = hydrology.artifact_records["BasinCatalog"].value
    drainage_graph = hydrology.artifact_records["DrainageGraph"].value
    waterbody_cells = hydrology.artifact_records["WaterBodyCellCatalog"].value
    reaches = hydrology.artifact_records["RiverReachCatalog"].value
    for table in (
        depression_catalog,
        lake_catalog,
        wetland_catalog,
        breach_catalog,
        basin_catalog,
        drainage_graph,
        waterbody_cells,
        reaches,
    ):
        assert isinstance(table, pa.Table)
    assert depression_catalog.num_rows > 0
    assert lake_catalog.num_rows > 0
    assert set(lake_catalog["class_code"].to_pylist()).isdisjoint({1})
    assert set(wetland_catalog["class_code"].to_pylist()) <= {1}
    np.testing.assert_array_equal(
        np.sort(np.asarray(lake_catalog["lake_id"])),
        np.arange(lake_catalog.num_rows, dtype=np.int32),
    )
    assert set(wetland_catalog["lake_id"].to_pylist()) <= {-1}
    depression_classes = np.asarray(depression_catalog["class_code"])
    preserved_depression_ids = np.asarray(depression_catalog["depression_id"])[
        depression_classes != 5
    ]
    preserved_support = np.isin(arrays["DepressionID"], preserved_depression_ids)
    assert np.all(arrays["RiverCorridor"][preserved_support] == 0.0)
    assert basin_catalog.num_rows > 0
    assert drainage_graph.num_rows == int(np.count_nonzero(land))
    assert waterbody_cells.num_rows == int(np.count_nonzero(water_fraction > 0.0))
    assert reaches.num_rows > 0
    assert {
        "cell_id",
        "receiver_id",
        "basin_id",
        "sink_type",
        "depression_id",
        "lake_id",
        "lake_fraction",
        "wetland_fraction",
        "contributing_area_km2",
        "mean_discharge_m3s",
    } == set(drainage_graph.column_names)
    assert {
        "cell_id",
        "waterbody_id",
        "depression_id",
        "lake_id",
        "class_code",
        "class_name",
        "lake_fraction",
        "wetland_fraction",
        "covered_area_km2",
    } == set(waterbody_cells.column_names)
    np.testing.assert_array_equal(
        np.asarray(waterbody_cells["waterbody_id"]),
        np.asarray(waterbody_cells["depression_id"]),
    )
    assert np.all(np.asarray(waterbody_cells["covered_area_km2"]) > 0.0)
    catalog_covered_area = float(np.sum(np.asarray(waterbody_cells["covered_area_km2"])))
    radius_km = 6_371.0
    physical_area_km2 = grid.cell_areas * radius_km * radius_km
    raster_covered_area = float(np.sum(physical_area_km2 * water_fraction))
    assert catalog_covered_area == pytest.approx(raster_covered_area, rel=1e-6)
    assert float(np.sum(np.asarray(lake_catalog["water_area_km2"]))) == pytest.approx(
        float(np.sum(physical_area_km2 * arrays["LakeFraction"])), rel=1e-6
    )
    assert float(np.sum(np.asarray(wetland_catalog["water_area_km2"]))) == pytest.approx(
        float(np.sum(physical_area_km2 * arrays["WetlandFraction"])), rel=1e-6
    )
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
        "reach_kind",
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
    connector = np.asarray(pc.equal(reaches["reach_kind"], "connector"))
    if np.any(connector):
        assert np.all(np.asarray(reaches["channel_width_m"])[connector] == 0.0)
        assert np.all(np.asarray(reaches["channel_depth_m"])[connector] == 0.0)
        assert np.all(np.asarray(reaches["valley_width_m"])[connector] == 0.0)
        assert np.all(np.asarray(reaches["floodplain_width_m"])[connector] == 0.0)
        assert np.all(np.asarray(reaches["incision_m"])[connector] == 0.0)
        assert np.all(np.asarray(reaches["velocity_mean"])[connector] == 0.0)
        assert np.all(np.asarray(reaches["stream_power"])[connector] == 0.0)
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
    assert metadata["wetland_count"] == wetland_catalog.num_rows
    assert metadata["waterbody_count"] == lake_catalog.num_rows + wetland_catalog.num_rows
    assert metadata["basin_count"] == basin_catalog.num_rows
    assert metadata["reach_count"] == reaches.num_rows
    assert metadata["connector_reach_count"] == int(np.count_nonzero(connector))
    assert metadata["reach_source_to_sink_ready"] == 1
    assert metadata["reach_terminal_unresolved_count"] == 0
    assert metadata["open_lake_count"] > 0
    assert 0.0 < metadata["lake_land_cell_fraction"] < 0.25
    assert 0.0 < metadata["lake_land_area_fraction"] < metadata["lake_land_cell_fraction"]
    assert 0.0 <= metadata["wetland_land_area_fraction"] < 0.25
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
        "LakeFraction",
        "WetlandFraction",
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
        "WetlandCatalog",
        "BreachCatalog",
        "BasinCatalog",
        "DrainageGraph",
        "WaterBodyCellCatalog",
        "RiverReachCatalog",
    ):
        assert first.artifact_records[name].value.equals(second.artifact_records[name].value)
    assert second.stats is not None and second.stats.cache_hit


def test_hydrology_config_and_cli_reject_invalid_controls():
    with pytest.raises(ValueError, match="Unknown hydrology controls"):
        HydrologyConfig.from_mapping({"draw_rivers": True})
    with pytest.raises(ValueError, match="wetland_mean_depth_m"):
        HydrologyConfig.from_mapping({"wetland_mean_depth_m": 0.0})
    with pytest.raises(ValueError, match="river_minimum_discharge_m3s"):
        HydrologyConfig.from_mapping(
            {"river_minimum_discharge_m3s": 500.0, "river_discharge_threshold_m3s": 100.0}
        )
    with pytest.raises(ValueError, match="subgrid_relief_scale"):
        HydrologyConfig.from_mapping({"subgrid_relief_scale": 0.0})
    with pytest.raises(ValueError, match="subgrid_connected_basin_fraction"):
        HydrologyConfig.from_mapping({"subgrid_connected_basin_fraction": 1.1})
    with pytest.raises(SystemExit):
        pipeline_tools_main(["--stage", "hydrology"])
