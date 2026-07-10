from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._world_age_native import run_cubed_sphere_world_age
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.tools import main as pipeline_tools_main


@pytest.fixture(autouse=True)
def _ensure_world_age_registered():
    reg = registry()
    reg.clear()
    geometry_module = importlib.import_module("map_maker.pipeline.stages.geometry")
    tectonics_module = importlib.import_module("map_maker.pipeline.stages.tectonics")
    world_age_module = importlib.import_module("map_maker.pipeline.stages.world_age")
    importlib.reload(geometry_module)
    importlib.reload(tectonics_module)
    importlib.reload(world_age_module)
    yield
    reg.clear()


def _make_config(
    tmp_path: Path,
    run_id: str,
    *,
    rng_seed: int = 0,
    overrides: dict | None = None,
) -> PipelineConfig:
    base = {
        "topology": "sphere",
        "resolutions": [{"height": 64, "width": 128}],
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "log_dir": str(tmp_path / "logs"),
        "run_id": run_id,
        "rng_seed": rng_seed,
    }
    if overrides:
        base["stage_overrides"] = overrides
    return PipelineConfig.from_mapping(base)


def _make_cubed_config(
    tmp_path: Path,
    run_id: str,
    *,
    rng_seed: int = 42,
    world_age: float = 4.1,
) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 32}],
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "run_id": run_id,
            "rng_seed": rng_seed,
            "stage_overrides": {
                "tectonics": {
                    "num_plates": 16,
                    "continental_fraction": 0.35,
                    "lloyd_iterations": 4,
                },
                "world_age": {
                    "world_age": world_age,
                    "thermal_decay_half_life": 1.8,
                    "hotspot_scale": 0.9,
                    "isostasy_factor": 0.6,
                    "radiogenic_heat_scale": 1.0,
                },
            },
        }
    )


def _cubed_ffi_arguments(face_resolution: int = 4) -> dict[str, object]:
    grid = CubedSphereGrid.create(face_resolution)
    shape = grid.face_shape
    plate = np.zeros((*shape, 7), dtype=np.float32)
    cell_ids = np.arange(grid.cell_count).reshape(shape)
    continental = cell_ids % 3 == 0
    plate[..., 0] = cell_ids
    plate[..., 1] = continental
    plate[..., 2] = np.where(continental, 35.0, 7.0)
    plate[..., 3] = np.where(continental, 2.75, 3.0)
    inputs = [np.full(shape, 0.1, dtype=np.float32) for _ in range(5)]
    outputs = [np.empty(shape, dtype=np.float32) for _ in range(10)]
    return {
        "seed": 3,
        "world_age": 4.1,
        "thermal_decay_half_life": 1.8,
        "hotspot_scale": 0.9,
        "isostasy_factor": 0.6,
        "radiogenic_heat_scale": 1.0,
        "areas": grid.cell_areas,
        "neighbors": grid.neighbor_indices,
        "plate_field": plate,
        "convergence_field": inputs[0],
        "divergence_field": inputs[1],
        "subduction_field": inputs[2],
        "shear_field": inputs[3],
        "hotspot_field": inputs[4],
        "crust_thickness_out": outputs[0],
        "isostatic_offset_out": outputs[1],
        "uplift_out": outputs[2],
        "subsidence_out": outputs[3],
        "compression_out": outputs[4],
        "extension_out": outputs[5],
        "shear_out": outputs[6],
        "margin_proximity_out": outputs[7],
        "lithosphere_stiffness_out": outputs[8],
        "proto_ocean_mask_out": outputs[9],
    }


def test_world_age_outputs_and_metadata(tmp_path: Path):
    overrides = {
        "tectonics": {
            "num_plates": 18,
            "hotspot_density": 0.04,
            "velocity_scale": 0.9,
        },
        "world_age": {
            "world_age": 2.6,
            "thermal_decay_half_life": 1.4,
            "hotspot_scale": 1.1,
            "isostasy_factor": 0.58,
            "radiogenic_heat_scale": 1.15,
        },
    }
    config = _make_config(tmp_path, "world-age-outputs", rng_seed=77, overrides=overrides)
    engine = ExecutionEngine(config)
    results = engine.run(["tectonics", "world_age"])
    world_age_res = results["world_age"]

    crust = np.array(world_age_res.artifact_records["CrustThickness"].value.array(), copy=False)
    offset = np.array(world_age_res.artifact_records["IsostaticOffset"].value.array(), copy=False)
    uplift = np.array(world_age_res.artifact_records["UpliftRate"].value.array(), copy=False)
    subsidence = np.array(
        world_age_res.artifact_records["SubsidenceRate"].value.array(), copy=False
    )
    compression = np.array(
        world_age_res.artifact_records["TectonicCompression"].value.array(), copy=False
    )
    extension = np.array(
        world_age_res.artifact_records["TectonicExtension"].value.array(), copy=False
    )
    shear = np.array(world_age_res.artifact_records["ShearMagnitude"].value.array(), copy=False)
    coastal = np.array(world_age_res.artifact_records["CoastalExposure"].value.array(), copy=False)
    lithosphere = np.array(
        world_age_res.artifact_records["LithosphereStiffness"].value.array(), copy=False
    )
    ocean_mask = np.array(world_age_res.artifact_records["BaseOceanMask"].value.array(), copy=False)

    assert crust.shape == (64, 128)
    assert crust.dtype == np.float32
    assert offset.shape == (64, 128)
    assert abs(float(offset.mean())) < 5e-3
    assert uplift.shape == (64, 128)
    assert subsidence.shape == (64, 128)
    assert np.all(crust > 0.0)
    assert np.all(uplift >= 0.0)
    assert np.all(subsidence >= 0.0)
    for field in (compression, extension, shear, coastal, lithosphere, ocean_mask):
        assert field.shape == (64, 128)
        assert field.dtype == np.float32

    assert np.all((compression >= 0.0) & (compression <= 1.0 + 1e-6))
    assert np.all((extension >= 0.0) & (extension <= 1.0 + 1e-6))
    assert np.all((shear >= 0.0) & (shear <= 1.0 + 1e-6))
    assert np.all((coastal >= 0.0) & (coastal <= 1.0 + 1e-6))
    assert np.all((lithosphere >= 0.0) & (lithosphere <= 1.0 + 1e-6))
    assert np.all((ocean_mask >= -1e-6) & (ocean_mask <= 1.0 + 1e-6))

    events_table = world_age_res.artifact_records["HotspotEvents"].value
    assert isinstance(events_table, pa.Table)
    assert set(events_table.schema.names) == {"row", "col", "strength", "plume_factor"}

    metadata = world_age_res.artifact_records["WorldAgeMetadata"].value
    assert isinstance(metadata, dict)
    assert pytest.approx(metadata["world_age"], rel=1e-6) == 2.6
    assert metadata["hotspot_count"] == events_table.num_rows
    assert metadata["convective_vigor"] > 0.0
    for key in (
        "water_fraction",
        "uplift_sigma_gt1",
        "uplift_sigma_gt2",
        "uplift_sigma_gt3",
        "subsidence_sigma_gt1",
        "subsidence_sigma_gt2",
        "subsidence_sigma_gt3",
        "hotspot_density",
    ):
        assert key in metadata
        assert 0.0 <= float(metadata[key]) <= 1.0 + 1e-6


def test_world_age_determinism(tmp_path: Path):
    overrides = {
        "tectonics": {
            "num_plates": 14,
            "velocity_scale": 1.1,
            "hotspot_density": 0.035,
        },
        "world_age": {
            "world_age": 3.1,
            "thermal_decay_half_life": 1.7,
            "hotspot_scale": 0.95,
            "isostasy_factor": 0.6,
            "radiogenic_heat_scale": 1.05,
        },
    }
    config1 = _make_config(tmp_path, "world-age-determinism-a", rng_seed=314, overrides=overrides)
    config2 = _make_config(tmp_path, "world-age-determinism-b", rng_seed=314, overrides=overrides)

    engine1 = ExecutionEngine(config1)
    engine2 = ExecutionEngine(config2)

    res1 = engine1.run(["tectonics", "world_age"])
    res2 = engine2.run(["tectonics", "world_age"])

    crust1 = np.array(
        res1["world_age"].artifact_records["CrustThickness"].value.array(), copy=False
    )
    crust2 = np.array(
        res2["world_age"].artifact_records["CrustThickness"].value.array(), copy=False
    )
    uplift1 = np.array(res1["world_age"].artifact_records["UpliftRate"].value.array(), copy=False)
    uplift2 = np.array(res2["world_age"].artifact_records["UpliftRate"].value.array(), copy=False)
    assert np.allclose(crust1, crust2)
    assert np.allclose(uplift1, uplift2)

    for artifact in (
        "TectonicCompression",
        "TectonicExtension",
        "ShearMagnitude",
        "CoastalExposure",
        "LithosphereStiffness",
        "BaseOceanMask",
    ):
        arr1 = np.array(res1["world_age"].artifact_records[artifact].value.array(), copy=False)
        arr2 = np.array(res2["world_age"].artifact_records[artifact].value.array(), copy=False)
        assert np.allclose(arr1, arr2)

    events1 = res1["world_age"].artifact_records["HotspotEvents"].value
    events2 = res2["world_age"].artifact_records["HotspotEvents"].value
    assert events1.equals(events2)


def test_world_age_age_dependence(tmp_path: Path):
    common_overrides = {
        "tectonics": {
            "num_plates": 16,
            "velocity_scale": 0.95,
            "hotspot_density": 0.03,
        }
    }

    young_overrides = {
        **common_overrides,
        "world_age": {
            "world_age": 1.0,
            "thermal_decay_half_life": 1.5,
            "hotspot_scale": 1.0,
            "isostasy_factor": 0.6,
            "radiogenic_heat_scale": 1.0,
        },
    }
    old_overrides = {
        **common_overrides,
        "world_age": {
            "world_age": 4.5,
            "thermal_decay_half_life": 1.5,
            "hotspot_scale": 1.0,
            "isostasy_factor": 0.6,
            "radiogenic_heat_scale": 1.0,
        },
    }

    young_config = _make_config(
        tmp_path, "world-age-young", rng_seed=4242, overrides=young_overrides
    )
    old_config = _make_config(tmp_path, "world-age-old", rng_seed=4242, overrides=old_overrides)

    young_engine = ExecutionEngine(young_config)
    old_engine = ExecutionEngine(old_config)

    young_results = young_engine.run(["tectonics", "world_age"])
    old_results = old_engine.run(["tectonics", "world_age"])

    plate = np.array(
        young_results["tectonics"].artifact_records["PlateField"].value.array(), copy=False
    )
    oceanic_mask = plate[..., 1] < 0.5

    crust_young = np.array(
        young_results["world_age"].artifact_records["CrustThickness"].value.array(), copy=False
    )
    crust_old = np.array(
        old_results["world_age"].artifact_records["CrustThickness"].value.array(), copy=False
    )
    assert crust_old[oceanic_mask].mean() < crust_young[oceanic_mask].mean()

    coastal_young = np.array(
        young_results["world_age"].artifact_records["CoastalExposure"].value.array(), copy=False
    )
    coastal_old = np.array(
        old_results["world_age"].artifact_records["CoastalExposure"].value.array(), copy=False
    )
    assert coastal_young.shape == coastal_old.shape

    events_young = young_results["world_age"].artifact_records["HotspotEvents"].value
    events_old = old_results["world_age"].artifact_records["HotspotEvents"].value
    assert events_old.num_rows >= events_young.num_rows

    metadata_young = young_results["world_age"].artifact_records["WorldAgeMetadata"].value
    metadata_old = old_results["world_age"].artifact_records["WorldAgeMetadata"].value
    assert metadata_old["convective_vigor"] < metadata_young["convective_vigor"]
    assert metadata_old["hotspot_count"] >= metadata_young["hotspot_count"]


def test_cubed_sphere_world_age_outputs_close_area_and_cross_faces(tmp_path: Path):
    config = _make_cubed_config(tmp_path, "cubed-world-age")
    engine = ExecutionEngine(config, generate_visuals=True)
    results = engine.run(["world_age"])
    grid = engine.context.topology
    assert isinstance(grid, CubedSphereGrid)
    world_age = results["world_age"]
    tectonics = results["tectonics"]

    arrays = {
        name: np.asarray(world_age.artifact_records[name].value.array())
        for name in (
            "CrustThickness",
            "IsostaticOffset",
            "UpliftRate",
            "SubsidenceRate",
            "TectonicCompression",
            "TectonicExtension",
            "ShearMagnitude",
            "CoastalExposure",
            "LithosphereStiffness",
            "BaseOceanMask",
        )
    }
    for array in arrays.values():
        assert array.shape == grid.face_shape
        assert array.dtype == np.float32
        assert np.all(np.isfinite(array))
    assert abs(float(np.sum(arrays["IsostaticOffset"] * grid.cell_areas))) < 1e-5
    assert np.all((arrays["LithosphereStiffness"] >= 0.0) & (arrays["LithosphereStiffness"] <= 1.0))
    assert np.all((arrays["CoastalExposure"] >= 0.0) & (arrays["CoastalExposure"] <= 1.0))

    plate = np.asarray(tectonics.artifact_records["PlateField"].value.array())
    np.testing.assert_array_equal(arrays["BaseOceanMask"], plate[..., 1] < 0.5)
    flat_ocean = arrays["BaseOceanMask"].reshape(-1) >= 0.5
    neighbors = grid.neighbor_indices.reshape(-1, 4)
    source = np.repeat(np.arange(grid.cell_count), 4)
    target = neighbors.reshape(-1)
    face_size = grid.face_resolution * grid.face_resolution
    cross_margin = (
        (source // face_size != target // face_size)
        & (flat_ocean[source] != flat_ocean[target])
        & (source < target)
    )
    assert np.any(cross_margin)
    margin = arrays["CoastalExposure"].reshape(-1)
    np.testing.assert_allclose(margin[source[cross_margin]], 1.0)
    np.testing.assert_allclose(margin[target[cross_margin]], 1.0)

    events = world_age.artifact_records["HotspotEvents"].value
    assert isinstance(events, pa.Table)
    assert events.column_names == [
        "global_cell_id",
        "face",
        "row",
        "col",
        "strength",
        "plume_factor",
    ]
    global_ids = events["global_cell_id"].to_numpy()
    assert np.all((global_ids >= 0) & (global_ids < grid.cell_count))
    for global_id, face, row, col in zip(
        global_ids,
        events["face"].to_numpy(),
        events["row"].to_numpy(),
        events["col"].to_numpy(),
        strict=True,
    ):
        assert grid.decode_index(int(global_id)) == (int(face), int(row), int(col))

    metadata = world_age.artifact_records["WorldAgeMetadata"].value
    assert metadata["crust_state_model"] == "cubed_sphere_area_weighted_v1"
    assert metadata["ocean_mask_semantics"] == "oceanic_crust_candidate_not_final_water"
    assert "water_fraction" not in metadata
    assert metadata["proto_ocean_area_fraction"] == pytest.approx(
        float(np.sum(arrays["BaseOceanMask"] * grid.cell_areas) / np.sum(grid.cell_areas))
    )
    assert metadata["event_semantics"] == "tectonic_thermal_anomaly_proxy_not_mantle_plume"
    assert metadata["spatial_scale_basis"] == "approximate_angular_radians"
    assert abs(metadata["mean_isostatic_offset"]) < 1e-5
    visuals = config.run_visual_dir() / "world_age"
    assert (visuals / "isostatic_potential.png").is_file()
    assert (visuals / "tectonic_rates.png").is_file()
    assert (visuals / "proto_crust.png").is_file()


def test_cubed_sphere_world_age_cools_with_age(tmp_path: Path):
    young = ExecutionEngine(_make_cubed_config(tmp_path, "cubed-young", world_age=1.0)).run(
        ["world_age"]
    )
    old = ExecutionEngine(_make_cubed_config(tmp_path, "cubed-old", world_age=4.5)).run(
        ["world_age"]
    )
    young_meta = young["world_age"].artifact_records["WorldAgeMetadata"].value
    old_meta = old["world_age"].artifact_records["WorldAgeMetadata"].value
    assert old_meta["thermal_decay_factor"] < young_meta["thermal_decay_factor"]
    assert old_meta["convective_vigor"] < young_meta["convective_vigor"]
    assert old_meta["hotspot_count"] <= young_meta["hotspot_count"]

    plate = np.asarray(young["tectonics"].artifact_records["PlateField"].value.array())
    oceanic = plate[..., 1] < 0.5
    young_crust = np.asarray(young["world_age"].artifact_records["CrustThickness"].value.array())
    old_crust = np.asarray(old["world_age"].artifact_records["CrustThickness"].value.array())
    assert float(np.mean(old_crust[oceanic])) < float(np.mean(young_crust[oceanic]))
    young_stiffness = np.asarray(
        young["world_age"].artifact_records["LithosphereStiffness"].value.array()
    )
    old_stiffness = np.asarray(
        old["world_age"].artifact_records["LithosphereStiffness"].value.array()
    )
    assert float(np.mean(old_stiffness)) > float(np.mean(young_stiffness))


def test_cubed_sphere_world_age_ffi_rejects_overlap_alignment_and_shape():
    arguments = _cubed_ffi_arguments()
    arguments["isostatic_offset_out"] = arguments["crust_thickness_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_cubed_sphere_world_age(**arguments)

    arguments = _cubed_ffi_arguments()
    shape = arguments["crust_thickness_out"].shape
    raw = bytearray(int(np.prod(shape)) * np.dtype(np.float32).itemsize + 1)
    arguments["crust_thickness_out"] = np.frombuffer(raw, dtype=np.float32, offset=1).reshape(shape)
    with pytest.raises(ValueError, match="must be aligned"):
        run_cubed_sphere_world_age(**arguments)

    arguments = _cubed_ffi_arguments()
    arguments["hotspot_field"] = np.empty((6, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        run_cubed_sphere_world_age(**arguments)


def test_cubed_sphere_hotspot_count_is_resolution_independent():
    low_events, low_metadata = run_cubed_sphere_world_age(**_cubed_ffi_arguments(4))
    high_events, high_metadata = run_cubed_sphere_world_age(**_cubed_ffi_arguments(8))

    assert low_events.num_rows == high_events.num_rows
    assert low_metadata["hotspot_count"] == high_metadata["hotspot_count"]


def test_cubed_sphere_world_age_cli_requires_and_runs_config(tmp_path: Path):
    with pytest.raises(SystemExit) as error:
        pipeline_tools_main(["--stage", "world_age"])
    assert error.value.code == 2

    config_path = tmp_path / "world-age.yaml"
    config_path.write_text(
        "\n".join(
            [
                "topology: cubed_sphere",
                "resolutions:",
                "  - face_resolution: 8",
                f"output_dir: {tmp_path / 'out'}",
                f"cache_dir: {tmp_path / 'cache'}",
                f"log_dir: {tmp_path / 'logs'}",
                "run_id: world-age-cli",
                "rng_seed: 19",
                "stage_overrides:",
                "  tectonics:",
                "    num_plates: 8",
                "    lloyd_iterations: 2",
                "  world_age:",
                "    world_age: 4.1",
            ]
        ),
        encoding="utf8",
    )
    assert pipeline_tools_main(["--stage", "world_age", "--config", str(config_path)]) == 0
    visual_dir = tmp_path / "out" / "world-age-cli" / "visuals" / "world_age"
    assert (visual_dir / "isostatic_potential.png").is_file()
    assert (visual_dir / "tectonic_rates.png").is_file()
    assert (visual_dir / "proto_crust.png").is_file()
