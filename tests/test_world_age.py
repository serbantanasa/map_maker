from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry


@pytest.fixture(autouse=True)
def _ensure_world_age_registered():
    reg = registry()
    reg.clear()
    tectonics_module = importlib.import_module("map_maker.pipeline.stages.tectonics")
    world_age_module = importlib.import_module("map_maker.pipeline.stages.world_age")
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
    subsidence = np.array(world_age_res.artifact_records["SubsidenceRate"].value.array(), copy=False)
    compression = np.array(world_age_res.artifact_records["TectonicCompression"].value.array(), copy=False)
    extension = np.array(world_age_res.artifact_records["TectonicExtension"].value.array(), copy=False)
    shear = np.array(world_age_res.artifact_records["ShearMagnitude"].value.array(), copy=False)
    coastal = np.array(world_age_res.artifact_records["CoastalExposure"].value.array(), copy=False)
    lithosphere = np.array(world_age_res.artifact_records["LithosphereStiffness"].value.array(), copy=False)
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

    crust1 = np.array(res1["world_age"].artifact_records["CrustThickness"].value.array(), copy=False)
    crust2 = np.array(res2["world_age"].artifact_records["CrustThickness"].value.array(), copy=False)
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

    young_config = _make_config(tmp_path, "world-age-young", rng_seed=4242, overrides=young_overrides)
    old_config = _make_config(tmp_path, "world-age-old", rng_seed=4242, overrides=old_overrides)

    young_engine = ExecutionEngine(young_config)
    old_engine = ExecutionEngine(old_config)

    young_results = young_engine.run(["tectonics", "world_age"])
    old_results = old_engine.run(["tectonics", "world_age"])

    plate = np.array(young_results["tectonics"].artifact_records["PlateField"].value.array(), copy=False)
    oceanic_mask = plate[..., 1] < 0.5

    crust_young = np.array(young_results["world_age"].artifact_records["CrustThickness"].value.array(), copy=False)
    crust_old = np.array(old_results["world_age"].artifact_records["CrustThickness"].value.array(), copy=False)
    assert crust_old[oceanic_mask].mean() < crust_young[oceanic_mask].mean()

    coastal_young = np.array(young_results["world_age"].artifact_records["CoastalExposure"].value.array(), copy=False)
    coastal_old = np.array(old_results["world_age"].artifact_records["CoastalExposure"].value.array(), copy=False)
    assert coastal_young.shape == coastal_old.shape

    events_young = young_results["world_age"].artifact_records["HotspotEvents"].value
    events_old = old_results["world_age"].artifact_records["HotspotEvents"].value
    assert events_old.num_rows >= events_young.num_rows

    metadata_young = young_results["world_age"].artifact_records["WorldAgeMetadata"].value
    metadata_old = old_results["world_age"].artifact_records["WorldAgeMetadata"].value
    assert metadata_old["convective_vigor"] < metadata_young["convective_vigor"]
    assert metadata_old["hotspot_count"] >= metadata_young["hotspot_count"]
