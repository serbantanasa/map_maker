from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline import _fluvial_native as fluvial_native
from map_maker.pipeline.stages.basin_erosion import BasinErosionConfig


@pytest.fixture(autouse=True)
def _ensure_stages_registered():
    registry().clear()
    for module_name in (
        "geometry",
        "planet",
        "atmosphere",
        "tectonics",
        "world_age",
        "geology",
        "elevation",
        "sea_level",
        "climate",
        "cryosphere",
        "hydrology",
        "basin_refinement",
        "basin_erosion",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(
    tmp_path: Path,
    run_id: str,
    *,
    root: str = "primary",
    face_resolution: int = 20,
    rng_seed: int = 37,
) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": face_resolution}],
            "rng_seed": rng_seed,
            "run_id": run_id,
            "output_dir": str(tmp_path / root / "out"),
            "cache_dir": str(tmp_path / root / "cache"),
            "log_dir": str(tmp_path / root / "logs"),
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
                "basin_refinement": {
                    "refinement_factor": 4,
                    "terrain_noise_fraction": 0.4,
                },
                "basin_erosion": {
                    "minimum_bed_slope": 1e-6,
                    "maximum_deposition_fraction": 0.35,
                    "deposition_slope_scale": 0.001,
                    "maximum_deposition_depth_m": 10.0,
                },
            },
        }
    )


def _table(result, name: str) -> pa.Table:
    value = result.artifact_records[name].value
    assert isinstance(value, pa.Table)
    return value


def _native_connector_fixture() -> dict[str, object]:
    return {
        "controls": {
            "planet_radius_m": 1_000.0,
            "minimum_bed_slope": 1e-6,
            "maximum_deposition_fraction": 0.25,
            "deposition_slope_scale": 0.001,
            "maximum_deposition_depth_m": 10.0,
            "bank_incision_fraction": 0.35,
        },
        "cell_ids": np.array([10, 11, 12, 13, 14, 99], dtype=np.int32),
        "cell_parent_ids": np.array([1, 1, 2, 3, 3, 9], dtype=np.int32),
        "cell_terrain_m": np.array([1000, 1001, 999, 500, 501, 100], dtype=np.float32),
        "cell_areas_km2": np.ones(6, dtype=np.float64),
        "cell_xyz": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9992, 0.04, 0.0],
                [0.9968, 0.08, 0.0],
                [0.9928, 0.12, 0.0],
                [0.9872, 0.16, 0.0],
                [0.98, 0.2, 0.0],
            ],
            dtype=np.float32,
        ),
        "reach_ids": np.array([1, 2, 3], dtype=np.int32),
        "downstream_reach_ids": np.array([2, 3, -1], dtype=np.int32),
        "reach_kinds": np.array([1, 2, 1], dtype=np.uint8),
        "terminal_kinds": np.array([0, 0, 1], dtype=np.uint8),
        "channel_width_m": np.array([10, 0, 20], dtype=np.float32),
        "reach_slope": np.array([0.01, 0, 0.001], dtype=np.float32),
        "membership_reach_ids": np.array([1, 1, 1, 3, 3], dtype=np.int32),
        "membership_cell_ids": np.array([10, 11, 12, 13, 14], dtype=np.int32),
        "membership_parent_ids": np.array([1, 1, 2, 3, 3], dtype=np.int32),
        "membership_path_order": np.array([0, 1, 3, 0, 1], dtype=np.int32),
        "membership_reach_length_m": np.full(5, 50.0, dtype=np.float64),
        "membership_channel_fraction": np.full(5, 0.0005, dtype=np.float32),
        "membership_valley_fraction": np.full(5, 0.01, dtype=np.float32),
        "membership_floodplain_fraction": np.full(5, 0.005, dtype=np.float32),
    }


def test_basin_erosion_profiles_and_routes_conservatively(tmp_path: Path):
    engine = ExecutionEngine(_config(tmp_path, "basin-erosion"), generate_visuals=True)
    result = engine.run(["basin_erosion"])["basin_erosion"]
    metadata = result.artifact_records["BasinErosionMetadata"].value
    profiles = _table(result, "ChannelBedProfileCatalog")
    reaches = _table(result, "FluvialRiverReachCatalog")
    cells = _table(result, "ErodedBasinCellCatalog")
    parents = _table(result, "BasinErosionParentCatalog")

    assert metadata["source_to_sink_ready"] == 1
    assert metadata["bed_profile_valid"] == 1
    assert metadata["sediment_conservation_valid"] == 1
    assert metadata["process_exclusion_valid"] == 1
    assert metadata["connector_process_valid"] == 1
    assert metadata["physical_node_count"] > 0
    assert metadata["profile_record_count"] == profiles.num_rows > 0
    assert metadata["maximum_junction_bed_error_m"] == 0.0
    assert metadata["minimum_realized_slope"] >= metadata["minimum_bed_slope"] - 1e-8
    assert metadata["emitted_minimum_realized_slope"] >= metadata["minimum_bed_slope"] - 1e-12
    assert metadata["total_eroded_volume_km3"] > 0.0
    assert metadata["bank_carve_valid"] == 1
    assert metadata["bank_carve_enabled"] == 0
    assert metadata["incision_plausibility_valid"] == 1
    assert metadata["raster_feedback_valid"] == 1
    assert metadata["raster_terrain_feedback_applied"] == 0
    assert metadata["regional_refinement_owns_physical_incision"] == 1
    assert metadata["local_dem_low_valid"] == 1
    assert metadata["vector_uphill_segment_count"] == 0
    assert metadata["total_bank_eroded_volume_m3"] == 0.0

    vectors = _table(result, "FluvialRiverVectorCatalog")
    assert vectors.num_rows == profiles.num_rows > 0
    vector_slopes = np.asarray(
        vectors["bed_slope_to_next"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    finite_vector_slopes = vector_slopes[np.isfinite(vector_slopes)]
    assert len(finite_vector_slopes)
    assert np.min(finite_vector_slopes) >= metadata["minimum_bed_slope"] - 1e-12
    assert np.all(
        np.asarray(vectors["bed_elevation_m"], dtype=np.float64)
        <= np.asarray(vectors["terrain_elevation_m"], dtype=np.float64) + 1e-5
    )

    terrain = np.asarray(profiles["terrain_elevation_m"], dtype=np.float64)
    bed = np.asarray(profiles["bed_elevation_m"], dtype=np.float64)
    depth = np.asarray(profiles["incision_depth_m"], dtype=np.float64)
    assert np.all(np.isfinite(bed))
    assert np.all(bed <= terrain + 1e-5)
    assert np.allclose(depth, terrain - bed, atol=1e-4)
    refined_ids = np.asarray(cells["fine_cell_id"], dtype=np.int32)
    row_by_id = {int(cell_id): row for row, cell_id in enumerate(refined_ids)}
    profile_rows = np.asarray(
        [row_by_id[int(cell_id)] for cell_id in profiles["fine_cell_id"].to_pylist()],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        terrain,
        np.asarray(cells["channel_surface_prior_m"], dtype=np.float64)[profile_rows],
        atol=1e-5,
    )
    width_by_reach = dict(
        zip(
            np.asarray(reaches["reach_id"], dtype=np.int32),
            np.asarray(reaches["channel_width_m"], dtype=np.float64),
            strict=True,
        )
    )
    expected_channel_volume = sum(
        width_by_reach[int(reach_id)] * length_m * incision_depth
        for reach_id, length_m, incision_depth in zip(
            np.asarray(profiles["reach_id"], dtype=np.int32),
            np.asarray(profiles["reach_length_m"], dtype=np.float64),
            depth,
            strict=True,
        )
    )
    profile_volume = np.sum(np.asarray(profiles["eroded_volume_m3"], dtype=np.float64))
    bank_volume = np.sum(np.asarray(reaches["bank_eroded_volume_m3"], dtype=np.float64))
    assert profile_volume == pytest.approx(expected_channel_volume, rel=1e-7)
    assert profile_volume == pytest.approx(metadata["total_channel_eroded_volume_m3"], rel=1e-9)
    assert bank_volume == pytest.approx(metadata["total_bank_eroded_volume_m3"], rel=1e-9)
    assert profile_volume + bank_volume == pytest.approx(
        metadata["total_eroded_volume_m3"], rel=1e-9
    )

    cell_ids = np.asarray(profiles["fine_cell_id"], dtype=np.int32)
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]
    sorted_beds = bed[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    ends = np.r_[starts[1:], len(sorted_ids)]
    for start, end in zip(starts, ends, strict=True):
        assert np.max(sorted_beds[start:end]) - np.min(sorted_beds[start:end]) <= 1e-5

    connector = np.asarray(reaches["reach_kind"].to_pylist()) == "connector"
    assert not np.any(np.asarray(reaches["has_physical_bed"])[connector])
    assert np.all(np.asarray(reaches["local_erosion_volume_m3"])[connector] == 0.0)
    assert np.all(np.asarray(reaches["floodplain_deposition_volume_m3"])[connector] == 0.0)

    eroded = np.asarray(cells["prospective_channel_excavation_volume_m3"], dtype=np.float64)
    deposited = np.asarray(cells["prospective_floodplain_deposition_volume_m3"], dtype=np.float64)
    applied_eroded = np.asarray(cells["applied_terrain_erosion_volume_m3"], dtype=np.float64)
    applied_deposited = np.asarray(cells["applied_terrain_deposition_volume_m3"], dtype=np.float64)
    area_m2 = np.asarray(cells["area_km2"], dtype=np.float64) * 1_000_000.0
    mean_delta = np.asarray(cells["terrain_mean_delta_m"], dtype=np.float64)
    assert np.array_equal(mean_delta, np.zeros_like(mean_delta))
    assert np.array_equal(applied_eroded, np.zeros_like(applied_eroded))
    assert np.array_equal(applied_deposited, np.zeros_like(applied_deposited))
    assert np.array_equal(
        np.asarray(cells["terrain_elevation_after_m"], dtype=np.float64),
        np.asarray(cells["terrain_elevation_m"], dtype=np.float64),
    )
    assert np.allclose(
        mean_delta * area_m2,
        applied_deposited - applied_eroded,
        rtol=0.0,
        atol=0.0,
    )
    assert np.sum(eroded) == pytest.approx(metadata["total_eroded_volume_m3"], rel=1e-9)
    assert np.sum(deposited) == pytest.approx(
        metadata["total_floodplain_deposition_volume_m3"], rel=1e-9
    )
    assert (
        metadata["total_floodplain_deposition_volume_m3"]
        + metadata["total_terminal_deposition_volume_m3"]
        + metadata["total_exported_sediment_volume_m3"]
    ) == pytest.approx(metadata["total_eroded_volume_m3"], rel=1e-9)

    assert np.sum(
        np.asarray(parents["prospective_child_channel_excavation_volume_m3"])
    ) == pytest.approx(np.sum(eroded), rel=1e-12)
    assert np.sum(
        np.asarray(parents["prospective_child_floodplain_deposition_volume_m3"])
    ) == pytest.approx(np.sum(deposited), rel=1e-12)
    assert np.all(np.asarray(parents["applied_child_terrain_erosion_volume_m3"]) == 0.0)
    assert np.all(np.asarray(parents["applied_child_terrain_deposition_volume_m3"]) == 0.0)
    assert np.all(np.asarray(parents["restricted_terrain_mean_delta_m"]) == 0.0)
    assert (engine.context.config.run_visual_dir() / "basin_erosion" / "eroded_basin.png").is_file()
    assert (
        engine.context.config.run_visual_dir() / "basin_erosion" / "longitudinal_profile.png"
    ).is_file()


def test_basin_erosion_is_independently_deterministic_and_cacheable(tmp_path: Path):
    first = ExecutionEngine(_config(tmp_path, "first", root="first")).run(["basin_erosion"])[
        "basin_erosion"
    ]
    second = ExecutionEngine(_config(tmp_path, "second", root="second")).run(["basin_erosion"])[
        "basin_erosion"
    ]
    for artifact in (
        "ChannelBedProfileCatalog",
        "FluvialRiverReachCatalog",
        "FluvialRiverVectorCatalog",
        "ErodedBasinCellCatalog",
        "BasinErosionParentCatalog",
    ):
        assert _table(first, artifact).equals(_table(second, artifact))
    assert (
        first.artifact_records["BasinErosionMetadata"].value
        == second.artifact_records["BasinErosionMetadata"].value
    )

    cached = ExecutionEngine(_config(tmp_path, "second", root="second")).run(["basin_erosion"])[
        "basin_erosion"
    ]
    assert cached.stats is not None and cached.stats.cache_hit


def test_basin_erosion_config_rejects_unknown_and_invalid_controls():
    with pytest.raises(ValueError, match="Unknown basin erosion controls"):
        BasinErosionConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="maximum_deposition_fraction"):
        BasinErosionConfig.from_mapping({"maximum_deposition_fraction": 1.1})
    with pytest.raises(ValueError, match="deposition_slope_scale"):
        BasinErosionConfig.from_mapping({"deposition_slope_scale": 0.0})
    with pytest.raises(ValueError, match="bank_incision_fraction"):
        BasinErosionConfig.from_mapping({"bank_incision_fraction": 1.5})
    with pytest.raises(ValueError, match="hard_maximum_channel_incision_m"):
        BasinErosionConfig.from_mapping({"hard_maximum_channel_incision_m": 0.0})


def test_stage_rejects_incision_outside_configured_envelope(tmp_path: Path):
    config = _config(tmp_path, "implausible-incision")
    overrides = {name: dict(values) for name, values in config.stage_overrides.items()}
    overrides["basin_erosion"]["hard_maximum_channel_incision_m"] = 1.0
    config.stage_overrides = overrides

    with pytest.raises(RuntimeError, match="physical plausibility envelope"):
        ExecutionEngine(config).run(["basin_erosion"])


def test_native_layout_round_trips_double_precision_profile_fields():
    record = fluvial_native._ffi.new("BedProfileRecord*")
    record.reach_id = 17
    record.fine_cell_id = 23
    record.bed_elevation_m = 1000.00004
    record.eroded_volume_m3 = 1234.5
    view = fluvial_native._ffi.buffer(record, fluvial_native._ffi.sizeof("BedProfileRecord"))
    parsed = np.frombuffer(view, dtype=fluvial_native.BED_PROFILE_DTYPE, count=1)[0]

    assert parsed["reach_id"] == 17
    assert parsed["fine_cell_id"] == 23
    assert parsed["bed_elevation_m"] == pytest.approx(1000.00004, rel=0.0, abs=1e-12)
    assert parsed["eroded_volume_m3"] == 1234.5


def test_native_connector_gap_and_persisted_grade_cross_cffi_boundary():
    profiles, reaches, cells, metadata = fluvial_native.run_fluvial_erosion(
        **_native_connector_fixture()
    )

    assert metadata["connector_reach_count"] == 1
    assert metadata["physical_component_count"] == 3
    connector = reaches[reaches["reach_id"] == 2][0]
    upstream = reaches[reaches["reach_id"] == 1][0]
    assert connector["has_physical_bed"] == 0
    assert connector["local_erosion_volume_m3"] == 0.0
    assert connector["upstream_input_volume_m3"] == upstream["downstream_transfer_volume_m3"]
    assert connector["downstream_transfer_volume_m3"] == connector["upstream_input_volume_m3"]
    assert 99 not in set(cells["fine_cell_id"])

    first_reach = profiles[profiles["reach_id"] == 1]
    first_reach.sort(order="path_order")
    bed_drop = first_reach["bed_elevation_m"][0] - first_reach["bed_elevation_m"][1]
    source = _native_connector_fixture()["cell_xyz"][0].astype(np.float64)
    target = _native_connector_fixture()["cell_xyz"][1].astype(np.float64)
    length_m = np.arccos(np.clip(np.dot(source, target), -1.0, 1.0)) * 1_000.0
    assert first_reach["bed_elevation_m"][0] != first_reach["bed_elevation_m"][1]
    assert bed_drop / length_m >= 1e-6 - 1e-12


def test_native_rejects_even_zero_length_connector_membership():
    fixture = _native_connector_fixture()
    fixture["membership_reach_ids"] = np.append(fixture["membership_reach_ids"], 2).astype(np.int32)
    fixture["membership_cell_ids"] = np.append(fixture["membership_cell_ids"], 99).astype(np.int32)
    fixture["membership_parent_ids"] = np.append(fixture["membership_parent_ids"], 9).astype(
        np.int32
    )
    fixture["membership_path_order"] = np.append(fixture["membership_path_order"], 0).astype(
        np.int32
    )
    fixture["membership_reach_length_m"] = np.append(
        fixture["membership_reach_length_m"], 0.0
    ).astype(np.float64)
    for field in (
        "membership_channel_fraction",
        "membership_valley_fraction",
        "membership_floodplain_fraction",
    ):
        fixture[field] = np.append(fixture[field], 0.0).astype(np.float32)

    with pytest.raises(RuntimeError, match="invalid channel/connector physical support"):
        fluvial_native.run_fluvial_erosion(**fixture)


def test_pipeline_fixture_exercises_connectors_and_process_exclusions(tmp_path: Path):
    # Deterministic face-16 fixture tuned for earth_relief_v1 terrain: connectors
    # and process exclusions depend on drainage topology and river thresholds.
    engine = ExecutionEngine(
        PipelineConfig.from_mapping(
            {
                "topology": "cubed_sphere",
                "resolutions": [{"face_resolution": 16}],
                "rng_seed": 34,
                "run_id": "connector-exclusion",
                "output_dir": str(tmp_path / "out"),
                "cache_dir": str(tmp_path / "cache"),
                "log_dir": str(tmp_path / "logs"),
                "stage_overrides": {
                    "tectonics": {
                        "num_plates": 14,
                        "continental_fraction": 0.42,
                        "lloyd_iterations": 3,
                    },
                    "world_age": {"world_age": 4.1},
                    "climate": {
                        "spinup_years": 8,
                        "moisture_spinup_years": 2,
                        "moisture_steps_per_month_at_face_128": 16,
                    },
                    "hydrology": {
                        "river_discharge_threshold_m3s": 20.0,
                        "river_contributing_area_threshold_km2": 8_000.0,
                        "river_minimum_discharge_m3s": 2.0,
                    },
                    "basin_refinement": {
                        "refinement_factor": 4,
                        "terrain_noise_fraction": 0.45,
                    },
                    "basin_erosion": {
                        "minimum_bed_slope": 1e-6,
                        "maximum_deposition_fraction": 0.35,
                        "deposition_slope_scale": 0.001,
                        "maximum_deposition_depth_m": 10.0,
                    },
                },
            }
        )
    )
    results = engine.run(["basin_refinement", "basin_erosion"])
    refinement = results["basin_refinement"]
    erosion = results["basin_erosion"]
    refinement_metadata = refinement.artifact_records["BasinRefinementMetadata"].value
    reaches = _table(erosion, "FluvialRiverReachCatalog")
    profiles = _table(erosion, "ChannelBedProfileCatalog")
    cells = _table(erosion, "ErodedBasinCellCatalog")
    memberships = _table(refinement, "RefinedReachCellCatalog")

    assert refinement_metadata["connector_reach_count"] > 0
    assert refinement_metadata["process_excluded_parent_count"] > 0
    connector_ids = np.asarray(reaches["reach_id"])[
        np.asarray(reaches["reach_kind"].to_pylist()) == "connector"
    ]
    assert not np.any(np.isin(np.asarray(memberships["reach_id"]), connector_ids))
    assert not np.any(np.isin(np.asarray(profiles["reach_id"]), connector_ids))
    connector = np.isin(np.asarray(reaches["reach_id"]), connector_ids)
    assert np.all(
        np.asarray(reaches["downstream_transfer_volume_m3"])[connector]
        == np.asarray(reaches["upstream_input_volume_m3"])[connector]
    )

    excluded = np.asarray(cells["process_excluded"])
    assert np.any(excluded)
    assert np.all(np.asarray(cells["prospective_channel_excavation_volume_m3"])[excluded] == 0.0)
    assert np.all(np.asarray(cells["prospective_floodplain_deposition_volume_m3"])[excluded] == 0.0)
    assert np.array_equal(
        np.asarray(cells["terrain_elevation_after_m"])[excluded],
        np.asarray(cells["terrain_elevation_m"])[excluded],
    )


def test_stage_rejects_cross_catalog_native_inconsistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = importlib.import_module("map_maker.pipeline.stages.basin_erosion")
    native_run = module.run_fluvial_erosion

    def inconsistent_native_run(**kwargs):
        profiles, reaches, cells, metadata = native_run(**kwargs)
        reaches = reaches.copy()
        physical = np.flatnonzero(reaches["has_physical_bed"] != 0)
        reaches["local_erosion_volume_m3"][physical[0]] += max(
            float(metadata["total_eroded_volume_m3"]) * 0.01, 1.0
        )
        return profiles, reaches, cells, metadata

    monkeypatch.setattr(module, "run_fluvial_erosion", inconsistent_native_run)
    engine = ExecutionEngine(_config(tmp_path, "inconsistent-native"))
    with pytest.raises(RuntimeError, match="catalogs disagree"):
        engine.run(["basin_erosion"])
