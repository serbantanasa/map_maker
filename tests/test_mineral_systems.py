from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline._mineral_systems_native import (
    COMMODITY_COUNT,
    SYSTEM_COUNT,
    run_mineral_systems,
)
from map_maker.pipeline.stages.mineral_systems import (
    CAUSAL_OUTPUTS,
    MineralSystemsConfig,
    _equatorial_scale_bar,
    _potential_reconstruction,
)
from map_maker.pipeline.tools import (
    _mineral_validation_failures,
    main as pipeline_tools_main,
)


@pytest.fixture
def registered_mineral_stages():
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
        "hydrology_pass2",
        "surface_water",
        "outlet_incision",
        "lake_hydrographs",
        "hydrology_validation",
        "surface_materials",
        "biosphere_envelope",
        "potential_biosphere",
        "biosphere_validation",
        "mineral_systems",
        "mineral_systems_validation",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield
    registry().clear()


def _integration_config(tmp_path: Path, run_id: str) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "cubed_sphere",
            "resolutions": [{"face_resolution": 16}],
            "rng_seed": 22,
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
                "basin_refinement": {
                    "refinement_factor": 4,
                    "terrain_noise_fraction": 0.4,
                    "maximum_tile_bubble_correlation_p50": 0.60,
                    "maximum_tile_bubble_correlation_p95": 0.95,
                },
                "basin_erosion": {
                    "minimum_bed_slope": 1e-5,
                    "maximum_deposition_fraction": 0.35,
                    "deposition_slope_scale": 0.001,
                    "maximum_deposition_depth_m": 10.0,
                },
                "hydrology_pass2": {
                    "minimum_depression_depth_m": 5.0,
                    "maximum_receiver_change_fraction": 0.15,
                    "maximum_receiver_change_cell_fraction": 0.15,
                    "maximum_new_depression_area_fraction": 0.02,
                },
                "surface_water": {
                    "minimum_solver_iterations": 8,
                    "maximum_solver_iterations": 64,
                    "maximum_connected_inundation_fraction": 0.25,
                    "outlet_erosion_score_threshold": 0.30,
                    "outlet_erosion_depth_scale_m": 200.0,
                    "minimum_outlet_erosion_discharge_m3s": 0.10,
                },
                "outlet_incision": {
                    "maximum_outlet_path_cells": 64,
                    "maximum_reroute_repair_rounds": 64,
                    "maximum_corrected_area_fraction": 0.10,
                    "maximum_receiver_change_area_fraction": 0.15,
                    "maximum_receiver_change_cell_fraction": 0.15,
                    "maximum_reroute_constraint_cell_fraction": 0.15,
                },
                "surface_water_final": {
                    "maximum_outlet_incision_rounds": 8,
                    "require_soil_readiness": True,
                },
                "mineral_systems": {
                    "system_catalog_potential_threshold": 0.10,
                    "deposit_minimum_potential": 0.15,
                    "deposit_minimum_confidence": 0.10,
                    "maximum_deposits_per_system": 32,
                    "minimum_deposit_spacing_steps": 1,
                },
                "mineral_systems_validation": {
                    "minimum_family_peak_potential": 0.10,
                    "minimum_directional_enrichment_ratio": 1.0,
                    "minimum_directional_enrichment_difference": 0.0,
                    "minimum_broad_cratonic_enrichment_ratio": 1.0,
                    "minimum_broad_cratonic_enrichment_difference": 0.0,
                },
            },
        }
    )


def _native_case(seed: int = 42) -> dict[str, np.ndarray]:
    shape = (8,)

    def f32(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    inputs: dict[str, np.ndarray] = {
        "xyz": np.ascontiguousarray(
            np.stack(
                (
                    np.cos(np.linspace(0.0, 2.0 * np.pi, shape[0], endpoint=False)),
                    np.sin(np.linspace(0.0, 2.0 * np.pi, shape[0], endpoint=False)),
                    np.zeros(shape[0]),
                ),
                axis=1,
            ),
            dtype=np.float32,
        ),
        "ocean": f32(0.0),
        "shelf": f32(0.0),
        "relief": f32(400.0),
        "elevation": f32(500.0),
        "terrain_slope": f32(0.004),
        "province_class": np.full(shape, 2, dtype=np.uint8),
        "crust_age": f32(2.2),
        "rock_strength": f32(0.7),
        "accommodation": f32(0.2),
        "province_confidence": f32(0.9),
        "elevation_confidence": f32(0.85),
        "convergence": f32(0.1),
        "divergence": f32(0.1),
        "shear": f32(0.1),
        "subduction": f32(0.0),
        "hotspot": f32(0.0),
        "uplift": f32(0.1),
        "subsidence": f32(0.1),
        "compression": f32(0.1),
        "extension": f32(0.1),
        "stiffness": f32(0.8),
        "temperature": f32(18.0),
        "precipitation": f32(900.0),
        "aridity": f32(0.35),
        "contributing_area": np.full(shape, 100.0, dtype=np.float64),
        "stream_power": f32(1_000.0),
        "river": f32(0.05),
        "floodplain": f32(0.05),
        "lake": f32(0.0),
        "wetland": f32(0.05),
        "bedrock": f32(0.25),
        "residual_regolith": f32(0.55),
        "alluvium": f32(0.08),
        "lacustrine": f32(0.02),
        "volcaniclastic": f32(0.02),
        "soil_depth": f32(1.0),
        "salinity": f32(0.05),
        "drainage": f32(0.65),
        "hydric_soil": f32(0.08),
        "soil_confidence": f32(0.85),
        "annual_npp": f32(0.35),
        "standing_biomass": f32(4.0),
        "vegetation_cover": f32(0.6),
        "biosphere_confidence": f32(0.85),
    }

    # Arc-magmatic setting.
    inputs["province_class"][1] = 6
    for name, value in (
        ("subduction", 0.95),
        ("convergence", 0.8),
        ("uplift", 0.7),
        ("volcaniclastic", 0.7),
    ):
        inputs[name][1] = value

    # Active ocean ridge for VMS; every terrestrial family must remain zero.
    inputs["ocean"][2] = 1.0
    inputs["shelf"][2] = 0.4
    inputs["province_class"][2] = 9
    inputs["divergence"][2] = 0.95
    inputs["hotspot"][2] = 0.6
    inputs["crust_age"][2] = 0.05

    # Productive, waterlogged, subsiding coal basin.
    inputs["province_class"][3] = 3
    inputs["accommodation"][3] = 0.9
    inputs["subsidence"][3] = 0.7
    inputs["wetland"][3] = 0.85
    inputs["hydric_soil"][3] = 0.9
    inputs["annual_npp"][3] = 0.9
    inputs["standing_biomass"][3] = 15.0
    inputs["vegetation_cover"][3] = 0.95
    inputs["drainage"][3] = 0.1

    # Warm/wet residual weathering profile.
    inputs["temperature"][4] = 27.0
    inputs["precipitation"][4] = 2_200.0
    inputs["aridity"][4] = 0.08
    inputs["residual_regolith"][4] = 0.92
    inputs["terrain_slope"][4] = 0.001

    # Energetic river entering a broad alluvial trap.
    inputs["province_class"][5] = 4
    inputs["relief"][5] = 1_600.0
    inputs["river"][5] = 0.95
    inputs["floodplain"][5] = 0.8
    inputs["alluvium"][5] = 0.85
    inputs["contributing_area"][5] = 100_000.0
    inputs["stream_power"][5] = 5.0e8

    # Arid saline sedimentary basin.
    inputs["province_class"][6] = 3
    inputs["accommodation"][6] = 0.9
    inputs["aridity"][6] = 0.95
    inputs["precipitation"][6] = 80.0
    inputs["salinity"][6] = 0.9
    inputs["lacustrine"][6] = 0.8
    inputs["lake"][6] = 0.5

    outputs = {
        "source_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "process_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "transport_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "trap_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "timing_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "preservation_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "unresolved_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "potential_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "confidence_out": np.zeros((SYSTEM_COUNT, *shape), dtype=np.float32),
        "commodity_out": np.zeros((COMMODITY_COUNT, *shape), dtype=np.float32),
        "dominant_system_out": np.zeros(shape, dtype=np.uint8),
    }
    run_mineral_systems(
        face_resolution=128,
        seed=seed,
        minimum_dominant_potential=0.14,
        **inputs,
        **outputs,
    )
    return outputs


def test_config_rejects_unknown_and_incoherent_thresholds():
    with pytest.raises(ValueError, match="Unknown mineral-system controls"):
        MineralSystemsConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="no lower"):
        MineralSystemsConfig.from_mapping(
            {
                "system_catalog_potential_threshold": 0.5,
                "deposit_minimum_potential": 0.4,
            }
        )


def test_pipeline_command_failure_catalog_tracks_hard_gate_metadata():
    passing = SimpleNamespace(
        artifact_records={
            "MineralSystemsValidationMetadata": SimpleNamespace(
                value={"hard_gate_pass": 1, "hard_failures": []}
            )
        }
    )
    failing = SimpleNamespace(
        artifact_records={
            "MineralSystemsValidationMetadata": SimpleNamespace(
                value={
                    "hard_gate_pass": 0,
                    "hard_failures": ["coal_basin_directional_or_noncollapse"],
                }
            )
        }
    )
    assert _mineral_validation_failures(passing) == ()
    assert _mineral_validation_failures(failing) == ("coal_basin_directional_or_noncollapse",)
    assert _mineral_validation_failures(SimpleNamespace(artifact_records={})) == (
        "missing_validation_metadata",
    )


def test_pipeline_command_returns_failure_for_red_mineral_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    config = _integration_config(tmp_path, "red-mineral-gate")
    failing = SimpleNamespace(
        artifact_records={
            "MineralSystemsValidationMetadata": SimpleNamespace(
                value={
                    "hard_gate_pass": 0,
                    "hard_failures": ["coal_basin_directional_or_noncollapse"],
                }
            )
        }
    )

    class FakeEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _stages):
            return {"mineral_systems_validation": failing}

    monkeypatch.setattr(
        PipelineConfig,
        "from_file",
        classmethod(lambda _cls, _path: config),
    )
    monkeypatch.setattr("map_maker.pipeline.tools.ExecutionEngine", FakeEngine)
    status = pipeline_tools_main(
        [
            "--stage",
            "mineral_systems_validation",
            "--config",
            str(tmp_path / "unused.yaml"),
            "--skip-visuals",
        ]
    )
    assert status == 1
    assert "failed hard validation" in capsys.readouterr().err


def test_equatorial_scale_bar_is_adaptive_and_physically_exact():
    circumference_km = 60_045.0
    distance_km, pixels = _equatorial_scale_bar(circumference_km, 512)
    assert distance_km == 5_000.0
    assert pixels == round(512 * distance_km / circumference_km)
    assert 32 <= pixels <= 64

    small_map_distance, small_map_pixels = _equatorial_scale_bar(40_030.0, 128)
    assert small_map_distance == 10_000.0
    assert small_map_pixels == round(128 * small_map_distance / 40_030.0)
    assert 24 <= small_map_pixels <= 64


def test_native_systems_are_bounded_causal_and_reconstructable():
    outputs = _native_case()
    causal = {
        artifact_name: outputs[native_name]
        for artifact_name, native_name in zip(
            CAUSAL_OUTPUTS,
            (
                "source_out",
                "process_out",
                "transport_out",
                "trap_out",
                "timing_out",
                "preservation_out",
            ),
            strict=True,
        )
    }
    for values in outputs.values():
        assert np.all(np.isfinite(values))
    for name in (
        "source_out",
        "process_out",
        "transport_out",
        "trap_out",
        "timing_out",
        "preservation_out",
        "unresolved_out",
        "potential_out",
        "confidence_out",
        "commodity_out",
    ):
        assert np.all((outputs[name] >= 0.0) & (outputs[name] <= 1.0))
    reconstructed = _potential_reconstruction(causal, outputs["unresolved_out"])
    np.testing.assert_allclose(reconstructed, outputs["potential_out"], atol=2e-6)

    potential = outputs["potential_out"]
    assert potential[0, 1] > potential[0, 0]  # arc
    assert potential[9, 3] > potential[9, 0]  # coal
    assert potential[6, 4] > potential[6, 0]  # supergene weathering
    assert potential[7, 5] > potential[7, 0]  # placer
    assert potential[8, 6] > potential[8, 0]  # evaporite
    assert potential[3, 2] > 0.0  # marine VMS
    assert potential[3, 1] == 0.0  # no inferred paleo-seafloor lineage on land
    np.testing.assert_array_equal(
        np.delete(potential[:, 2], 3),
        np.zeros(SYSTEM_COUNT - 1, dtype=np.float32),
    )


def test_native_replay_is_exact_and_seed_only_changes_unresolved_detail():
    first = _native_case(seed=42)
    replay = _native_case(seed=42)
    changed = _native_case(seed=43)
    for name in first:
        np.testing.assert_array_equal(first[name], replay[name])
    assert not np.array_equal(first["unresolved_out"], changed["unresolved_out"])
    np.testing.assert_array_equal(first["source_out"], changed["source_out"])
    assert not np.array_equal(first["potential_out"], changed["potential_out"])


def test_pipeline_persists_catalogs_validates_and_replays(
    tmp_path: Path,
    registered_mineral_stages,
):
    del registered_mineral_stages
    config = _integration_config(tmp_path, "minerals-e2e")
    first = ExecutionEngine(config, generate_visuals=True).run(["mineral_systems_validation"])
    mineral = first["mineral_systems"]
    validation = first["mineral_systems_validation"]
    potential = np.asarray(mineral.artifact_records["MineralSystemPotential"].value.array())
    assert potential.shape == (SYSTEM_COUNT, 6, 16, 16)
    system_catalog = mineral.artifact_records["MineralSystemCatalog"].value
    deposit_catalog = mineral.artifact_records["MajorDepositCandidateCatalog"].value
    assert isinstance(system_catalog, pa.Table)
    assert isinstance(deposit_catalog, pa.Table)
    assert system_catalog.num_rows >= SYSTEM_COUNT
    assert deposit_catalog.num_rows >= SYSTEM_COUNT
    validation_metadata = validation.artifact_records["MineralSystemsValidationMetadata"].value
    assert validation_metadata["hard_gate_pass"] == 1
    visual_dir = config.run_visual_dir() / "mineral_systems"
    assert (visual_dir / "dominant_mineral_systems.png").is_file()
    assert (visual_dir / "commodity_prospectivity_atlas.png").is_file()
    assert (visual_dir / "major_deposit_candidates.png").is_file()

    replay = ExecutionEngine(config, generate_visuals=False).run(["mineral_systems_validation"])
    for artifact_name in (
        "MineralSystemPotential",
        "CommodityProspectivity",
        "MineralSystemCatalog",
        "MajorDepositCandidateCatalog",
    ):
        assert (
            mineral.artifact_records[artifact_name].checksum
            == replay["mineral_systems"].artifact_records[artifact_name].checksum
        )
    assert all(result.stats is not None and result.stats.cache_hit for result in replay.values())

    cold_config = _integration_config(tmp_path / "isolated-cold", "minerals-e2e-cold")
    cold = ExecutionEngine(cold_config, generate_visuals=False).run(["mineral_systems_validation"])
    for artifact_name in (
        *CAUSAL_OUTPUTS,
        "MineralUnresolvedSupport",
        "MineralSystemPotential",
        "MineralSystemConfidence",
        "CommodityProspectivity",
        "DominantMineralSystemCode",
        "MineralSystemCatalog",
        "MajorDepositCandidateCatalog",
        "MineralSystemsMetadata",
    ):
        assert (
            mineral.artifact_records[artifact_name].checksum
            == cold["mineral_systems"].artifact_records[artifact_name].checksum
        )
    for artifact_name in (
        "MineralSystemsValidationCatalog",
        "MineralSystemsValidationMetadata",
    ):
        assert (
            validation.artifact_records[artifact_name].checksum
            == cold["mineral_systems_validation"].artifact_records[artifact_name].checksum
        )
