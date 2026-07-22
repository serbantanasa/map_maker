from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.pipeline import ExecutionEngine, PipelineConfig, registry
from map_maker.pipeline.stages.hydrology_pass2 import NO_FIXED_RECEIVER
from map_maker.pipeline.stages.hydrology_validation import HydrologyValidationConfig
from map_maker.pipeline.stages.lake_hydrographs import (
    LakeHydrographConfig,
    _find_effective_start,
)
from map_maker.pipeline.stages.outlet_incision import OutletIncisionConfig, _cyclic_rows


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
        "hydrology_pass2",
        "surface_water",
        "outlet_incision",
        "lake_hydrographs",
        "hydrology_validation",
        "surface_materials",
        "biosphere_envelope",
        "potential_biosphere",
        "biosphere_validation",
    ):
        module = importlib.import_module(f"map_maker.pipeline.stages.{module_name}")
        importlib.reload(module)
    yield


def _config(
    tmp_path: Path, run_id: str, *, minimum_outlet_discharge_m3s: float = 0.10
) -> PipelineConfig:
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
                    "minimum_outlet_erosion_discharge_m3s": minimum_outlet_discharge_m3s,
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
            },
        }
    )


def _table(result, name: str) -> pa.Table:
    value = result.artifact_records[name].value
    assert isinstance(value, pa.Table)
    return value


def test_cycle_detector_returns_only_cyclic_support():
    cell_ids = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    receivers = np.array([11, 12, 11, 12, -1], dtype=np.int32)

    assert _cyclic_rows(cell_ids, receivers).tolist() == [1, 2]


def test_outlet_config_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="Unknown outlet-incision controls"):
        OutletIncisionConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="maximum_reroute_repair_rounds"):
        OutletIncisionConfig.from_mapping({"maximum_reroute_repair_rounds": 0})
    with pytest.raises(ValueError, match="maximum_reroute_constraint_cell_fraction"):
        OutletIncisionConfig.from_mapping({"maximum_reroute_constraint_cell_fraction": 1.1})


def test_hydrology_validation_config_rejects_invalid_controls():
    with pytest.raises(ValueError, match="Unknown hydrology-validation controls"):
        HydrologyValidationConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="major_river_discharge_threshold_m3s"):
        HydrologyValidationConfig.from_mapping({"major_river_discharge_threshold_m3s": 0.0})


def test_lake_hydrograph_config_rejects_invalid_controls():
    with pytest.raises(ValueError, match="Unknown lake-hydrograph controls"):
        LakeHydrographConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="maximum_negative_discharge_m3s"):
        LakeHydrographConfig.from_mapping({"maximum_negative_discharge_m3s": 2.0})


def test_lake_loss_projection_is_bounded_by_available_channel_discharge():
    base_entry = np.full((2, 12), 8.0)
    base_exit = np.vstack((np.full(12, 4.0), np.full(12, 8.0)))
    adjustment = np.zeros((2, 12), dtype=np.float64)

    limited = _find_effective_start(
        [0, 1],
        0,
        -10.0,
        False,
        base_entry,
        base_exit,
        adjustment,
        adjustment,
        1e-6,
    )
    represented = _find_effective_start(
        [0, 1],
        0,
        -3.0,
        False,
        base_entry,
        base_exit,
        adjustment,
        adjustment,
        1e-6,
    )

    assert limited.start_index == 0
    assert limited.applied_delta_m3s == -4.0
    assert not limited.fully_represented
    assert represented.start_index == 0
    assert represented.applied_delta_m3s == -3.0
    assert represented.fully_represented


def test_final_surface_water_converges_with_bounded_persistent_outlets(tmp_path: Path):
    config = _config(tmp_path, "outlet-final")
    results = ExecutionEngine(config).run(["biosphere_validation"])
    outlet = results["outlet_incision"]
    final = results["surface_water_final"]
    lake_hydrographs = results["lake_hydrographs"]
    validation = results["hydrology_validation"]
    surface_materials = results["surface_materials"]
    biosphere_envelope = results["biosphere_envelope"]
    potential_biosphere = results["potential_biosphere"]
    biosphere_validation = results["biosphere_validation"]
    outlet_metadata = outlet.artifact_records["OutletIncisionMetadata"].value
    final_metadata = final.artifact_records["SurfaceWaterMetadata"].value
    cells = _table(final, "FinalOutletCorrectedBasinCellCatalog")
    iterations = _table(final, "OutletIncisionIterationCatalog")
    kpis = _table(validation, "HydrologyKpiCatalog")
    reach_losses = _table(validation, "HydrologyReachLossCatalog")
    coupled_reaches = _table(lake_hydrographs, "LakeCoupledRiverReachCatalog")
    hydrograph_adjustments = _table(lake_hydrographs, "LakeHydrographAdjustmentCatalog")
    lake_hydrograph_metadata = lake_hydrographs.artifact_records["LakeHydrographMetadata"].value
    validation_metadata = validation.artifact_records["HydrologyValidationMetadata"].value
    soil_metadata = surface_materials.artifact_records["SurfaceMaterialsMetadata"].value
    envelope_metadata = biosphere_envelope.artifact_records["BiosphereEnvelopeMetadata"].value
    potential_metadata = potential_biosphere.artifact_records["PotentialBiosphereMetadata"].value
    biosphere_validation_metadata = biosphere_validation.artifact_records[
        "BiosphereValidationMetadata"
    ].value
    biosphere_kpis = _table(biosphere_validation, "BiosphereKpiCatalog")
    biosphere_distributions = _table(biosphere_validation, "BiosphereClimateDistributionCatalog")

    assert outlet_metadata["graph_valid"] == 1
    assert outlet_metadata["independent_graph_valid"] == 1
    assert outlet_metadata["trunk_preserved_valid"] == 1
    assert outlet_metadata["process_exclusion_valid"] == 1
    assert outlet_metadata["corrected_area_fraction"] <= 0.10
    assert outlet_metadata["independent_receiver_changed_area_fraction"] <= 0.15
    assert final_metadata["outlet_erosion_required_count"] == 0
    assert final_metadata["outlet_correction_converged"] == 1
    assert final_metadata["outlet_resolution_contract_satisfied"] == 1
    assert final_metadata["regional_refinement_deferred_outlet_candidate_count"] == 0
    assert final_metadata["surface_water_ready_for_soils"] == 1
    assert iterations["residual_feedback_candidate_count"][-1].as_py() == 0
    assert len(cells.column_names) == len(set(cells.column_names))
    assert validation_metadata["kpi_count"] == kpis.num_rows
    assert validation_metadata["reference_profile_version"] == "earth_hydrology_v1"
    assert validation_metadata["hard_gate_pass"] == 1
    assert validation_metadata["unaccounted_reach_loss_count"] == 0
    assert lake_hydrograph_metadata["lake_reach_hydrograph_coupling_implemented"] == 1
    assert lake_hydrograph_metadata["network_balance_relative_error"] <= 1e-9
    assert lake_hydrograph_metadata["minimum_coupled_discharge_m3s"] >= -1e-6
    assert hydrograph_adjustments.num_rows == 12 * int(
        lake_hydrograph_metadata["terminal_lake_network_count"]
    )
    assert {
        "requested_hydrograph_adjustment_m3s",
        "pre_channel_interception_km3",
        "fully_represented_in_channel",
    }.issubset(hydrograph_adjustments.column_names)
    assert lake_hydrograph_metadata["pre_channel_interception_km3"] >= 0.0
    coupled_monthly = np.asarray(
        coupled_reaches["discharge_seasonal"].combine_chunks().values,
        dtype=np.float64,
    )
    coupled_exit = np.asarray(
        coupled_reaches["exit_discharge_seasonal"].combine_chunks().values,
        dtype=np.float64,
    )
    assert np.all(coupled_monthly >= 0.0)
    assert np.all(coupled_exit >= 0.0)
    if reach_losses.num_rows:
        assert all(reach_losses["accounted_by_registered_storage"].to_pylist())
    kpi_by_id = {
        row["kpi_id"]: row
        for row in kpis.select(["kpi_id", "value", "gate_kind", "comparison_status"]).to_pylist()
    }
    assert kpi_by_id["candidate_graph_valid"]["comparison_status"] == "hard_pass"
    assert kpi_by_id["global_river_reach_graph_issue_count"]["comparison_status"] == "hard_pass"
    assert kpi_by_id["global_longest_source_to_terminal_river_path_km"]["value"] > 0.0
    assert kpi_by_id["global_longest_source_to_terminal_channel_km"]["value"] > 0.0
    assert kpi_by_id["seasonal_snow_storage_implemented"]["value"] == 1.0
    assert kpi_by_id["glacier_mass_balance_implemented"]["value"] == 1.0
    assert kpi_by_id["seasonal_sea_ice_implemented"]["value"] == 1.0
    assert kpi_by_id["global_mean_sea_ice_ocean_area_fraction"]["comparison_status"] in {
        "within_reference",
        "outside_reference",
    }
    assert kpi_by_id["lake_reach_hydrograph_coupling_implemented"]["value"] == 1.0
    assert kpi_by_id["floodplain_inundation_implemented"]["value"] == 0.0
    assert soil_metadata["surface_materials_ready_for_biomes"] == 1
    assert (
        soil_metadata["refined_surface_projection"]
        == "conservative_applied_child_process_area_km2_v3"
    )
    assert soil_metadata["coarse_prospective_fluvial_budgets_consumed"] == 0
    assert 0.0 <= soil_metadata["effective_lake_land_area_fraction"] <= 1.0
    assert 0.0 <= soil_metadata["effective_wetland_land_area_fraction"] <= 1.0
    assert soil_metadata["material_balance_max_error"] <= 1e-5
    assert soil_metadata["texture_balance_max_error"] <= 1e-5
    assert soil_metadata["water_balance_relative_error"] <= 1e-5
    assert envelope_metadata["hard_gate_pass"] == 1
    assert envelope_metadata["biosphere_envelope_ready_for_traits"] == 1
    assert envelope_metadata["combined_energy_is_universal_habitability_score"] == 0
    assert potential_metadata["hard_gate_pass"] == 1
    assert potential_metadata["potential_biosphere_ready_for_functional_types"] == 1
    assert potential_metadata["actual_vegetation_state_implemented"] == 0
    assert (
        potential_metadata["actual_maximum_rooting_depth_m"]
        <= potential_metadata["maximum_rooting_depth_m"]
    )
    assert (
        potential_metadata["actual_maximum_canopy_height_m"]
        <= potential_metadata["maximum_canopy_height_m"]
    )
    assert biosphere_validation_metadata["reference_profile_version"] == "earth_biosphere_v1"
    assert biosphere_validation_metadata["hard_gate_pass"] == 1
    assert biosphere_validation_metadata["earth_profile_status"] in {
        "within_reference",
        "outside_reference",
    }
    assert biosphere_kpis.num_rows == biosphere_validation_metadata["kpi_count"]
    assert (
        biosphere_distributions.num_rows
        == biosphere_validation_metadata["climate_distribution_row_count"]
    )
    biosphere_kpi_by_id = {
        row["kpi_id"]: row
        for row in biosphere_kpis.select(
            ["kpi_id", "value", "gate_kind", "comparison_status"]
        ).to_pylist()
    }
    assert (
        biosphere_kpi_by_id["finite_nonnegative_biosphere_fields"]["comparison_status"]
        == "hard_pass"
    )
    assert biosphere_kpi_by_id["global_potential_npp_pg_c_year"]["gate_kind"] == (
        "earth_diagnostic"
    )
    annual_energy = np.asarray(
        biosphere_envelope.artifact_records[
            "AnnualTerrestrialPrimaryEnergyPotentialMJm2"
        ].value.array()
    )
    ocean = (
        np.asarray(results["sea_level"].artifact_records["SurfaceOceanMask"].value.array()) >= 0.5
    )
    assert np.all(annual_energy[ocean] == 0.0)
    assert np.any(annual_energy[~ocean] > 0.0)
    potential_cover = np.asarray(
        potential_biosphere.artifact_records["PotentialVegetationCoverFraction"].value.array()
    )
    assert np.all(potential_cover[ocean] == 0.0)
    assert np.any(potential_cover[~ocean] > 0.0)

    fixed = np.asarray(cells["outlet_fixed_receiver_id"], dtype=np.int32)
    constrained = fixed != NO_FIXED_RECEIVER
    anchors = np.asarray(cells["routing_anchor_kind"].to_pylist())
    assert np.any(constrained)
    assert np.all(anchors[constrained] == "ordinary")
    assert np.all(
        np.asarray(cells["stabilized_receiver_id"], dtype=np.int32)[constrained]
        == fixed[constrained]
    )

    cached = ExecutionEngine(config).run(["surface_water_final"])["surface_water_final"]
    assert cached.stats is not None and cached.stats.cache_hit
    cached_validation = ExecutionEngine(config).run(["hydrology_validation"])[
        "hydrology_validation"
    ]
    assert cached_validation.stats is not None and cached_validation.stats.cache_hit


def test_final_surface_water_defers_round_limited_outlets_to_regional_refinement(
    tmp_path: Path,
):
    config = _config(tmp_path, "outlet-resolution-deferral")
    overrides = {name: dict(values) for name, values in config.stage_overrides.items()}
    overrides["surface_water_final"]["maximum_outlet_incision_rounds"] = 1
    config.stage_overrides = overrides

    results = ExecutionEngine(config).run(["surface_water_final", "hydrology_validation"])
    result = results["surface_water_final"]
    metadata = result.artifact_records["SurfaceWaterMetadata"].value
    candidates = _table(result, "SurfaceWaterCandidateCatalog")
    iterations = _table(result, "OutletIncisionIterationCatalog")
    validation_metadata = (
        results["hydrology_validation"].artifact_records["HydrologyValidationMetadata"].value
    )
    validation_kpis = _table(results["hydrology_validation"], "HydrologyKpiCatalog")

    assert metadata["regional_refinement_deferred_outlet_candidate_count"] > 0
    assert metadata["regional_refinement_deferred_outlet_mean_water_area_km2"] > 0.0
    assert metadata["outlet_erosion_required_count"] == 0
    assert metadata["outlet_correction_converged"] == 0
    assert metadata["outlet_resolution_contract_satisfied"] == 1
    assert metadata["surface_water_ready_for_soils"] == 1
    assert iterations["residual_feedback_candidate_count"][-1].as_py() > 0
    assert "regional_refinement_deferred" in candidates["classification_reason"].to_pylist()
    assert validation_metadata["hard_gate_pass"] == 1
    kpi_by_id = {
        row["kpi_id"]: row for row in validation_kpis.select(["kpi_id", "value"]).to_pylist()
    }
    assert kpi_by_id["outlet_resolution_contract_satisfied"]["value"] == 1.0
    assert kpi_by_id["outlet_correction_converged"]["value"] == 0.0


def test_final_surface_water_accepts_a_zero_correction_world(tmp_path: Path):
    config = _config(
        tmp_path,
        "outlet-noop",
        minimum_outlet_discharge_m3s=100_000.0,
    )

    results = ExecutionEngine(config).run(["surface_water_final"])
    outlet_metadata = results["outlet_incision"].artifact_records["OutletIncisionMetadata"].value
    final = results["surface_water_final"]
    final_metadata = final.artifact_records["SurfaceWaterMetadata"].value
    iterations = _table(final, "OutletIncisionIterationCatalog")

    assert outlet_metadata["requested_candidate_count"] == 0
    assert outlet_metadata["corrected_cell_count"] == 0
    assert final_metadata["outlet_incision_iteration_count"] == 1
    assert final_metadata["outlet_erosion_required_count"] == 0
    assert final_metadata["surface_water_ready_for_soils"] == 1
    assert iterations["residual_feedback_candidate_count"].to_pylist() == [0]
