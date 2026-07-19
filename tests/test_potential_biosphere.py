from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline._potential_biosphere_native import run_potential_biosphere
from map_maker.pipeline.stages.potential_biosphere import PotentialBiosphereConfig

MONTHLY_OUTPUTS = {"monthly_npp_out"}
SCALAR_OUTPUTS = {
    "annual_npp_out",
    "vegetation_cover_out",
    "standing_biomass_out",
    "growing_season_out",
    "productivity_seasonality_out",
    "drought_pressure_out",
    "cold_pressure_out",
    "heat_pressure_out",
    "waterlogging_pressure_out",
    "salinity_pressure_out",
    "woody_trait_out",
    "resource_conservative_trait_out",
    "rooting_depth_out",
    "canopy_height_out",
    "leaf_area_index_out",
    "fuel_continuity_out",
    "confidence_out",
}


def _native_arguments() -> dict[str, object]:
    shape = (6, 2, 2)
    monthly_shape = (12, *shape)
    fields = {
        "areas": np.full(shape, 1.0e10, dtype=np.float64),
        "ocean": np.zeros(shape, dtype=np.float32),
        "monthly_primary_energy": np.full(monthly_shape, 0.4, dtype=np.float32),
        "monthly_thermal_opportunity": np.full(monthly_shape, 0.9, dtype=np.float32),
        "monthly_water_opportunity": np.full(monthly_shape, 0.8, dtype=np.float32),
        "monthly_temperature": np.full(monthly_shape, 22.0, dtype=np.float32),
        "monthly_soil_saturation": np.full(monthly_shape, 0.65, dtype=np.float32),
        "surface_support": np.full(shape, 0.9, dtype=np.float32),
        "nutrient_support": np.full(shape, 0.8, dtype=np.float32),
        "environmental_stress": np.full(shape, 0.2, dtype=np.float32),
        "soil_depth": np.full(shape, 1.2, dtype=np.float32),
        "regolith_depth": np.full(shape, 3.0, dtype=np.float32),
        "salinity": np.full(shape, 0.05, dtype=np.float32),
        "hydric_fraction": np.full(shape, 0.05, dtype=np.float32),
        "soil_confidence": np.full(shape, 0.9, dtype=np.float32),
        "envelope_confidence": np.full(shape, 0.8, dtype=np.float32),
    }
    fields["ocean"].reshape(-1)[0] = 1.0
    fields["monthly_primary_energy"].reshape(12, -1)[:, 0] = 0.0

    # Dry and cold cells have no inherited primary-energy opportunity.
    fields["monthly_primary_energy"].reshape(12, -1)[:, 1:3] = 0.0
    fields["monthly_water_opportunity"].reshape(12, -1)[:, 1] = 0.0
    fields["monthly_soil_saturation"].reshape(12, -1)[:, 1] = 0.0
    fields["monthly_thermal_opportunity"].reshape(12, -1)[:, 2] = 0.0
    fields["monthly_temperature"].reshape(12, -1)[:, 2] = -20.0

    # Cell 3 is wet and productive; cell 4 has the same annual energy in one pulse.
    fields["monthly_primary_energy"].reshape(12, -1)[:, 3] = 0.8
    fields["monthly_soil_saturation"].reshape(12, -1)[:, 3] = 0.95
    fields["hydric_fraction"].reshape(-1)[3] = 0.7
    fields["monthly_primary_energy"].reshape(12, -1)[:, 4] = 0.0
    fields["monthly_primary_energy"].reshape(12, -1)[5, 4] = 4.8

    # Cell 5 is saline despite otherwise favorable forcing.
    fields["salinity"].reshape(-1)[5] = 0.9

    outputs = {
        name: np.zeros(monthly_shape if name in MONTHLY_OUTPUTS else shape, dtype=np.float32)
        for name in MONTHLY_OUTPUTS | SCALAR_OUTPUTS
    }
    return {
        "energy_per_kg_carbon_mj": 39.9,
        "cover_half_saturation_npp_kg_c_m2_year": 0.25,
        "active_month_thermal_threshold": 0.15,
        "active_month_water_threshold": 0.10,
        "cold_pressure_reference_c": -15.0,
        "cold_pressure_release_c": 10.0,
        "heat_pressure_onset_c": 30.0,
        "heat_pressure_reference_c": 50.0,
        "minimum_biomass_residence_years": 0.5,
        "maximum_biomass_residence_years": 50.0,
        "biomass_residence_baseline_fraction": 0.10,
        "woody_biomass_residence_weight": 0.60,
        "resource_conservative_biomass_residence_weight": 0.40,
        "low_productivity_biomass_residence_weight": 2.50,
        "maximum_rooting_depth_m": 6.0,
        "maximum_canopy_height_m": 45.0,
        "maximum_leaf_area_index": 8.0,
        "maximum_standing_biomass_kg_c_m2": 40.0,
        **fields,
        **outputs,
    }


def test_native_potential_biosphere_is_trait_first_and_energy_bounded():
    args = _native_arguments()
    metadata = run_potential_biosphere(**args)
    monthly_energy = np.asarray(args["monthly_primary_energy"])
    monthly_npp = np.asarray(args["monthly_npp_out"])
    annual_npp = np.asarray(args["annual_npp_out"])
    np.testing.assert_allclose(monthly_npp, monthly_energy / 39.9, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(annual_npp, np.sum(monthly_npp, axis=0), rtol=1e-6, atol=1e-7)

    ocean = np.asarray(args["ocean"]) >= 0.5
    for name in (
        "annual_npp_out",
        "vegetation_cover_out",
        "standing_biomass_out",
        "rooting_depth_out",
        "canopy_height_out",
        "leaf_area_index_out",
        "fuel_continuity_out",
    ):
        assert np.all(np.asarray(args[name])[ocean] == 0.0)

    cover = np.asarray(args["vegetation_cover_out"]).reshape(-1)
    woody = np.asarray(args["woody_trait_out"]).reshape(-1)
    seasonality = np.asarray(args["productivity_seasonality_out"]).reshape(-1)
    assert cover[1] == 0.0 and cover[2] == 0.0
    assert cover[3] > 0.0
    assert seasonality[4] > seasonality[3]
    assert np.asarray(args["drought_pressure_out"]).reshape(-1)[1] > 0.9
    assert np.asarray(args["cold_pressure_out"]).reshape(-1)[2] > 0.9
    assert np.asarray(args["waterlogging_pressure_out"]).reshape(-1)[3] > 0.6
    assert np.asarray(args["salinity_pressure_out"]).reshape(-1)[5] > 0.8
    assert woody[5] < woody[6]

    roots = np.asarray(args["rooting_depth_out"])
    regolith = np.asarray(args["regolith_depth"])
    assert np.all(roots <= regolith + 1e-6)
    assert metadata["land_mean_annual_npp_kg_c_m2"] > 0.0
    assert metadata["maximum_rooting_depth_m"] <= 3.0
    assert metadata["maximum_canopy_height_m"] <= 45.0


def test_native_potential_biosphere_rejects_overlapping_outputs():
    args = _native_arguments()
    args["annual_npp_out"] = args["vegetation_cover_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_potential_biosphere(**args)


def test_potential_biosphere_config_rejects_invalid_response_contracts():
    with pytest.raises(ValueError, match="Unknown potential-biosphere controls"):
        PotentialBiosphereConfig.from_mapping({"paint_forest": True})
    with pytest.raises(ValueError, match="cold pressure reference"):
        PotentialBiosphereConfig.from_mapping(
            {"cold_pressure_reference_c": 15.0, "cold_pressure_release_c": 10.0}
        )
    with pytest.raises(ValueError, match="residence-year bounds"):
        PotentialBiosphereConfig.from_mapping(
            {
                "minimum_biomass_residence_years": 10.0,
                "maximum_biomass_residence_years": 5.0,
            }
        )
    with pytest.raises(ValueError, match="biomass_residence_baseline_fraction"):
        PotentialBiosphereConfig.from_mapping({"biomass_residence_baseline_fraction": 1.1})
    with pytest.raises(ValueError, match="biomass residence weights"):
        PotentialBiosphereConfig.from_mapping(
            {
                "woody_biomass_residence_weight": 0.0,
                "resource_conservative_biomass_residence_weight": 0.0,
                "low_productivity_biomass_residence_weight": 0.0,
            }
        )
