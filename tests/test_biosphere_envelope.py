from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline._biosphere_envelope_native import run_biosphere_envelope
from map_maker.pipeline.stages.biosphere_envelope import BiosphereEnvelopeConfig


def _native_case() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    shape = (6, 2, 2)
    monthly_shape = (12, *shape)
    fields = {
        "areas": np.full(shape, 1.0e10, dtype=np.float64),
        "ocean": np.zeros(shape, dtype=np.float32),
        "monthly_insolation": np.full(monthly_shape, 300.0, dtype=np.float32),
        "monthly_temperature": np.full(monthly_shape, 20.0, dtype=np.float32),
        "monthly_liquid_input": np.full(monthly_shape, 100.0, dtype=np.float32),
        "monthly_soil_saturation": np.full(monthly_shape, 0.8, dtype=np.float32),
        "soil_bearing": np.full(shape, 0.9, dtype=np.float32),
        "nutrient_potential": np.full(shape, 0.8, dtype=np.float32),
        "fertility_potential": np.full(shape, 0.8, dtype=np.float32),
        "salinity": np.full(shape, 0.1, dtype=np.float32),
        "soil_confidence": np.full(shape, 0.9, dtype=np.float32),
        "co2_partial_pressure": np.full(shape, 28.371, dtype=np.float32),
        "oxygen_partial_pressure": np.full(shape, 21.22, dtype=np.float32),
    }
    fields["ocean"].reshape(-1)[0] = 1.0
    fields["monthly_liquid_input"].reshape(12, -1)[:, 1] = 0.0
    fields["monthly_soil_saturation"].reshape(12, -1)[:, 1] = 0.0
    fields["monthly_temperature"].reshape(12, -1)[:, 2] = -20.0
    fields["co2_partial_pressure"].reshape(-1)[3] = 56.742

    monthly_outputs = {
        "monthly_par_out",
        "monthly_liquid_opportunity_out",
        "monthly_thermal_opportunity_out",
        "monthly_primary_energy_out",
    }
    output_names = monthly_outputs | {
        "annual_par_out",
        "annual_primary_energy_out",
        "carbon_substrate_relative_out",
        "aerobic_oxygen_relative_out",
        "terrestrial_surface_support_out",
        "nutrient_support_out",
        "environmental_stress_out",
        "confidence_out",
    }
    outputs = {
        name: np.zeros(monthly_shape if name in monthly_outputs else shape, dtype=np.float32)
        for name in output_names
    }
    metadata = run_biosphere_envelope(
        seconds_per_month=365.2422 * 86_400.0 / 12.0,
        par_fraction=0.43,
        shortwave_transmission=0.65,
        thermal_minimum_c=-10.0,
        thermal_optimum_low_c=15.0,
        thermal_optimum_high_c=30.0,
        thermal_maximum_c=50.0,
        water_input_half_saturation_mm=50.0,
        nutrient_half_saturation_index=0.5,
        co2_half_saturation_pa=20.0,
        reference_co2_partial_pressure_pa=28.371,
        reference_oxygen_partial_pressure_kpa=21.22,
        photosynthetic_conversion_efficiency=0.02,
        minimum_productive_energy_mj_m2_year=5.0,
        confidence_multiplier=1.0,
        **fields,
        **outputs,
    )
    return outputs, metadata


def test_native_envelope_separates_resources_and_respects_energy_bounds():
    outputs, metadata = _native_case()
    flat_annual = outputs["annual_primary_energy_out"].reshape(-1)

    assert outputs["monthly_par_out"].reshape(12, -1)[:, 0].min() > 0.0
    assert flat_annual[0] == 0.0
    assert outputs["terrestrial_surface_support_out"].reshape(-1)[0] == 0.0
    assert flat_annual[1] == 0.0
    assert flat_annual[2] == 0.0
    assert flat_annual[3] > flat_annual[4]
    assert outputs["carbon_substrate_relative_out"].reshape(-1)[3] > 1.0
    assert outputs["aerobic_oxygen_relative_out"].reshape(-1)[3] == pytest.approx(1.0)
    np.testing.assert_allclose(
        outputs["annual_par_out"],
        np.sum(outputs["monthly_par_out"], axis=0),
        rtol=1e-6,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        outputs["annual_primary_energy_out"],
        np.sum(outputs["monthly_primary_energy_out"], axis=0),
        rtol=1e-6,
        atol=1e-5,
    )
    assert np.all(outputs["monthly_primary_energy_out"] <= outputs["monthly_par_out"] * 0.02 + 1e-6)
    assert 0.0 < metadata["potentially_productive_land_area_fraction"] < 1.0


def test_native_envelope_rejects_overlapping_outputs():
    shape = (6, 1, 1)
    monthly_shape = (12, *shape)
    shared = np.zeros(monthly_shape, dtype=np.float32)
    args = {
        "seconds_per_month": 2.6e6,
        "par_fraction": 0.43,
        "shortwave_transmission": 0.65,
        "thermal_minimum_c": -10.0,
        "thermal_optimum_low_c": 15.0,
        "thermal_optimum_high_c": 30.0,
        "thermal_maximum_c": 50.0,
        "water_input_half_saturation_mm": 50.0,
        "nutrient_half_saturation_index": 0.5,
        "co2_half_saturation_pa": 20.0,
        "reference_co2_partial_pressure_pa": 28.0,
        "reference_oxygen_partial_pressure_kpa": 21.22,
        "photosynthetic_conversion_efficiency": 0.02,
        "minimum_productive_energy_mj_m2_year": 5.0,
        "confidence_multiplier": 1.0,
        "areas": np.ones(shape, dtype=np.float64),
        "ocean": np.zeros(shape, dtype=np.float32),
        "monthly_insolation": np.ones(monthly_shape, dtype=np.float32),
        "monthly_temperature": np.ones(monthly_shape, dtype=np.float32),
        "monthly_liquid_input": np.ones(monthly_shape, dtype=np.float32),
        "monthly_soil_saturation": np.ones(monthly_shape, dtype=np.float32),
        "soil_bearing": np.ones(shape, dtype=np.float32),
        "nutrient_potential": np.ones(shape, dtype=np.float32),
        "fertility_potential": np.ones(shape, dtype=np.float32),
        "salinity": np.zeros(shape, dtype=np.float32),
        "soil_confidence": np.ones(shape, dtype=np.float32),
        "co2_partial_pressure": np.full(shape, 28.0, dtype=np.float32),
        "oxygen_partial_pressure": np.full(shape, 21.22, dtype=np.float32),
        "monthly_par_out": shared,
        "monthly_liquid_opportunity_out": shared,
        "monthly_thermal_opportunity_out": np.zeros(monthly_shape, dtype=np.float32),
        "monthly_primary_energy_out": np.zeros(monthly_shape, dtype=np.float32),
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name in (
                "annual_par_out",
                "annual_primary_energy_out",
                "carbon_substrate_relative_out",
                "aerobic_oxygen_relative_out",
                "terrestrial_surface_support_out",
                "nutrient_support_out",
                "environmental_stress_out",
                "confidence_out",
            )
        },
    }
    with pytest.raises(ValueError, match="must not overlap"):
        run_biosphere_envelope(**args)


def test_biosphere_envelope_config_validates_response_contract():
    with pytest.raises(ValueError, match="Unknown biosphere-envelope controls"):
        BiosphereEnvelopeConfig.from_mapping({"paint_biomes": True})
    with pytest.raises(ValueError, match="temperatures must be ordered"):
        BiosphereEnvelopeConfig.from_mapping(
            {"thermal_optimum_low_c": 35.0, "thermal_optimum_high_c": 25.0}
        )
