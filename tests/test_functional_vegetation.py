from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline._functional_vegetation_native import (
    FUNCTIONAL_TYPE_COUNT,
    NONVEGETATED_TYPE_COUNT,
    RESOURCE_POTENTIAL_COUNT,
    run_functional_vegetation,
)
from map_maker.pipeline.stages.functional_vegetation import FunctionalVegetationConfig


def _native_arguments() -> dict[str, object]:
    shape = (6, 2, 2)

    def field(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    inputs = {
        "areas": np.full(shape, 1.0e10, dtype=np.float64),
        "ocean": field(0.0),
        "vegetation_cover": field(0.70),
        "annual_npp": field(0.50),
        "standing_biomass": field(8.0),
        "growing_season": field(0.80),
        "productivity_seasonality": field(0.30),
        "drought_pressure": field(0.20),
        "cold_pressure": field(0.10),
        "heat_pressure": field(0.05),
        "waterlogging_pressure": field(0.10),
        "salinity_pressure": field(0.05),
        "woody_trait": field(0.40),
        "resource_conservative_trait": field(0.40),
        "fuel_continuity": field(0.40),
        "biosphere_confidence": field(0.80),
        "annual_temperature": field(15.0),
        "soil_fertility": field(0.70),
        "soil_depth": field(1.0),
        "soil_bearing": field(0.90),
        "soil_drainage": field(0.65),
        "glacier_fraction": field(0.0),
        "lake_fraction": field(0.0),
        "wetland_fraction": field(0.0),
        "terrain_relief": field(200.0),
    }
    flat = {name: value.reshape(-1) for name, value in inputs.items() if name != "areas"}

    flat["ocean"][0] = 1.0
    flat["vegetation_cover"][0] = 0.0
    flat["glacier_fraction"][1] = 0.80
    flat["lake_fraction"][2] = 0.60

    flat["annual_temperature"][3] = 27.0
    flat["drought_pressure"][3] = 0.05
    flat["cold_pressure"][3] = 0.0
    flat["woody_trait"][3] = 0.80
    flat["productivity_seasonality"][3] = 0.10

    flat["annual_temperature"][4] = -5.0
    flat["cold_pressure"][4] = 0.90
    flat["heat_pressure"][4] = 0.0
    flat["woody_trait"][4] = 0.60
    flat["resource_conservative_trait"][4] = 0.70

    flat["waterlogging_pressure"][5] = 0.95
    flat["wetland_fraction"][5] = 0.80
    flat["woody_trait"][5] = 0.10

    flat["drought_pressure"][6] = 0.95
    flat["waterlogging_pressure"][6] = 0.0
    flat["woody_trait"][6] = 0.10
    flat["resource_conservative_trait"][6] = 0.80

    outputs = {
        "functional_type_fractions_out": np.zeros(
            (FUNCTIONAL_TYPE_COUNT, *shape), dtype=np.float32
        ),
        "nonvegetated_fractions_out": np.zeros((NONVEGETATED_TYPE_COUNT, *shape), dtype=np.float32),
        "resource_potentials_out": np.zeros((RESOURCE_POTENTIAL_COUNT, *shape), dtype=np.float32),
        "confidence_out": np.zeros(shape, dtype=np.float32),
        "dominant_cover_code_out": np.zeros(shape, dtype=np.uint8),
    }
    return {
        "warm_transition_midpoint_c": 18.0,
        "warm_transition_width_c": 12.0,
        "npp_response_half_saturation_kg_c_m2_year": 0.25,
        "biomass_response_half_saturation_kg_c_m2": 5.0,
        "terrain_relief_half_saturation_m": 500.0,
        "crop_soil_depth_half_saturation_m": 0.5,
        "strategy_confidence_multiplier": 0.75,
        **inputs,
        **outputs,
    }


def test_native_functional_vegetation_closes_cover_and_responds_directionally():
    args = _native_arguments()

    metadata = run_functional_vegetation(**args)

    functional = np.asarray(args["functional_type_fractions_out"])
    nonvegetated = np.asarray(args["nonvegetated_fractions_out"])
    resources = np.asarray(args["resource_potentials_out"])
    dominant = np.asarray(args["dominant_cover_code_out"])
    ocean = np.asarray(args["ocean"]) >= 0.5
    land = ~ocean
    np.testing.assert_allclose(
        np.sum(functional, axis=0)[land] + np.sum(nonvegetated, axis=0)[land],
        1.0,
        atol=1e-6,
    )
    assert np.all(functional[:, ocean] == 0.0)
    assert np.all(nonvegetated[:, ocean] == 0.0)
    assert np.all(resources[:, ocean] == 0.0)
    assert np.all(dominant[ocean] == 0)

    flat_functional = functional.reshape(FUNCTIONAL_TYPE_COUNT, -1)
    flat_nonvegetated = nonvegetated.reshape(NONVEGETATED_TYPE_COUNT, -1)
    assert flat_nonvegetated[2, 1] == pytest.approx(0.80)
    assert flat_nonvegetated[3, 2] == pytest.approx(0.60)
    assert flat_nonvegetated[4, 7] == pytest.approx(0.10)
    assert flat_functional[1, 3] > flat_functional[0, 3]
    assert flat_functional[0, 4] > flat_functional[1, 4]
    assert flat_functional[6, 5] == np.max(flat_functional[:, 5])
    assert flat_functional[3, 6] > flat_functional[1, 6]
    assert dominant.reshape(-1)[7] <= FUNCTIONAL_TYPE_COUNT
    assert (
        resources.reshape(RESOURCE_POTENTIAL_COUNT, -1)[4, 2]
        < resources.reshape(RESOURCE_POTENTIAL_COUNT, -1)[4, 7]
    )
    assert np.all((resources >= 0.0) & (resources <= 1.0))
    assert metadata["maximum_partition_absolute_error"] < 1e-6


def test_native_functional_vegetation_is_deterministic():
    first = _native_arguments()
    second = _native_arguments()

    first_metadata = run_functional_vegetation(**first)
    second_metadata = run_functional_vegetation(**second)

    assert first_metadata == second_metadata
    for name in (
        "functional_type_fractions_out",
        "nonvegetated_fractions_out",
        "resource_potentials_out",
        "confidence_out",
        "dominant_cover_code_out",
    ):
        np.testing.assert_array_equal(first[name], second[name])


def test_native_functional_vegetation_rejects_overlapping_outputs():
    args = _native_arguments()
    args["confidence_out"] = np.asarray(args["functional_type_fractions_out"])[0]

    with pytest.raises(ValueError, match="must not overlap"):
        run_functional_vegetation(**args)


def test_functional_vegetation_config_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="Unknown functional-vegetation controls"):
        FunctionalVegetationConfig.from_mapping({"paint_biome": True})
    with pytest.raises(ValueError, match="warm_transition_width_c"):
        FunctionalVegetationConfig.from_mapping({"warm_transition_width_c": 0.0})
    with pytest.raises(ValueError, match="strategy_confidence_multiplier"):
        FunctionalVegetationConfig.from_mapping({"strategy_confidence_multiplier": 1.1})
