from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline._derived_biomes_native import BIOME_COUNT, run_derived_biomes
from map_maker.pipeline.stages.derived_biomes import DerivedBiomeConfig


def _native_arguments() -> dict[str, object]:
    shape = (6, 2, 2)

    def field(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float32)

    functional = np.zeros((8, *shape), dtype=np.float32)
    functional[2] = 0.35
    functional[4] = 0.20
    functional[7] = 0.10
    nonvegetated = np.zeros((5, *shape), dtype=np.float32)
    nonvegetated[0] = 0.25
    nonvegetated[4] = 0.10
    resources = np.full((5, *shape), 0.25, dtype=np.float32)
    inputs = {
        "areas": np.full(shape, 1.0e10, dtype=np.float64),
        "ocean": field(0.0),
        "annual_temperature": field(15.0),
        "annual_precipitation": field(800.0),
        "growing_season": field(0.80),
        "seasonality": field(0.10),
        "drought": field(0.30),
        "waterlogging": field(0.0),
        "biosphere_confidence": field(0.85),
        "functional_confidence": field(0.80),
        "wetland_fraction": field(0.0),
        "elevation": field(300.0),
        "relief": field(150.0),
        "functional_type_fractions": functional,
        "nonvegetated_fractions": nonvegetated,
        "resource_potentials": resources,
    }
    flat = {name: value.reshape(value.shape[0], -1) for name, value in inputs.items() if name in {
        "functional_type_fractions",
        "nonvegetated_fractions",
        "resource_potentials",
    }}
    scalar = {
        name: value.reshape(-1)
        for name, value in inputs.items()
        if name not in {"areas", *flat}
    }

    scalar["ocean"][0] = 1.0
    flat["functional_type_fractions"][:, 0] = 0.0
    flat["nonvegetated_fractions"][:, 0] = 0.0
    flat["resource_potentials"][:, 0] = 0.0

    scalar["annual_temperature"][1] = 26.0
    scalar["annual_precipitation"][1] = 2_400.0
    scalar["drought"][1] = 0.05
    scalar["seasonality"][1] = 0.01
    flat["functional_type_fractions"][:, 1] = 0.0
    flat["functional_type_fractions"][1, 1] = 0.75
    flat["nonvegetated_fractions"][:, 1] = 0.0
    flat["nonvegetated_fractions"][0, 1] = 0.25

    scalar["annual_temperature"][2] = 28.0
    scalar["annual_precipitation"][2] = 80.0
    scalar["drought"][2] = 0.95
    flat["functional_type_fractions"][:, 2] = 0.0
    flat["functional_type_fractions"][3, 2] = 0.06
    flat["functional_type_fractions"][7, 2] = 0.04
    flat["nonvegetated_fractions"][:, 2] = 0.0
    flat["nonvegetated_fractions"][0, 2] = 0.75
    flat["nonvegetated_fractions"][1, 2] = 0.05
    flat["nonvegetated_fractions"][4, 2] = 0.10

    scalar["waterlogging"][3] = 0.95
    scalar["wetland_fraction"][3] = 0.50
    flat["functional_type_fractions"][:, 3] = 0.0
    flat["functional_type_fractions"][6, 3] = 0.55
    flat["functional_type_fractions"][4, 3] = 0.10
    flat["nonvegetated_fractions"][:, 3] = 0.0
    flat["nonvegetated_fractions"][0, 3] = 0.25
    flat["nonvegetated_fractions"][3, 3] = 0.10

    scalar["annual_temperature"][4] = -3.0
    scalar["growing_season"][4] = 0.20
    scalar["elevation"][4] = 2_800.0
    scalar["relief"][4] = 700.0
    flat["functional_type_fractions"][:, 4] = 0.0
    flat["functional_type_fractions"][4, 4] = 0.08
    flat["functional_type_fractions"][7, 4] = 0.25
    flat["nonvegetated_fractions"][:, 4] = 0.0
    flat["nonvegetated_fractions"][0, 4] = 0.29
    flat["nonvegetated_fractions"][4, 4] = 0.38

    scalar["annual_temperature"][5] = -8.0
    scalar["growing_season"][5] = 0.15
    scalar["drought"][5] = 0.20
    flat["functional_type_fractions"][:, 5] = 0.0
    flat["functional_type_fractions"][4, 5] = 0.08
    flat["functional_type_fractions"][7, 5] = 0.27
    flat["nonvegetated_fractions"][:, 5] = 0.0
    flat["nonvegetated_fractions"][0, 5] = 0.45
    flat["nonvegetated_fractions"][4, 5] = 0.20

    outputs = {
        "biome_fractions_out": np.zeros((BIOME_COUNT, *shape), dtype=np.float32),
        "classification_confidence_out": np.zeros(shape, dtype=np.float32),
        "dominance_margin_out": np.zeros(shape, dtype=np.float32),
        "transition_index_out": np.zeros(shape, dtype=np.float32),
        "primary_biome_code_out": np.zeros(shape, dtype=np.uint8),
        "secondary_biome_code_out": np.zeros(shape, dtype=np.uint8),
        "dominant_landscape_code_out": np.zeros(shape, dtype=np.uint8),
    }
    return {
        "highland_elevation_start_m": 1_000.0,
        "highland_elevation_full_m": 3_000.0,
        "highland_relief_start_m": 250.0,
        "highland_relief_full_m": 800.0,
        "minimum_classifiable_ground_fraction": 0.05,
        "ambiguity_margin_threshold": 0.12,
        "transition_confidence_weight": 0.45,
        **inputs,
        **outputs,
    }


def test_native_derived_biomes_close_partition_and_respond_directionally():
    args = _native_arguments()

    metadata = run_derived_biomes(**args)

    fractions = np.asarray(args["biome_fractions_out"])
    nonvegetated = np.asarray(args["nonvegetated_fractions"])
    dominant = np.asarray(args["primary_biome_code_out"])
    secondary = np.asarray(args["secondary_biome_code_out"])
    ocean = np.asarray(args["ocean"]) >= 0.5
    land = ~ocean
    np.testing.assert_allclose(
        np.sum(fractions, axis=0)[land] + nonvegetated[2][land] + nonvegetated[3][land],
        1.0,
        atol=1e-6,
    )
    assert np.all(fractions[:, ocean] == 0.0)
    assert np.all(dominant[ocean] == 0)
    assert np.all(secondary[ocean] == 0)
    assert dominant.reshape(-1)[1] == 1
    assert dominant.reshape(-1)[2] == 4
    assert dominant.reshape(-1)[3] == 13
    assert dominant.reshape(-1)[4] == 12
    assert dominant.reshape(-1)[5] == 10
    assert np.all(dominant[land] != secondary[land])
    assert metadata["maximum_partition_absolute_error"] < 1e-6


def test_native_derived_biomes_are_deterministic():
    first = _native_arguments()
    second = _native_arguments()

    assert run_derived_biomes(**first) == run_derived_biomes(**second)
    for name in (
        "biome_fractions_out",
        "classification_confidence_out",
        "dominance_margin_out",
        "transition_index_out",
        "primary_biome_code_out",
        "secondary_biome_code_out",
        "dominant_landscape_code_out",
    ):
        np.testing.assert_array_equal(first[name], second[name])


def test_native_derived_biomes_reject_overlapping_outputs():
    args = _native_arguments()
    args["dominance_margin_out"] = np.asarray(args["biome_fractions_out"])[0]

    with pytest.raises(ValueError, match="must not overlap"):
        run_derived_biomes(**args)


def test_derived_biome_config_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="Unknown derived-biome controls"):
        DerivedBiomeConfig.from_mapping({"paint_random_biomes": True})
    with pytest.raises(ValueError, match="highland_elevation_full_m"):
        DerivedBiomeConfig.from_mapping(
            {"highland_elevation_start_m": 3_000.0, "highland_elevation_full_m": 1_000.0}
        )
    with pytest.raises(ValueError, match="ambiguity_margin_threshold"):
        DerivedBiomeConfig.from_mapping({"ambiguity_margin_threshold": 1.1})
