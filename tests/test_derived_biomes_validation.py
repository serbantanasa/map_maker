from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline.stages.derived_biomes_validation import (
    REFERENCE_PROFILE_VERSION,
    DerivedBiomeValidationConfig,
    _climate_distribution_catalog,
)


def test_derived_biome_reference_profile_is_versioned():
    assert REFERENCE_PROFILE_VERSION == "earth_biomes_v1"


def test_derived_biome_climate_catalog_reconstructs_weighted_global_means():
    zones = np.arange(7, dtype=np.int8)
    land = np.ones(7, dtype=bool)
    areas = np.arange(1.0, 8.0)
    forest = np.arange(0.1, 0.8, 0.1)
    transition = forest[::-1].copy()

    catalog, summaries = _climate_distribution_catalog(
        zones=zones,
        land=land,
        areas_m2=areas,
        minimum_reportable_fraction=0.0,
        fields={
            "forest_fraction": (forest, "fraction"),
            "transition_index": (transition, "index"),
        },
    )

    reconstructed = sum(
        float(summary["land_area_fraction"]) * float(summary["forest_fraction_mean"])
        for summary in summaries.values()
    )
    assert reconstructed == pytest.approx(float(np.sum(forest * areas) / np.sum(areas)))
    assert catalog.num_rows == 7 * 2 * 4
    assert set(catalog["statistic"].to_pylist()) == {"mean", "p10", "p50", "p90"}


def test_derived_biome_validation_config_keeps_reference_ranges_immutable():
    with pytest.raises(ValueError, match="Unknown derived-biome-validation controls"):
        DerivedBiomeValidationConfig.from_mapping({"forest_fraction_minimum": 0.0})
    with pytest.raises(ValueError, match="minimum_reportable_zone_land_fraction"):
        DerivedBiomeValidationConfig.from_mapping(
            {"minimum_reportable_zone_land_fraction": 1.1}
        )
    with pytest.raises(ValueError, match="dry_support_maximum"):
        DerivedBiomeValidationConfig.from_mapping(
            {"dry_support_maximum": 0.3, "wet_support_threshold": 0.2}
        )
