from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline.stages.functional_vegetation_validation import (
    REFERENCE_PROFILE_VERSION,
    FunctionalVegetationValidationConfig,
    _climate_distribution_catalog,
)


def test_functional_reference_profile_is_versioned():
    assert REFERENCE_PROFILE_VERSION == "earth_functional_vegetation_v1"


def test_functional_climate_catalog_reconstructs_weighted_global_means():
    zones = np.arange(7, dtype=np.int8)
    land = np.ones(7, dtype=bool)
    areas = np.arange(1.0, 8.0)
    woody = np.arange(0.1, 0.8, 0.1)
    fire = woody[::-1].copy()

    catalog, summaries = _climate_distribution_catalog(
        zones=zones,
        land=land,
        areas_m2=areas,
        minimum_reportable_fraction=0.0,
        fields={
            "functional_woody_fraction": (woody, "fraction"),
            "resource_fire_tendency": (fire, "index"),
        },
    )

    reconstructed = sum(
        float(summary["land_area_fraction"]) * float(summary["functional_woody_fraction_mean"])
        for summary in summaries.values()
    )
    assert reconstructed == pytest.approx(float(np.sum(woody * areas) / np.sum(areas)))
    assert catalog.num_rows == 7 * 2 * 4
    assert set(catalog["statistic"].to_pylist()) == {"mean", "p10", "p50", "p90"}


def test_functional_validation_config_keeps_reference_ranges_immutable():
    with pytest.raises(ValueError, match="Unknown functional-vegetation-validation controls"):
        FunctionalVegetationValidationConfig.from_mapping({"woody_cover_minimum": 0.0})
    with pytest.raises(ValueError, match="minimum_reportable_zone_land_fraction"):
        FunctionalVegetationValidationConfig.from_mapping(
            {"minimum_reportable_zone_land_fraction": 1.1}
        )
