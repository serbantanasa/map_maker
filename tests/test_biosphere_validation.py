from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline.stages.biosphere_validation import (
    BiosphereValidationConfig,
    CLIMATE_STRATA,
    EARTHLIKE_LAND_FRACTION_RANGE,
    _climate_distribution_catalog,
    _weighted_quantile,
    classify_climate_strata,
)


def test_earthlike_land_fraction_profile_accepts_configured_baseline():
    minimum, maximum = EARTHLIKE_LAND_FRACTION_RANGE

    assert minimum <= 0.35 <= maximum


def test_climate_strata_are_exclusive_complete_and_upstream_only():
    temperature = np.array([-10.0, 0.0, 10.0, 10.0, 22.0, 22.0, 22.0, 30.0])
    precipitation = np.array([100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 2_000.0, 0.0])
    land = np.array([True, True, True, True, True, True, True, False])

    zones = classify_climate_strata(temperature, precipitation, land)

    np.testing.assert_array_equal(zones, np.array([0, 1, 2, 3, 4, 5, 6, -1], dtype=np.int8))
    assert len(CLIMATE_STRATA) == 7


def test_weighted_quantile_respects_physical_area():
    values = np.array([0.0, 10.0, 20.0])
    weights = np.array([1.0, 8.0, 1.0])

    assert _weighted_quantile(values, weights, 0.5) == pytest.approx(10.0)
    assert _weighted_quantile(values, weights, 0.1) < 10.0
    assert _weighted_quantile(values, weights, 0.9) > 10.0


def test_climate_distribution_catalog_reconstructs_global_weighted_mean():
    shape = (7,)
    zones = np.arange(7, dtype=np.int8)
    land = np.ones(shape, dtype=bool)
    areas = np.arange(1.0, 8.0)
    npp = np.arange(0.1, 0.8, 0.1)

    catalog, summaries = _climate_distribution_catalog(
        zones=zones,
        land=land,
        areas_m2=areas,
        minimum_reportable_fraction=0.0,
        fields={"annual_potential_npp": (npp, "kg C/m2/year")},
    )

    reconstructed = sum(
        summary["land_area_fraction"] * summary["annual_potential_npp_mean"]
        for summary in summaries.values()
    )
    assert reconstructed == pytest.approx(float(np.sum(npp * areas) / np.sum(areas)))
    assert catalog.num_rows == len(CLIMATE_STRATA) * 4
    assert set(catalog["statistic"].to_pylist()) == {"mean", "p10", "p50", "p90"}


def test_biosphere_validation_config_rejects_mutable_profile_bounds():
    with pytest.raises(ValueError, match="Unknown biosphere-validation controls"):
        BiosphereValidationConfig.from_mapping({"global_npp_minimum": 0.0})
    with pytest.raises(ValueError, match="minimum_reportable_zone_land_fraction"):
        BiosphereValidationConfig.from_mapping({"minimum_reportable_zone_land_fraction": 1.1})
