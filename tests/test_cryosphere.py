from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline._cryosphere_native import run_cryosphere
from map_maker.pipeline.cubed_sphere import CubedSphereGrid
from map_maker.pipeline.stages.cryosphere import CryosphereConfig


def _kernel_arguments(face_resolution: int = 4) -> dict[str, object]:
    grid = CubedSphereGrid.create(face_resolution)
    shape = grid.face_shape
    monthly_shape = (12, *shape)
    ocean = np.ascontiguousarray((grid.longitude < -2.0).astype(np.float32))
    temperature = np.empty(monthly_shape, dtype=np.float32)
    precipitation = np.full(monthly_shape, 40.0, dtype=np.float32)
    for month in range(12):
        temperature[month] = -10.0 if month < 9 else 5.0
    outputs = {
        "snowfall_out": np.empty(monthly_shape, dtype=np.float32),
        "snowmelt_out": np.empty(monthly_shape, dtype=np.float32),
        "snowpack_out": np.empty(monthly_shape, dtype=np.float32),
        "firn_to_ice_out": np.empty(monthly_shape, dtype=np.float32),
        "glacier_melt_out": np.empty(monthly_shape, dtype=np.float32),
        "glacier_ice_out": np.empty(monthly_shape, dtype=np.float32),
        "runoff_out": np.empty(monthly_shape, dtype=np.float32),
        "annual_mass_balance_out": np.empty(shape, dtype=np.float32),
        "annual_flow_export_out": np.empty(shape, dtype=np.float32),
        "annual_flow_import_out": np.empty(shape, dtype=np.float32),
        "annual_calving_out": np.empty(shape, dtype=np.float32),
        "annual_sublimation_out": np.empty(shape, dtype=np.float32),
        "glacier_fraction_out": np.empty(shape, dtype=np.float32),
    }
    return {
        "spinup_years": 30,
        "lapse_rate_c_per_km": 6.5,
        "relief_elevation_multiplier": 1.0,
        "maximum_highland_fraction": 0.4,
        "snow_degree_day_melt_mm_c_month": 12.0,
        "glacier_degree_day_melt_mm_c_month": 16.0,
        "firn_conversion_fraction_year": 0.65,
        "snow_sublimation_fraction_month": 0.004,
        "glacier_sublimation_fraction_month": 0.0002,
        "glacier_flow_activation_mm": 1_000.0,
        "glacier_flow_fraction_year": 0.08,
        "glacier_reference_thickness_mm": 100_000.0,
        "runoff_base_fraction": 0.35,
        "areas": grid.cell_areas,
        "neighbors": grid.neighbor_indices,
        "ocean": ocean,
        "elevation": np.ascontiguousarray(
            np.where(ocean >= 0.5, -3_000.0, 2_000.0 + 500.0 * grid.xyz[..., 2]),
            dtype=np.float32,
        ),
        "relief": np.full(shape, 500.0, dtype=np.float32),
        "temperature": temperature,
        "precipitation": precipitation,
        "evaporation": np.zeros(monthly_shape, dtype=np.float32),
        **outputs,
    }


def test_cryosphere_retains_firn_ice_and_releases_separate_glacier_melt():
    arguments = _kernel_arguments()
    metadata = run_cryosphere(**arguments)
    land = np.asarray(arguments["ocean"]) < 0.5

    assert np.max(np.asarray(arguments["firn_to_ice_out"])[:, land]) > 0.0
    assert np.max(np.asarray(arguments["glacier_ice_out"])[:, land]) > 0.0
    assert np.max(np.asarray(arguments["glacier_melt_out"])[:, land]) > 0.0
    assert np.all(np.asarray(arguments["glacier_ice_out"]) >= 0.0)
    assert np.all(np.asarray(arguments["runoff_out"]) >= 0.0)
    assert metadata["maximum_glacier_ice_water_equivalent_mm"] > 0.0
    assert metadata["land_mean_annual_glacier_melt_mm"] > 0.0


def test_cryosphere_config_and_ffi_reject_invalid_controls_and_buffers():
    with pytest.raises(ValueError, match="Unknown cryosphere controls"):
        CryosphereConfig.from_mapping({"paint_glaciers": True})
    with pytest.raises(ValueError, match="relief_elevation_multiplier"):
        CryosphereConfig.from_mapping({"relief_elevation_multiplier": 7.0})

    invalid = _kernel_arguments()
    invalid["snowpack_out"] = np.empty((12, 6, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        run_cryosphere(**invalid)

    overlapping = _kernel_arguments()
    overlapping["annual_flow_import_out"] = overlapping["annual_flow_export_out"]
    with pytest.raises(ValueError, match="must not overlap"):
        run_cryosphere(**overlapping)
