from __future__ import annotations

import numpy as np
import pytest

from map_maker.pipeline._surface_materials_native import run_surface_materials
from map_maker.pipeline.stages.surface_materials import (
    MATERIAL_OUTPUTS,
    MONTHLY_OUTPUTS,
    SOIL_OUTPUTS,
    SurfaceMaterialsConfig,
)


def _native_case(
    shape: tuple[int, ...] = (6, 2, 2),
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    monthly_shape = (12, *shape)
    fields = {
        "areas": np.full(shape, 1.0e10, dtype=np.float64),
        "ocean": np.zeros(shape, dtype=np.float32),
        "province_class": np.full(shape, 2, dtype=np.uint8),
        "crust_age": np.full(shape, 2.5, dtype=np.float32),
        "rock_strength": np.full(shape, 0.5, dtype=np.float32),
        "accommodation": np.full(shape, 0.35, dtype=np.float32),
        "province_confidence": np.full(shape, 0.9, dtype=np.float32),
        "elevation_confidence": np.full(shape, 0.9, dtype=np.float32),
        "relief": np.full(shape, 250.0, dtype=np.float32),
        "flow_slope": np.full(shape, 0.002, dtype=np.float32),
        "river_corridor": np.zeros(shape, dtype=np.float32),
        "floodplain": np.zeros(shape, dtype=np.float32),
        "lake_fraction": np.zeros(shape, dtype=np.float32),
        "wetland_fraction": np.zeros(shape, dtype=np.float32),
        "depression_fill_depth": np.zeros(shape, dtype=np.float32),
        "refined_mask": np.zeros(shape, dtype=np.float32),
        "refined_lake_fraction": np.zeros(shape, dtype=np.float32),
        "refined_wetland_fraction": np.zeros(shape, dtype=np.float32),
        "refined_hydroperiod": np.zeros(shape, dtype=np.float32),
        "refined_salinity": np.zeros(shape, dtype=np.float32),
        "recent_erosion_depth": np.zeros(shape, dtype=np.float32),
        "recent_deposition_depth": np.zeros(shape, dtype=np.float32),
        "glacier_fraction": np.zeros(shape, dtype=np.float32),
        "annual_temperature": np.full(shape, 18.0, dtype=np.float32),
        "annual_precipitation": np.full(shape, 1_200.0, dtype=np.float32),
        "monthly_temperature": np.full(monthly_shape, 18.0, dtype=np.float32),
        "monthly_precipitation": np.full(monthly_shape, 100.0, dtype=np.float32),
        "monthly_evaporation": np.full(monthly_shape, 55.0, dtype=np.float32),
        "monthly_snowfall": np.zeros(monthly_shape, dtype=np.float32),
        "monthly_snowmelt": np.zeros(monthly_shape, dtype=np.float32),
        "monthly_glacier_melt": np.zeros(monthly_shape, dtype=np.float32),
    }
    flat = {name: value.reshape(-1) for name, value in fields.items() if value.shape == shape}
    flat["river_corridor"][1] = 1.0
    flat["floodplain"][1] = 0.8
    flat["refined_mask"][2] = 1.0
    flat["refined_lake_fraction"][2] = 0.65
    flat["refined_hydroperiod"][2] = 1.0
    flat["refined_salinity"][2] = 0.8
    flat["province_class"][3] = 6
    flat["glacier_fraction"][4] = 0.20
    flat["annual_temperature"][4] = -8.0
    flat["flow_slope"][5] = 0.08
    flat["relief"][5] = 1_100.0
    flat["annual_temperature"][5] = 28.0
    flat["annual_precipitation"][5] = 80.0
    fields["monthly_temperature"].reshape(12, -1)[:, 4] = -8.0
    fields["monthly_temperature"].reshape(12, -1)[:, 5] = 28.0
    fields["monthly_precipitation"].reshape(12, -1)[:, 5] = 80.0 / 12.0
    fields["monthly_evaporation"].reshape(12, -1)[:, 5] = 120.0
    fields["ocean"].reshape(-1)[0] = 1.0

    output_names = {
        "bedrock_out": MATERIAL_OUTPUTS[0],
        "residual_out": MATERIAL_OUTPUTS[1],
        "colluvium_out": MATERIAL_OUTPUTS[2],
        "alluvium_out": MATERIAL_OUTPUTS[3],
        "lacustrine_out": MATERIAL_OUTPUTS[4],
        "glacial_out": MATERIAL_OUTPUTS[5],
        "volcaniclastic_out": MATERIAL_OUTPUTS[6],
        "dominant_material_out": "DominantSurfaceMaterialCode",
        "soil_bearing_out": SOIL_OUTPUTS[0],
        "regolith_depth_out": SOIL_OUTPUTS[1],
        "soil_depth_out": SOIL_OUTPUTS[2],
        "sand_out": SOIL_OUTPUTS[3],
        "silt_out": SOIL_OUTPUTS[4],
        "clay_out": SOIL_OUTPUTS[5],
        "coarse_fragments_out": SOIL_OUTPUTS[6],
        "bulk_density_out": SOIL_OUTPUTS[7],
        "organic_carbon_out": SOIL_OUTPUTS[8],
        "soil_ph_out": SOIL_OUTPUTS[9],
        "carbonate_out": SOIL_OUTPUTS[10],
        "salinity_out": SOIL_OUTPUTS[11],
        "drainage_out": SOIL_OUTPUTS[12],
        "available_water_capacity_out": SOIL_OUTPUTS[13],
        "nutrient_potential_out": SOIL_OUTPUTS[14],
        "fertility_potential_out": SOIL_OUTPUTS[15],
        "erodibility_out": SOIL_OUTPUTS[16],
        "reset_age_out": SOIL_OUTPUTS[17],
        "hydric_fraction_out": SOIL_OUTPUTS[18],
        "soil_confidence_out": SOIL_OUTPUTS[19],
        "annual_storage_change_out": SOIL_OUTPUTS[20],
        "monthly_liquid_input_out": MONTHLY_OUTPUTS[0],
        "monthly_soil_water_out": MONTHLY_OUTPUTS[1],
        "monthly_saturation_out": MONTHLY_OUTPUTS[2],
        "monthly_evapotranspiration_out": MONTHLY_OUTPUTS[3],
        "monthly_runoff_out": MONTHLY_OUTPUTS[4],
        "monthly_deep_drainage_out": MONTHLY_OUTPUTS[5],
    }
    outputs = {
        native_name: np.zeros(
            monthly_shape if artifact_name in MONTHLY_OUTPUTS else shape,
            dtype=np.uint8 if artifact_name == "DominantSurfaceMaterialCode" else np.float32,
        )
        for native_name, artifact_name in output_names.items()
    }
    metadata = run_surface_materials(
        spinup_years=12,
        maximum_regolith_depth_m=20.0,
        maximum_soil_depth_m=3.0,
        maximum_alluvial_fraction=0.65,
        maximum_lacustrine_fraction=0.85,
        maximum_glacial_fraction=0.80,
        weathering_temperature_scale_c=22.0,
        weathering_precipitation_scale_mm=1_600.0,
        soil_evaporation_factor=1.0,
        monthly_deep_drainage_fraction=0.06,
        **fields,
        **outputs,
    )
    return {artifact: outputs[native] for native, artifact in output_names.items()}, metadata


def test_surface_material_config_rejects_invalid_controls():
    with pytest.raises(ValueError, match="Unknown surface-material controls"):
        SurfaceMaterialsConfig.from_mapping({"magic": 1})
    with pytest.raises(ValueError, match="maximum_soil_depth_m"):
        SurfaceMaterialsConfig.from_mapping(
            {"maximum_regolith_depth_m": 1.0, "maximum_soil_depth_m": 2.0}
        )


def test_native_surface_materials_are_fractional_causal_and_conservative():
    outputs, metadata = _native_case()
    land = np.ones((6, 2, 2), dtype=bool)
    land.reshape(-1)[0] = False
    material_sum = sum(outputs[name] for name in MATERIAL_OUTPUTS)
    texture_sum = sum(outputs[name] for name in ("SandFraction", "SiltFraction", "ClayFraction"))
    assert np.allclose(material_sum[land], 1.0, atol=1e-6)
    assert np.allclose(texture_sum[land], 1.0, atol=1e-6)
    assert np.all(material_sum[~land] == 0.0)
    assert outputs["AlluviumFraction"].reshape(-1)[1] > outputs["AlluviumFraction"].reshape(-1)[6]
    assert outputs["LacustrineSedimentFraction"].reshape(-1)[2] > 0.4
    assert (
        outputs["VolcaniclasticFraction"].reshape(-1)[3]
        > outputs["VolcaniclasticFraction"].reshape(-1)[6]
    )
    assert outputs["GlacialDepositFraction"].reshape(-1)[4] > 0.1
    assert outputs["SoilDepthM"].reshape(-1)[6] > outputs["SoilDepthM"].reshape(-1)[5]
    assert outputs["SoilSalinityIndex"].reshape(-1)[2] > outputs["SoilSalinityIndex"].reshape(-1)[6]

    annual_input = np.sum(outputs["MonthlySoilLiquidInputMm"], axis=0)
    annual_losses = sum(
        np.sum(outputs[name], axis=0)
        for name in (
            "MonthlyActualEvapotranspirationMm",
            "MonthlySoilRunoffMm",
            "MonthlyDeepDrainageMm",
        )
    )
    residual = annual_input - annual_losses - outputs["AnnualSoilWaterStorageChangeMm"]
    assert np.max(np.abs(residual[land])) < 2e-4
    assert metadata["water_balance_relative_error"] < 1e-12


def test_native_surface_materials_accept_sparse_flat_batches():
    outputs, metadata = _native_case((24,))
    assert outputs["SoilDepthM"].shape == (24,)
    assert outputs["MonthlySoilWaterMm"].shape == (12, 24)
    assert metadata["material_balance_max_error"] < 1e-6
