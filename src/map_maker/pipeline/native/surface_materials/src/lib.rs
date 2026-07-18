use std::slice;

const MONTHS: usize = 12;
const MATERIAL_COUNT: usize = 7;

const MATERIAL_BEDROCK: u8 = 1;
const MATERIAL_RESIDUAL: u8 = 2;
const MATERIAL_COLLUVIUM: u8 = 3;
const MATERIAL_ALLUVIUM: u8 = 4;
const MATERIAL_LACUSTRINE: u8 = 5;
const MATERIAL_GLACIAL: u8 = 6;
const MATERIAL_VOLCANICLASTIC: u8 = 7;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SurfaceMaterialsStats {
    pub soil_bearing_land_area_fraction: f32,
    pub bedrock_land_area_fraction: f32,
    pub residual_land_area_fraction: f32,
    pub colluvium_land_area_fraction: f32,
    pub alluvium_land_area_fraction: f32,
    pub lacustrine_land_area_fraction: f32,
    pub glacial_land_area_fraction: f32,
    pub volcaniclastic_land_area_fraction: f32,
    pub hydric_soil_land_area_fraction: f32,
    pub land_mean_regolith_depth_m: f32,
    pub land_mean_soil_depth_m: f32,
    pub land_mean_organic_carbon_kg_m2: f32,
    pub material_balance_max_error: f32,
    pub texture_balance_max_error: f32,
    pub water_balance_relative_error: f64,
}

#[no_mangle]
pub extern "C" fn surface_materials_native_abi_version() -> u32 {
    2
}

#[derive(Clone, Copy)]
struct Controls {
    spinup_years: usize,
    maximum_regolith_depth_m: f64,
    maximum_soil_depth_m: f64,
    maximum_alluvial_fraction: f64,
    maximum_lacustrine_fraction: f64,
    maximum_glacial_fraction: f64,
    weathering_temperature_scale_c: f64,
    weathering_precipitation_scale_mm: f64,
    soil_evaporation_factor: f64,
    monthly_deep_drainage_fraction: f64,
}

struct Inputs<'a> {
    areas: &'a [f64],
    ocean: &'a [f32],
    province_class: &'a [u8],
    crust_age: &'a [f32],
    rock_strength: &'a [f32],
    accommodation: &'a [f32],
    province_confidence: &'a [f32],
    elevation_confidence: &'a [f32],
    relief: &'a [f32],
    flow_slope: &'a [f32],
    river_corridor: &'a [f32],
    floodplain: &'a [f32],
    lake_fraction: &'a [f32],
    wetland_fraction: &'a [f32],
    depression_fill_depth: &'a [f32],
    refined_mask: &'a [f32],
    refined_lake_fraction: &'a [f32],
    refined_wetland_fraction: &'a [f32],
    refined_hydroperiod: &'a [f32],
    refined_salinity: &'a [f32],
    recent_erosion_depth: &'a [f32],
    recent_deposition_depth: &'a [f32],
    glacier_fraction: &'a [f32],
    annual_temperature: &'a [f32],
    annual_precipitation: &'a [f32],
    monthly_temperature: &'a [f32],
    monthly_precipitation: &'a [f32],
    monthly_evaporation: &'a [f32],
    monthly_snowfall: &'a [f32],
    monthly_snowmelt: &'a [f32],
    monthly_glacier_melt: &'a [f32],
}

struct Outputs<'a> {
    bedrock: &'a mut [f32],
    residual: &'a mut [f32],
    colluvium: &'a mut [f32],
    alluvium: &'a mut [f32],
    lacustrine: &'a mut [f32],
    glacial: &'a mut [f32],
    volcaniclastic: &'a mut [f32],
    dominant_material: &'a mut [u8],
    soil_bearing: &'a mut [f32],
    regolith_depth: &'a mut [f32],
    soil_depth: &'a mut [f32],
    sand: &'a mut [f32],
    silt: &'a mut [f32],
    clay: &'a mut [f32],
    coarse_fragments: &'a mut [f32],
    bulk_density: &'a mut [f32],
    organic_carbon: &'a mut [f32],
    soil_ph: &'a mut [f32],
    carbonate: &'a mut [f32],
    salinity: &'a mut [f32],
    drainage: &'a mut [f32],
    available_water_capacity: &'a mut [f32],
    nutrient_potential: &'a mut [f32],
    fertility_potential: &'a mut [f32],
    erodibility: &'a mut [f32],
    reset_age: &'a mut [f32],
    hydric_fraction: &'a mut [f32],
    soil_confidence: &'a mut [f32],
    monthly_liquid_input: &'a mut [f32],
    monthly_soil_water: &'a mut [f32],
    monthly_saturation: &'a mut [f32],
    monthly_evapotranspiration: &'a mut [f32],
    monthly_runoff: &'a mut [f32],
    monthly_deep_drainage: &'a mut [f32],
    annual_storage_change: &'a mut [f32],
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn class_texture(class_code: u8) -> (f64, f64, f64, f64, f64) {
    match class_code {
        1 => (0.55, 0.25, 0.20, 0.34, 0.03),
        2 => (0.42, 0.35, 0.23, 0.52, 0.12),
        3 => (0.32, 0.36, 0.32, 0.58, 0.22),
        4 => (0.58, 0.27, 0.15, 0.42, 0.03),
        5 => (0.46, 0.28, 0.26, 0.66, 0.04),
        6 => (0.42, 0.34, 0.24, 0.76, 0.03),
        7 => (0.35, 0.38, 0.27, 0.61, 0.34),
        10 | 11 => (0.42, 0.34, 0.24, 0.78, 0.02),
        _ => (0.45, 0.31, 0.24, 0.50, 0.05),
    }
}

fn volcanic_signal(class_code: u8, crust_age_ga: f64) -> f64 {
    let class_signal: f64 = match class_code {
        6 => 0.60,
        10 => 0.78,
        11 => 1.0,
        5 => 0.18,
        _ => 0.0,
    };
    class_signal * (1.0 - 0.35 * clamp01(crust_age_ga / 4.0))
}

fn run_model(
    controls: Controls,
    inputs: &Inputs<'_>,
    outputs: &mut Outputs<'_>,
) -> SurfaceMaterialsStats {
    let total = inputs.areas.len();
    outputs.bedrock.fill(0.0);
    outputs.residual.fill(0.0);
    outputs.colluvium.fill(0.0);
    outputs.alluvium.fill(0.0);
    outputs.lacustrine.fill(0.0);
    outputs.glacial.fill(0.0);
    outputs.volcaniclastic.fill(0.0);
    outputs.dominant_material.fill(0);
    outputs.soil_bearing.fill(0.0);
    outputs.regolith_depth.fill(0.0);
    outputs.soil_depth.fill(0.0);
    outputs.sand.fill(0.0);
    outputs.silt.fill(0.0);
    outputs.clay.fill(0.0);
    outputs.coarse_fragments.fill(0.0);
    outputs.bulk_density.fill(0.0);
    outputs.organic_carbon.fill(0.0);
    outputs.soil_ph.fill(0.0);
    outputs.carbonate.fill(0.0);
    outputs.salinity.fill(0.0);
    outputs.drainage.fill(0.0);
    outputs.available_water_capacity.fill(0.0);
    outputs.nutrient_potential.fill(0.0);
    outputs.fertility_potential.fill(0.0);
    outputs.erodibility.fill(0.0);
    outputs.reset_age.fill(0.0);
    outputs.hydric_fraction.fill(0.0);
    outputs.soil_confidence.fill(0.0);
    outputs.monthly_liquid_input.fill(0.0);
    outputs.monthly_soil_water.fill(0.0);
    outputs.monthly_saturation.fill(0.0);
    outputs.monthly_evapotranspiration.fill(0.0);
    outputs.monthly_runoff.fill(0.0);
    outputs.monthly_deep_drainage.fill(0.0);
    outputs.annual_storage_change.fill(0.0);

    let mut land_area = 0.0f64;
    let mut material_area = [0.0f64; MATERIAL_COUNT];
    let mut soil_bearing_area = 0.0f64;
    let mut hydric_area = 0.0f64;
    let mut regolith_area_sum = 0.0f64;
    let mut soil_depth_area_sum = 0.0f64;
    let mut organic_carbon_area_sum = 0.0f64;
    let mut maximum_material_error = 0.0f64;
    let mut maximum_texture_error = 0.0f64;
    let mut water_residual_area_sum = 0.0f64;
    let mut water_reference_area_sum = 0.0f64;

    for cell in 0..total {
        if inputs.ocean[cell] >= 0.5 {
            continue;
        }
        let area = inputs.areas[cell];
        land_area += area;
        let class_code = inputs.province_class[cell];
        let crust_age = f64::from(inputs.crust_age[cell]).max(0.0);
        let rock_strength = clamp01(f64::from(inputs.rock_strength[cell]));
        let accommodation = clamp01(f64::from(inputs.accommodation[cell]));
        let relief = clamp01(f64::from(inputs.relief[cell]) / 1_200.0);
        let slope = clamp01(f64::from(inputs.flow_slope[cell]) / 0.025);
        let river = clamp01(f64::from(inputs.river_corridor[cell]));
        let floodplain = clamp01(f64::from(inputs.floodplain[cell]));
        let depression = clamp01(f64::from(inputs.depression_fill_depth[cell]) / 180.0);
        let annual_temperature = f64::from(inputs.annual_temperature[cell]);
        let annual_precipitation = f64::from(inputs.annual_precipitation[cell]).max(0.0);
        let warmth =
            clamp01((annual_temperature + 5.0) / controls.weathering_temperature_scale_c.max(1e-6));
        let moisture =
            clamp01(annual_precipitation / controls.weathering_precipitation_scale_mm.max(1e-6));
        let aridity = 1.0 - annual_precipitation / (annual_precipitation + 650.0);
        let age_factor = 0.55 + 0.45 * clamp01(crust_age / 3.5);
        let weathering =
            clamp01(warmth * moisture * age_factor * (0.62 + 0.38 * (1.0 - rock_strength)));
        let refined = inputs.refined_mask[cell] >= 0.5;
        let lake = if refined {
            clamp01(f64::from(inputs.refined_lake_fraction[cell]))
        } else {
            clamp01(f64::from(inputs.lake_fraction[cell]))
        };
        let wetland = if refined {
            clamp01(f64::from(inputs.refined_wetland_fraction[cell]))
        } else {
            clamp01(f64::from(inputs.wetland_fraction[cell]))
        };
        let hydroperiod = if refined {
            clamp01(f64::from(inputs.refined_hydroperiod[cell]))
        } else {
            clamp01(lake + 0.65 * wetland)
        };
        let recent_erosion = clamp01(f64::from(inputs.recent_erosion_depth[cell]) / 1.0);
        let recent_deposition = clamp01(f64::from(inputs.recent_deposition_depth[cell]) / 2.0);
        let glacier = clamp01(f64::from(inputs.glacier_fraction[cell]));
        let cold_highland = clamp01((-annual_temperature - 2.0) / 18.0) * relief;
        let volcanic = volcanic_signal(class_code, crust_age);

        let lacustrine_target = clamp01(
            0.78 * lake
                + 0.38 * wetland
                + 0.10 * depression * accommodation
                + 0.20 * recent_deposition,
        )
        .min(controls.maximum_lacustrine_fraction);
        let mut remaining = 1.0 - lacustrine_target;
        let alluvium_target = clamp01(0.58 * river + 0.52 * floodplain + 0.78 * recent_deposition)
            .min(controls.maximum_alluvial_fraction);
        let alluvium = remaining * alluvium_target;
        remaining -= alluvium;
        let glacial_target =
            clamp01(3.5 * glacier + 0.22 * cold_highland).min(controls.maximum_glacial_fraction);
        let glacial = remaining * glacial_target;
        remaining -= glacial;
        let volcaniclastic_target = clamp01(volcanic * (0.52 - 0.22 * weathering)).min(0.65);
        let volcaniclastic = remaining * volcaniclastic_target;
        remaining -= volcaniclastic;

        let bedrock_weight =
            0.05 + 0.85 * slope + 0.38 * relief + 0.16 * aridity + 0.42 * recent_erosion;
        let residual_weight =
            0.45 + 2.70 * weathering * (1.0 - 0.78 * slope) + 0.45 * (1.0 - relief);
        let colluvium_weight = 0.08 + 1.75 * slope * relief + 0.32 * recent_erosion * (1.0 - river);
        let weight_sum = bedrock_weight + residual_weight + colluvium_weight;
        let bedrock = remaining * bedrock_weight / weight_sum;
        let residual = remaining * residual_weight / weight_sum;
        let colluvium = remaining * colluvium_weight / weight_sum;
        let materials = [
            bedrock,
            residual,
            colluvium,
            alluvium,
            lacustrine_target,
            glacial,
            volcaniclastic,
        ];
        let material_sum: f64 = materials.iter().sum();
        maximum_material_error = maximum_material_error.max((material_sum - 1.0).abs());
        outputs.bedrock[cell] = bedrock as f32;
        outputs.residual[cell] = residual as f32;
        outputs.colluvium[cell] = colluvium as f32;
        outputs.alluvium[cell] = alluvium as f32;
        outputs.lacustrine[cell] = lacustrine_target as f32;
        outputs.glacial[cell] = glacial as f32;
        outputs.volcaniclastic[cell] = volcaniclastic as f32;
        for (index, fraction) in materials.iter().enumerate() {
            material_area[index] += area * fraction;
        }
        let dominant_index = materials
            .iter()
            .enumerate()
            .max_by(|first, second| first.1.total_cmp(second.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        outputs.dominant_material[cell] = [
            MATERIAL_BEDROCK,
            MATERIAL_RESIDUAL,
            MATERIAL_COLLUVIUM,
            MATERIAL_ALLUVIUM,
            MATERIAL_LACUSTRINE,
            MATERIAL_GLACIAL,
            MATERIAL_VOLCANICLASTIC,
        ][dominant_index];

        let (base_sand, base_silt, base_clay, base_cations, base_carbonate) =
            class_texture(class_code);
        let residual_sand = (base_sand - 0.16 * weathering).max(0.05);
        let residual_silt = base_silt;
        let residual_clay = base_clay + 0.16 * weathering;
        let mut sand = bedrock * base_sand
            + residual * residual_sand
            + colluvium * 0.50
            + alluvium * 0.32
            + lacustrine_target * 0.16
            + glacial * 0.48
            + volcaniclastic * 0.38;
        let mut silt = bedrock * base_silt
            + residual * residual_silt
            + colluvium * 0.32
            + alluvium * 0.46
            + lacustrine_target * 0.38
            + glacial * 0.32
            + volcaniclastic * 0.35;
        let mut clay = bedrock * base_clay
            + residual * residual_clay
            + colluvium * 0.18
            + alluvium * 0.22
            + lacustrine_target * 0.46
            + glacial * 0.20
            + volcaniclastic * 0.27;
        let texture_sum = sand + silt + clay;
        sand /= texture_sum;
        silt /= texture_sum;
        clay /= texture_sum;
        maximum_texture_error = maximum_texture_error.max((sand + silt + clay - 1.0).abs());
        let coarse_fragments = clamp01(
            bedrock * 0.72
                + residual * (0.20 - 0.08 * weathering)
                + colluvium * 0.46
                + alluvium * 0.12
                + lacustrine_target * 0.04
                + glacial * 0.52
                + volcaniclastic * 0.30,
        );
        outputs.sand[cell] = sand as f32;
        outputs.silt[cell] = silt as f32;
        outputs.clay[cell] = clay as f32;
        outputs.coarse_fragments[cell] = coarse_fragments as f32;

        let bedrock_depth = 0.04 + 0.45 * (1.0 - rock_strength);
        let residual_depth =
            0.25 + 8.0 * weathering * (1.0 - 0.72 * slope) * (0.70 + 0.30 * accommodation);
        let colluvium_depth = 0.45 + 3.8 * slope * relief;
        let alluvium_depth = 0.8
            + 7.5 * clamp01(0.55 * river + 0.45 * floodplain)
            + f64::from(inputs.recent_deposition_depth[cell]);
        let lacustrine_depth =
            1.2 + 14.0 * clamp01(0.50 * depression + 0.30 * lake + 0.20 * accommodation);
        let glacial_depth = 0.8 + 5.0 * clamp01(3.0 * glacier + cold_highland);
        let volcanic_depth = 0.6 + 4.0 * volcanic;
        let regolith_depth = (bedrock * bedrock_depth
            + residual * residual_depth
            + colluvium * colluvium_depth
            + alluvium * alluvium_depth
            + lacustrine_target * lacustrine_depth
            + glacial * glacial_depth
            + volcaniclastic * volcanic_depth)
            .clamp(0.0, controls.maximum_regolith_depth_m);
        let pedogenic_fraction =
            clamp01(0.18 + 0.62 * weathering + 0.16 * alluvium + 0.12 * lacustrine_target);
        let soil_depth = regolith_depth
            .min(0.10 + controls.maximum_soil_depth_m * pedogenic_fraction)
            .min(controls.maximum_soil_depth_m);
        let non_open_fraction = clamp01(1.0 - lake - glacier);
        let soil_bearing = non_open_fraction * clamp01(1.0 - 0.82 * bedrock);
        outputs.regolith_depth[cell] = regolith_depth as f32;
        outputs.soil_depth[cell] = soil_depth as f32;
        outputs.soil_bearing[cell] = soil_bearing as f32;

        let carbonate = clamp01(
            base_carbonate * (0.55 + 0.90 * aridity)
                + 0.14 * lacustrine_target * aridity * accommodation,
        )
        .min(0.80);
        let refined_salinity = if refined {
            clamp01(f64::from(inputs.refined_salinity[cell]))
        } else {
            0.0
        };
        let salinity = clamp01(
            aridity * (0.08 + 0.55 * depression * (0.35 + 0.65 * accommodation) + 0.22 * wetland)
                + refined_salinity * (lake + wetland),
        );
        let drainage = clamp01(
            0.18 + 0.50 * sand + 0.30 * slope + 0.15 * coarse_fragments
                - 0.48 * clay
                - 0.42 * lacustrine_target
                - 0.22 * wetland,
        );
        let base_status = clamp01(base_cations + 0.20 * volcaniclastic + 0.10 * alluvium);
        let leaching = weathering * moisture * (1.0 - carbonate);
        let soil_ph = (4.15 + 2.55 * base_status + 1.55 * carbonate + 0.55 * salinity
            - 0.85 * leaching)
            .clamp(3.5, 9.5);
        let bulk_density = (1_470.0 + 230.0 * sand - 260.0 * clay + 130.0 * coarse_fragments
            - 120.0 * wetland)
            .clamp(700.0, 1_950.0);
        let available_water_capacity = (soil_depth
            * 1_000.0
            * (0.07 * sand + 0.18 * silt + 0.23 * clay)
            * (1.0 - 0.68 * coarse_fragments))
            .clamp(0.0, 650.0);
        outputs.carbonate[cell] = carbonate as f32;
        outputs.salinity[cell] = salinity as f32;
        outputs.drainage[cell] = drainage as f32;
        outputs.soil_ph[cell] = soil_ph as f32;
        outputs.bulk_density[cell] = bulk_density as f32;
        outputs.available_water_capacity[cell] = available_water_capacity as f32;

        let reset_ages = [
            80.0 + 380.0 * clamp01(crust_age / 4.0) * (1.0 - recent_erosion),
            240.0 + 760.0 * clamp01(crust_age / 4.0),
            20.0 + 120.0 * (1.0 - slope),
            0.5 + 18.0 * (1.0 - river),
            2.0 + 45.0 * (1.0 - hydroperiod),
            1.0 + 28.0 * (1.0 - glacier),
            8.0 + 180.0 * clamp01(crust_age / 4.0),
        ];
        let reset_age = materials
            .iter()
            .zip(reset_ages)
            .map(|(fraction, age)| fraction * age)
            .sum::<f64>()
            .clamp(0.0, 1_000.0);
        outputs.reset_age[cell] = reset_age as f32;

        let capacity_cell_equivalent = available_water_capacity * soil_bearing;
        let mut storage = 0.5 * capacity_cell_equivalent;
        let mut final_start_storage = storage;
        let mut final_input = 0.0f64;
        let mut final_evapotranspiration = 0.0f64;
        let mut final_runoff = 0.0f64;
        let mut final_deep_drainage = 0.0f64;
        let mut saturated_months = 0usize;
        for year in 0..controls.spinup_years {
            let persist = year + 1 == controls.spinup_years;
            if persist {
                final_start_storage = storage;
            }
            for month in 0..MONTHS {
                let offset = month * total + cell;
                let rain = (f64::from(inputs.monthly_precipitation[offset])
                    - f64::from(inputs.monthly_snowfall[offset]))
                .max(0.0);
                let liquid = (rain
                    + f64::from(inputs.monthly_snowmelt[offset])
                    + f64::from(inputs.monthly_glacier_melt[offset]))
                .max(0.0);
                let modeled_input = liquid * non_open_fraction;
                let frozen = clamp01(-f64::from(inputs.monthly_temperature[offset]) / 10.0);
                let infiltration_fraction =
                    clamp01(0.36 + 0.74 * soil_bearing * (0.62 + 0.38 * drainage) - 0.24 * frozen)
                        .clamp(0.05, 0.99);
                let mut runoff = modeled_input * (1.0 - infiltration_fraction);
                storage += modeled_input * infiltration_fraction;
                if storage > capacity_cell_equivalent {
                    runoff += storage - capacity_cell_equivalent;
                    storage = capacity_cell_equivalent;
                }
                let saturation = if capacity_cell_equivalent > 1e-12 {
                    clamp01(storage / capacity_cell_equivalent)
                } else {
                    0.0
                };
                let deep_drainage = storage
                    .min(storage * controls.monthly_deep_drainage_fraction * drainage * saturation);
                storage -= deep_drainage;
                let post_drainage_saturation = if capacity_cell_equivalent > 1e-12 {
                    clamp01(storage / capacity_cell_equivalent)
                } else {
                    0.0
                };
                let evaporation_demand = f64::from(inputs.monthly_evaporation[offset]).max(0.0)
                    * non_open_fraction
                    * controls.soil_evaporation_factor
                    * (0.35 + 0.65 * post_drainage_saturation);
                let evapotranspiration = storage.min(evaporation_demand);
                storage -= evapotranspiration;
                let final_saturation = if capacity_cell_equivalent > 1e-12 {
                    clamp01(storage / capacity_cell_equivalent)
                } else {
                    0.0
                };
                if persist {
                    outputs.monthly_liquid_input[offset] = modeled_input as f32;
                    outputs.monthly_soil_water[offset] = storage as f32;
                    outputs.monthly_saturation[offset] = final_saturation as f32;
                    outputs.monthly_evapotranspiration[offset] = evapotranspiration as f32;
                    outputs.monthly_runoff[offset] = runoff as f32;
                    outputs.monthly_deep_drainage[offset] = deep_drainage as f32;
                    final_input += modeled_input;
                    final_evapotranspiration += evapotranspiration;
                    final_runoff += runoff;
                    final_deep_drainage += deep_drainage;
                    if final_saturation >= 0.85 {
                        saturated_months += 1;
                    }
                }
            }
        }
        let storage_change = storage - final_start_storage;
        outputs.annual_storage_change[cell] = storage_change as f32;
        let water_residual = final_input
            - final_evapotranspiration
            - final_runoff
            - final_deep_drainage
            - storage_change;
        water_residual_area_sum += water_residual.abs() * area;
        water_reference_area_sum += final_input * area;

        let saturation_frequency = saturated_months as f64 / MONTHS as f64;
        let hydric_fraction = soil_bearing
            * clamp01(
                0.65 * saturation_frequency
                    + 0.22 * wetland
                    + 0.08 * floodplain
                    + 0.05 * hydroperiod,
            );
        let temperature_productivity = (-((annual_temperature - 18.0) / 18.0).powi(2)).exp();
        let moisture_productivity = clamp01(annual_precipitation / 1_400.0);
        let productivity = temperature_productivity
            * moisture_productivity
            * (1.0 - 0.55 * salinity)
            * (1.0 - 0.30 * saturation_frequency);
        let organic_carbon = ((0.8
            + 10.0 * productivity * (0.55 + 0.45 * (1.0 - warmth))
            + 9.0 * hydric_fraction * (0.45 + 0.55 * (1.0 - warmth)))
            * clamp01(soil_depth / 1.2))
        .clamp(0.0, 45.0);
        let ph_suitability = clamp01(1.0 - (soil_ph - 6.6).abs() / 3.2);
        let nutrient_potential = clamp01(
            base_status * (0.42 + 0.58 * weathering)
                + 0.16 * alluvium
                + 0.18 * volcaniclastic
                + 0.08 * organic_carbon / 20.0,
        );
        let fertility_potential = clamp01(
            nutrient_potential
                * (0.45 + 0.55 * ph_suitability)
                * (1.0 - 0.72 * salinity)
                * (1.0 - 0.35 * saturation_frequency)
                * (0.65 + 0.35 * clamp01(available_water_capacity / 180.0)),
        );
        let erodibility = clamp01(
            0.08 + 0.55 * silt + 0.18 * sand - 0.22 * clay
                + 0.18 * alluvium
                + 0.12 * lacustrine_target
                - 0.22 * coarse_fragments
                - 0.10 * clamp01(organic_carbon / 20.0),
        );
        let confidence = clamp01(
            f64::from(inputs.province_confidence[cell])
                * f64::from(inputs.elevation_confidence[cell])
                * (0.72 + 0.28 * if refined { 1.0 } else { 0.65 }),
        );
        outputs.hydric_fraction[cell] = hydric_fraction as f32;
        outputs.organic_carbon[cell] = organic_carbon as f32;
        outputs.nutrient_potential[cell] = nutrient_potential as f32;
        outputs.fertility_potential[cell] = fertility_potential as f32;
        outputs.erodibility[cell] = erodibility as f32;
        outputs.soil_confidence[cell] = confidence as f32;

        soil_bearing_area += area * soil_bearing;
        hydric_area += area * hydric_fraction;
        regolith_area_sum += area * regolith_depth;
        soil_depth_area_sum += area * soil_depth;
        organic_carbon_area_sum += area * organic_carbon;
    }

    SurfaceMaterialsStats {
        soil_bearing_land_area_fraction: (soil_bearing_area / land_area.max(1e-12)) as f32,
        bedrock_land_area_fraction: (material_area[0] / land_area.max(1e-12)) as f32,
        residual_land_area_fraction: (material_area[1] / land_area.max(1e-12)) as f32,
        colluvium_land_area_fraction: (material_area[2] / land_area.max(1e-12)) as f32,
        alluvium_land_area_fraction: (material_area[3] / land_area.max(1e-12)) as f32,
        lacustrine_land_area_fraction: (material_area[4] / land_area.max(1e-12)) as f32,
        glacial_land_area_fraction: (material_area[5] / land_area.max(1e-12)) as f32,
        volcaniclastic_land_area_fraction: (material_area[6] / land_area.max(1e-12)) as f32,
        hydric_soil_land_area_fraction: (hydric_area / land_area.max(1e-12)) as f32,
        land_mean_regolith_depth_m: (regolith_area_sum / land_area.max(1e-12)) as f32,
        land_mean_soil_depth_m: (soil_depth_area_sum / land_area.max(1e-12)) as f32,
        land_mean_organic_carbon_kg_m2: (organic_carbon_area_sum / land_area.max(1e-12)) as f32,
        material_balance_max_error: maximum_material_error as f32,
        texture_balance_max_error: maximum_texture_error as f32,
        water_balance_relative_error: water_residual_area_sum / water_reference_area_sum.max(1e-12),
    }
}

/// Generate fractional L2 surface materials, initial mineral-soil properties,
/// and a conservative monthly soil-water partition.
///
/// Returns 0 on success, 1 for invalid dimensions or pointers, 2 for invalid
/// controls, and 3 for invalid numeric inputs.
///
/// # Safety
///
/// Every pointer must reference the documented number of elements. Output
/// buffers must be writable, disjoint, and may not alias inputs.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn surface_materials_run(
    cell_count: i32,
    spinup_years: i32,
    maximum_regolith_depth_m: f64,
    maximum_soil_depth_m: f64,
    maximum_alluvial_fraction: f64,
    maximum_lacustrine_fraction: f64,
    maximum_glacial_fraction: f64,
    weathering_temperature_scale_c: f64,
    weathering_precipitation_scale_mm: f64,
    soil_evaporation_factor: f64,
    monthly_deep_drainage_fraction: f64,
    areas_ptr: *const f64,
    ocean_ptr: *const f32,
    province_class_ptr: *const u8,
    crust_age_ptr: *const f32,
    rock_strength_ptr: *const f32,
    accommodation_ptr: *const f32,
    province_confidence_ptr: *const f32,
    elevation_confidence_ptr: *const f32,
    relief_ptr: *const f32,
    flow_slope_ptr: *const f32,
    river_corridor_ptr: *const f32,
    floodplain_ptr: *const f32,
    lake_fraction_ptr: *const f32,
    wetland_fraction_ptr: *const f32,
    depression_fill_depth_ptr: *const f32,
    refined_mask_ptr: *const f32,
    refined_lake_fraction_ptr: *const f32,
    refined_wetland_fraction_ptr: *const f32,
    refined_hydroperiod_ptr: *const f32,
    refined_salinity_ptr: *const f32,
    recent_erosion_depth_ptr: *const f32,
    recent_deposition_depth_ptr: *const f32,
    glacier_fraction_ptr: *const f32,
    annual_temperature_ptr: *const f32,
    annual_precipitation_ptr: *const f32,
    monthly_temperature_ptr: *const f32,
    monthly_precipitation_ptr: *const f32,
    monthly_evaporation_ptr: *const f32,
    monthly_snowfall_ptr: *const f32,
    monthly_snowmelt_ptr: *const f32,
    monthly_glacier_melt_ptr: *const f32,
    bedrock_ptr: *mut f32,
    residual_ptr: *mut f32,
    colluvium_ptr: *mut f32,
    alluvium_ptr: *mut f32,
    lacustrine_ptr: *mut f32,
    glacial_ptr: *mut f32,
    volcaniclastic_ptr: *mut f32,
    dominant_material_ptr: *mut u8,
    soil_bearing_ptr: *mut f32,
    regolith_depth_ptr: *mut f32,
    soil_depth_ptr: *mut f32,
    sand_ptr: *mut f32,
    silt_ptr: *mut f32,
    clay_ptr: *mut f32,
    coarse_fragments_ptr: *mut f32,
    bulk_density_ptr: *mut f32,
    organic_carbon_ptr: *mut f32,
    soil_ph_ptr: *mut f32,
    carbonate_ptr: *mut f32,
    salinity_ptr: *mut f32,
    drainage_ptr: *mut f32,
    available_water_capacity_ptr: *mut f32,
    nutrient_potential_ptr: *mut f32,
    fertility_potential_ptr: *mut f32,
    erodibility_ptr: *mut f32,
    reset_age_ptr: *mut f32,
    hydric_fraction_ptr: *mut f32,
    soil_confidence_ptr: *mut f32,
    monthly_liquid_input_ptr: *mut f32,
    monthly_soil_water_ptr: *mut f32,
    monthly_saturation_ptr: *mut f32,
    monthly_evapotranspiration_ptr: *mut f32,
    monthly_runoff_ptr: *mut f32,
    monthly_deep_drainage_ptr: *mut f32,
    annual_storage_change_ptr: *mut f32,
    stats_out: *mut SurfaceMaterialsStats,
) -> i32 {
    let pointers = [
        areas_ptr.cast::<()>(),
        ocean_ptr.cast::<()>(),
        province_class_ptr.cast::<()>(),
        crust_age_ptr.cast::<()>(),
        rock_strength_ptr.cast::<()>(),
        accommodation_ptr.cast::<()>(),
        province_confidence_ptr.cast::<()>(),
        elevation_confidence_ptr.cast::<()>(),
        relief_ptr.cast::<()>(),
        flow_slope_ptr.cast::<()>(),
        river_corridor_ptr.cast::<()>(),
        floodplain_ptr.cast::<()>(),
        lake_fraction_ptr.cast::<()>(),
        wetland_fraction_ptr.cast::<()>(),
        depression_fill_depth_ptr.cast::<()>(),
        refined_mask_ptr.cast::<()>(),
        refined_lake_fraction_ptr.cast::<()>(),
        refined_wetland_fraction_ptr.cast::<()>(),
        refined_hydroperiod_ptr.cast::<()>(),
        refined_salinity_ptr.cast::<()>(),
        recent_erosion_depth_ptr.cast::<()>(),
        recent_deposition_depth_ptr.cast::<()>(),
        glacier_fraction_ptr.cast::<()>(),
        annual_temperature_ptr.cast::<()>(),
        annual_precipitation_ptr.cast::<()>(),
        monthly_temperature_ptr.cast::<()>(),
        monthly_precipitation_ptr.cast::<()>(),
        monthly_evaporation_ptr.cast::<()>(),
        monthly_snowfall_ptr.cast::<()>(),
        monthly_snowmelt_ptr.cast::<()>(),
        monthly_glacier_melt_ptr.cast::<()>(),
        bedrock_ptr.cast::<()>(),
        residual_ptr.cast::<()>(),
        colluvium_ptr.cast::<()>(),
        alluvium_ptr.cast::<()>(),
        lacustrine_ptr.cast::<()>(),
        glacial_ptr.cast::<()>(),
        volcaniclastic_ptr.cast::<()>(),
        dominant_material_ptr.cast::<()>(),
        soil_bearing_ptr.cast::<()>(),
        regolith_depth_ptr.cast::<()>(),
        soil_depth_ptr.cast::<()>(),
        sand_ptr.cast::<()>(),
        silt_ptr.cast::<()>(),
        clay_ptr.cast::<()>(),
        coarse_fragments_ptr.cast::<()>(),
        bulk_density_ptr.cast::<()>(),
        organic_carbon_ptr.cast::<()>(),
        soil_ph_ptr.cast::<()>(),
        carbonate_ptr.cast::<()>(),
        salinity_ptr.cast::<()>(),
        drainage_ptr.cast::<()>(),
        available_water_capacity_ptr.cast::<()>(),
        nutrient_potential_ptr.cast::<()>(),
        fertility_potential_ptr.cast::<()>(),
        erodibility_ptr.cast::<()>(),
        reset_age_ptr.cast::<()>(),
        hydric_fraction_ptr.cast::<()>(),
        soil_confidence_ptr.cast::<()>(),
        monthly_liquid_input_ptr.cast::<()>(),
        monthly_soil_water_ptr.cast::<()>(),
        monthly_saturation_ptr.cast::<()>(),
        monthly_evapotranspiration_ptr.cast::<()>(),
        monthly_runoff_ptr.cast::<()>(),
        monthly_deep_drainage_ptr.cast::<()>(),
        annual_storage_change_ptr.cast::<()>(),
        stats_out.cast::<()>(),
    ];
    if cell_count <= 0 || pointers.iter().any(|pointer| pointer.is_null()) {
        return 1;
    }
    let controls = Controls {
        spinup_years: spinup_years as usize,
        maximum_regolith_depth_m,
        maximum_soil_depth_m,
        maximum_alluvial_fraction,
        maximum_lacustrine_fraction,
        maximum_glacial_fraction,
        weathering_temperature_scale_c,
        weathering_precipitation_scale_mm,
        soil_evaporation_factor,
        monthly_deep_drainage_fraction,
    };
    let numeric_controls = [
        maximum_regolith_depth_m,
        maximum_soil_depth_m,
        maximum_alluvial_fraction,
        maximum_lacustrine_fraction,
        maximum_glacial_fraction,
        weathering_temperature_scale_c,
        weathering_precipitation_scale_mm,
        soil_evaporation_factor,
        monthly_deep_drainage_fraction,
    ];
    if controls.spinup_years < 2
        || numeric_controls.iter().any(|value| !value.is_finite())
        || maximum_regolith_depth_m <= 0.0
        || maximum_soil_depth_m <= 0.0
        || maximum_soil_depth_m > maximum_regolith_depth_m
        || !(0.0..=1.0).contains(&maximum_alluvial_fraction)
        || !(0.0..=1.0).contains(&maximum_lacustrine_fraction)
        || !(0.0..=1.0).contains(&maximum_glacial_fraction)
        || weathering_temperature_scale_c <= 0.0
        || weathering_precipitation_scale_mm <= 0.0
        || soil_evaporation_factor < 0.0
        || !(0.0..=1.0).contains(&monthly_deep_drainage_fraction)
    {
        return 2;
    }

    let total = cell_count as usize;
    let monthly = total * MONTHS;
    let inputs = Inputs {
        areas: unsafe { slice::from_raw_parts(areas_ptr, total) },
        ocean: unsafe { slice::from_raw_parts(ocean_ptr, total) },
        province_class: unsafe { slice::from_raw_parts(province_class_ptr, total) },
        crust_age: unsafe { slice::from_raw_parts(crust_age_ptr, total) },
        rock_strength: unsafe { slice::from_raw_parts(rock_strength_ptr, total) },
        accommodation: unsafe { slice::from_raw_parts(accommodation_ptr, total) },
        province_confidence: unsafe { slice::from_raw_parts(province_confidence_ptr, total) },
        elevation_confidence: unsafe { slice::from_raw_parts(elevation_confidence_ptr, total) },
        relief: unsafe { slice::from_raw_parts(relief_ptr, total) },
        flow_slope: unsafe { slice::from_raw_parts(flow_slope_ptr, total) },
        river_corridor: unsafe { slice::from_raw_parts(river_corridor_ptr, total) },
        floodplain: unsafe { slice::from_raw_parts(floodplain_ptr, total) },
        lake_fraction: unsafe { slice::from_raw_parts(lake_fraction_ptr, total) },
        wetland_fraction: unsafe { slice::from_raw_parts(wetland_fraction_ptr, total) },
        depression_fill_depth: unsafe { slice::from_raw_parts(depression_fill_depth_ptr, total) },
        refined_mask: unsafe { slice::from_raw_parts(refined_mask_ptr, total) },
        refined_lake_fraction: unsafe { slice::from_raw_parts(refined_lake_fraction_ptr, total) },
        refined_wetland_fraction: unsafe {
            slice::from_raw_parts(refined_wetland_fraction_ptr, total)
        },
        refined_hydroperiod: unsafe { slice::from_raw_parts(refined_hydroperiod_ptr, total) },
        refined_salinity: unsafe { slice::from_raw_parts(refined_salinity_ptr, total) },
        recent_erosion_depth: unsafe { slice::from_raw_parts(recent_erosion_depth_ptr, total) },
        recent_deposition_depth: unsafe {
            slice::from_raw_parts(recent_deposition_depth_ptr, total)
        },
        glacier_fraction: unsafe { slice::from_raw_parts(glacier_fraction_ptr, total) },
        annual_temperature: unsafe { slice::from_raw_parts(annual_temperature_ptr, total) },
        annual_precipitation: unsafe { slice::from_raw_parts(annual_precipitation_ptr, total) },
        monthly_temperature: unsafe { slice::from_raw_parts(monthly_temperature_ptr, monthly) },
        monthly_precipitation: unsafe { slice::from_raw_parts(monthly_precipitation_ptr, monthly) },
        monthly_evaporation: unsafe { slice::from_raw_parts(monthly_evaporation_ptr, monthly) },
        monthly_snowfall: unsafe { slice::from_raw_parts(monthly_snowfall_ptr, monthly) },
        monthly_snowmelt: unsafe { slice::from_raw_parts(monthly_snowmelt_ptr, monthly) },
        monthly_glacier_melt: unsafe { slice::from_raw_parts(monthly_glacier_melt_ptr, monthly) },
    };
    let finite_inputs = inputs
        .areas
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        && [
            inputs.ocean,
            inputs.crust_age,
            inputs.rock_strength,
            inputs.accommodation,
            inputs.province_confidence,
            inputs.elevation_confidence,
            inputs.relief,
            inputs.flow_slope,
            inputs.river_corridor,
            inputs.floodplain,
            inputs.lake_fraction,
            inputs.wetland_fraction,
            inputs.depression_fill_depth,
            inputs.refined_mask,
            inputs.refined_lake_fraction,
            inputs.refined_wetland_fraction,
            inputs.refined_hydroperiod,
            inputs.refined_salinity,
            inputs.recent_erosion_depth,
            inputs.recent_deposition_depth,
            inputs.glacier_fraction,
            inputs.annual_temperature,
            inputs.annual_precipitation,
            inputs.monthly_temperature,
            inputs.monthly_precipitation,
            inputs.monthly_evaporation,
            inputs.monthly_snowfall,
            inputs.monthly_snowmelt,
            inputs.monthly_glacier_melt,
        ]
        .iter()
        .all(|values| values.iter().all(|value| value.is_finite()));
    if !finite_inputs {
        return 3;
    }

    let mut outputs = Outputs {
        bedrock: unsafe { slice::from_raw_parts_mut(bedrock_ptr, total) },
        residual: unsafe { slice::from_raw_parts_mut(residual_ptr, total) },
        colluvium: unsafe { slice::from_raw_parts_mut(colluvium_ptr, total) },
        alluvium: unsafe { slice::from_raw_parts_mut(alluvium_ptr, total) },
        lacustrine: unsafe { slice::from_raw_parts_mut(lacustrine_ptr, total) },
        glacial: unsafe { slice::from_raw_parts_mut(glacial_ptr, total) },
        volcaniclastic: unsafe { slice::from_raw_parts_mut(volcaniclastic_ptr, total) },
        dominant_material: unsafe { slice::from_raw_parts_mut(dominant_material_ptr, total) },
        soil_bearing: unsafe { slice::from_raw_parts_mut(soil_bearing_ptr, total) },
        regolith_depth: unsafe { slice::from_raw_parts_mut(regolith_depth_ptr, total) },
        soil_depth: unsafe { slice::from_raw_parts_mut(soil_depth_ptr, total) },
        sand: unsafe { slice::from_raw_parts_mut(sand_ptr, total) },
        silt: unsafe { slice::from_raw_parts_mut(silt_ptr, total) },
        clay: unsafe { slice::from_raw_parts_mut(clay_ptr, total) },
        coarse_fragments: unsafe { slice::from_raw_parts_mut(coarse_fragments_ptr, total) },
        bulk_density: unsafe { slice::from_raw_parts_mut(bulk_density_ptr, total) },
        organic_carbon: unsafe { slice::from_raw_parts_mut(organic_carbon_ptr, total) },
        soil_ph: unsafe { slice::from_raw_parts_mut(soil_ph_ptr, total) },
        carbonate: unsafe { slice::from_raw_parts_mut(carbonate_ptr, total) },
        salinity: unsafe { slice::from_raw_parts_mut(salinity_ptr, total) },
        drainage: unsafe { slice::from_raw_parts_mut(drainage_ptr, total) },
        available_water_capacity: unsafe {
            slice::from_raw_parts_mut(available_water_capacity_ptr, total)
        },
        nutrient_potential: unsafe { slice::from_raw_parts_mut(nutrient_potential_ptr, total) },
        fertility_potential: unsafe { slice::from_raw_parts_mut(fertility_potential_ptr, total) },
        erodibility: unsafe { slice::from_raw_parts_mut(erodibility_ptr, total) },
        reset_age: unsafe { slice::from_raw_parts_mut(reset_age_ptr, total) },
        hydric_fraction: unsafe { slice::from_raw_parts_mut(hydric_fraction_ptr, total) },
        soil_confidence: unsafe { slice::from_raw_parts_mut(soil_confidence_ptr, total) },
        monthly_liquid_input: unsafe {
            slice::from_raw_parts_mut(monthly_liquid_input_ptr, monthly)
        },
        monthly_soil_water: unsafe { slice::from_raw_parts_mut(monthly_soil_water_ptr, monthly) },
        monthly_saturation: unsafe { slice::from_raw_parts_mut(monthly_saturation_ptr, monthly) },
        monthly_evapotranspiration: unsafe {
            slice::from_raw_parts_mut(monthly_evapotranspiration_ptr, monthly)
        },
        monthly_runoff: unsafe { slice::from_raw_parts_mut(monthly_runoff_ptr, monthly) },
        monthly_deep_drainage: unsafe {
            slice::from_raw_parts_mut(monthly_deep_drainage_ptr, monthly)
        },
        annual_storage_change: unsafe {
            slice::from_raw_parts_mut(annual_storage_change_ptr, total)
        },
    };
    let stats = run_model(controls, &inputs, &mut outputs);
    unsafe { *stats_out = stats };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn province_texture_priors_are_normalized() {
        for class_code in 0..=11 {
            let (sand, silt, clay, base, carbonate) = class_texture(class_code);
            assert!((sand + silt + clay - 1.0).abs() < 1e-12);
            assert!((0.0..=1.0).contains(&base));
            assert!((0.0..=1.0).contains(&carbonate));
        }
    }

    #[test]
    fn active_arcs_have_more_volcanic_parent_signal() {
        assert!(volcanic_signal(6, 0.5) > volcanic_signal(2, 0.5));
        assert!(volcanic_signal(11, 0.2) > volcanic_signal(6, 3.0));
    }
}
