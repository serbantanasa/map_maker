use std::slice;

const FUNCTIONAL_TYPE_COUNT: usize = 8;
const NONVEGETATED_TYPE_COUNT: usize = 5;
const RESOURCE_POTENTIAL_COUNT: usize = 5;

const COLD_WOODY: usize = 0;
const WARM_EVERGREEN_WOODY: usize = 1;
const SEASONAL_WOODY: usize = 2;
const XERIC_SHRUB: usize = 3;
const COOL_HERBACEOUS: usize = 4;
const WARM_HERBACEOUS: usize = 5;
const HYDROPHYTIC: usize = 6;
const LOW_STATURE_CONSERVATIVE: usize = 7;

const BARE_GROUND: usize = 0;
const SALINE_BARREN: usize = 1;
const PERSISTENT_ICE: usize = 2;
const INLAND_OPEN_WATER: usize = 3;
const UNSUPPORTED_SURFACE: usize = 4;

const FIRE_TENDENCY: usize = 0;
const GRAZING_POTENTIAL: usize = 1;
const FOREST_RESOURCE_POTENTIAL: usize = 2;
const PASTURE_POTENTIAL: usize = 3;
const CROP_POTENTIAL: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FunctionalVegetationStats {
    pub land_mean_functional_vegetated_fraction: f32,
    pub vegetated_woody_fraction: f32,
    pub vegetated_herbaceous_fraction: f32,
    pub vegetated_hydrophytic_fraction: f32,
    pub land_mean_bare_ground_fraction: f32,
    pub land_mean_saline_barren_fraction: f32,
    pub land_mean_ice_fraction: f32,
    pub land_mean_inland_water_fraction: f32,
    pub land_mean_unsupported_surface_fraction: f32,
    pub land_mean_fire_tendency: f32,
    pub land_mean_crop_potential: f32,
    pub maximum_partition_absolute_error: f32,
}

#[no_mangle]
pub extern "C" fn functional_vegetation_native_abi_version() -> u32 {
    1
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn smoothstep01(value: f64) -> f64 {
    let bounded = clamp01(value);
    bounded * bounded * (3.0 - 2.0 * bounded)
}

fn saturating_response(value: f64, half_saturation: f64) -> f64 {
    if value <= 0.0 {
        0.0
    } else {
        clamp01(value / (value + half_saturation))
    }
}

#[allow(clippy::too_many_arguments)]
fn strategy_scores(
    annual_temperature_c: f64,
    warm_transition_midpoint_c: f64,
    warm_transition_width_c: f64,
    productivity_response: f64,
    growing_season: f64,
    seasonality: f64,
    drought: f64,
    cold: f64,
    heat: f64,
    waterlogging: f64,
    salinity: f64,
    woody: f64,
    conservative: f64,
    wetland: f64,
) -> [f64; FUNCTIONAL_TYPE_COUNT] {
    let warm_start = warm_transition_midpoint_c - 0.5 * warm_transition_width_c;
    let warm = smoothstep01((annual_temperature_c - warm_start) / warm_transition_width_c);
    let humid = 1.0 - drought;
    let wet = waterlogging.max(wetland);
    let pressure = drought.max(cold).max(heat);
    let cool = 1.0 - warm;

    let woody_budget = clamp01(0.03 + 0.97 * woody);
    let hydrophytic_budget = (1.0 - woody_budget) * 0.65 * wet.powf(1.5) * (1.0 - 0.85 * salinity);
    let ground_budget = (1.0 - woody_budget - hydrophytic_budget).max(0.0);

    let woody_weights = [
        (0.10 + 0.90 * cold) * cool * (0.40 + 0.60 * conservative) * (1.0 - 0.75 * heat),
        warm * (0.25 + 0.75 * humid) * (1.0 - 0.70 * seasonality) * (1.0 - 0.85 * cold),
        (0.15 + 0.85 * (cool * (1.0 - cold)).max(seasonality).max(0.50 * drought))
            * (0.40 + 0.60 * productivity_response)
            * (1.0 - 0.55 * heat),
    ];
    let woody_weight_sum = woody_weights.iter().sum::<f64>().max(1e-30);

    let ground_weights = [
        (0.10 + 0.90 * drought) * (0.25 + 0.75 * conservative) * (1.0 - 0.80 * wet),
        cool * (0.20 + 0.80 * growing_season) * (1.0 - 0.50 * drought) * (1.0 - 0.60 * wet),
        warm * (0.20 + 0.80 * growing_season)
            * (0.20 + 0.80 * seasonality.max(drought))
            * (1.0 - 0.65 * wet)
            * (1.0 - 0.40 * cold),
        (0.10 + 0.90 * conservative)
            * (0.20 + 0.80 * pressure)
            * (1.0 - 0.70 * wet)
            * (1.0 - 0.35 * warm),
    ];
    let ground_weight_sum = ground_weights.iter().sum::<f64>().max(1e-30);

    let mut scores = [0.0f64; FUNCTIONAL_TYPE_COUNT];
    scores[COLD_WOODY] = woody_budget * woody_weights[0] / woody_weight_sum;
    scores[WARM_EVERGREEN_WOODY] = woody_budget * woody_weights[1] / woody_weight_sum;
    scores[SEASONAL_WOODY] = woody_budget * woody_weights[2] / woody_weight_sum;
    scores[XERIC_SHRUB] = ground_budget * ground_weights[0] / ground_weight_sum;
    scores[COOL_HERBACEOUS] = ground_budget * ground_weights[1] / ground_weight_sum;
    scores[WARM_HERBACEOUS] = ground_budget * ground_weights[2] / ground_weight_sum;
    scores[HYDROPHYTIC] = hydrophytic_budget;
    scores[LOW_STATURE_CONSERVATIVE] = ground_budget * ground_weights[3] / ground_weight_sum;
    scores
}

fn nonvegetated_partition(
    potential_cover: f64,
    glacier_fraction: f64,
    lake_fraction: f64,
    salinity: f64,
    soil_bearing: f64,
) -> (f64, [f64; NONVEGETATED_TYPE_COUNT]) {
    let ice = clamp01(glacier_fraction);
    let water = clamp01(lake_fraction).min(1.0 - ice);
    let available_ground = (1.0 - ice - water).max(0.0);
    let supported_ground = clamp01(soil_bearing).min(available_ground);
    let vegetated = clamp01(potential_cover).min(supported_ground);
    let unsupported = (available_ground - supported_ground).max(0.0);
    let supported_nonvegetated = (supported_ground - vegetated).max(0.0);
    let saline = supported_nonvegetated * clamp01(salinity);
    let bare = (supported_nonvegetated - saline).max(0.0);
    let mut nonvegetated = [0.0; NONVEGETATED_TYPE_COUNT];
    nonvegetated[BARE_GROUND] = bare;
    nonvegetated[SALINE_BARREN] = saline;
    nonvegetated[PERSISTENT_ICE] = ice;
    nonvegetated[INLAND_OPEN_WATER] = water;
    nonvegetated[UNSUPPORTED_SURFACE] = unsupported;
    (vegetated, nonvegetated)
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// Convert continuous potential-biosphere traits into conservative functional mixtures.
///
/// # Safety
/// Every pointer must reference the declared number of contiguous elements.
/// Input and output buffers must not overlap.
pub unsafe extern "C" fn functional_vegetation_run(
    cell_count: i32,
    warm_transition_midpoint_c: f64,
    warm_transition_width_c: f64,
    npp_response_half_saturation_kg_c_m2_year: f64,
    biomass_response_half_saturation_kg_c_m2: f64,
    terrain_relief_half_saturation_m: f64,
    crop_soil_depth_half_saturation_m: f64,
    strategy_confidence_multiplier: f64,
    area_ptr: *const f64,
    ocean_ptr: *const f32,
    vegetation_cover_ptr: *const f32,
    annual_npp_ptr: *const f32,
    standing_biomass_ptr: *const f32,
    growing_season_ptr: *const f32,
    productivity_seasonality_ptr: *const f32,
    drought_pressure_ptr: *const f32,
    cold_pressure_ptr: *const f32,
    heat_pressure_ptr: *const f32,
    waterlogging_pressure_ptr: *const f32,
    salinity_pressure_ptr: *const f32,
    woody_trait_ptr: *const f32,
    resource_conservative_trait_ptr: *const f32,
    fuel_continuity_ptr: *const f32,
    biosphere_confidence_ptr: *const f32,
    annual_temperature_ptr: *const f32,
    soil_fertility_ptr: *const f32,
    soil_depth_ptr: *const f32,
    soil_bearing_ptr: *const f32,
    soil_drainage_ptr: *const f32,
    glacier_fraction_ptr: *const f32,
    lake_fraction_ptr: *const f32,
    wetland_fraction_ptr: *const f32,
    terrain_relief_ptr: *const f32,
    functional_type_fractions_ptr: *mut f32,
    nonvegetated_fractions_ptr: *mut f32,
    resource_potentials_ptr: *mut f32,
    confidence_ptr: *mut f32,
    dominant_cover_code_ptr: *mut u8,
    stats_out: *mut FunctionalVegetationStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || ocean_ptr.is_null()
        || vegetation_cover_ptr.is_null()
        || annual_npp_ptr.is_null()
        || standing_biomass_ptr.is_null()
        || growing_season_ptr.is_null()
        || productivity_seasonality_ptr.is_null()
        || drought_pressure_ptr.is_null()
        || cold_pressure_ptr.is_null()
        || heat_pressure_ptr.is_null()
        || waterlogging_pressure_ptr.is_null()
        || salinity_pressure_ptr.is_null()
        || woody_trait_ptr.is_null()
        || resource_conservative_trait_ptr.is_null()
        || fuel_continuity_ptr.is_null()
        || biosphere_confidence_ptr.is_null()
        || annual_temperature_ptr.is_null()
        || soil_fertility_ptr.is_null()
        || soil_depth_ptr.is_null()
        || soil_bearing_ptr.is_null()
        || soil_drainage_ptr.is_null()
        || glacier_fraction_ptr.is_null()
        || lake_fraction_ptr.is_null()
        || wetland_fraction_ptr.is_null()
        || terrain_relief_ptr.is_null()
        || functional_type_fractions_ptr.is_null()
        || nonvegetated_fractions_ptr.is_null()
        || resource_potentials_ptr.is_null()
        || confidence_ptr.is_null()
        || dominant_cover_code_ptr.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let controls = [
        warm_transition_midpoint_c,
        warm_transition_width_c,
        npp_response_half_saturation_kg_c_m2_year,
        biomass_response_half_saturation_kg_c_m2,
        terrain_relief_half_saturation_m,
        crop_soil_depth_half_saturation_m,
        strategy_confidence_multiplier,
    ];
    if controls.iter().any(|value| !value.is_finite())
        || warm_transition_width_c <= 0.0
        || npp_response_half_saturation_kg_c_m2_year <= 0.0
        || biomass_response_half_saturation_kg_c_m2 <= 0.0
        || terrain_relief_half_saturation_m <= 0.0
        || crop_soil_depth_half_saturation_m <= 0.0
        || !(0.0..=1.0).contains(&strategy_confidence_multiplier)
    {
        return 2;
    }

    let total = cell_count as usize;
    let functional_len = match total.checked_mul(FUNCTIONAL_TYPE_COUNT) {
        Some(value) => value,
        None => return 1,
    };
    let nonvegetated_len = match total.checked_mul(NONVEGETATED_TYPE_COUNT) {
        Some(value) => value,
        None => return 1,
    };
    let resource_len = match total.checked_mul(RESOURCE_POTENTIAL_COUNT) {
        Some(value) => value,
        None => return 1,
    };

    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let ocean = unsafe { slice::from_raw_parts(ocean_ptr, total) };
    let vegetation_cover = unsafe { slice::from_raw_parts(vegetation_cover_ptr, total) };
    let annual_npp = unsafe { slice::from_raw_parts(annual_npp_ptr, total) };
    let standing_biomass = unsafe { slice::from_raw_parts(standing_biomass_ptr, total) };
    let growing_season = unsafe { slice::from_raw_parts(growing_season_ptr, total) };
    let productivity_seasonality =
        unsafe { slice::from_raw_parts(productivity_seasonality_ptr, total) };
    let drought_pressure = unsafe { slice::from_raw_parts(drought_pressure_ptr, total) };
    let cold_pressure = unsafe { slice::from_raw_parts(cold_pressure_ptr, total) };
    let heat_pressure = unsafe { slice::from_raw_parts(heat_pressure_ptr, total) };
    let waterlogging_pressure = unsafe { slice::from_raw_parts(waterlogging_pressure_ptr, total) };
    let salinity_pressure = unsafe { slice::from_raw_parts(salinity_pressure_ptr, total) };
    let woody_trait = unsafe { slice::from_raw_parts(woody_trait_ptr, total) };
    let resource_conservative_trait =
        unsafe { slice::from_raw_parts(resource_conservative_trait_ptr, total) };
    let fuel_continuity = unsafe { slice::from_raw_parts(fuel_continuity_ptr, total) };
    let biosphere_confidence = unsafe { slice::from_raw_parts(biosphere_confidence_ptr, total) };
    let annual_temperature = unsafe { slice::from_raw_parts(annual_temperature_ptr, total) };
    let soil_fertility = unsafe { slice::from_raw_parts(soil_fertility_ptr, total) };
    let soil_depth = unsafe { slice::from_raw_parts(soil_depth_ptr, total) };
    let soil_bearing = unsafe { slice::from_raw_parts(soil_bearing_ptr, total) };
    let soil_drainage = unsafe { slice::from_raw_parts(soil_drainage_ptr, total) };
    let glacier_fraction = unsafe { slice::from_raw_parts(glacier_fraction_ptr, total) };
    let lake_fraction = unsafe { slice::from_raw_parts(lake_fraction_ptr, total) };
    let wetland_fraction = unsafe { slice::from_raw_parts(wetland_fraction_ptr, total) };
    let terrain_relief = unsafe { slice::from_raw_parts(terrain_relief_ptr, total) };

    let functional_type_fractions =
        unsafe { slice::from_raw_parts_mut(functional_type_fractions_ptr, functional_len) };
    let nonvegetated_fractions =
        unsafe { slice::from_raw_parts_mut(nonvegetated_fractions_ptr, nonvegetated_len) };
    let resource_potentials =
        unsafe { slice::from_raw_parts_mut(resource_potentials_ptr, resource_len) };
    let confidence = unsafe { slice::from_raw_parts_mut(confidence_ptr, total) };
    let dominant_cover_code = unsafe { slice::from_raw_parts_mut(dominant_cover_code_ptr, total) };

    let bounded_inputs = [
        ocean,
        vegetation_cover,
        growing_season,
        productivity_seasonality,
        drought_pressure,
        cold_pressure,
        heat_pressure,
        waterlogging_pressure,
        salinity_pressure,
        woody_trait,
        resource_conservative_trait,
        fuel_continuity,
        biosphere_confidence,
        soil_fertility,
        soil_bearing,
        soil_drainage,
        glacier_fraction,
        lake_fraction,
        wetland_fraction,
    ];
    if areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || bounded_inputs
            .iter()
            .flat_map(|values| values.iter())
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || annual_npp
            .iter()
            .chain(standing_biomass.iter())
            .chain(soil_depth.iter())
            .chain(terrain_relief.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        || annual_temperature.iter().any(|value| !value.is_finite())
    {
        return 3;
    }

    functional_type_fractions.fill(0.0);
    nonvegetated_fractions.fill(0.0);
    resource_potentials.fill(0.0);
    confidence.fill(0.0);
    dominant_cover_code.fill(0);

    let mut land_area = 0.0f64;
    let mut vegetated_area = 0.0f64;
    let mut woody_area = 0.0f64;
    let mut herbaceous_area = 0.0f64;
    let mut hydrophytic_area = 0.0f64;
    let mut bare_area = 0.0f64;
    let mut saline_area = 0.0f64;
    let mut ice_area = 0.0f64;
    let mut water_area = 0.0f64;
    let mut unsupported_area = 0.0f64;
    let mut fire_area = 0.0f64;
    let mut crop_area = 0.0f64;
    let mut maximum_partition_error = 0.0f64;

    for cell in 0..total {
        if ocean[cell] >= 0.5 {
            continue;
        }
        let area = areas[cell];
        land_area += area;
        let salinity = clamp01(f64::from(salinity_pressure[cell]));
        let wetland = clamp01(f64::from(wetland_fraction[cell]));
        let (vegetated, nonvegetated) = nonvegetated_partition(
            f64::from(vegetation_cover[cell]),
            f64::from(glacier_fraction[cell]),
            f64::from(lake_fraction[cell]),
            salinity,
            f64::from(soil_bearing[cell]),
        );
        for (class_index, value) in nonvegetated.iter().enumerate() {
            nonvegetated_fractions[class_index * total + cell] = *value as f32;
        }

        let npp_response = saturating_response(
            f64::from(annual_npp[cell]),
            npp_response_half_saturation_kg_c_m2_year,
        );
        let biomass_response = saturating_response(
            f64::from(standing_biomass[cell]),
            biomass_response_half_saturation_kg_c_m2,
        );
        let scores = strategy_scores(
            f64::from(annual_temperature[cell]),
            warm_transition_midpoint_c,
            warm_transition_width_c,
            npp_response,
            f64::from(growing_season[cell]),
            f64::from(productivity_seasonality[cell]),
            f64::from(drought_pressure[cell]),
            f64::from(cold_pressure[cell]),
            f64::from(heat_pressure[cell]),
            f64::from(waterlogging_pressure[cell]),
            salinity,
            f64::from(woody_trait[cell]),
            f64::from(resource_conservative_trait[cell]),
            wetland,
        );
        let score_sum = scores.iter().sum::<f64>();
        if vegetated > 0.0 {
            if score_sum <= 1e-30 {
                functional_type_fractions[LOW_STATURE_CONSERVATIVE * total + cell] =
                    vegetated as f32;
            } else {
                for (strategy_index, score) in scores.iter().enumerate() {
                    functional_type_fractions[strategy_index * total + cell] =
                        (vegetated * score / score_sum) as f32;
                }
            }
        }

        let woody_fraction = (COLD_WOODY..=SEASONAL_WOODY)
            .map(|index| f64::from(functional_type_fractions[index * total + cell]))
            .sum::<f64>();
        let herbaceous_fraction = f64::from(
            functional_type_fractions[COOL_HERBACEOUS * total + cell]
                + functional_type_fractions[WARM_HERBACEOUS * total + cell],
        );
        let shrub_fraction = f64::from(functional_type_fractions[XERIC_SHRUB * total + cell]);
        let hydrophytic_fraction = f64::from(functional_type_fractions[HYDROPHYTIC * total + cell]);
        let vegetation_denominator = vegetated.max(0.05);
        let grazeable_share =
            clamp01((herbaceous_fraction + 0.35 * shrub_fraction) / vegetation_denominator);
        let pasture_share =
            clamp01((herbaceous_fraction + 0.25 * shrub_fraction) / vegetation_denominator);
        let woody_share = clamp01(woody_fraction / vegetation_denominator);
        let wet = f64::from(waterlogging_pressure[cell]).max(wetland);
        let growing = f64::from(growing_season[cell]);
        let seasonality = f64::from(productivity_seasonality[cell]);
        let fertility = f64::from(soil_fertility[cell]);
        let relief_suitability = terrain_relief_half_saturation_m
            / (terrain_relief_half_saturation_m + f64::from(terrain_relief[cell]));
        let soil_depth_response = saturating_response(
            f64::from(soil_depth[cell]),
            crop_soil_depth_half_saturation_m,
        );
        let drainage_suitability =
            clamp01(1.0 - (f64::from(soil_drainage[cell]) - 0.65).abs() / 0.65);
        let fire = clamp01(
            f64::from(fuel_continuity[cell]).sqrt()
                * (0.35 + 0.65 * seasonality)
                * (1.0 - 0.80 * wet)
                * (1.0 - nonvegetated[PERSISTENT_ICE] - nonvegetated[INLAND_OPEN_WATER]),
        );
        let grazing = clamp01(
            grazeable_share
                * (npp_response * growing).sqrt()
                * (1.0 - 0.80 * salinity)
                * (1.0 - 0.50 * wet)
                * (1.0 - nonvegetated[PERSISTENT_ICE] - nonvegetated[INLAND_OPEN_WATER]),
        );
        let forest_resource = clamp01(
            woody_share
                * (biomass_response * f64::from(biosphere_confidence[cell])).sqrt()
                * (1.0 - nonvegetated[PERSISTENT_ICE] - nonvegetated[INLAND_OPEN_WATER]),
        );
        let pasture = clamp01(
            (pasture_share * npp_response * fertility * growing).sqrt()
                * relief_suitability.sqrt()
                * (1.0 - 0.65 * wet)
                * (1.0 - 0.80 * salinity)
                * (1.0 - nonvegetated[PERSISTENT_ICE] - nonvegetated[INLAND_OPEN_WATER]),
        );
        let crop_resource_base = clamp01(
            npp_response
                * fertility
                * soil_depth_response
                * drainage_suitability
                * growing
                * f64::from(biosphere_confidence[cell]),
        )
        .powf(1.0 / 6.0);
        let crop = clamp01(
            (1.0 - nonvegetated[PERSISTENT_ICE] - nonvegetated[INLAND_OPEN_WATER])
                * crop_resource_base
                * relief_suitability.sqrt()
                * (1.0 - 0.85 * salinity)
                * (1.0 - 0.65 * wet)
                * (1.0 - 0.50 * f64::from(cold_pressure[cell]))
                * (1.0 - 0.40 * f64::from(heat_pressure[cell])),
        );
        resource_potentials[FIRE_TENDENCY * total + cell] = fire as f32;
        resource_potentials[GRAZING_POTENTIAL * total + cell] = grazing as f32;
        resource_potentials[FOREST_RESOURCE_POTENTIAL * total + cell] = forest_resource as f32;
        resource_potentials[PASTURE_POTENTIAL * total + cell] = pasture as f32;
        resource_potentials[CROP_POTENTIAL * total + cell] = crop as f32;
        confidence[cell] = (clamp01(f64::from(biosphere_confidence[cell]))
            * strategy_confidence_multiplier) as f32;

        let functional_sum = (0..FUNCTIONAL_TYPE_COUNT)
            .map(|index| f64::from(functional_type_fractions[index * total + cell]))
            .sum::<f64>();
        let mut dominant_strategy = 0usize;
        let mut dominant_strategy_fraction = -1.0f64;
        for strategy_index in 0..FUNCTIONAL_TYPE_COUNT {
            let fraction = f64::from(functional_type_fractions[strategy_index * total + cell]);
            if fraction > dominant_strategy_fraction {
                dominant_strategy_fraction = fraction;
                dominant_strategy = strategy_index;
            }
        }
        let mut dominant_nonvegetated = 0usize;
        let mut dominant_nonvegetated_fraction = -1.0f64;
        for class_index in 0..NONVEGETATED_TYPE_COUNT {
            let fraction = f64::from(nonvegetated_fractions[class_index * total + cell]);
            if fraction > dominant_nonvegetated_fraction {
                dominant_nonvegetated_fraction = fraction;
                dominant_nonvegetated = class_index;
            }
        }
        dominant_cover_code[cell] = if functional_sum > dominant_nonvegetated_fraction {
            (dominant_strategy + 1) as u8
        } else {
            (FUNCTIONAL_TYPE_COUNT + dominant_nonvegetated + 1) as u8
        };

        let nonvegetated_sum = nonvegetated.iter().sum::<f64>();
        maximum_partition_error =
            maximum_partition_error.max((functional_sum + nonvegetated_sum - 1.0).abs());

        vegetated_area += area * functional_sum;
        woody_area += area * woody_fraction;
        herbaceous_area += area * herbaceous_fraction;
        hydrophytic_area += area * hydrophytic_fraction;
        bare_area += area * nonvegetated[BARE_GROUND];
        saline_area += area * nonvegetated[SALINE_BARREN];
        ice_area += area * nonvegetated[PERSISTENT_ICE];
        water_area += area * nonvegetated[INLAND_OPEN_WATER];
        unsupported_area += area * nonvegetated[UNSUPPORTED_SURFACE];
        fire_area += area * fire;
        crop_area += area * crop;
    }

    unsafe {
        *stats_out = FunctionalVegetationStats {
            land_mean_functional_vegetated_fraction: (vegetated_area / land_area.max(1e-30)) as f32,
            vegetated_woody_fraction: (woody_area / vegetated_area.max(1e-30)) as f32,
            vegetated_herbaceous_fraction: (herbaceous_area / vegetated_area.max(1e-30)) as f32,
            vegetated_hydrophytic_fraction: (hydrophytic_area / vegetated_area.max(1e-30)) as f32,
            land_mean_bare_ground_fraction: (bare_area / land_area.max(1e-30)) as f32,
            land_mean_saline_barren_fraction: (saline_area / land_area.max(1e-30)) as f32,
            land_mean_ice_fraction: (ice_area / land_area.max(1e-30)) as f32,
            land_mean_inland_water_fraction: (water_area / land_area.max(1e-30)) as f32,
            land_mean_unsupported_surface_fraction: (unsupported_area / land_area.max(1e-30))
                as f32,
            land_mean_fire_tendency: (fire_area / land_area.max(1e-30)) as f32,
            land_mean_crop_potential: (crop_area / land_area.max(1e-30)) as f32,
            maximum_partition_absolute_error: maximum_partition_error as f32,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{nonvegetated_partition, strategy_scores, UNSUPPORTED_SURFACE};

    #[test]
    fn strategy_partition_preserves_woody_budget_and_closes() {
        let scores = strategy_scores(
            12.0, 18.0, 12.0, 0.7, 0.8, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0, 0.40, 0.3, 0.0,
        );
        let woody = scores[0] + scores[1] + scores[2];

        assert!((scores.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((woody - (0.03 + 0.97 * 0.40)).abs() < 1e-12);
        assert_eq!(scores[6], 0.0);
    }

    #[test]
    fn nonvegetated_partition_closes_land_area() {
        let (vegetated, nonvegetated) = nonvegetated_partition(0.40, 0.10, 0.20, 0.25, 0.50);
        let total = vegetated + nonvegetated.iter().sum::<f64>();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((vegetated - 0.40).abs() < 1e-12);
        assert!((nonvegetated[UNSUPPORTED_SURFACE] - 0.20).abs() < 1e-12);
    }

    #[test]
    fn strategies_respond_directionally_to_climate() {
        let cold = strategy_scores(
            -5.0, 18.0, 12.0, 0.7, 0.6, 0.5, 0.2, 0.9, 0.0, 0.1, 0.0, 0.6, 0.7, 0.0,
        );
        assert!(cold[0] > cold[1]);

        let warm_humid = strategy_scores(
            27.0, 18.0, 12.0, 0.9, 0.9, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.8, 0.2, 0.0,
        );
        assert!(warm_humid[1] > warm_humid[0]);
        assert!(warm_humid[1] > warm_humid[3]);

        let wet = strategy_scores(
            20.0, 18.0, 12.0, 0.8, 0.9, 0.2, 0.0, 0.0, 0.0, 0.95, 0.0, 0.1, 0.2, 0.9,
        );
        assert_eq!(
            wet[6],
            wet.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }
}
