use std::slice;

const FUNCTIONAL_TYPE_COUNT: usize = 8;
const NONVEGETATED_TYPE_COUNT: usize = 5;
const RESOURCE_POTENTIAL_COUNT: usize = 5;
const BIOME_COUNT: usize = 13;

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

const TROPICAL_RAINFOREST: usize = 0;
const TROPICAL_SEASONAL_FOREST: usize = 1;
const SAVANNA: usize = 2;
const HOT_DESERT: usize = 3;
const XERIC_SHRUBLAND: usize = 4;
const TEMPERATE_FOREST: usize = 5;
const TEMPERATE_GRASSLAND: usize = 6;
const STEPPE: usize = 7;
const BOREAL_FOREST: usize = 8;
const TUNDRA: usize = 9;
const COLD_DESERT: usize = 10;
const ALPINE: usize = 11;
const WETLAND: usize = 12;

const INLAND_WATER_LANDSCAPE_CODE: u8 = 14;
const PERSISTENT_ICE_LANDSCAPE_CODE: u8 = 15;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DerivedBiomeStats {
    pub land_mean_classification_confidence: f32,
    pub land_mean_dominance_margin: f32,
    pub land_mean_transition_index: f32,
    pub ambiguous_land_area_fraction: f32,
    pub classifiable_land_area_fraction: f32,
    pub maximum_partition_absolute_error: f32,
}

#[no_mangle]
pub extern "C" fn derived_biomes_native_abi_version() -> u32 {
    1
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn smoothstep01(value: f64) -> f64 {
    let bounded = clamp01(value);
    bounded * bounded * (3.0 - 2.0 * bounded)
}

fn ramp(value: f64, lower: f64, upper: f64) -> f64 {
    smoothstep01((value - lower) / (upper - lower))
}

fn window(value: f64, lower: f64, plateau_lower: f64, plateau_upper: f64, upper: f64) -> f64 {
    ramp(value, lower, plateau_lower) * (1.0 - ramp(value, plateau_upper, upper))
}

fn intermediate(value: f64, center: f64, half_width: f64) -> f64 {
    1.0 - smoothstep01((value - center).abs() / half_width)
}

#[derive(Clone, Copy)]
struct BiomeInputs {
    temperature_c: f64,
    precipitation_mm: f64,
    growing_season: f64,
    seasonality: f64,
    drought: f64,
    waterlogging: f64,
    wetland_fraction: f64,
    elevation_m: f64,
    relief_m: f64,
    fire_tendency: f64,
    functional: [f64; FUNCTIONAL_TYPE_COUNT],
    nonvegetated: [f64; NONVEGETATED_TYPE_COUNT],
}

fn biome_scores(
    input: BiomeInputs,
    highland_elevation_start_m: f64,
    highland_elevation_full_m: f64,
    highland_relief_start_m: f64,
    highland_relief_full_m: f64,
) -> [f64; BIOME_COUNT] {
    let vegetated = input.functional.iter().sum::<f64>();
    let vegetation_denominator = vegetated.max(0.02);
    let share = |index: usize| clamp01(input.functional[index] / vegetation_denominator);
    let cold_woody = share(COLD_WOODY);
    let warm_evergreen = share(WARM_EVERGREEN_WOODY);
    let seasonal_woody = share(SEASONAL_WOODY);
    let xeric_shrub = share(XERIC_SHRUB);
    let cool_herbaceous = share(COOL_HERBACEOUS);
    let warm_herbaceous = share(WARM_HERBACEOUS);
    let hydrophytic = share(HYDROPHYTIC);
    let low_stature = share(LOW_STATURE_CONSERVATIVE);

    let ice = clamp01(input.nonvegetated[PERSISTENT_ICE]);
    let water = clamp01(input.nonvegetated[INLAND_OPEN_WATER]).min(1.0 - ice);
    let ecological_ground = (1.0 - ice - water).max(0.0);
    let bare_ground = input.nonvegetated[BARE_GROUND]
        + input.nonvegetated[SALINE_BARREN]
        + input.nonvegetated[UNSUPPORTED_SURFACE];
    let bare_share = clamp01(bare_ground / ecological_ground.max(0.02));
    let unsupported_share =
        clamp01(input.nonvegetated[UNSUPPORTED_SURFACE] / ecological_ground.max(0.02));
    let cover_share = clamp01(vegetated / ecological_ground.max(0.02));

    let tropical = ramp(input.temperature_c, 17.0, 24.0);
    let warm = ramp(input.temperature_c, 10.0, 22.0);
    let temperate = window(input.temperature_c, -7.0, 4.0, 18.0, 27.0);
    let boreal_temperature = window(input.temperature_c, -22.0, -8.0, 5.0, 13.0);
    let tundra_temperature =
        ramp(input.temperature_c, -25.0, -12.0) * (1.0 - ramp(input.temperature_c, -2.0, 8.0));
    let cool = 1.0 - ramp(input.temperature_c, 8.0, 20.0);

    let drought = clamp01(input.drought);
    let moist = 1.0 - drought;
    let very_wet = ramp(input.precipitation_mm, 750.0, 1_800.0).max(moist.powf(2.5));
    let extreme_dry = ramp(drought, 0.48, 0.88)
        * (0.35 + 0.65 * (1.0 - ramp(input.precipitation_mm, 250.0, 900.0)));
    let mesic = intermediate(drought, 0.34, 0.40);
    let seasonal =
        ramp(input.seasonality, 0.015, 0.16).max(intermediate(drought, 0.48, 0.34) * 0.55);
    let wet = clamp01(input.waterlogging.max(input.wetland_fraction));
    let highland = ramp(
        input.elevation_m,
        highland_elevation_start_m,
        highland_elevation_full_m,
    )
    .max(
        0.80 * ramp(
            input.relief_m,
            highland_relief_start_m,
            highland_relief_full_m,
        ),
    );
    let short_growing_season = 1.0 - clamp01(input.growing_season);

    let mut scores = [0.0f64; BIOME_COUNT];
    scores[TROPICAL_RAINFOREST] = tropical.powf(1.4)
        * very_wet.powf(1.35)
        * (0.15 + 0.85 * warm_evergreen)
        * (0.35 + 0.65 * cover_share)
        * (1.0 - 0.65 * seasonal)
        * (1.0 - 0.70 * wet);
    scores[TROPICAL_SEASONAL_FOREST] = tropical
        * moist.sqrt()
        * (0.15 + 0.85 * (0.55 * warm_evergreen + seasonal_woody))
        * (0.30 + 0.70 * seasonal)
        * (0.35 + 0.65 * cover_share)
        * (1.0 - 0.65 * wet);
    scores[SAVANNA] = tropical
        * (0.20 + 0.80 * (warm_herbaceous + 0.45 * seasonal_woody).min(1.0))
        * (0.25 + 0.75 * intermediate(drought, 0.52, 0.38))
        * (0.55 + 0.45 * clamp01(input.fire_tendency))
        * (0.30 + 0.70 * cover_share)
        * (1.0 - 0.75 * wet)
        * 1.25;
    scores[HOT_DESERT] = warm
        * extreme_dry.powf(1.25)
        * (0.25 + 0.75 * (1.0 - cover_share + xeric_shrub + low_stature).min(1.0))
        * (0.35 + 0.65 * bare_share)
        * (1.0 - 0.80 * wet);
    scores[XERIC_SHRUBLAND] = ramp(input.temperature_c, 2.0, 16.0)
        * drought
        * (1.0 - 0.70 * extreme_dry)
        * (0.20 + 0.80 * (xeric_shrub + 0.20 * seasonal_woody).min(1.0))
        * (0.35 + 0.65 * cover_share)
        * (1.0 - 0.75 * wet);
    scores[TEMPERATE_FOREST] = temperate
        * moist.powf(1.2)
        * (0.15 + 0.85 * (cold_woody + seasonal_woody).min(1.0))
        * (0.35 + 0.65 * cover_share)
        * (1.0 - 0.60 * highland)
        * (1.0 - 0.65 * wet)
        * (1.0 - 0.55 * tropical);
    scores[TEMPERATE_GRASSLAND] = temperate
        * mesic
        * (0.20 + 0.80 * (cool_herbaceous + 0.35 * warm_herbaceous).min(1.0))
        * (0.35 + 0.65 * cover_share)
        * (1.0 - 0.65 * wet)
        * (1.0 - 0.80 * tropical)
        * 0.82;
    scores[STEPPE] = temperate
        * drought
        * (1.0 - 0.55 * extreme_dry)
        * (0.20 + 0.80 * (cool_herbaceous + 0.45 * xeric_shrub).min(1.0))
        * (0.30 + 0.70 * cover_share)
        * (1.0 - 0.65 * wet)
        * (1.0 - 0.65 * tropical);
    scores[BOREAL_FOREST] = boreal_temperature
        * moist
        * (0.18 + 0.82 * cold_woody)
        * (0.30 + 0.70 * cover_share)
        * (1.0 - 0.70 * highland)
        * (1.0 - 0.60 * wet)
        * 1.80;
    scores[TUNDRA] = tundra_temperature
        * (0.20 + 0.80 * (low_stature + 0.45 * cool_herbaceous).min(1.0))
        * (0.25 + 0.75 * short_growing_season)
        * (0.35 + 0.65 * cover_share)
        * (1.0 - 0.75 * highland)
        * (1.0 - 0.60 * wet);
    scores[COLD_DESERT] = cool
        * extreme_dry
        * (0.25 + 0.75 * (low_stature + bare_share).min(1.0))
        * (1.0 - 0.55 * highland)
        * (1.0 - 0.75 * wet);
    scores[ALPINE] = highland
        * (0.25 + 0.75 * cold_or_short_growing(input.temperature_c, short_growing_season))
        * (0.20 + 0.80 * (low_stature + 0.45 * cool_herbaceous + unsupported_share).min(1.0))
        * (1.0 - 0.65 * wet);
    scores[WETLAND] =
        wet.powf(1.4) * (0.15 + 0.85 * hydrophytic) * (0.35 + 0.65 * moist) * (1.0 - 0.65 * water);

    scores
}

fn cold_or_short_growing(temperature_c: f64, short_growing_season: f64) -> f64 {
    (1.0 - ramp(temperature_c, 4.0, 18.0)).max(short_growing_season)
}

fn fallback_biome(input: BiomeInputs, highland: f64) -> usize {
    if input.waterlogging.max(input.wetland_fraction) >= 0.5 {
        WETLAND
    } else if highland >= 0.5 {
        ALPINE
    } else if input.temperature_c < 3.0 {
        if input.drought >= 0.65 {
            COLD_DESERT
        } else {
            TUNDRA
        }
    } else if input.temperature_c >= 18.0 {
        if input.drought >= 0.65 {
            HOT_DESERT
        } else {
            SAVANNA
        }
    } else if input.drought >= 0.60 {
        STEPPE
    } else {
        TEMPERATE_FOREST
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// Derive familiar biome mixtures from causal climate and functional vegetation.
///
/// # Safety
/// Every pointer must reference the declared number of contiguous elements.
/// Input and output buffers must not overlap.
pub unsafe extern "C" fn derived_biomes_run(
    cell_count: i32,
    highland_elevation_start_m: f64,
    highland_elevation_full_m: f64,
    highland_relief_start_m: f64,
    highland_relief_full_m: f64,
    minimum_classifiable_ground_fraction: f64,
    ambiguity_margin_threshold: f64,
    transition_confidence_weight: f64,
    area_ptr: *const f64,
    ocean_ptr: *const f32,
    annual_temperature_ptr: *const f32,
    annual_precipitation_ptr: *const f32,
    growing_season_ptr: *const f32,
    seasonality_ptr: *const f32,
    drought_ptr: *const f32,
    waterlogging_ptr: *const f32,
    biosphere_confidence_ptr: *const f32,
    functional_confidence_ptr: *const f32,
    wetland_fraction_ptr: *const f32,
    elevation_ptr: *const f32,
    relief_ptr: *const f32,
    functional_type_fractions_ptr: *const f32,
    nonvegetated_fractions_ptr: *const f32,
    resource_potentials_ptr: *const f32,
    biome_fractions_ptr: *mut f32,
    classification_confidence_ptr: *mut f32,
    dominance_margin_ptr: *mut f32,
    transition_index_ptr: *mut f32,
    primary_biome_code_ptr: *mut u8,
    secondary_biome_code_ptr: *mut u8,
    dominant_landscape_code_ptr: *mut u8,
    stats_out: *mut DerivedBiomeStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || ocean_ptr.is_null()
        || annual_temperature_ptr.is_null()
        || annual_precipitation_ptr.is_null()
        || growing_season_ptr.is_null()
        || seasonality_ptr.is_null()
        || drought_ptr.is_null()
        || waterlogging_ptr.is_null()
        || biosphere_confidence_ptr.is_null()
        || functional_confidence_ptr.is_null()
        || wetland_fraction_ptr.is_null()
        || elevation_ptr.is_null()
        || relief_ptr.is_null()
        || functional_type_fractions_ptr.is_null()
        || nonvegetated_fractions_ptr.is_null()
        || resource_potentials_ptr.is_null()
        || biome_fractions_ptr.is_null()
        || classification_confidence_ptr.is_null()
        || dominance_margin_ptr.is_null()
        || transition_index_ptr.is_null()
        || primary_biome_code_ptr.is_null()
        || secondary_biome_code_ptr.is_null()
        || dominant_landscape_code_ptr.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let controls = [
        highland_elevation_start_m,
        highland_elevation_full_m,
        highland_relief_start_m,
        highland_relief_full_m,
        minimum_classifiable_ground_fraction,
        ambiguity_margin_threshold,
        transition_confidence_weight,
    ];
    if controls.iter().any(|value| !value.is_finite())
        || highland_elevation_full_m <= highland_elevation_start_m
        || highland_relief_full_m <= highland_relief_start_m
        || !(0.0..=1.0).contains(&minimum_classifiable_ground_fraction)
        || !(0.0..=1.0).contains(&ambiguity_margin_threshold)
        || !(0.0..=1.0).contains(&transition_confidence_weight)
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
    let biome_len = match total.checked_mul(BIOME_COUNT) {
        Some(value) => value,
        None => return 1,
    };

    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let ocean = unsafe { slice::from_raw_parts(ocean_ptr, total) };
    let annual_temperature = unsafe { slice::from_raw_parts(annual_temperature_ptr, total) };
    let annual_precipitation = unsafe { slice::from_raw_parts(annual_precipitation_ptr, total) };
    let growing_season = unsafe { slice::from_raw_parts(growing_season_ptr, total) };
    let seasonality = unsafe { slice::from_raw_parts(seasonality_ptr, total) };
    let drought = unsafe { slice::from_raw_parts(drought_ptr, total) };
    let waterlogging = unsafe { slice::from_raw_parts(waterlogging_ptr, total) };
    let biosphere_confidence = unsafe { slice::from_raw_parts(biosphere_confidence_ptr, total) };
    let functional_confidence = unsafe { slice::from_raw_parts(functional_confidence_ptr, total) };
    let wetland_fraction = unsafe { slice::from_raw_parts(wetland_fraction_ptr, total) };
    let elevation = unsafe { slice::from_raw_parts(elevation_ptr, total) };
    let relief = unsafe { slice::from_raw_parts(relief_ptr, total) };
    let functional_type_fractions =
        unsafe { slice::from_raw_parts(functional_type_fractions_ptr, functional_len) };
    let nonvegetated_fractions =
        unsafe { slice::from_raw_parts(nonvegetated_fractions_ptr, nonvegetated_len) };
    let resource_potentials =
        unsafe { slice::from_raw_parts(resource_potentials_ptr, resource_len) };

    let biome_fractions = unsafe { slice::from_raw_parts_mut(biome_fractions_ptr, biome_len) };
    let classification_confidence =
        unsafe { slice::from_raw_parts_mut(classification_confidence_ptr, total) };
    let dominance_margin = unsafe { slice::from_raw_parts_mut(dominance_margin_ptr, total) };
    let transition_index = unsafe { slice::from_raw_parts_mut(transition_index_ptr, total) };
    let primary_biome_code = unsafe { slice::from_raw_parts_mut(primary_biome_code_ptr, total) };
    let secondary_biome_code =
        unsafe { slice::from_raw_parts_mut(secondary_biome_code_ptr, total) };
    let dominant_landscape_code =
        unsafe { slice::from_raw_parts_mut(dominant_landscape_code_ptr, total) };

    let bounded_inputs = [
        ocean,
        growing_season,
        seasonality,
        drought,
        waterlogging,
        biosphere_confidence,
        functional_confidence,
        wetland_fraction,
        functional_type_fractions,
        nonvegetated_fractions,
        resource_potentials,
    ];
    if areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || bounded_inputs
            .iter()
            .flat_map(|values| values.iter())
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || annual_temperature.iter().any(|value| !value.is_finite())
        || annual_precipitation
            .iter()
            .chain(relief.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        || elevation.iter().any(|value| !value.is_finite())
    {
        return 3;
    }

    biome_fractions.fill(0.0);
    classification_confidence.fill(0.0);
    dominance_margin.fill(0.0);
    transition_index.fill(0.0);
    primary_biome_code.fill(0);
    secondary_biome_code.fill(0);
    dominant_landscape_code.fill(0);

    let mut land_area = 0.0f64;
    let mut classifiable_area = 0.0f64;
    let mut confidence_area = 0.0f64;
    let mut margin_area = 0.0f64;
    let mut transition_area = 0.0f64;
    let mut ambiguous_area = 0.0f64;
    let mut maximum_partition_error = 0.0f64;

    for cell in 0..total {
        if ocean[cell] >= 0.5 {
            continue;
        }
        let area = areas[cell];
        land_area += area;
        let functional =
            std::array::from_fn(|index| f64::from(functional_type_fractions[index * total + cell]));
        let nonvegetated =
            std::array::from_fn(|index| f64::from(nonvegetated_fractions[index * total + cell]));
        let input = BiomeInputs {
            temperature_c: f64::from(annual_temperature[cell]),
            precipitation_mm: f64::from(annual_precipitation[cell]),
            growing_season: f64::from(growing_season[cell]),
            seasonality: f64::from(seasonality[cell]),
            drought: f64::from(drought[cell]),
            waterlogging: f64::from(waterlogging[cell]),
            wetland_fraction: f64::from(wetland_fraction[cell]),
            elevation_m: f64::from(elevation[cell]),
            relief_m: f64::from(relief[cell]),
            fire_tendency: f64::from(resource_potentials[FIRE_TENDENCY * total + cell]),
            functional,
            nonvegetated,
        };
        let ice = clamp01(nonvegetated[PERSISTENT_ICE]);
        let water = clamp01(nonvegetated[INLAND_OPEN_WATER]).min(1.0 - ice);
        let ground = (1.0 - ice - water).max(0.0);
        let highland = ramp(
            input.elevation_m,
            highland_elevation_start_m,
            highland_elevation_full_m,
        )
        .max(
            0.80 * ramp(
                input.relief_m,
                highland_relief_start_m,
                highland_relief_full_m,
            ),
        );
        let mut scores = biome_scores(
            input,
            highland_elevation_start_m,
            highland_elevation_full_m,
            highland_relief_start_m,
            highland_relief_full_m,
        );
        let mut score_sum = scores.iter().sum::<f64>();
        if score_sum <= 1e-30 {
            scores[fallback_biome(input, highland)] = 1.0;
            score_sum = 1.0;
        }

        let mut first_index = 0usize;
        let mut second_index = 0usize;
        let mut first_probability = -1.0f64;
        let mut second_probability = -1.0f64;
        let mut entropy = 0.0f64;
        for (index, score) in scores.iter().enumerate() {
            let probability = score / score_sum;
            biome_fractions[index * total + cell] = (ground * probability) as f32;
            if probability > 0.0 {
                entropy -= probability * probability.ln();
            }
            if probability > first_probability {
                second_probability = first_probability;
                second_index = first_index;
                first_probability = probability;
                first_index = index;
            } else if probability > second_probability {
                second_probability = probability;
                second_index = index;
            }
        }
        let margin = clamp01(first_probability - second_probability.max(0.0));
        let transition = clamp01(entropy / (BIOME_COUNT as f64).ln());
        let ground_support = if minimum_classifiable_ground_fraction <= 0.0 {
            1.0
        } else {
            ramp(ground, 0.0, minimum_classifiable_ground_fraction)
        };
        let upstream_confidence =
            (f64::from(biosphere_confidence[cell]) * f64::from(functional_confidence[cell])).sqrt();
        let confidence = clamp01(
            upstream_confidence
                * ground_support
                * (1.0 - transition_confidence_weight * transition),
        );
        dominance_margin[cell] = margin as f32;
        transition_index[cell] = transition as f32;
        classification_confidence[cell] = confidence as f32;

        if ground >= minimum_classifiable_ground_fraction {
            primary_biome_code[cell] = (first_index + 1) as u8;
            secondary_biome_code[cell] = (second_index + 1) as u8;
            classifiable_area += area;
            confidence_area += confidence * area;
            margin_area += margin * area;
            transition_area += transition * area;
            if margin < ambiguity_margin_threshold {
                ambiguous_area += area;
            }
        }
        dominant_landscape_code[cell] = if ground > ice.max(water) {
            primary_biome_code[cell]
        } else if water >= ice {
            INLAND_WATER_LANDSCAPE_CODE
        } else {
            PERSISTENT_ICE_LANDSCAPE_CODE
        };

        let represented = (0..BIOME_COUNT)
            .map(|index| f64::from(biome_fractions[index * total + cell]))
            .sum::<f64>()
            + ice
            + water;
        maximum_partition_error = maximum_partition_error.max((represented - 1.0).abs());
    }

    let classifiable_denominator = classifiable_area.max(1e-30);
    unsafe {
        *stats_out = DerivedBiomeStats {
            land_mean_classification_confidence: (confidence_area / classifiable_denominator)
                as f32,
            land_mean_dominance_margin: (margin_area / classifiable_denominator) as f32,
            land_mean_transition_index: (transition_area / classifiable_denominator) as f32,
            ambiguous_land_area_fraction: (ambiguous_area / land_area.max(1e-30)) as f32,
            classifiable_land_area_fraction: (classifiable_area / land_area.max(1e-30)) as f32,
            maximum_partition_absolute_error: maximum_partition_error as f32,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> BiomeInputs {
        BiomeInputs {
            temperature_c: 15.0,
            precipitation_mm: 800.0,
            growing_season: 0.8,
            seasonality: 0.1,
            drought: 0.3,
            waterlogging: 0.0,
            wetland_fraction: 0.0,
            elevation_m: 300.0,
            relief_m: 150.0,
            fire_tendency: 0.2,
            functional: [0.05, 0.05, 0.35, 0.05, 0.15, 0.05, 0.0, 0.05],
            nonvegetated: [0.20, 0.0, 0.0, 0.0, 0.05],
        }
    }

    fn scores(input: BiomeInputs) -> [f64; BIOME_COUNT] {
        biome_scores(input, 1_000.0, 3_000.0, 250.0, 800.0)
    }

    #[test]
    fn humid_tropical_evergreen_state_prefers_rainforest() {
        let mut input = base_inputs();
        input.temperature_c = 26.0;
        input.precipitation_mm = 2_400.0;
        input.drought = 0.05;
        input.seasonality = 0.01;
        input.functional = [0.0, 0.70, 0.05, 0.0, 0.02, 0.03, 0.02, 0.0];
        let values = scores(input);
        assert_eq!(
            values
                .iter()
                .position(|value| *value == values.iter().copied().fold(0.0, f64::max)),
            Some(TROPICAL_RAINFOREST)
        );
    }

    #[test]
    fn hot_arid_sparse_state_prefers_hot_desert() {
        let mut input = base_inputs();
        input.temperature_c = 28.0;
        input.precipitation_mm = 80.0;
        input.drought = 0.95;
        input.functional = [0.0, 0.0, 0.01, 0.06, 0.0, 0.01, 0.0, 0.02];
        input.nonvegetated = [0.75, 0.05, 0.0, 0.0, 0.10];
        let values = scores(input);
        assert!(values[HOT_DESERT] > values[SAVANNA]);
        assert!(values[HOT_DESERT] > values[XERIC_SHRUBLAND]);
    }

    #[test]
    fn wet_hydrophytic_state_prefers_wetland() {
        let mut input = base_inputs();
        input.waterlogging = 0.95;
        input.wetland_fraction = 0.50;
        input.functional = [0.0, 0.02, 0.03, 0.0, 0.05, 0.05, 0.50, 0.05];
        let values = scores(input);
        assert_eq!(
            values
                .iter()
                .position(|value| *value == values.iter().copied().fold(0.0, f64::max)),
            Some(WETLAND)
        );
    }

    #[test]
    fn cold_high_relief_state_prefers_alpine_over_tundra() {
        let mut input = base_inputs();
        input.temperature_c = -3.0;
        input.growing_season = 0.2;
        input.elevation_m = 2_800.0;
        input.relief_m = 700.0;
        input.functional = [0.02, 0.0, 0.01, 0.01, 0.08, 0.0, 0.0, 0.25];
        input.nonvegetated = [0.25, 0.0, 0.0, 0.0, 0.38];
        let values = scores(input);
        assert!(values[ALPINE] > values[TUNDRA]);
    }
}
