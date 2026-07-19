use std::slice;

const MONTHS: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PotentialBiosphereStats {
    pub land_mean_annual_npp_kg_c_m2: f32,
    pub land_mean_vegetation_cover_fraction: f32,
    pub land_mean_standing_biomass_kg_c_m2: f32,
    pub potentially_vegetated_land_area_fraction: f32,
    pub land_mean_growing_season_fraction: f32,
    pub land_mean_woody_allocation_trait: f32,
    pub maximum_rooting_depth_m: f32,
    pub maximum_canopy_height_m: f32,
}

#[no_mangle]
pub extern "C" fn potential_biosphere_native_abi_version() -> u32 {
    3
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn normalized_seasonality(monthly: &[f64; MONTHS], annual: f64) -> f64 {
    if annual <= 1e-15 {
        return 0.0;
    }
    let concentration = monthly
        .iter()
        .map(|value| (value / annual).powi(2))
        .sum::<f64>();
    ((MONTHS as f64 * concentration - 1.0) / (MONTHS as f64 - 1.0)).clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn biomass_residence_score(
    productivity_response: f64,
    woody: f64,
    conservative: f64,
    baseline_fraction: f64,
    woody_weight: f64,
    conservative_weight: f64,
    low_productivity_weight: f64,
) -> f64 {
    let weight_sum = woody_weight + conservative_weight + low_productivity_weight;
    let weighted_response = (woody_weight * woody
        + conservative_weight * conservative
        + low_productivity_weight * (1.0 - productivity_response))
        / weight_sum;
    baseline_fraction + (1.0 - baseline_fraction) * clamp01(weighted_response)
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// Compute continuous terrestrial producer-community potentials.
///
/// # Safety
/// Every pointer must reference the declared number of contiguous elements.
/// Input and output buffers must not overlap.
pub unsafe extern "C" fn potential_biosphere_run(
    cell_count: i32,
    energy_per_kg_carbon_mj: f64,
    cover_half_saturation_npp_kg_c_m2_year: f64,
    active_month_thermal_threshold: f64,
    active_month_water_threshold: f64,
    cold_pressure_reference_c: f64,
    cold_pressure_release_c: f64,
    heat_pressure_onset_c: f64,
    heat_pressure_reference_c: f64,
    minimum_biomass_residence_years: f64,
    maximum_biomass_residence_years: f64,
    biomass_residence_baseline_fraction: f64,
    woody_biomass_residence_weight: f64,
    resource_conservative_biomass_residence_weight: f64,
    low_productivity_biomass_residence_weight: f64,
    maximum_rooting_depth_m: f64,
    maximum_canopy_height_m: f64,
    maximum_leaf_area_index: f64,
    maximum_standing_biomass_kg_c_m2: f64,
    area_ptr: *const f64,
    ocean_ptr: *const f32,
    monthly_primary_energy_ptr: *const f32,
    monthly_thermal_opportunity_ptr: *const f32,
    monthly_water_opportunity_ptr: *const f32,
    monthly_temperature_ptr: *const f32,
    monthly_soil_saturation_ptr: *const f32,
    surface_support_ptr: *const f32,
    nutrient_support_ptr: *const f32,
    environmental_stress_ptr: *const f32,
    soil_depth_ptr: *const f32,
    regolith_depth_ptr: *const f32,
    salinity_ptr: *const f32,
    hydric_fraction_ptr: *const f32,
    soil_confidence_ptr: *const f32,
    envelope_confidence_ptr: *const f32,
    monthly_npp_ptr: *mut f32,
    annual_npp_ptr: *mut f32,
    vegetation_cover_ptr: *mut f32,
    standing_biomass_ptr: *mut f32,
    growing_season_ptr: *mut f32,
    productivity_seasonality_ptr: *mut f32,
    drought_pressure_ptr: *mut f32,
    cold_pressure_ptr: *mut f32,
    heat_pressure_ptr: *mut f32,
    waterlogging_pressure_ptr: *mut f32,
    salinity_pressure_ptr: *mut f32,
    woody_trait_ptr: *mut f32,
    resource_conservative_trait_ptr: *mut f32,
    rooting_depth_ptr: *mut f32,
    canopy_height_ptr: *mut f32,
    leaf_area_index_ptr: *mut f32,
    fuel_continuity_ptr: *mut f32,
    confidence_ptr: *mut f32,
    stats_out: *mut PotentialBiosphereStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || ocean_ptr.is_null()
        || monthly_primary_energy_ptr.is_null()
        || monthly_thermal_opportunity_ptr.is_null()
        || monthly_water_opportunity_ptr.is_null()
        || monthly_temperature_ptr.is_null()
        || monthly_soil_saturation_ptr.is_null()
        || surface_support_ptr.is_null()
        || nutrient_support_ptr.is_null()
        || environmental_stress_ptr.is_null()
        || soil_depth_ptr.is_null()
        || regolith_depth_ptr.is_null()
        || salinity_ptr.is_null()
        || hydric_fraction_ptr.is_null()
        || soil_confidence_ptr.is_null()
        || envelope_confidence_ptr.is_null()
        || monthly_npp_ptr.is_null()
        || annual_npp_ptr.is_null()
        || vegetation_cover_ptr.is_null()
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
        || rooting_depth_ptr.is_null()
        || canopy_height_ptr.is_null()
        || leaf_area_index_ptr.is_null()
        || fuel_continuity_ptr.is_null()
        || confidence_ptr.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let controls = [
        energy_per_kg_carbon_mj,
        cover_half_saturation_npp_kg_c_m2_year,
        active_month_thermal_threshold,
        active_month_water_threshold,
        cold_pressure_reference_c,
        cold_pressure_release_c,
        heat_pressure_onset_c,
        heat_pressure_reference_c,
        minimum_biomass_residence_years,
        maximum_biomass_residence_years,
        biomass_residence_baseline_fraction,
        woody_biomass_residence_weight,
        resource_conservative_biomass_residence_weight,
        low_productivity_biomass_residence_weight,
        maximum_rooting_depth_m,
        maximum_canopy_height_m,
        maximum_leaf_area_index,
        maximum_standing_biomass_kg_c_m2,
    ];
    if controls.iter().any(|value| !value.is_finite())
        || energy_per_kg_carbon_mj <= 0.0
        || cover_half_saturation_npp_kg_c_m2_year <= 0.0
        || !(0.0..=1.0).contains(&active_month_thermal_threshold)
        || !(0.0..=1.0).contains(&active_month_water_threshold)
        || cold_pressure_reference_c >= cold_pressure_release_c
        || heat_pressure_onset_c >= heat_pressure_reference_c
        || minimum_biomass_residence_years < 0.0
        || maximum_biomass_residence_years < minimum_biomass_residence_years
        || !(0.0..=1.0).contains(&biomass_residence_baseline_fraction)
        || woody_biomass_residence_weight < 0.0
        || resource_conservative_biomass_residence_weight < 0.0
        || low_productivity_biomass_residence_weight < 0.0
        || woody_biomass_residence_weight
            + resource_conservative_biomass_residence_weight
            + low_productivity_biomass_residence_weight
            <= 0.0
        || maximum_rooting_depth_m <= 0.0
        || maximum_canopy_height_m <= 0.0
        || maximum_leaf_area_index <= 0.0
        || maximum_standing_biomass_kg_c_m2 <= 0.0
    {
        return 2;
    }

    let total = cell_count as usize;
    let monthly_len = match total.checked_mul(MONTHS) {
        Some(value) => value,
        None => return 1,
    };
    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let ocean = unsafe { slice::from_raw_parts(ocean_ptr, total) };
    let monthly_primary_energy =
        unsafe { slice::from_raw_parts(monthly_primary_energy_ptr, monthly_len) };
    let monthly_thermal_opportunity =
        unsafe { slice::from_raw_parts(monthly_thermal_opportunity_ptr, monthly_len) };
    let monthly_water_opportunity =
        unsafe { slice::from_raw_parts(monthly_water_opportunity_ptr, monthly_len) };
    let monthly_temperature =
        unsafe { slice::from_raw_parts(monthly_temperature_ptr, monthly_len) };
    let monthly_soil_saturation =
        unsafe { slice::from_raw_parts(monthly_soil_saturation_ptr, monthly_len) };
    let surface_support = unsafe { slice::from_raw_parts(surface_support_ptr, total) };
    let nutrient_support = unsafe { slice::from_raw_parts(nutrient_support_ptr, total) };
    let environmental_stress = unsafe { slice::from_raw_parts(environmental_stress_ptr, total) };
    let soil_depth = unsafe { slice::from_raw_parts(soil_depth_ptr, total) };
    let regolith_depth = unsafe { slice::from_raw_parts(regolith_depth_ptr, total) };
    let salinity = unsafe { slice::from_raw_parts(salinity_ptr, total) };
    let hydric_fraction = unsafe { slice::from_raw_parts(hydric_fraction_ptr, total) };
    let soil_confidence = unsafe { slice::from_raw_parts(soil_confidence_ptr, total) };
    let envelope_confidence = unsafe { slice::from_raw_parts(envelope_confidence_ptr, total) };
    let monthly_npp = unsafe { slice::from_raw_parts_mut(monthly_npp_ptr, monthly_len) };
    let annual_npp = unsafe { slice::from_raw_parts_mut(annual_npp_ptr, total) };
    let vegetation_cover = unsafe { slice::from_raw_parts_mut(vegetation_cover_ptr, total) };
    let standing_biomass = unsafe { slice::from_raw_parts_mut(standing_biomass_ptr, total) };
    let growing_season = unsafe { slice::from_raw_parts_mut(growing_season_ptr, total) };
    let productivity_seasonality =
        unsafe { slice::from_raw_parts_mut(productivity_seasonality_ptr, total) };
    let drought_pressure = unsafe { slice::from_raw_parts_mut(drought_pressure_ptr, total) };
    let cold_pressure = unsafe { slice::from_raw_parts_mut(cold_pressure_ptr, total) };
    let heat_pressure = unsafe { slice::from_raw_parts_mut(heat_pressure_ptr, total) };
    let waterlogging_pressure =
        unsafe { slice::from_raw_parts_mut(waterlogging_pressure_ptr, total) };
    let salinity_pressure = unsafe { slice::from_raw_parts_mut(salinity_pressure_ptr, total) };
    let woody_trait = unsafe { slice::from_raw_parts_mut(woody_trait_ptr, total) };
    let resource_conservative_trait =
        unsafe { slice::from_raw_parts_mut(resource_conservative_trait_ptr, total) };
    let rooting_depth = unsafe { slice::from_raw_parts_mut(rooting_depth_ptr, total) };
    let canopy_height = unsafe { slice::from_raw_parts_mut(canopy_height_ptr, total) };
    let leaf_area_index = unsafe { slice::from_raw_parts_mut(leaf_area_index_ptr, total) };
    let fuel_continuity = unsafe { slice::from_raw_parts_mut(fuel_continuity_ptr, total) };
    let confidence = unsafe { slice::from_raw_parts_mut(confidence_ptr, total) };

    if areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || ocean
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || monthly_primary_energy
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || monthly_thermal_opportunity
            .iter()
            .chain(monthly_water_opportunity.iter())
            .chain(monthly_soil_saturation.iter())
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || monthly_temperature.iter().any(|value| !value.is_finite())
        || surface_support
            .iter()
            .chain(nutrient_support.iter())
            .chain(environmental_stress.iter())
            .chain(salinity.iter())
            .chain(hydric_fraction.iter())
            .chain(soil_confidence.iter())
            .chain(envelope_confidence.iter())
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || soil_depth
            .iter()
            .chain(regolith_depth.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return 3;
    }

    let mut total_land_area = 0.0f64;
    let mut annual_npp_area = 0.0f64;
    let mut cover_area = 0.0f64;
    let mut biomass_area = 0.0f64;
    let mut vegetated_area = 0.0f64;
    let mut growing_area = 0.0f64;
    let mut woody_area = 0.0f64;
    let mut maximum_rooting = 0.0f64;
    let mut maximum_canopy = 0.0f64;

    for cell in 0..total {
        let is_land = ocean[cell] < 0.5;
        let mut cell_monthly_npp = [0.0f64; MONTHS];
        let mut thermal_sum = 0.0f64;
        let mut water_sum = 0.0f64;
        let mut drought_sum = 0.0f64;
        let mut cold_sum = 0.0f64;
        let mut heat_sum = 0.0f64;
        let mut saturation_excess_sum = 0.0f64;
        let mut active_months = 0usize;
        for month in 0..MONTHS {
            let index = month * total + cell;
            let npp = if is_land {
                f64::from(monthly_primary_energy[index]) / energy_per_kg_carbon_mj
            } else {
                0.0
            };
            monthly_npp[index] = npp as f32;
            cell_monthly_npp[month] = npp;
            if !is_land {
                continue;
            }
            let thermal = clamp01(f64::from(monthly_thermal_opportunity[index]));
            let water = clamp01(f64::from(monthly_water_opportunity[index]));
            let temperature = f64::from(monthly_temperature[index]);
            let saturation = clamp01(f64::from(monthly_soil_saturation[index]));
            thermal_sum += thermal;
            water_sum += water;
            drought_sum += thermal * (1.0 - water);
            cold_sum += clamp01(
                (cold_pressure_release_c - temperature)
                    / (cold_pressure_release_c - cold_pressure_reference_c),
            );
            heat_sum += clamp01(
                (temperature - heat_pressure_onset_c)
                    / (heat_pressure_reference_c - heat_pressure_onset_c),
            );
            saturation_excess_sum += clamp01((saturation - 0.75) / 0.25);
            if thermal >= active_month_thermal_threshold && water >= active_month_water_threshold {
                active_months += 1;
            }
        }

        if !is_land {
            annual_npp[cell] = 0.0;
            vegetation_cover[cell] = 0.0;
            standing_biomass[cell] = 0.0;
            growing_season[cell] = 0.0;
            productivity_seasonality[cell] = 0.0;
            drought_pressure[cell] = 0.0;
            cold_pressure[cell] = 0.0;
            heat_pressure[cell] = 0.0;
            waterlogging_pressure[cell] = 0.0;
            salinity_pressure[cell] = 0.0;
            woody_trait[cell] = 0.0;
            resource_conservative_trait[cell] = 0.0;
            rooting_depth[cell] = 0.0;
            canopy_height[cell] = 0.0;
            leaf_area_index[cell] = 0.0;
            fuel_continuity[cell] = 0.0;
            confidence[cell] = 0.0;
            continue;
        }

        let cell_annual_npp = cell_monthly_npp.iter().sum::<f64>();
        let surface = clamp01(f64::from(surface_support[cell]));
        let nutrient = clamp01(f64::from(nutrient_support[cell]));
        let stress = clamp01(f64::from(environmental_stress[cell]));
        let salinity_value = clamp01(f64::from(salinity[cell]));
        let productivity_response =
            1.0 - (-cell_annual_npp / cover_half_saturation_npp_kg_c_m2_year).exp();
        let cover = clamp01(surface * productivity_response);
        let growing = active_months as f64 / MONTHS as f64;
        let seasonality = normalized_seasonality(&cell_monthly_npp, cell_annual_npp);
        let mean_thermal = thermal_sum / MONTHS as f64;
        let mean_water = water_sum / MONTHS as f64;
        let drought = if thermal_sum > 1e-12 {
            clamp01(drought_sum / thermal_sum)
        } else {
            0.0
        };
        let cold = clamp01(cold_sum / MONTHS as f64);
        let heat = clamp01(heat_sum / MONTHS as f64);
        let waterlogging =
            clamp01(f64::from(hydric_fraction[cell]).max(saturation_excess_sum / MONTHS as f64));
        let conservative = if cover > 1e-8 {
            clamp01(0.34 * drought + 0.20 * cold + 0.18 * salinity_value + 0.28 * (1.0 - nutrient))
        } else {
            0.0
        };
        let woody = clamp01(
            cover
                * productivity_response.sqrt()
                * growing.sqrt()
                * (0.45 + 0.55 * mean_thermal)
                * (1.0 - 0.62 * drought)
                * (1.0 - 0.45 * waterlogging)
                * (1.0 - 0.60 * salinity_value)
                * (1.0 - 0.30 * seasonality),
        );
        let residence_score = biomass_residence_score(
            productivity_response,
            woody,
            conservative,
            biomass_residence_baseline_fraction,
            woody_biomass_residence_weight,
            resource_conservative_biomass_residence_weight,
            low_productivity_biomass_residence_weight,
        );
        let residence = minimum_biomass_residence_years
            + (maximum_biomass_residence_years - minimum_biomass_residence_years) * residence_score;
        let biomass = (cell_annual_npp * residence).min(maximum_standing_biomass_kg_c_m2);
        let regolith_limit = f64::from(regolith_depth[cell]).min(maximum_rooting_depth_m);
        let soil_limit = f64::from(soil_depth[cell]).min(regolith_limit);
        let deeper_access = clamp01(0.25 + 0.50 * woody + 0.25 * drought);
        let accessible_root_zone = soil_limit + (regolith_limit - soil_limit) * deeper_access;
        let root_fraction = clamp01(cover * (0.25 + 0.45 * drought + 0.30 * woody));
        let roots = accessible_root_zone * root_fraction;
        let canopy = maximum_canopy_height_m
            * woody.sqrt()
            * productivity_response.sqrt()
            * (0.45 + 0.55 * mean_water)
            * (1.0 - 0.45 * stress);
        let lai = maximum_leaf_area_index
            * cover
            * (0.35 + 0.65 * nutrient)
            * (0.55 + 0.45 * mean_water)
            * (1.0 - 0.45 * salinity_value);
        let fuel = clamp01(
            cover * productivity_response * (0.25 + 0.75 * drought) * (1.0 - 0.70 * waterlogging),
        );
        let trait_confidence = clamp01(
            (f64::from(soil_confidence[cell]) * f64::from(envelope_confidence[cell])).sqrt(),
        );

        annual_npp[cell] = cell_annual_npp as f32;
        vegetation_cover[cell] = cover as f32;
        standing_biomass[cell] = biomass as f32;
        growing_season[cell] = growing as f32;
        productivity_seasonality[cell] = seasonality as f32;
        drought_pressure[cell] = drought as f32;
        cold_pressure[cell] = cold as f32;
        heat_pressure[cell] = heat as f32;
        waterlogging_pressure[cell] = waterlogging as f32;
        salinity_pressure[cell] = salinity_value as f32;
        woody_trait[cell] = woody as f32;
        resource_conservative_trait[cell] = conservative as f32;
        rooting_depth[cell] = roots as f32;
        canopy_height[cell] = canopy.min(maximum_canopy_height_m) as f32;
        leaf_area_index[cell] = lai.min(maximum_leaf_area_index) as f32;
        fuel_continuity[cell] = fuel as f32;
        confidence[cell] = trait_confidence as f32;

        let area = areas[cell];
        total_land_area += area;
        annual_npp_area += area * cell_annual_npp;
        cover_area += area * cover;
        biomass_area += area * biomass;
        growing_area += area * growing;
        woody_area += area * woody;
        maximum_rooting = maximum_rooting.max(roots);
        maximum_canopy = maximum_canopy.max(canopy);
        if cover >= 0.10 {
            vegetated_area += area;
        }
    }

    unsafe {
        *stats_out = PotentialBiosphereStats {
            land_mean_annual_npp_kg_c_m2: (annual_npp_area / total_land_area.max(1e-12)) as f32,
            land_mean_vegetation_cover_fraction: (cover_area / total_land_area.max(1e-12)) as f32,
            land_mean_standing_biomass_kg_c_m2: (biomass_area / total_land_area.max(1e-12)) as f32,
            potentially_vegetated_land_area_fraction: (vegetated_area / total_land_area.max(1e-12))
                as f32,
            land_mean_growing_season_fraction: (growing_area / total_land_area.max(1e-12)) as f32,
            land_mean_woody_allocation_trait: (woody_area / total_land_area.max(1e-12)) as f32,
            maximum_rooting_depth_m: maximum_rooting as f32,
            maximum_canopy_height_m: maximum_canopy as f32,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{biomass_residence_score, normalized_seasonality, MONTHS};

    #[test]
    fn seasonality_distinguishes_uniform_and_single_month_production() {
        let uniform = [1.0; MONTHS];
        assert!(normalized_seasonality(&uniform, 12.0) < 1e-12);
        let mut pulse = [0.0; MONTHS];
        pulse[4] = 1.0;
        assert!((normalized_seasonality(&pulse, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn low_productivity_extends_residence_for_equal_structure() {
        let low_productivity = biomass_residence_score(0.1, 0.3, 0.5, 0.1, 0.6, 0.4, 2.5);
        let high_productivity = biomass_residence_score(0.9, 0.3, 0.5, 0.1, 0.6, 0.4, 2.5);
        assert!(low_productivity > high_productivity);
        assert!((0.0..=1.0).contains(&low_productivity));
        assert!((0.0..=1.0).contains(&high_productivity));
    }
}
