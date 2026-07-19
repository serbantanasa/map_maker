use std::slice;

const MONTHS: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BiosphereEnvelopeStats {
    pub land_area_fraction: f32,
    pub land_mean_annual_par_mj_m2: f32,
    pub land_mean_annual_primary_energy_mj_m2: f32,
    pub land_mean_thermal_opportunity: f32,
    pub land_mean_liquid_water_opportunity: f32,
    pub land_mean_carbon_substrate_relative: f32,
    pub land_mean_aerobic_oxygen_relative: f32,
    pub potentially_productive_land_area_fraction: f32,
}

#[no_mangle]
pub extern "C" fn biosphere_envelope_native_abi_version() -> u32 {
    3
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn nutrient_opportunity(raw_support: f64, half_saturation: f64) -> f64 {
    clamp01((1.0 + half_saturation) * raw_support / (raw_support + half_saturation))
}

fn thermal_opportunity(
    temperature_c: f64,
    minimum_c: f64,
    optimum_low_c: f64,
    optimum_high_c: f64,
    maximum_c: f64,
) -> f64 {
    if temperature_c <= minimum_c || temperature_c >= maximum_c {
        0.0
    } else if temperature_c < optimum_low_c {
        (temperature_c - minimum_c) / (optimum_low_c - minimum_c)
    } else if temperature_c <= optimum_high_c {
        1.0
    } else {
        (maximum_c - temperature_c) / (maximum_c - optimum_high_c)
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// Compute a provisional terrestrial biosphere resource envelope.
///
/// # Safety
/// Every pointer must reference the declared number of contiguous elements.
/// Input and output buffers must not overlap.
pub unsafe extern "C" fn biosphere_envelope_run(
    cell_count: i32,
    seconds_per_month: f64,
    par_fraction: f64,
    shortwave_transmission: f64,
    thermal_minimum_c: f64,
    thermal_optimum_low_c: f64,
    thermal_optimum_high_c: f64,
    thermal_maximum_c: f64,
    water_input_half_saturation_mm: f64,
    nutrient_half_saturation_index: f64,
    co2_half_saturation_pa: f64,
    reference_co2_partial_pressure_pa: f64,
    reference_oxygen_partial_pressure_kpa: f64,
    photosynthetic_conversion_efficiency: f64,
    minimum_productive_energy_mj_m2_year: f64,
    confidence_multiplier: f64,
    area_ptr: *const f64,
    ocean_ptr: *const f32,
    monthly_insolation_ptr: *const f32,
    monthly_temperature_ptr: *const f32,
    monthly_liquid_input_ptr: *const f32,
    monthly_soil_saturation_ptr: *const f32,
    soil_bearing_ptr: *const f32,
    nutrient_potential_ptr: *const f32,
    fertility_potential_ptr: *const f32,
    salinity_ptr: *const f32,
    soil_confidence_ptr: *const f32,
    co2_partial_pressure_ptr: *const f32,
    oxygen_partial_pressure_ptr: *const f32,
    monthly_par_ptr: *mut f32,
    monthly_liquid_opportunity_ptr: *mut f32,
    monthly_thermal_opportunity_ptr: *mut f32,
    monthly_primary_energy_ptr: *mut f32,
    annual_par_ptr: *mut f32,
    annual_primary_energy_ptr: *mut f32,
    carbon_substrate_relative_ptr: *mut f32,
    aerobic_oxygen_relative_ptr: *mut f32,
    terrestrial_surface_support_ptr: *mut f32,
    nutrient_support_ptr: *mut f32,
    environmental_stress_ptr: *mut f32,
    confidence_ptr: *mut f32,
    stats_out: *mut BiosphereEnvelopeStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || ocean_ptr.is_null()
        || monthly_insolation_ptr.is_null()
        || monthly_temperature_ptr.is_null()
        || monthly_liquid_input_ptr.is_null()
        || monthly_soil_saturation_ptr.is_null()
        || soil_bearing_ptr.is_null()
        || nutrient_potential_ptr.is_null()
        || fertility_potential_ptr.is_null()
        || salinity_ptr.is_null()
        || soil_confidence_ptr.is_null()
        || co2_partial_pressure_ptr.is_null()
        || oxygen_partial_pressure_ptr.is_null()
        || monthly_par_ptr.is_null()
        || monthly_liquid_opportunity_ptr.is_null()
        || monthly_thermal_opportunity_ptr.is_null()
        || monthly_primary_energy_ptr.is_null()
        || annual_par_ptr.is_null()
        || annual_primary_energy_ptr.is_null()
        || carbon_substrate_relative_ptr.is_null()
        || aerobic_oxygen_relative_ptr.is_null()
        || terrestrial_surface_support_ptr.is_null()
        || nutrient_support_ptr.is_null()
        || environmental_stress_ptr.is_null()
        || confidence_ptr.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let controls = [
        seconds_per_month,
        par_fraction,
        shortwave_transmission,
        thermal_minimum_c,
        thermal_optimum_low_c,
        thermal_optimum_high_c,
        thermal_maximum_c,
        water_input_half_saturation_mm,
        nutrient_half_saturation_index,
        co2_half_saturation_pa,
        reference_co2_partial_pressure_pa,
        reference_oxygen_partial_pressure_kpa,
        photosynthetic_conversion_efficiency,
        minimum_productive_energy_mj_m2_year,
        confidence_multiplier,
    ];
    if controls.iter().any(|value| !value.is_finite())
        || seconds_per_month <= 0.0
        || !(0.0..=1.0).contains(&par_fraction)
        || !(0.0..=1.0).contains(&shortwave_transmission)
        || !(thermal_minimum_c < thermal_optimum_low_c
            && thermal_optimum_low_c <= thermal_optimum_high_c
            && thermal_optimum_high_c < thermal_maximum_c)
        || water_input_half_saturation_mm <= 0.0
        || nutrient_half_saturation_index <= 0.0
        || co2_half_saturation_pa <= 0.0
        || reference_co2_partial_pressure_pa <= 0.0
        || reference_oxygen_partial_pressure_kpa <= 0.0
        || !(0.0..=1.0).contains(&photosynthetic_conversion_efficiency)
        || minimum_productive_energy_mj_m2_year < 0.0
        || !(0.0..=1.0).contains(&confidence_multiplier)
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
    let monthly_insolation = unsafe { slice::from_raw_parts(monthly_insolation_ptr, monthly_len) };
    let monthly_temperature =
        unsafe { slice::from_raw_parts(monthly_temperature_ptr, monthly_len) };
    let monthly_liquid_input =
        unsafe { slice::from_raw_parts(monthly_liquid_input_ptr, monthly_len) };
    let monthly_soil_saturation =
        unsafe { slice::from_raw_parts(monthly_soil_saturation_ptr, monthly_len) };
    let soil_bearing = unsafe { slice::from_raw_parts(soil_bearing_ptr, total) };
    let nutrient_potential = unsafe { slice::from_raw_parts(nutrient_potential_ptr, total) };
    let fertility_potential = unsafe { slice::from_raw_parts(fertility_potential_ptr, total) };
    let salinity = unsafe { slice::from_raw_parts(salinity_ptr, total) };
    let soil_confidence = unsafe { slice::from_raw_parts(soil_confidence_ptr, total) };
    let co2_partial_pressure = unsafe { slice::from_raw_parts(co2_partial_pressure_ptr, total) };
    let oxygen_partial_pressure =
        unsafe { slice::from_raw_parts(oxygen_partial_pressure_ptr, total) };
    let monthly_par = unsafe { slice::from_raw_parts_mut(monthly_par_ptr, monthly_len) };
    let monthly_liquid_opportunity =
        unsafe { slice::from_raw_parts_mut(monthly_liquid_opportunity_ptr, monthly_len) };
    let monthly_thermal_opportunity =
        unsafe { slice::from_raw_parts_mut(monthly_thermal_opportunity_ptr, monthly_len) };
    let monthly_primary_energy =
        unsafe { slice::from_raw_parts_mut(monthly_primary_energy_ptr, monthly_len) };
    let annual_par = unsafe { slice::from_raw_parts_mut(annual_par_ptr, total) };
    let annual_primary_energy =
        unsafe { slice::from_raw_parts_mut(annual_primary_energy_ptr, total) };
    let carbon_substrate_relative =
        unsafe { slice::from_raw_parts_mut(carbon_substrate_relative_ptr, total) };
    let aerobic_oxygen_relative =
        unsafe { slice::from_raw_parts_mut(aerobic_oxygen_relative_ptr, total) };
    let terrestrial_surface_support =
        unsafe { slice::from_raw_parts_mut(terrestrial_surface_support_ptr, total) };
    let nutrient_support = unsafe { slice::from_raw_parts_mut(nutrient_support_ptr, total) };
    let environmental_stress =
        unsafe { slice::from_raw_parts_mut(environmental_stress_ptr, total) };
    let confidence = unsafe { slice::from_raw_parts_mut(confidence_ptr, total) };

    if areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || ocean
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || monthly_insolation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || monthly_temperature.iter().any(|value| !value.is_finite())
        || monthly_liquid_input
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || monthly_soil_saturation
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || soil_bearing
            .iter()
            .chain(nutrient_potential.iter())
            .chain(fertility_potential.iter())
            .chain(salinity.iter())
            .chain(soil_confidence.iter())
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || co2_partial_pressure
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || oxygen_partial_pressure
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return 3;
    }

    let reference_carbon_limitation = reference_co2_partial_pressure_pa
        / (reference_co2_partial_pressure_pa + co2_half_saturation_pa);
    let mut total_area = 0.0f64;
    let mut land_area = 0.0f64;
    let mut productive_area = 0.0f64;
    let mut annual_par_area = 0.0f64;
    let mut annual_primary_area = 0.0f64;
    let mut thermal_area_months = 0.0f64;
    let mut water_area_months = 0.0f64;
    let mut carbon_area = 0.0f64;
    let mut oxygen_area = 0.0f64;

    for cell in 0..total {
        let area = areas[cell];
        total_area += area;
        let is_land = ocean[cell] < 0.5;
        let surface = if is_land {
            clamp01(f64::from(soil_bearing[cell]))
        } else {
            0.0
        };
        let nutrient = if is_land {
            let base = 0.5 * f64::from(nutrient_potential[cell])
                + 0.5 * f64::from(fertility_potential[cell]);
            let raw_support = clamp01(base / (1.0 + 2.0 * f64::from(salinity[cell])));
            nutrient_opportunity(raw_support, nutrient_half_saturation_index)
        } else {
            0.0
        };
        let co2_pa = f64::from(co2_partial_pressure[cell]);
        let carbon_limitation = co2_pa / (co2_pa + co2_half_saturation_pa);
        let carbon_relative = carbon_limitation / reference_carbon_limitation;
        let oxygen_relative =
            f64::from(oxygen_partial_pressure[cell]) / reference_oxygen_partial_pressure_kpa;
        carbon_substrate_relative[cell] = carbon_relative as f32;
        aerobic_oxygen_relative[cell] = oxygen_relative as f32;
        terrestrial_surface_support[cell] = surface as f32;
        nutrient_support[cell] = nutrient as f32;

        let mut cell_annual_par = 0.0f64;
        let mut cell_annual_primary = 0.0f64;
        let mut cell_thermal = 0.0f64;
        let mut cell_water = 0.0f64;
        for month in 0..MONTHS {
            let index = month * total + cell;
            let par = f64::from(monthly_insolation[index]) * seconds_per_month / 1.0e6
                * par_fraction
                * shortwave_transmission;
            let thermal = thermal_opportunity(
                f64::from(monthly_temperature[index]),
                thermal_minimum_c,
                thermal_optimum_low_c,
                thermal_optimum_high_c,
                thermal_maximum_c,
            );
            let saturation = clamp01(f64::from(monthly_soil_saturation[index]));
            let liquid_input = f64::from(monthly_liquid_input[index]);
            let input_response = liquid_input / (liquid_input + water_input_half_saturation_mm);
            let water = if is_land {
                clamp01(0.65 * saturation.sqrt() + 0.35 * input_response)
            } else {
                0.0
            };
            let primary = par
                * photosynthetic_conversion_efficiency
                * thermal
                * water
                * surface
                * nutrient
                * carbon_limitation;
            monthly_par[index] = par as f32;
            monthly_liquid_opportunity[index] = water as f32;
            monthly_thermal_opportunity[index] = thermal as f32;
            monthly_primary_energy[index] = primary as f32;
            cell_annual_par += par;
            cell_annual_primary += primary;
            cell_thermal += thermal;
            cell_water += water;
        }
        annual_par[cell] = cell_annual_par as f32;
        annual_primary_energy[cell] = cell_annual_primary as f32;
        let possible_primary =
            cell_annual_par * photosynthetic_conversion_efficiency * surface * nutrient;
        environmental_stress[cell] = if is_land {
            (1.0 - cell_annual_primary / possible_primary.max(1.0e-12)).clamp(0.0, 1.0) as f32
        } else {
            1.0
        };
        confidence[cell] = if is_land {
            (clamp01(f64::from(soil_confidence[cell])) * confidence_multiplier) as f32
        } else {
            0.0
        };

        if is_land {
            land_area += area;
            annual_par_area += area * cell_annual_par;
            annual_primary_area += area * cell_annual_primary;
            thermal_area_months += area * cell_thermal;
            water_area_months += area * cell_water;
            carbon_area += area * carbon_relative;
            oxygen_area += area * oxygen_relative;
            if cell_annual_primary >= minimum_productive_energy_mj_m2_year {
                productive_area += area;
            }
        }
    }

    unsafe {
        *stats_out = BiosphereEnvelopeStats {
            land_area_fraction: (land_area / total_area.max(1.0e-12)) as f32,
            land_mean_annual_par_mj_m2: (annual_par_area / land_area.max(1.0e-12)) as f32,
            land_mean_annual_primary_energy_mj_m2: (annual_primary_area / land_area.max(1.0e-12))
                as f32,
            land_mean_thermal_opportunity: (thermal_area_months
                / (land_area * MONTHS as f64).max(1.0e-12))
                as f32,
            land_mean_liquid_water_opportunity: (water_area_months
                / (land_area * MONTHS as f64).max(1.0e-12))
                as f32,
            land_mean_carbon_substrate_relative: (carbon_area / land_area.max(1.0e-12)) as f32,
            land_mean_aerobic_oxygen_relative: (oxygen_area / land_area.max(1.0e-12)) as f32,
            potentially_productive_land_area_fraction: (productive_area / land_area.max(1.0e-12))
                as f32,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{nutrient_opportunity, thermal_opportunity};

    #[test]
    fn thermal_response_is_bounded_and_trapezoidal() {
        assert_eq!(thermal_opportunity(-10.0, -10.0, 15.0, 30.0, 50.0), 0.0);
        assert_eq!(thermal_opportunity(15.0, -10.0, 15.0, 30.0, 50.0), 1.0);
        assert_eq!(thermal_opportunity(25.0, -10.0, 15.0, 30.0, 50.0), 1.0);
        assert_eq!(thermal_opportunity(50.0, -10.0, 15.0, 30.0, 50.0), 0.0);
        assert!((thermal_opportunity(40.0, -10.0, 15.0, 30.0, 50.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn nutrient_response_is_saturating_and_normalized() {
        assert_eq!(nutrient_opportunity(0.0, 0.5), 0.0);
        assert_eq!(nutrient_opportunity(1.0, 0.5), 1.0);
        assert!(nutrient_opportunity(0.2, 0.5) > 0.2);
        assert!(nutrient_opportunity(0.8, 0.5) < 1.0);
    }
}
