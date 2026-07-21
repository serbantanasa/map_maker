use std::slice;

const MONTHS: usize = 12;
const NEIGHBORS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CryosphereStats {
    pub seasonal_snow_land_area_fraction: f32,
    pub perennial_snow_land_area_fraction: f32,
    pub glacierized_land_area_fraction: f32,
    pub glacier_ice_land_area_fraction: f32,
    pub maximum_glacier_ice_water_equivalent_mm: f32,
    pub land_mean_annual_glacier_melt_mm: f32,
    pub land_mean_annual_glacier_calving_mm: f32,
}

#[no_mangle]
pub extern "C" fn cryosphere_native_abi_version() -> u32 {
    2
}

struct Inputs<'a> {
    areas: &'a [f64],
    neighbors: &'a [i32],
    ocean: &'a [f32],
    elevation: &'a [f32],
    relief: &'a [f32],
    temperature: &'a [f32],
    precipitation: &'a [f32],
    evaporation: &'a [f32],
}

struct Outputs<'a> {
    snowfall: &'a mut [f32],
    snowmelt: &'a mut [f32],
    snowpack: &'a mut [f32],
    firn_to_ice: &'a mut [f32],
    glacier_melt: &'a mut [f32],
    glacier_ice: &'a mut [f32],
    runoff: &'a mut [f32],
    annual_mass_balance: &'a mut [f32],
    annual_flow_export: &'a mut [f32],
    annual_flow_import: &'a mut [f32],
    annual_calving: &'a mut [f32],
    annual_sublimation: &'a mut [f32],
    glacier_fraction: &'a mut [f32],
}

#[derive(Clone, Copy)]
struct Controls {
    spinup_years: usize,
    lapse_rate_c_per_km: f64,
    relief_elevation_multiplier: f64,
    maximum_highland_fraction: f64,
    snow_degree_day_melt_mm_c_month: f64,
    glacier_degree_day_melt_mm_c_month: f64,
    firn_conversion_fraction_year: f64,
    snow_sublimation_fraction_month: f64,
    glacier_sublimation_fraction_month: f64,
    glacier_flow_activation_mm: f64,
    glacier_flow_fraction_year: f64,
    glacier_reference_thickness_mm: f64,
    runoff_base_fraction: f64,
}

fn snow_fraction(temperature_c: f64) -> f64 {
    ((2.0 - temperature_c) / 5.0).clamp(0.0, 1.0)
}

fn remove_snow(snow: &mut f64, snow_age_mass: &mut f64, requested_mm: f64) -> f64 {
    let removed = snow.min(requested_mm.max(0.0));
    if *snow > 0.0 {
        let remaining_fraction = ((*snow - removed) / *snow).clamp(0.0, 1.0);
        *snow_age_mass *= remaining_fraction;
    }
    *snow -= removed;
    if *snow <= 1e-12 {
        *snow = 0.0;
        *snow_age_mass = 0.0;
    }
    removed
}

fn update_snow_reservoir(
    snow: &mut f64,
    snow_age_mass: &mut f64,
    fallen_mm: f64,
    melt_capacity_mm: f64,
    monthly_firn_fraction: f64,
    sublimation_fraction: f64,
) -> (f64, f64) {
    *snow_age_mass += *snow;
    *snow += fallen_mm;
    let mean_age_months = *snow_age_mass / snow.max(1e-12);
    let mature_fraction = ((mean_age_months - 9.0) / 6.0).clamp(0.0, 1.0);
    let requested_conversion = *snow * mature_fraction * monthly_firn_fraction;
    let converted = remove_snow(snow, snow_age_mass, requested_conversion);
    let melted = remove_snow(snow, snow_age_mass, melt_capacity_mm);
    let requested_sublimation = *snow * sublimation_fraction;
    remove_snow(snow, snow_age_mass, requested_sublimation);
    (converted, melted)
}

#[allow(clippy::too_many_arguments)]
fn route_glacier_ice(
    controls: Controls,
    inputs: &Inputs<'_>,
    ice: &mut [f64],
    flow_export: &mut [f64],
    flow_import: &mut [f64],
    calving: &mut [f64],
) {
    let total = ice.len();
    let monthly_flow_fraction =
        1.0 - (1.0 - controls.glacier_flow_fraction_year).powf(1.0 / MONTHS as f64);
    let mut delta = vec![0.0f64; total];
    for cell in 0..total {
        if inputs.ocean[cell] >= 0.5 || ice[cell] <= controls.glacier_flow_activation_mm {
            continue;
        }
        let source_height = f64::from(inputs.elevation[cell]);
        let mut target = cell;
        let mut target_height = source_height;
        for edge in 0..NEIGHBORS {
            let neighbor = inputs.neighbors[cell * NEIGHBORS + edge] as usize;
            let height = if inputs.ocean[neighbor] >= 0.5 {
                -1.0
            } else {
                f64::from(inputs.elevation[neighbor])
            };
            if height < target_height {
                target = neighbor;
                target_height = height;
            }
        }
        if target == cell {
            continue;
        }
        let drop_m = (source_height - target_height).max(0.0);
        // Require real slope for ice motion; suppress thin-sheet drainage that
        // empties polar ice caps into the ocean during spinup.
        if drop_m < 40.0 {
            continue;
        }
        let slope_factor = ((drop_m - 40.0) / 1_200.0).clamp(0.02, 1.0);
        let transfer = (ice[cell] - controls.glacier_flow_activation_mm)
            * monthly_flow_fraction
            * slope_factor;
        if transfer <= 0.0 {
            continue;
        }
        delta[cell] -= transfer;
        flow_export[cell] += transfer;
        if inputs.ocean[target] >= 0.5 {
            calving[cell] += transfer;
        } else {
            let received = transfer * inputs.areas[cell] / inputs.areas[target];
            delta[target] += received;
            flow_import[target] += received;
        }
    }
    for (storage, change) in ice.iter_mut().zip(delta) {
        *storage = (*storage + change).max(0.0);
    }
}

fn run_model(
    controls: Controls,
    inputs: &Inputs<'_>,
    outputs: &mut Outputs<'_>,
) -> CryosphereStats {
    let total = inputs.areas.len();
    outputs.snowfall.fill(0.0);
    outputs.snowmelt.fill(0.0);
    outputs.snowpack.fill(0.0);
    outputs.firn_to_ice.fill(0.0);
    outputs.glacier_melt.fill(0.0);
    outputs.glacier_ice.fill(0.0);
    outputs.runoff.fill(0.0);
    outputs.annual_mass_balance.fill(0.0);
    outputs.annual_flow_export.fill(0.0);
    outputs.annual_flow_import.fill(0.0);
    outputs.annual_calving.fill(0.0);
    outputs.annual_sublimation.fill(0.0);
    outputs.glacier_fraction.fill(0.0);

    let mut lowland_snow = vec![0.0f64; total];
    let mut lowland_snow_age_mass = vec![0.0f64; total];
    let mut highland_snow = vec![0.0f64; total];
    let mut highland_snow_age_mass = vec![0.0f64; total];
    let mut ice = vec![0.0f64; total];
    let mut final_start_ice = vec![0.0f64; total];
    let monthly_firn_fraction =
        1.0 - (1.0 - controls.firn_conversion_fraction_year).powf(1.0 / MONTHS as f64);

    // Annual climate for ice caps (must not flicker with single warm months).
    let mut annual_mean_temperature = vec![0.0f64; total];
    let mut coldest_month_temperature = vec![f64::INFINITY; total];
    let mut annual_precipitation = vec![0.0f64; total];
    for cell in 0..total {
        if inputs.ocean[cell] >= 0.5 {
            continue;
        }
        let mut sum_t = 0.0f64;
        let mut coldest = f64::INFINITY;
        let mut sum_p = 0.0f64;
        for month in 0..MONTHS {
            let offset = month * total + cell;
            let temperature = f64::from(inputs.temperature[offset]);
            sum_t += temperature;
            coldest = coldest.min(temperature);
            sum_p += f64::from(inputs.precipitation[offset]);
        }
        annual_mean_temperature[cell] = sum_t / MONTHS as f64;
        coldest_month_temperature[cell] = coldest;
        annual_precipitation[cell] = sum_p;
    }

    for year in 0..controls.spinup_years {
        let persist = year + 1 == controls.spinup_years;
        if persist {
            final_start_ice.copy_from_slice(&ice);
        }
        let mut flow_export = vec![0.0f64; total];
        let mut flow_import = vec![0.0f64; total];
        let mut calving = vec![0.0f64; total];
        for month in 0..MONTHS {
            for cell in 0..total {
                let offset = month * total + cell;
                if inputs.ocean[cell] >= 0.5 {
                    lowland_snow[cell] = 0.0;
                    lowland_snow_age_mass[cell] = 0.0;
                    highland_snow[cell] = 0.0;
                    highland_snow_age_mass[cell] = 0.0;
                    ice[cell] = 0.0;
                    continue;
                }
                let temperature_c = f64::from(inputs.temperature[offset]);
                let relief_m = f64::from(inputs.relief[cell]);
                // Climate temperature already reflects cell-mean elevation.
                // Mountain glaciers: peak-cooled relief fraction.
                // Polar / near-permanent ice caps: cold cell-mean climate can
                // ice over whole tiles without multi-km mean altitude.
                let surface_elev_m = f64::from(inputs.elevation[cell]).max(0.0);
                let mountain_fraction = (relief_m / 1_400.0 + surface_elev_m / 4_500.0)
                    .clamp(0.0, controls.maximum_highland_fraction);
                // Ice-cap potential from annual climate (routing only — no free water).
                let mean_t = annual_mean_temperature[cell];
                let coldest_t = coldest_month_temperature[cell];
                let mean_cap = ((-2.0 - mean_t) / 14.0).clamp(0.0, 1.0);
                let winter_cap = ((-12.0 - coldest_t) / 25.0).clamp(0.0, 1.0);
                // Dry polar land can still host ice if precipitation exists;
                // do not invent precip — only scale how much of real snow stays.
                let precip_support = (annual_precipitation[cell] / 250.0).clamp(0.0, 1.0);
                let ice_cap_fraction =
                    (0.55 * mean_cap + 0.45 * winter_cap).min(1.0) * precip_support.sqrt();
                let accumulation_fraction = mountain_fraction
                    .max(ice_cap_fraction * 0.95)
                    .clamp(0.0, 0.98);
                let peak_cooling_km = controls.relief_elevation_multiplier * relief_m / 1_000.0;
                let highland_temperature =
                    temperature_c - controls.lapse_rate_c_per_km * peak_cooling_km;
                let melt_temperature = if ice_cap_fraction > mountain_fraction {
                    temperature_c
                } else {
                    highland_temperature
                };
                // Conserved precipitation only (no synthetic floor).
                let precipitation = f64::from(inputs.precipitation[offset]).max(0.0);
                let lowland_share = (1.0 - accumulation_fraction).max(0.0);
                let snow_temp = melt_temperature.min(temperature_c);
                let lowland_fallen = precipitation * lowland_share * snow_fraction(temperature_c);
                let highland_snowfall =
                    precipitation * accumulation_fraction * snow_fraction(snow_temp);
                // Partition real highland snowfall: cold climates convert a share
                // of new snow directly to ice (still sourced from precipitation).
                let direct_ice_fraction = ice_cap_fraction.max(mountain_fraction * 0.35)
                    * ((-1.0 - snow_temp.min(mean_t)) / 12.0).clamp(0.0, 0.65);
                let direct_ice = highland_snowfall * direct_ice_fraction;
                let highland_fallen = highland_snowfall - direct_ice;
                // Faster firn conversion of stored snow in cold climates (no new water).
                let cold_firn_boost = 1.0 + 1.6 * ((-2.0 - mean_t) / 16.0).clamp(0.0, 1.0);
                let effective_firn_fraction =
                    (monthly_firn_fraction * cold_firn_boost).clamp(0.0, 0.55);
                let (lowland_converted, lowland_melt) = update_snow_reservoir(
                    &mut lowland_snow[cell],
                    &mut lowland_snow_age_mass[cell],
                    lowland_fallen,
                    lowland_share
                        * temperature_c.max(0.0)
                        * controls.snow_degree_day_melt_mm_c_month,
                    effective_firn_fraction,
                    controls.snow_sublimation_fraction_month,
                );
                let (highland_converted, highland_melt) = update_snow_reservoir(
                    &mut highland_snow[cell],
                    &mut highland_snow_age_mass[cell],
                    highland_fallen,
                    accumulation_fraction
                        * melt_temperature.max(0.0)
                        * controls.snow_degree_day_melt_mm_c_month
                        * if mean_t < -5.0 { 0.20 } else { 1.0 },
                    effective_firn_fraction,
                    controls.snow_sublimation_fraction_month,
                );
                let fallen = lowland_fallen + highland_snowfall;
                let converted = lowland_converted + highland_converted + direct_ice;
                let snow_melt = lowland_melt + highland_melt;
                ice[cell] += converted;

                let ice_fraction =
                    (ice[cell] / controls.glacier_reference_thickness_mm).clamp(0.0, 1.0);
                let exposed_ice_fraction = accumulation_fraction.max(ice_fraction);
                // Permanent ice climates melt little; water still leaves only via
                // melt/sublimation/calving of existing ice (no free source).
                let melt_degree = if mean_t <= -6.0 {
                    0.0
                } else if mean_t < 0.0 {
                    melt_temperature.max(0.0)
                        * ((mean_t + 6.0) / 6.0).clamp(0.0, 1.0)
                        * (1.0 - 0.75 * ice_cap_fraction)
                } else {
                    melt_temperature.max(0.0) * (1.0 - 0.55 * ice_cap_fraction)
                };
                let glacier_melt = ice[cell].min(
                    exposed_ice_fraction
                        * melt_degree
                        * controls.glacier_degree_day_melt_mm_c_month,
                );
                ice[cell] -= glacier_melt;
                let glacier_sublimation = ice[cell]
                    * controls.glacier_sublimation_fraction_month
                    * (1.0 - 0.5 * ice_cap_fraction);
                ice[cell] -= glacier_sublimation;

                if persist {
                    outputs.snowfall[offset] = fallen as f32;
                    outputs.snowmelt[offset] = snow_melt as f32;
                    outputs.firn_to_ice[offset] = converted as f32;
                    outputs.glacier_melt[offset] = glacier_melt as f32;
                    outputs.annual_sublimation[cell] += glacier_sublimation as f32;
                }
            }

            route_glacier_ice(
                controls,
                inputs,
                &mut ice,
                &mut flow_export,
                &mut flow_import,
                &mut calving,
            );
            if persist {
                for cell in 0..total {
                    let offset = month * total + cell;
                    outputs.snowpack[offset] = (lowland_snow[cell] + highland_snow[cell]) as f32;
                    outputs.glacier_ice[offset] = ice[cell] as f32;
                }
            }
        }
        if persist {
            for cell in 0..total {
                outputs.annual_mass_balance[cell] = (ice[cell] - final_start_ice[cell]) as f32;
                outputs.annual_flow_export[cell] = flow_export[cell] as f32;
                outputs.annual_flow_import[cell] = flow_import[cell] as f32;
                outputs.annual_calving[cell] = calving[cell] as f32;
                outputs.glacier_fraction[cell] =
                    (ice[cell] / controls.glacier_reference_thickness_mm).clamp(0.0, 1.0) as f32;
            }
        }
    }

    for month in 0..MONTHS {
        for cell in 0..total {
            let offset = month * total + cell;
            if inputs.ocean[cell] >= 0.5 {
                continue;
            }
            let rain = (f64::from(inputs.precipitation[offset])
                - f64::from(outputs.snowfall[offset]))
            .max(0.0);
            let available = rain
                + f64::from(outputs.snowmelt[offset])
                + f64::from(outputs.glacier_melt[offset]);
            let effective_evaporation = f64::from(inputs.evaporation[offset]).min(0.85 * available);
            let relief_factor = (f64::from(inputs.relief[cell]) / 1_500.0).clamp(0.0, 1.0);
            let storm_factor = (f64::from(inputs.precipitation[offset]) / 200.0).clamp(0.0, 1.0);
            let runoff_fraction =
                (controls.runoff_base_fraction + 0.25 * relief_factor + 0.15 * storm_factor)
                    .clamp(0.0, 0.95);
            outputs.runoff[offset] =
                ((available - effective_evaporation).max(0.0) * runoff_fraction) as f32;
        }
    }

    let mut land_area = 0.0;
    let mut seasonal_snow_area = 0.0;
    let mut perennial_snow_area = 0.0;
    let mut glacierized_area = 0.0;
    let mut glacier_fraction_area = 0.0;
    let mut glacier_melt_area = 0.0;
    let mut calving_area = 0.0;
    let mut maximum_ice = 0.0f32;
    for cell in 0..total {
        if inputs.ocean[cell] >= 0.5 {
            continue;
        }
        let area = inputs.areas[cell];
        land_area += area;
        let mut maximum_snow = 0.0f32;
        let mut minimum_snow = f32::INFINITY;
        let mut annual_melt = 0.0f32;
        for month in 0..MONTHS {
            let offset = month * total + cell;
            maximum_snow = maximum_snow.max(outputs.snowpack[offset]);
            minimum_snow = minimum_snow.min(outputs.snowpack[offset]);
            annual_melt += outputs.glacier_melt[offset];
            maximum_ice = maximum_ice.max(outputs.glacier_ice[offset]);
        }
        seasonal_snow_area += area * if maximum_snow >= 10.0 { 1.0 } else { 0.0 };
        perennial_snow_area += area * if minimum_snow >= 10.0 { 1.0 } else { 0.0 };
        glacierized_area += area
            * if outputs.glacier_fraction[cell] >= 0.01 {
                1.0
            } else {
                0.0
            };
        glacier_fraction_area += area * f64::from(outputs.glacier_fraction[cell]);
        glacier_melt_area += area * f64::from(annual_melt);
        calving_area += area * f64::from(outputs.annual_calving[cell]);
    }
    CryosphereStats {
        seasonal_snow_land_area_fraction: (seasonal_snow_area / land_area) as f32,
        perennial_snow_land_area_fraction: (perennial_snow_area / land_area) as f32,
        glacierized_land_area_fraction: (glacierized_area / land_area) as f32,
        glacier_ice_land_area_fraction: (glacier_fraction_area / land_area) as f32,
        maximum_glacier_ice_water_equivalent_mm: maximum_ice,
        land_mean_annual_glacier_melt_mm: (glacier_melt_area / land_area) as f32,
        land_mean_annual_glacier_calving_mm: (calving_area / land_area) as f32,
    }
}

/// Run the bounded V1 seasonal snow, firn, and glacier-reservoir model.
///
/// Returns 0 on success, 1 for invalid dimensions or pointers, 2 for invalid
/// controls, 3 for invalid numeric inputs, and 4 for invalid topology.
///
/// # Safety
///
/// Every pointer must reference the documented number of values. Output
/// buffers must be writable, disjoint, and may not alias any input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cryosphere_run(
    cell_count: i32,
    spinup_years: i32,
    lapse_rate_c_per_km: f64,
    relief_elevation_multiplier: f64,
    maximum_highland_fraction: f64,
    snow_degree_day_melt_mm_c_month: f64,
    glacier_degree_day_melt_mm_c_month: f64,
    firn_conversion_fraction_year: f64,
    snow_sublimation_fraction_month: f64,
    glacier_sublimation_fraction_month: f64,
    glacier_flow_activation_mm: f64,
    glacier_flow_fraction_year: f64,
    glacier_reference_thickness_mm: f64,
    runoff_base_fraction: f64,
    area_ptr: *const f64,
    neighbor_ptr: *const i32,
    ocean_ptr: *const f32,
    elevation_ptr: *const f32,
    relief_ptr: *const f32,
    temperature_ptr: *const f32,
    precipitation_ptr: *const f32,
    evaporation_ptr: *const f32,
    snowfall_ptr: *mut f32,
    snowmelt_ptr: *mut f32,
    snowpack_ptr: *mut f32,
    firn_to_ice_ptr: *mut f32,
    glacier_melt_ptr: *mut f32,
    glacier_ice_ptr: *mut f32,
    runoff_ptr: *mut f32,
    annual_mass_balance_ptr: *mut f32,
    annual_flow_export_ptr: *mut f32,
    annual_flow_import_ptr: *mut f32,
    annual_calving_ptr: *mut f32,
    annual_sublimation_ptr: *mut f32,
    glacier_fraction_ptr: *mut f32,
    stats_out: *mut CryosphereStats,
) -> i32 {
    let pointers_valid = !area_ptr.is_null()
        && !neighbor_ptr.is_null()
        && !ocean_ptr.is_null()
        && !elevation_ptr.is_null()
        && !relief_ptr.is_null()
        && !temperature_ptr.is_null()
        && !precipitation_ptr.is_null()
        && !evaporation_ptr.is_null()
        && !snowfall_ptr.is_null()
        && !snowmelt_ptr.is_null()
        && !snowpack_ptr.is_null()
        && !firn_to_ice_ptr.is_null()
        && !glacier_melt_ptr.is_null()
        && !glacier_ice_ptr.is_null()
        && !runoff_ptr.is_null()
        && !annual_mass_balance_ptr.is_null()
        && !annual_flow_export_ptr.is_null()
        && !annual_flow_import_ptr.is_null()
        && !annual_calving_ptr.is_null()
        && !annual_sublimation_ptr.is_null()
        && !glacier_fraction_ptr.is_null();
    if cell_count <= 0 || !pointers_valid {
        return 1;
    }
    let controls = Controls {
        spinup_years: spinup_years as usize,
        lapse_rate_c_per_km,
        relief_elevation_multiplier,
        maximum_highland_fraction,
        snow_degree_day_melt_mm_c_month,
        glacier_degree_day_melt_mm_c_month,
        firn_conversion_fraction_year,
        snow_sublimation_fraction_month,
        glacier_sublimation_fraction_month,
        glacier_flow_activation_mm,
        glacier_flow_fraction_year,
        glacier_reference_thickness_mm,
        runoff_base_fraction,
    };
    let numeric_controls = [
        lapse_rate_c_per_km,
        relief_elevation_multiplier,
        maximum_highland_fraction,
        snow_degree_day_melt_mm_c_month,
        glacier_degree_day_melt_mm_c_month,
        firn_conversion_fraction_year,
        snow_sublimation_fraction_month,
        glacier_sublimation_fraction_month,
        glacier_flow_activation_mm,
        glacier_flow_fraction_year,
        glacier_reference_thickness_mm,
        runoff_base_fraction,
    ];
    if spinup_years < 2
        || numeric_controls.iter().any(|value| !value.is_finite())
        || lapse_rate_c_per_km < 0.0
        || !(0.0..=6.0).contains(&relief_elevation_multiplier)
        || !(0.0..=1.0).contains(&maximum_highland_fraction)
        || snow_degree_day_melt_mm_c_month <= 0.0
        || glacier_degree_day_melt_mm_c_month <= 0.0
        || !(0.0..=1.0).contains(&firn_conversion_fraction_year)
        || !(0.0..=1.0).contains(&snow_sublimation_fraction_month)
        || !(0.0..=1.0).contains(&glacier_sublimation_fraction_month)
        || glacier_flow_activation_mm < 0.0
        || !(0.0..=1.0).contains(&glacier_flow_fraction_year)
        || glacier_reference_thickness_mm <= 0.0
        || !(0.0..=1.0).contains(&runoff_base_fraction)
    {
        return 2;
    }
    let total = cell_count as usize;
    let monthly_len = match total.checked_mul(MONTHS) {
        Some(value) => value,
        None => return 1,
    };
    let edge_len = match total.checked_mul(NEIGHBORS) {
        Some(value) => value,
        None => return 1,
    };
    let inputs = Inputs {
        areas: unsafe { slice::from_raw_parts(area_ptr, total) },
        neighbors: unsafe { slice::from_raw_parts(neighbor_ptr, edge_len) },
        ocean: unsafe { slice::from_raw_parts(ocean_ptr, total) },
        elevation: unsafe { slice::from_raw_parts(elevation_ptr, total) },
        relief: unsafe { slice::from_raw_parts(relief_ptr, total) },
        temperature: unsafe { slice::from_raw_parts(temperature_ptr, monthly_len) },
        precipitation: unsafe { slice::from_raw_parts(precipitation_ptr, monthly_len) },
        evaporation: unsafe { slice::from_raw_parts(evaporation_ptr, monthly_len) },
    };
    if inputs
        .areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || inputs
            .neighbors
            .iter()
            .any(|value| *value < 0 || *value as usize >= total)
        || inputs
            .ocean
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || inputs.elevation.iter().any(|value| !value.is_finite())
        || inputs
            .relief
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || inputs.temperature.iter().any(|value| !value.is_finite())
        || inputs
            .precipitation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || inputs
            .evaporation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return 3;
    }
    let mut outputs = Outputs {
        snowfall: unsafe { slice::from_raw_parts_mut(snowfall_ptr, monthly_len) },
        snowmelt: unsafe { slice::from_raw_parts_mut(snowmelt_ptr, monthly_len) },
        snowpack: unsafe { slice::from_raw_parts_mut(snowpack_ptr, monthly_len) },
        firn_to_ice: unsafe { slice::from_raw_parts_mut(firn_to_ice_ptr, monthly_len) },
        glacier_melt: unsafe { slice::from_raw_parts_mut(glacier_melt_ptr, monthly_len) },
        glacier_ice: unsafe { slice::from_raw_parts_mut(glacier_ice_ptr, monthly_len) },
        runoff: unsafe { slice::from_raw_parts_mut(runoff_ptr, monthly_len) },
        annual_mass_balance: unsafe { slice::from_raw_parts_mut(annual_mass_balance_ptr, total) },
        annual_flow_export: unsafe { slice::from_raw_parts_mut(annual_flow_export_ptr, total) },
        annual_flow_import: unsafe { slice::from_raw_parts_mut(annual_flow_import_ptr, total) },
        annual_calving: unsafe { slice::from_raw_parts_mut(annual_calving_ptr, total) },
        annual_sublimation: unsafe { slice::from_raw_parts_mut(annual_sublimation_ptr, total) },
        glacier_fraction: unsafe { slice::from_raw_parts_mut(glacier_fraction_ptr, total) },
    };
    let stats = run_model(controls, &inputs, &mut outputs);
    if !stats_out.is_null() {
        unsafe { *stats_out = stats };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snow_fraction_transitions_around_freezing() {
        assert_eq!(snow_fraction(3.0), 0.0);
        assert!(snow_fraction(0.0) > 0.0);
        assert_eq!(snow_fraction(-3.0), 1.0);
    }

    #[test]
    fn snow_removal_preserves_nonnegative_storage() {
        let mut snow = 10.0;
        let mut age = 60.0;
        assert_eq!(remove_snow(&mut snow, &mut age, 12.0), 10.0);
        assert_eq!(snow, 0.0);
        assert_eq!(age, 0.0);
    }
}
