use std::f64::consts::{PI, TAU};
use std::slice;

const MONTHS: usize = 12;
const SOLAR_CONSTANT_W_M2: f64 = 1361.0;

#[repr(C)]
pub struct PlanetStats {
    pub global_mean_insolation_w_m2: f32,
    pub equatorial_mean_insolation_w_m2: f32,
    pub polar_mean_insolation_w_m2: f32,
    pub mean_seasonality_w_m2: f32,
    pub maximum_monthly_insolation_w_m2: f32,
    pub minimum_orbital_distance_au: f32,
    pub maximum_orbital_distance_au: f32,
    pub tide_strength_index: f32,
    pub obliquity_stability_index: f32,
}

#[no_mangle]
pub extern "C" fn planet_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_planet_abi_version() -> u32 {
    1
}

fn solve_eccentric_anomaly(mean_anomaly: f64, eccentricity: f64) -> f64 {
    let mut eccentric_anomaly = mean_anomaly + eccentricity * mean_anomaly.sin();
    for _ in 0..10 {
        let residual = eccentric_anomaly - eccentricity * eccentric_anomaly.sin() - mean_anomaly;
        let derivative = 1.0 - eccentricity * eccentric_anomaly.cos();
        eccentric_anomaly -= residual / derivative.max(1e-12);
    }
    eccentric_anomaly
}

fn orbital_state(day: f64, period: f64, perihelion_day: f64, eccentricity: f64) -> (f64, f64) {
    let mean_anomaly = (TAU * (day - perihelion_day) / period).rem_euclid(TAU);
    let eccentric_anomaly = solve_eccentric_anomaly(mean_anomaly, eccentricity);
    let true_anomaly = 2.0
        * ((1.0 + eccentricity).sqrt() * (0.5 * eccentric_anomaly).sin())
            .atan2((1.0 - eccentricity).sqrt() * (0.5 * eccentric_anomaly).cos());
    let normalized_distance = 1.0 - eccentricity * eccentric_anomaly.cos();
    (true_anomaly, normalized_distance)
}

fn daily_mean_insolation(latitude: f64, declination: f64, solar_flux: f64) -> (f64, f64) {
    let cosine_hour_angle = -latitude.tan() * declination.tan();
    let sunset_hour_angle = if cosine_hour_angle >= 1.0 {
        0.0
    } else if cosine_hour_angle <= -1.0 {
        PI
    } else {
        cosine_hour_angle.acos()
    };
    let daily_mean = solar_flux / PI
        * (sunset_hour_angle * latitude.sin() * declination.sin()
            + latitude.cos() * declination.cos() * sunset_hour_angle.sin());
    (daily_mean.max(0.0), sunset_hour_angle / PI)
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_planet(
    total: usize,
    star_luminosity_solar: f64,
    semi_major_axis_au: f64,
    eccentricity: f64,
    obliquity_radians: f64,
    rotation_period_hours: f64,
    orbital_period_days: f64,
    perihelion_day: f64,
    northern_vernal_equinox_day: f64,
    moon_mass_lunar: f64,
    moon_distance_km: f64,
    areas: &[f64],
    latitudes: &[f64],
    monthly_insolation: &mut [f32],
    monthly_daylight: &mut [f32],
    annual_mean: &mut [f32],
    seasonality: &mut [f32],
    polar_extreme_fraction: &mut [f32],
    orbital_distance: &mut [f32],
    solar_declination: &mut [f32],
) -> PlanetStats {
    let (vernal_true_anomaly, _) = orbital_state(
        northern_vernal_equinox_day,
        orbital_period_days,
        perihelion_day,
        eccentricity,
    );
    let baseline_solar_flux =
        SOLAR_CONSTANT_W_M2 * star_luminosity_solar / semi_major_axis_au.powi(2);

    for month in 0..MONTHS {
        let day = (month as f64 + 0.5) / MONTHS as f64 * orbital_period_days;
        let (true_anomaly, normalized_distance) =
            orbital_state(day, orbital_period_days, perihelion_day, eccentricity);
        let distance_au = semi_major_axis_au * normalized_distance;
        let solar_longitude = true_anomaly - vernal_true_anomaly;
        let declination = (obliquity_radians.sin() * solar_longitude.sin()).asin();
        orbital_distance[month] = distance_au as f32;
        solar_declination[month] = declination as f32;
        let solar_flux = baseline_solar_flux / normalized_distance.powi(2);
        for cell in 0..total {
            let (insolation, daylight_fraction) =
                daily_mean_insolation(latitudes[cell], declination, solar_flux);
            monthly_insolation[month * total + cell] = insolation as f32;
            monthly_daylight[month * total + cell] =
                (rotation_period_hours * daylight_fraction) as f32;
        }
    }

    let mut global_sum = 0.0f64;
    let mut equatorial_sum = 0.0f64;
    let mut equatorial_area = 0.0f64;
    let mut polar_sum = 0.0f64;
    let mut polar_area = 0.0f64;
    let mut seasonality_sum = 0.0f64;
    let mut maximum_monthly = 0.0f32;
    let total_area: f64 = areas.iter().sum();
    for cell in 0..total {
        let mut sum = 0.0f32;
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut extreme_months = 0usize;
        for month in 0..MONTHS {
            let value = monthly_insolation[month * total + cell];
            let daylight = monthly_daylight[month * total + cell];
            sum += value;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            maximum_monthly = maximum_monthly.max(value);
            if daylight <= rotation_period_hours as f32 * 0.01
                || daylight >= rotation_period_hours as f32 * 0.99
            {
                extreme_months += 1;
            }
        }
        let mean = sum / MONTHS as f32;
        let amplitude = maximum - minimum;
        annual_mean[cell] = mean;
        seasonality[cell] = amplitude;
        polar_extreme_fraction[cell] = extreme_months as f32 / MONTHS as f32;
        global_sum += mean as f64 * areas[cell];
        seasonality_sum += amplitude as f64 * areas[cell];
        let absolute_latitude = latitudes[cell].abs();
        if absolute_latitude <= 15.0f64.to_radians() {
            equatorial_sum += mean as f64 * areas[cell];
            equatorial_area += areas[cell];
        }
        if absolute_latitude >= 66.5f64.to_radians() {
            polar_sum += mean as f64 * areas[cell];
            polar_area += areas[cell];
        }
    }

    let tide_strength = if moon_mass_lunar <= 0.0 {
        0.0
    } else {
        moon_mass_lunar * (384_400.0 / moon_distance_km).powi(3)
    };
    let stability = (0.35 + 0.65 * tide_strength / (0.5 + tide_strength)).clamp(0.0, 1.0);
    PlanetStats {
        global_mean_insolation_w_m2: (global_sum / total_area) as f32,
        equatorial_mean_insolation_w_m2: (equatorial_sum / equatorial_area) as f32,
        polar_mean_insolation_w_m2: (polar_sum / polar_area) as f32,
        mean_seasonality_w_m2: (seasonality_sum / total_area) as f32,
        maximum_monthly_insolation_w_m2: maximum_monthly,
        minimum_orbital_distance_au: orbital_distance
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        maximum_orbital_distance_au: orbital_distance
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        tide_strength_index: tide_strength as f32,
        obliquity_stability_index: stability as f32,
    }
}

/// Compute twelve equal-time monthly orbital forcing fields.
///
/// Returns 0 on success, 1 for invalid dimensions or pointers, 2 for invalid
/// parameters, and 3 for invalid numeric inputs.
///
/// # Safety
///
/// Input pointers must reference `cell_count` readable values. Monthly output
/// pointers must reference `12 * cell_count` distinct writable values; annual
/// outputs must reference `cell_count` values; orbital outputs must reference
/// twelve values. Outputs must not alias inputs or each other. `stats_out` may
/// be null or must reference one writable `PlanetStats`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn planet_run_cubed_sphere(
    cell_count: i32,
    star_luminosity_solar: f64,
    semi_major_axis_au: f64,
    eccentricity: f64,
    obliquity_radians: f64,
    rotation_period_hours: f64,
    orbital_period_days: f64,
    perihelion_day: f64,
    northern_vernal_equinox_day: f64,
    moon_mass_lunar: f64,
    moon_distance_km: f64,
    area_ptr: *const f64,
    latitude_ptr: *const f64,
    monthly_insolation_ptr: *mut f32,
    monthly_daylight_ptr: *mut f32,
    annual_mean_ptr: *mut f32,
    seasonality_ptr: *mut f32,
    polar_extreme_fraction_ptr: *mut f32,
    orbital_distance_ptr: *mut f32,
    solar_declination_ptr: *mut f32,
    stats_out: *mut PlanetStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || latitude_ptr.is_null()
        || monthly_insolation_ptr.is_null()
        || monthly_daylight_ptr.is_null()
        || annual_mean_ptr.is_null()
        || seasonality_ptr.is_null()
        || polar_extreme_fraction_ptr.is_null()
        || orbital_distance_ptr.is_null()
        || solar_declination_ptr.is_null()
    {
        return 1;
    }
    let parameters = [
        star_luminosity_solar,
        semi_major_axis_au,
        eccentricity,
        obliquity_radians,
        rotation_period_hours,
        orbital_period_days,
        perihelion_day,
        northern_vernal_equinox_day,
        moon_mass_lunar,
        moon_distance_km,
    ];
    if parameters.iter().any(|value| !value.is_finite())
        || star_luminosity_solar <= 0.0
        || semi_major_axis_au <= 0.0
        || !(0.0..0.95).contains(&eccentricity)
        || !(0.0..=PI / 2.0).contains(&obliquity_radians)
        || rotation_period_hours <= 0.0
        || orbital_period_days <= 0.0
        || !(0.0..orbital_period_days).contains(&perihelion_day)
        || !(0.0..orbital_period_days).contains(&northern_vernal_equinox_day)
        || moon_mass_lunar < 0.0
        || moon_distance_km <= 0.0
    {
        return 2;
    }
    let total = cell_count as usize;
    let monthly_len = match total.checked_mul(MONTHS) {
        Some(value) => value,
        None => return 1,
    };
    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let latitudes = unsafe { slice::from_raw_parts(latitude_ptr, total) };
    if areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || latitudes
            .iter()
            .any(|value| !value.is_finite() || value.abs() > PI / 2.0 + 1e-9)
    {
        return 3;
    }
    let monthly_insolation =
        unsafe { slice::from_raw_parts_mut(monthly_insolation_ptr, monthly_len) };
    let monthly_daylight = unsafe { slice::from_raw_parts_mut(monthly_daylight_ptr, monthly_len) };
    let annual_mean = unsafe { slice::from_raw_parts_mut(annual_mean_ptr, total) };
    let seasonality = unsafe { slice::from_raw_parts_mut(seasonality_ptr, total) };
    let polar_extreme_fraction =
        unsafe { slice::from_raw_parts_mut(polar_extreme_fraction_ptr, total) };
    let orbital_distance = unsafe { slice::from_raw_parts_mut(orbital_distance_ptr, MONTHS) };
    let solar_declination = unsafe { slice::from_raw_parts_mut(solar_declination_ptr, MONTHS) };
    let stats = unsafe {
        run_planet(
            total,
            star_luminosity_solar,
            semi_major_axis_au,
            eccentricity,
            obliquity_radians,
            rotation_period_hours,
            orbital_period_days,
            perihelion_day,
            northern_vernal_equinox_day,
            moon_mass_lunar,
            moon_distance_km,
            areas,
            latitudes,
            monthly_insolation,
            monthly_daylight,
            annual_mean,
            seasonality,
            polar_extreme_fraction,
            orbital_distance,
            solar_declination,
        )
    };
    if !stats_out.is_null() {
        unsafe { *stats_out = stats };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_orbit_has_constant_distance() {
        let (_, first) = orbital_state(10.0, 365.0, 3.0, 0.0);
        let (_, second) = orbital_state(200.0, 365.0, 3.0, 0.0);
        assert!((first - 1.0).abs() < 1e-12);
        assert!((second - first).abs() < 1e-12);
    }

    #[test]
    fn polar_daylight_extremes_are_opposite() {
        let north = daily_mean_insolation(80.0f64.to_radians(), 23.0f64.to_radians(), 1361.0);
        let south = daily_mean_insolation(-80.0f64.to_radians(), 23.0f64.to_radians(), 1361.0);
        assert!(north.1 > 0.99);
        assert!(south.1 < 0.01);
        assert!(north.0 > south.0);
    }
}
