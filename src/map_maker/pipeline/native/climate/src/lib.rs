use std::f64::consts::PI;
use std::slice;

const MONTHS: usize = 12;
const NEIGHBORS: usize = 4;
const VECTOR_COMPONENTS: usize = 3;
const DAYS_PER_MONTH: f64 = 365.2422 / 12.0;
const REFERENCE_MOISTURE_STEPS_PER_MONTH: f64 = 16.0;
const REFERENCE_SUPERSATURATION_FRACTION: f64 = 0.03;
const REFERENCE_MAXIMUM_CONDENSATION_FRACTION: f64 = 0.05;

#[repr(C)]
pub struct ClimateStats {
    pub global_mean_temperature_c: f32,
    pub land_mean_temperature_c: f32,
    pub ocean_mean_temperature_c: f32,
    pub minimum_monthly_temperature_c: f32,
    pub maximum_monthly_temperature_c: f32,
    pub global_mean_annual_precipitation_mm: f32,
    pub land_mean_annual_precipitation_mm: f32,
    pub dry_land_area_fraction: f32,
    pub wet_land_area_fraction: f32,
    pub persistent_snow_land_area_fraction: f32,
    pub maximum_wind_speed_m_s: f32,
}

#[no_mangle]
pub extern "C" fn climate_native_abi_version() -> u32 {
    3
}

#[no_mangle]
pub extern "C" fn cubed_sphere_climate_abi_version() -> u32 {
    2
}

fn dot(first: [f64; 3], second: [f64; 3]) -> f64 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn cross(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn normalized(vector: [f64; 3]) -> [f64; 3] {
    let length = norm(vector).max(1e-12);
    [vector[0] / length, vector[1] / length, vector[2] / length]
}

fn radial_at(xyz: &[f32], cell: usize) -> [f64; 3] {
    let offset = cell * VECTOR_COMPONENTS;
    [
        f64::from(xyz[offset]),
        f64::from(xyz[offset + 1]),
        f64::from(xyz[offset + 2]),
    ]
}

fn tangent_basis(radial: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let horizontal = (radial[0] * radial[0] + radial[1] * radial[1]).sqrt();
    let east = if horizontal > 1e-10 {
        [-radial[1] / horizontal, radial[0] / horizontal, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let north = normalized(cross(radial, east));
    (east, north)
}

fn tangent_direction(source: [f64; 3], target: [f64; 3]) -> ([f64; 3], f64) {
    let cosine = dot(source, target).clamp(-1.0, 1.0);
    let angle = cosine.acos().max(1e-9);
    let projected = [
        target[0] - cosine * source[0],
        target[1] - cosine * source[1],
        target[2] - cosine * source[2],
    ];
    (normalized(projected), angle)
}

fn interpolate(points: &[(f64, f64)], position: f64) -> f64 {
    if position <= points[0].0 {
        return points[0].1;
    }
    for pair in points.windows(2) {
        let (left_x, left_y) = pair[0];
        let (right_x, right_y) = pair[1];
        if position <= right_x {
            let weight = (position - left_x) / (right_x - left_x);
            return left_y + weight * (right_y - left_y);
        }
    }
    points[points.len() - 1].1
}

fn circulation_components(latitude: f64, declination: f64) -> (f64, f64) {
    let shifted = (latitude - 0.25 * declination).clamp(-PI / 2.0, PI / 2.0);
    let absolute_degrees = shifted.abs().to_degrees();
    let zonal = interpolate(
        &[
            (0.0, -2.0),
            (15.0, -7.5),
            (30.0, 0.0),
            (50.0, 11.5),
            (65.0, 0.0),
            (80.0, -5.0),
            (90.0, 0.0),
        ],
        absolute_degrees,
    );
    let northward_in_northern_hemisphere = interpolate(
        &[
            (0.0, 0.0),
            (15.0, -2.2),
            (30.0, 0.0),
            (45.0, 1.5),
            (60.0, 0.0),
            (75.0, -1.0),
            (90.0, 0.0),
        ],
        absolute_degrees,
    );
    let meridional = northward_in_northern_hemisphere * shifted.signum();
    (zonal, meridional)
}

fn moisture_capacity_mm(temperature_c: f64) -> f64 {
    (7.5 * (0.07 * temperature_c).exp()).clamp(1.5, 75.0)
}

fn equivalent_step_fraction(reference_fraction: f64, steps_per_month: usize) -> f64 {
    1.0 - (1.0 - reference_fraction)
        .powf(REFERENCE_MOISTURE_STEPS_PER_MONTH / steps_per_month as f64)
}

fn smooth_orography(
    total: usize,
    neighbors: &[i32],
    ocean: &[f32],
    elevation: &[f32],
    output: &mut [f32],
) {
    output.copy_from_slice(elevation);
    let mut next = vec![0.0f32; total];
    for _ in 0..5 {
        for cell in 0..total {
            let is_ocean = ocean[cell] >= 0.5;
            let mut neighbor_sum = 0.0f64;
            let mut neighbor_count = 0usize;
            for edge in 0..NEIGHBORS {
                let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                if (ocean[neighbor] >= 0.5) == is_ocean {
                    neighbor_sum += f64::from(output[neighbor]);
                    neighbor_count += 1;
                }
            }
            next[cell] = if neighbor_count == 0 {
                output[cell]
            } else {
                (0.48 * f64::from(output[cell]) + 0.52 * neighbor_sum / neighbor_count as f64)
                    as f32
            };
        }
        output.copy_from_slice(&next);
    }
    for cell in 0..total {
        output[cell] = if ocean[cell] >= 0.5 {
            0.0
        } else {
            output[cell].max(0.0)
        };
    }
}

fn mix_monthly_field(
    total: usize,
    passes: usize,
    neighbors: &[i32],
    areas: &[f64],
    field: &mut [f32],
) {
    let mut mixed = vec![0.0f32; total];
    for month in 0..MONTHS {
        let month_offset = month * total;
        let original_total = (0..total)
            .map(|cell| f64::from(field[month_offset + cell]) * areas[cell])
            .sum::<f64>();
        // Monthly climatology represents unresolved synoptic weather, so mix
        // the cell-scale transport signal over a roughly 400 km footprint.
        for _ in 0..passes {
            for cell in 0..total {
                let neighbor_mean = (0..NEIGHBORS)
                    .map(|edge| {
                        let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                        f64::from(field[month_offset + neighbor])
                    })
                    .sum::<f64>()
                    / NEIGHBORS as f64;
                mixed[cell] =
                    (0.72 * f64::from(field[month_offset + cell]) + 0.28 * neighbor_mean) as f32;
            }
            field[month_offset..month_offset + total].copy_from_slice(&mixed);
        }
        let mixed_total = (0..total)
            .map(|cell| f64::from(field[month_offset + cell]) * areas[cell])
            .sum::<f64>();
        let correction = if mixed_total > 0.0 {
            original_total / mixed_total
        } else {
            1.0
        };
        for cell in 0..total {
            field[month_offset + cell] =
                (f64::from(field[month_offset + cell]) * correction) as f32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_temperature(
    total: usize,
    spinup_years: usize,
    greenhouse_offset_c: f64,
    land_albedo: f64,
    ocean_albedo: f64,
    olr_intercept_w_m2: f64,
    olr_slope_w_m2_c: f64,
    heat_transport_w_m2: f64,
    land_thermal_response: f64,
    ocean_thermal_response: f64,
    atmospheric_exchange: f64,
    lapse_rate_c_per_km: f64,
    neighbors: &[i32],
    latitudes: &[f64],
    elevation: &[f32],
    ocean: &[f32],
    insolation: &[f32],
    temperature: &mut [f32],
) {
    let equilibrium = |cell: usize, forcing: f64| -> f64 {
        let latitude_sine = latitudes[cell].sin();
        let transport = heat_transport_w_m2 * (3.0 * latitude_sine.powi(2) - 1.0);
        let albedo = if ocean[cell] >= 0.5 {
            ocean_albedo
        } else {
            land_albedo
        };
        let lapse = lapse_rate_c_per_km * f64::from(elevation[cell]).max(0.0) / 1000.0;
        (((1.0 - albedo) * forcing + transport - olr_intercept_w_m2) / olr_slope_w_m2_c
            + greenhouse_offset_c
            - lapse)
            .clamp(-95.0, 75.0)
    };

    let mut state = vec![0.0f64; total];
    let mut next = vec![0.0f64; total];
    for cell in 0..total {
        let annual_forcing = (0..MONTHS)
            .map(|month| f64::from(insolation[month * total + cell]))
            .sum::<f64>()
            / MONTHS as f64;
        state[cell] = equilibrium(cell, annual_forcing);
    }

    for year in 0..spinup_years {
        for month in 0..MONTHS {
            for cell in 0..total {
                let neighbor_mean = (0..NEIGHBORS)
                    .map(|edge| state[neighbors[cell * NEIGHBORS + edge] as usize])
                    .sum::<f64>()
                    / NEIGHBORS as f64;
                let response = if ocean[cell] >= 0.5 {
                    ocean_thermal_response
                } else {
                    land_thermal_response
                };
                let target = equilibrium(cell, f64::from(insolation[month * total + cell]));
                next[cell] = (state[cell]
                    + response * (target - state[cell])
                    + atmospheric_exchange * (neighbor_mean - state[cell]))
                    .clamp(-95.0, 70.0);
            }
            state.copy_from_slice(&next);
            if year + 1 == spinup_years {
                for cell in 0..total {
                    temperature[month * total + cell] = state[cell] as f32;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_wind(
    total: usize,
    wind_scale: f64,
    neighbors: &[i32],
    xyz: &[f32],
    latitudes: &[f64],
    elevation: &[f32],
    declination: &[f32],
    temperature: &[f32],
    wind_xyz: &mut [f32],
    wind_speed: &mut [f32],
) {
    for month in 0..MONTHS {
        for cell in 0..total {
            let radial = radial_at(xyz, cell);
            let (east, north) = tangent_basis(radial);
            let (zonal, meridional) =
                circulation_components(latitudes[cell], f64::from(declination[month]));
            let mut thermal_gradient = [0.0f64; 3];
            let mut terrain_gradient = [0.0f64; 3];
            let center_temperature = f64::from(temperature[month * total + cell]);
            let center_elevation = f64::from(elevation[cell]);
            for edge in 0..NEIGHBORS {
                let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                let (direction, angle) = tangent_direction(radial, radial_at(xyz, neighbor));
                let temperature_delta =
                    f64::from(temperature[month * total + neighbor]) - center_temperature;
                let elevation_delta = f64::from(elevation[neighbor]) - center_elevation;
                for component in 0..VECTOR_COMPONENTS {
                    thermal_gradient[component] += temperature_delta / angle * direction[component];
                    terrain_gradient[component] += elevation_delta / angle * direction[component];
                }
            }
            let thermal_magnitude = norm(thermal_gradient);
            let thermal_direction = normalized(thermal_gradient);
            let geostrophic_direction = normalized(cross(radial, thermal_direction));
            let thermal_flow = (thermal_magnitude * 0.025).clamp(0.0, 2.5);
            let geostrophic_flow =
                (thermal_magnitude * 0.035).clamp(0.0, 3.5) * latitudes[cell].signum();
            let mut wind = [0.0f64; 3];
            for component in 0..VECTOR_COMPONENTS {
                wind[component] = wind_scale
                    * (zonal * east[component]
                        + meridional * north[component]
                        + thermal_flow * thermal_direction[component]
                        + geostrophic_flow * geostrophic_direction[component]);
            }

            let terrain_magnitude = norm(terrain_gradient);
            if terrain_magnitude > 1e-9 {
                let terrain_direction = normalized(terrain_gradient);
                let uphill_component = dot(wind, terrain_direction).max(0.0);
                for component in 0..VECTOR_COMPONENTS {
                    wind[component] -= 0.3 * uphill_component * terrain_direction[component];
                }
            }
            let radial_leak = dot(wind, radial);
            for component in 0..VECTOR_COMPONENTS {
                wind[component] -= radial_leak * radial[component];
            }
            let speed = norm(wind);
            wind_speed[month * total + cell] = speed as f32;
            let output_offset = (month * total + cell) * VECTOR_COMPONENTS;
            for component in 0..VECTOR_COMPONENTS {
                wind_xyz[output_offset + component] = wind[component] as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_moisture(
    total: usize,
    spinup_years: usize,
    steps_per_month: usize,
    advection_fraction: f64,
    diffusion_fraction: f64,
    orographic_factor: f64,
    rain_shadow_factor: f64,
    neighbors: &[i32],
    xyz: &[f32],
    elevation: &[f32],
    ocean: &[f32],
    temperature: &[f32],
    wind_xyz: &[f32],
    precipitation: &mut [f32],
    humidity: &mut [f32],
    evaporation: &mut [f32],
) {
    precipitation.fill(0.0);
    humidity.fill(0.0);
    evaporation.fill(0.0);
    let mut moisture = vec![0.0f64; total];
    let mut next = vec![0.0f64; total];
    let mut upslope = vec![0.0f64; total];
    let mut downslope = vec![0.0f64; total];
    for cell in 0..total {
        let capacity = moisture_capacity_mm(f64::from(temperature[cell]));
        moisture[cell] = capacity * if ocean[cell] >= 0.5 { 0.78 } else { 0.45 };
    }
    let retained_fraction = 1.0 - advection_fraction - diffusion_fraction;
    let step_days = DAYS_PER_MONTH / steps_per_month as f64;
    let supersaturation_fraction =
        equivalent_step_fraction(REFERENCE_SUPERSATURATION_FRACTION, steps_per_month);
    let maximum_condensation_fraction =
        equivalent_step_fraction(REFERENCE_MAXIMUM_CONDENSATION_FRACTION, steps_per_month);

    for year in 0..spinup_years {
        for month in 0..MONTHS {
            for cell in 0..total {
                let radial = radial_at(xyz, cell);
                let wind_offset = (month * total + cell) * VECTOR_COMPONENTS;
                let wind = [
                    f64::from(wind_xyz[wind_offset]),
                    f64::from(wind_xyz[wind_offset + 1]),
                    f64::from(wind_xyz[wind_offset + 2]),
                ];
                let wind_direction = normalized(wind);
                let mut weight_sum = 0.0;
                let mut upwind_elevation = 0.0;
                for edge in 0..NEIGHBORS {
                    let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                    let (direction, _) = tangent_direction(radial, radial_at(xyz, neighbor));
                    let weight = (-dot(wind_direction, direction)).max(0.0).powi(2);
                    weight_sum += weight;
                    upwind_elevation += weight * f64::from(elevation[neighbor]);
                }
                if weight_sum > 1e-10 {
                    upwind_elevation /= weight_sum;
                } else {
                    upwind_elevation = (0..NEIGHBORS)
                        .map(|edge| {
                            let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                            f64::from(elevation[neighbor])
                        })
                        .sum::<f64>()
                        / NEIGHBORS as f64;
                }
                let elevation_delta = f64::from(elevation[cell]) - upwind_elevation;
                upslope[cell] = (elevation_delta / 1000.0).max(0.0);
                downslope[cell] = (-elevation_delta / 1000.0).max(0.0);
            }

            for _ in 0..steps_per_month {
                next.fill(0.0);
                for cell in 0..total {
                    let temperature_c = f64::from(temperature[month * total + cell]);
                    let capacity = moisture_capacity_mm(temperature_c);
                    let relative_humidity = (moisture[cell] / capacity).clamp(0.0, 1.2);
                    let warmth = ((temperature_c + 15.0) / 35.0).clamp(0.15, 1.5);
                    let evaporation_rate = if ocean[cell] >= 0.5 {
                        6.2 * warmth * (1.0 - 0.45 * relative_humidity.min(1.0))
                    } else {
                        0.85 * warmth * relative_humidity.min(1.0)
                    };
                    let evaporated = evaporation_rate.max(0.0) * step_days;
                    if year + 1 == spinup_years {
                        evaporation[month * total + cell] += evaporated as f32;
                    }
                    let available = moisture[cell] + evaporated;
                    next[cell] += available * retained_fraction;
                    let diffuse_packet = available * diffusion_fraction / NEIGHBORS as f64;
                    for edge in 0..NEIGHBORS {
                        let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                        next[neighbor] += diffuse_packet;
                    }

                    let radial = radial_at(xyz, cell);
                    let wind_offset = (month * total + cell) * VECTOR_COMPONENTS;
                    let wind = normalized([
                        f64::from(wind_xyz[wind_offset]),
                        f64::from(wind_xyz[wind_offset + 1]),
                        f64::from(wind_xyz[wind_offset + 2]),
                    ]);
                    let mut weights = [0.0f64; NEIGHBORS];
                    let mut weight_sum = 0.0;
                    for (edge, weight) in weights.iter_mut().enumerate() {
                        let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                        let (direction, _) = tangent_direction(radial, radial_at(xyz, neighbor));
                        *weight = dot(wind, direction).max(0.0).powi(2);
                        weight_sum += *weight;
                    }
                    if weight_sum <= 1e-10 {
                        next[cell] += available * advection_fraction;
                    } else {
                        for (edge, weight) in weights.iter().enumerate() {
                            let neighbor = neighbors[cell * NEIGHBORS + edge] as usize;
                            next[neighbor] += available * advection_fraction * *weight / weight_sum;
                        }
                    }
                }

                for cell in 0..total {
                    let capacity =
                        moisture_capacity_mm(f64::from(temperature[month * total + cell]));
                    let is_ocean = ocean[cell] >= 0.5;
                    let shadow = (-rain_shadow_factor * downslope[cell]).exp();
                    let background_rate: f64 = if is_ocean { 0.0001 } else { 0.015 };
                    let background = next[cell] * (background_rate * step_days).min(0.22) * shadow;
                    let saturation_threshold = if is_ocean { 10.0 } else { 1.1 };
                    let supersaturation = (next[cell] - saturation_threshold * capacity).max(0.0)
                        * supersaturation_fraction;
                    let orographic = if is_ocean {
                        0.0
                    } else {
                        next[cell] * (1.0 - (-orographic_factor * upslope[cell]).exp()) * 0.65
                    };
                    let condensed = (background + supersaturation + orographic)
                        .min(next[cell] * maximum_condensation_fraction);
                    next[cell] -= condensed;
                    if year + 1 == spinup_years {
                        precipitation[month * total + cell] += condensed as f32;
                    }
                }
                moisture.copy_from_slice(&next);
            }

            if year + 1 == spinup_years {
                for cell in 0..total {
                    let capacity =
                        moisture_capacity_mm(f64::from(temperature[month * total + cell]));
                    humidity[month * total + cell] =
                        (moisture[cell] / capacity).clamp(0.0, 1.0) as f32;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_surface_water(
    total: usize,
    runoff_base_fraction: f64,
    ocean: &[f32],
    relief: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
    evaporation: &[f32],
    snowfall: &mut [f32],
    snowmelt: &mut [f32],
    snowpack: &mut [f32],
    runoff: &mut [f32],
) {
    snowfall.fill(0.0);
    snowmelt.fill(0.0);
    snowpack.fill(0.0);
    runoff.fill(0.0);
    let mut stored_snow = vec![0.0f64; total];
    for year in 0..24 {
        for month in 0..MONTHS {
            for cell in 0..total {
                let offset = month * total + cell;
                if ocean[cell] >= 0.5 {
                    stored_snow[cell] = 0.0;
                    continue;
                }
                let temperature_c = f64::from(temperature[offset]);
                let snow_fraction = ((2.0 - temperature_c) / 5.0).clamp(0.0, 1.0);
                let fallen = f64::from(precipitation[offset]) * snow_fraction;
                stored_snow[cell] += fallen;
                let melt_capacity = temperature_c.max(0.0) * 12.0;
                let melted = stored_snow[cell].min(melt_capacity);
                stored_snow[cell] -= melted;
                let sublimated = stored_snow[cell].min(0.004 * stored_snow[cell] + 0.2);
                stored_snow[cell] = (stored_snow[cell] - sublimated).min(5000.0);
                if year == 23 {
                    snowfall[offset] = fallen as f32;
                    snowmelt[offset] = melted as f32;
                    snowpack[offset] = stored_snow[cell] as f32;
                }
            }
        }
    }

    for month in 0..MONTHS {
        for cell in 0..total {
            let offset = month * total + cell;
            if ocean[cell] >= 0.5 {
                continue;
            }
            let rain = (f64::from(precipitation[offset]) - f64::from(snowfall[offset])).max(0.0);
            let available = rain + f64::from(snowmelt[offset]);
            let effective_evaporation = f64::from(evaporation[offset]).min(0.85 * available);
            let relief_factor = (f64::from(relief[cell]) / 1500.0).clamp(0.0, 1.0);
            let storm_factor = (f64::from(precipitation[offset]) / 200.0).clamp(0.0, 1.0);
            let runoff_fraction =
                (runoff_base_fraction + 0.25 * relief_factor + 0.15 * storm_factor)
                    .clamp(0.0, 0.95);
            runoff[offset] =
                ((available - effective_evaporation).max(0.0) * runoff_fraction) as f32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize(
    total: usize,
    areas: &[f64],
    ocean: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
    evaporation: &[f32],
    snowpack: &[f32],
    wind_speed: &[f32],
    annual_temperature: &mut [f32],
    annual_precipitation: &mut [f32],
    aridity: &mut [f32],
) -> ClimateStats {
    let mut total_area = 0.0;
    let mut land_area = 0.0;
    let mut ocean_area = 0.0;
    let mut global_temperature_sum = 0.0;
    let mut land_temperature_sum = 0.0;
    let mut ocean_temperature_sum = 0.0;
    let mut global_precipitation_sum = 0.0;
    let mut land_precipitation_sum = 0.0;
    let mut dry_land_area = 0.0;
    let mut wet_land_area = 0.0;
    let mut snow_land_area = 0.0;
    let mut minimum_temperature = f32::INFINITY;
    let mut maximum_temperature = f32::NEG_INFINITY;
    let mut maximum_wind = 0.0f32;

    for cell in 0..total {
        let mut temperature_sum = 0.0f32;
        let mut precipitation_sum = 0.0f32;
        let mut evaporation_sum = 0.0f32;
        let mut maximum_snowpack = 0.0f32;
        for month in 0..MONTHS {
            let offset = month * total + cell;
            let temperature_value = temperature[offset];
            temperature_sum += temperature_value;
            precipitation_sum += precipitation[offset];
            evaporation_sum += evaporation[offset];
            maximum_snowpack = maximum_snowpack.max(snowpack[offset]);
            minimum_temperature = minimum_temperature.min(temperature_value);
            maximum_temperature = maximum_temperature.max(temperature_value);
            maximum_wind = maximum_wind.max(wind_speed[offset]);
        }
        let mean_temperature = temperature_sum / MONTHS as f32;
        annual_temperature[cell] = mean_temperature;
        annual_precipitation[cell] = precipitation_sum;
        aridity[cell] = precipitation_sum / evaporation_sum.max(1.0);
        let area = areas[cell];
        total_area += area;
        global_temperature_sum += f64::from(mean_temperature) * area;
        global_precipitation_sum += f64::from(precipitation_sum) * area;
        if ocean[cell] >= 0.5 {
            ocean_area += area;
            ocean_temperature_sum += f64::from(mean_temperature) * area;
        } else {
            land_area += area;
            land_temperature_sum += f64::from(mean_temperature) * area;
            land_precipitation_sum += f64::from(precipitation_sum) * area;
            if precipitation_sum < 250.0 {
                dry_land_area += area;
            }
            if precipitation_sum > 2000.0 {
                wet_land_area += area;
            }
            if maximum_snowpack > 100.0 {
                snow_land_area += area;
            }
        }
    }

    ClimateStats {
        global_mean_temperature_c: (global_temperature_sum / total_area) as f32,
        land_mean_temperature_c: (land_temperature_sum / land_area) as f32,
        ocean_mean_temperature_c: (ocean_temperature_sum / ocean_area) as f32,
        minimum_monthly_temperature_c: minimum_temperature,
        maximum_monthly_temperature_c: maximum_temperature,
        global_mean_annual_precipitation_mm: (global_precipitation_sum / total_area) as f32,
        land_mean_annual_precipitation_mm: (land_precipitation_sum / land_area) as f32,
        dry_land_area_fraction: (dry_land_area / land_area) as f32,
        wet_land_area_fraction: (wet_land_area / land_area) as f32,
        persistent_snow_land_area_fraction: (snow_land_area / land_area) as f32,
        maximum_wind_speed_m_s: maximum_wind,
    }
}

/// Compute a structurally realistic monthly climate on a canonical cubed sphere.
///
/// Returns 0 on success, 1 for invalid dimensions/pointers, 2 for invalid
/// controls, 3 for invalid numeric inputs, and 4 for invalid topology.
///
/// # Safety
///
/// All input pointers must reference the documented readable lengths. Outputs
/// must reference distinct writable buffers of the documented lengths and must
/// not alias any input. `stats_out` may be null or reference one writable value.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn climate_run_cubed_sphere(
    cell_count: i32,
    spinup_years: i32,
    moisture_spinup_years: i32,
    moisture_steps_per_month: i32,
    synoptic_mixing_passes: i32,
    greenhouse_offset_c: f64,
    land_albedo: f64,
    ocean_albedo: f64,
    olr_intercept_w_m2: f64,
    olr_slope_w_m2_c: f64,
    heat_transport_w_m2: f64,
    land_thermal_response: f64,
    ocean_thermal_response: f64,
    atmospheric_exchange: f64,
    lapse_rate_c_per_km: f64,
    wind_scale: f64,
    moisture_advection_fraction: f64,
    moisture_diffusion_fraction: f64,
    orographic_factor: f64,
    rain_shadow_factor: f64,
    runoff_base_fraction: f64,
    area_ptr: *const f64,
    neighbor_ptr: *const i32,
    xyz_ptr: *const f32,
    latitude_ptr: *const f64,
    elevation_ptr: *const f32,
    relief_ptr: *const f32,
    ocean_ptr: *const f32,
    insolation_ptr: *const f32,
    declination_ptr: *const f32,
    climate_orography_ptr: *mut f32,
    temperature_ptr: *mut f32,
    wind_xyz_ptr: *mut f32,
    wind_speed_ptr: *mut f32,
    precipitation_ptr: *mut f32,
    humidity_ptr: *mut f32,
    snowfall_ptr: *mut f32,
    snowmelt_ptr: *mut f32,
    snowpack_ptr: *mut f32,
    evaporation_ptr: *mut f32,
    runoff_ptr: *mut f32,
    annual_temperature_ptr: *mut f32,
    annual_precipitation_ptr: *mut f32,
    aridity_ptr: *mut f32,
    stats_out: *mut ClimateStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || neighbor_ptr.is_null()
        || xyz_ptr.is_null()
        || latitude_ptr.is_null()
        || elevation_ptr.is_null()
        || relief_ptr.is_null()
        || ocean_ptr.is_null()
        || insolation_ptr.is_null()
        || declination_ptr.is_null()
        || climate_orography_ptr.is_null()
        || temperature_ptr.is_null()
        || wind_xyz_ptr.is_null()
        || wind_speed_ptr.is_null()
        || precipitation_ptr.is_null()
        || humidity_ptr.is_null()
        || snowfall_ptr.is_null()
        || snowmelt_ptr.is_null()
        || snowpack_ptr.is_null()
        || evaporation_ptr.is_null()
        || runoff_ptr.is_null()
        || annual_temperature_ptr.is_null()
        || annual_precipitation_ptr.is_null()
        || aridity_ptr.is_null()
    {
        return 1;
    }
    let controls = [
        greenhouse_offset_c,
        land_albedo,
        ocean_albedo,
        olr_intercept_w_m2,
        olr_slope_w_m2_c,
        heat_transport_w_m2,
        land_thermal_response,
        ocean_thermal_response,
        atmospheric_exchange,
        lapse_rate_c_per_km,
        wind_scale,
        moisture_advection_fraction,
        moisture_diffusion_fraction,
        orographic_factor,
        rain_shadow_factor,
        runoff_base_fraction,
    ];
    if spinup_years < 2
        || moisture_spinup_years < 1
        || moisture_steps_per_month < 2
        || synoptic_mixing_passes < 1
        || controls.iter().any(|value| !value.is_finite())
        || !(0.0..1.0).contains(&land_albedo)
        || !(0.0..1.0).contains(&ocean_albedo)
        || olr_slope_w_m2_c <= 0.0
        || !(0.0..=1.0).contains(&land_thermal_response)
        || !(0.0..=1.0).contains(&ocean_thermal_response)
        || !(0.0..=1.0).contains(&atmospheric_exchange)
        || land_thermal_response + atmospheric_exchange > 1.0
        || ocean_thermal_response + atmospheric_exchange > 1.0
        || lapse_rate_c_per_km < 0.0
        || wind_scale <= 0.0
        || moisture_advection_fraction < 0.0
        || moisture_diffusion_fraction < 0.0
        || moisture_advection_fraction + moisture_diffusion_fraction > 1.0
        || orographic_factor < 0.0
        || rain_shadow_factor < 0.0
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
    let xyz_len = match total.checked_mul(VECTOR_COMPONENTS) {
        Some(value) => value,
        None => return 1,
    };
    let wind_len = match monthly_len.checked_mul(VECTOR_COMPONENTS) {
        Some(value) => value,
        None => return 1,
    };

    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let neighbors = unsafe { slice::from_raw_parts(neighbor_ptr, edge_len) };
    let xyz = unsafe { slice::from_raw_parts(xyz_ptr, xyz_len) };
    let latitudes = unsafe { slice::from_raw_parts(latitude_ptr, total) };
    let elevation = unsafe { slice::from_raw_parts(elevation_ptr, total) };
    let relief = unsafe { slice::from_raw_parts(relief_ptr, total) };
    let ocean = unsafe { slice::from_raw_parts(ocean_ptr, total) };
    let insolation = unsafe { slice::from_raw_parts(insolation_ptr, monthly_len) };
    let declination = unsafe { slice::from_raw_parts(declination_ptr, MONTHS) };
    if areas
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || xyz.iter().any(|value| !value.is_finite())
        || latitudes
            .iter()
            .any(|value| !value.is_finite() || value.abs() > PI / 2.0 + 1e-9)
        || elevation.iter().any(|value| !value.is_finite())
        || relief
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || ocean
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || insolation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || declination.iter().any(|value| !value.is_finite())
    {
        return 3;
    }
    let ocean_cells = ocean.iter().filter(|value| **value >= 0.5).count();
    if ocean_cells == 0 || ocean_cells == total {
        return 3;
    }
    if neighbors
        .iter()
        .any(|neighbor| *neighbor < 0 || *neighbor as usize >= total)
    {
        return 4;
    }

    let climate_orography = unsafe { slice::from_raw_parts_mut(climate_orography_ptr, total) };
    let temperature = unsafe { slice::from_raw_parts_mut(temperature_ptr, monthly_len) };
    let wind_xyz = unsafe { slice::from_raw_parts_mut(wind_xyz_ptr, wind_len) };
    let wind_speed = unsafe { slice::from_raw_parts_mut(wind_speed_ptr, monthly_len) };
    let precipitation = unsafe { slice::from_raw_parts_mut(precipitation_ptr, monthly_len) };
    let humidity = unsafe { slice::from_raw_parts_mut(humidity_ptr, monthly_len) };
    let snowfall = unsafe { slice::from_raw_parts_mut(snowfall_ptr, monthly_len) };
    let snowmelt = unsafe { slice::from_raw_parts_mut(snowmelt_ptr, monthly_len) };
    let snowpack = unsafe { slice::from_raw_parts_mut(snowpack_ptr, monthly_len) };
    let evaporation = unsafe { slice::from_raw_parts_mut(evaporation_ptr, monthly_len) };
    let runoff = unsafe { slice::from_raw_parts_mut(runoff_ptr, monthly_len) };
    let annual_temperature = unsafe { slice::from_raw_parts_mut(annual_temperature_ptr, total) };
    let annual_precipitation =
        unsafe { slice::from_raw_parts_mut(annual_precipitation_ptr, total) };
    let aridity = unsafe { slice::from_raw_parts_mut(aridity_ptr, total) };

    smooth_orography(total, neighbors, ocean, elevation, climate_orography);
    run_temperature(
        total,
        spinup_years as usize,
        greenhouse_offset_c,
        land_albedo,
        ocean_albedo,
        olr_intercept_w_m2,
        olr_slope_w_m2_c,
        heat_transport_w_m2,
        land_thermal_response,
        ocean_thermal_response,
        atmospheric_exchange,
        lapse_rate_c_per_km,
        neighbors,
        latitudes,
        climate_orography,
        ocean,
        insolation,
        temperature,
    );
    run_wind(
        total,
        wind_scale,
        neighbors,
        xyz,
        latitudes,
        climate_orography,
        declination,
        temperature,
        wind_xyz,
        wind_speed,
    );
    run_moisture(
        total,
        moisture_spinup_years as usize,
        moisture_steps_per_month as usize,
        moisture_advection_fraction,
        moisture_diffusion_fraction,
        orographic_factor,
        rain_shadow_factor,
        neighbors,
        xyz,
        climate_orography,
        ocean,
        temperature,
        wind_xyz,
        precipitation,
        humidity,
        evaporation,
    );
    mix_monthly_field(
        total,
        synoptic_mixing_passes as usize,
        neighbors,
        areas,
        precipitation,
    );
    run_surface_water(
        total,
        runoff_base_fraction,
        ocean,
        relief,
        temperature,
        precipitation,
        evaporation,
        snowfall,
        snowmelt,
        snowpack,
        runoff,
    );
    let stats = summarize(
        total,
        areas,
        ocean,
        temperature,
        precipitation,
        evaporation,
        snowpack,
        wind_speed,
        annual_temperature,
        annual_precipitation,
        aridity,
    );
    if !stats_out.is_null() {
        unsafe { *stats_out = stats };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moisture_capacity_increases_with_temperature() {
        assert!(moisture_capacity_mm(25.0) > moisture_capacity_mm(10.0));
        assert!(moisture_capacity_mm(10.0) > moisture_capacity_mm(-10.0));
    }

    #[test]
    fn moisture_step_fractions_preserve_monthly_relaxation() {
        let coarse = equivalent_step_fraction(0.05, 8);
        let canonical = equivalent_step_fraction(0.05, 16);
        let coarse_month = 1.0 - (1.0 - coarse).powi(8);
        let canonical_month = 1.0 - (1.0 - canonical).powi(16);
        assert!((coarse_month - canonical_month).abs() < 1e-12);
    }

    #[test]
    fn circulation_has_trades_and_westerlies() {
        let tropical = circulation_components(15.0f64.to_radians(), 0.0);
        let midlatitude = circulation_components(50.0f64.to_radians(), 0.0);
        assert!(tropical.0 < 0.0);
        assert!(midlatitude.0 > 0.0);
    }

    #[test]
    fn tangent_basis_is_orthonormal() {
        let radial = normalized([0.6, 0.4, 0.7]);
        let (east, north) = tangent_basis(radial);
        assert!(dot(radial, east).abs() < 1e-12);
        assert!(dot(radial, north).abs() < 1e-12);
        assert!(dot(east, north).abs() < 1e-12);
    }
}
