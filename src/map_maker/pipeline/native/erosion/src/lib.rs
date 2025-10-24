use std::mem;
use std::slice;

const PLATE_COMPONENTS: usize = 6;

#[repr(C)]
pub struct IterDiagnostic {
    pub step: i32,
    pub mean_elevation: f32,
    pub mass_removed: f32,
    pub mass_deposited: f32,
}

#[repr(C)]
pub struct IterDiagnosticArray {
    pub data: *mut IterDiagnostic,
    pub len: usize,
}

#[repr(C)]
pub struct ErosionStats {
    pub total_mass_removed: f32,
    pub total_mass_deposited: f32,
    pub sediment_mass: f32,
    pub final_mean_elevation: f32,
    pub final_min_elevation: f32,
    pub final_max_elevation: f32,
    pub mass_residual: f32,
    pub steps_run: i32,
}

fn clamp_unit(value: f32) -> f32 {
    value.max(0.0).min(1.0)
}

unsafe fn read_slice<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    slice::from_raw_parts(ptr, len)
}

unsafe fn read_slice_mut<'a>(ptr: *mut f32, len: usize) -> &'a mut [f32] {
    slice::from_raw_parts_mut(ptr, len)
}

#[no_mangle]
pub unsafe extern "C" fn erosion_free_diagnostics(array: IterDiagnosticArray) {
    if !array.data.is_null() && array.len > 0 {
        let _ = Vec::from_raw_parts(array.data, array.len, array.len);
    }
}

#[no_mangle]
pub unsafe extern "C" fn erosion_run(
    height: i32,
    width: i32,
    steps: i32,
    dt: f32,
    stream_power_k: f32,
    sediment_capacity: f32,
    coastal_wave_energy: f32,
    plate_components: i32,
    plate_field_ptr: *const f32,
    crust_ptr: *const f32,
    isostasy_ptr: *const f32,
    uplift_ptr: *const f32,
    subsidence_ptr: *const f32,
    compression_ptr: *const f32,
    extension_ptr: *const f32,
    shear_ptr: *const f32,
    coastal_ptr: *const f32,
    lithosphere_ptr: *const f32,
    ocean_mask_ptr: *const f32,
    hotspot_influence_ptr: *const f32,
    elevation_out_ptr: *mut f32,
    sediment_out_ptr: *mut f32,
    incision_out_ptr: *mut f32,
    diagnostics_out: *mut IterDiagnosticArray,
    stats_out: *mut ErosionStats,
) -> i32 {
    if height <= 0 || width <= 0 {
        return -1;
    }
    if steps <= 0 {
        return -2;
    }
    if dt <= 0.0 {
        return -3;
    }
    if plate_components <= 0 {
        return -4;
    }
    if plate_field_ptr.is_null()
        || crust_ptr.is_null()
        || isostasy_ptr.is_null()
        || uplift_ptr.is_null()
        || subsidence_ptr.is_null()
        || compression_ptr.is_null()
        || extension_ptr.is_null()
        || shear_ptr.is_null()
        || coastal_ptr.is_null()
        || lithosphere_ptr.is_null()
        || ocean_mask_ptr.is_null()
        || hotspot_influence_ptr.is_null()
        || elevation_out_ptr.is_null()
        || sediment_out_ptr.is_null()
        || incision_out_ptr.is_null()
        || diagnostics_out.is_null()
    {
        return -5;
    }

    let h = height as usize;
    let w = width as usize;
    let total = h * w;
    let plate_components_usize = plate_components as usize;
    if plate_components_usize != PLATE_COMPONENTS {
        if plate_components_usize == 0 {
            return -6;
        }
        let expected = total * plate_components_usize;
        let _ = read_slice(plate_field_ptr, expected);
    }

    let plate_field = read_slice(plate_field_ptr, total * plate_components_usize);
    let crust = read_slice(crust_ptr, total);
    let isostasy = read_slice(isostasy_ptr, total);
    let uplift = read_slice(uplift_ptr, total);
    let subsidence = read_slice(subsidence_ptr, total);
    let compression = read_slice(compression_ptr, total);
    let extension = read_slice(extension_ptr, total);
    let shear = read_slice(shear_ptr, total);
    let coastal = read_slice(coastal_ptr, total);
    let lithosphere = read_slice(lithosphere_ptr, total);
    let ocean_mask = read_slice(ocean_mask_ptr, total);
    let hotspot_influence = read_slice(hotspot_influence_ptr, total);

    let elevation = read_slice_mut(elevation_out_ptr, total);
    let sediment = read_slice_mut(sediment_out_ptr, total);
    let incision = read_slice_mut(incision_out_ptr, total);

    for idx in 0..total {
        let crust_term = crust[idx] * 0.12;
        let litho_term = (1.0 - clamp_unit(lithosphere[idx])) * 0.05;
        let plate_idx = idx * plate_components_usize;
        let vel_u = plate_field[plate_idx + 4];
        let vel_v = plate_field[plate_idx + 5];
        let velocity_term = (vel_u * vel_u + vel_v * vel_v).sqrt() * 0.02;
        elevation[idx] = isostasy[idx] + crust_term + litho_term + hotspot_influence[idx] * 0.08 + velocity_term;
    }
    sediment.fill(0.0);
    incision.fill(0.0);

    let mut diagnostics: Vec<IterDiagnostic> = Vec::with_capacity(steps as usize);

    let mut total_mass_removed = 0.0f64;
    let mut total_mass_deposited = 0.0f64;

    let weight_n = 1.0f32;
    let weight_s = 1.0f32;
    let weight_e = 1.0f32;
    let weight_w = 1.0f32;

    for step in 0..steps {
        for idx in 0..total {
            let net = uplift[idx] - subsidence[idx];
            elevation[idx] += net * dt;
        }

        let mut removal_step = 0.0f64;
        let mut deposition_step = 0.0f64;

        let elevation_snapshot = elevation.to_vec();

        for row in 0..h {
            for col in 0..w {
                let idx = row * w + col;
                let center = elevation_snapshot[idx];
                let north = if row > 0 {
                    elevation_snapshot[idx - w]
                } else {
                    center
                };
                let south = if row + 1 < h {
                    elevation_snapshot[idx + w]
                } else {
                    center
                };
                let west = if col > 0 {
                    elevation_snapshot[idx - 1]
                } else {
                    center
                };
                let east = if col + 1 < w {
                    elevation_snapshot[idx + 1]
                } else {
                    center
                };

                let dx = (east - west) * 0.5;
                let dy = (south - north) * 0.5;
                let slope = (dx * dx + dy * dy).sqrt();

                let laplacian = (north * weight_n + south * weight_s + east * weight_e + west * weight_w - 4.0 * center)
                    * 0.25;

                let compression_factor = 1.0 + compression[idx] * 0.75;
                let shear_factor = 1.0 + shear[idx] * 0.5;
                let litho_factor = 1.0 / (0.4 + lithosphere[idx].max(0.05));
                let hotspot_factor = 1.0 + hotspot_influence[idx] * 0.3;
                let flow_factor = slope.max(1e-4);

                let erosion_potential = stream_power_k
                    * flow_factor
                    * compression_factor
                    * shear_factor
                    * (1.0 + coastal[idx] * coastal_wave_energy.max(0.0))
                    * hotspot_factor
                    * litho_factor
                    * dt;

                let ocean_bias = if ocean_mask[idx] >= 0.5 { 1.0 + coastal[idx] * 0.5 } else { 1.0 };
                let capacity = sediment_capacity * (1.0 + extension[idx] * 0.6) * ocean_bias;

                let diffusive = (-laplacian).max(0.0) * 0.2 * dt;
                let raw_removal = (diffusive + erosion_potential).max(0.0);
                let removal = (raw_removal - capacity).max(0.0);

                if removal > 0.0 {
                    elevation[idx] -= removal;
                    sediment[idx] += removal;
                    incision[idx] += removal;
                    removal_step += removal as f64;
                } else {
                    let desired_deposit = (capacity - raw_removal).max(0.0);
                    if desired_deposit > 0.0 {
                        let available = sediment[idx];
                        let deposit = desired_deposit.min(available);
                        elevation[idx] += deposit;
                        sediment[idx] -= deposit;
                        deposition_step += deposit as f64;
                    }
                }
            }
        }

        total_mass_removed += removal_step;
        total_mass_deposited += deposition_step;

        let mean_elevation = if total > 0 {
            elevation.iter().fold(0.0f64, |acc, &v| acc + v as f64) / total as f64
        } else {
            0.0
        } as f32;

        diagnostics.push(IterDiagnostic {
            step: step,
            mean_elevation,
            mass_removed: removal_step as f32,
            mass_deposited: deposition_step as f32,
        });
    }

    let mut min_elev = f32::INFINITY;
    let mut max_elev = f32::NEG_INFINITY;
    let mut sum_elev = 0.0f64;
    let mut sediment_mass = 0.0f64;
    for idx in 0..total {
        let value = elevation[idx];
        if value < min_elev {
            min_elev = value;
        }
        if value > max_elev {
            max_elev = value;
        }
        sum_elev += value as f64;
        sediment_mass += sediment[idx] as f64;
    }

    let final_mean = if total > 0 {
        sum_elev / total as f64
    } else {
        0.0
    } as f32;

    let total_deposited = total_mass_deposited + sediment_mass;
    let residual = total_mass_removed - total_deposited;

    if !diagnostics_out.is_null() {
        (*diagnostics_out).data = std::ptr::null_mut();
        (*diagnostics_out).len = 0;
        if !diagnostics.is_empty() {
            let mut owned = diagnostics;
            let len = owned.len();
            let ptr = owned.as_mut_ptr();
            mem::forget(owned);
            (*diagnostics_out).data = ptr;
            (*diagnostics_out).len = len;
        }
    }

    if !stats_out.is_null() {
        (*stats_out) = ErosionStats {
            total_mass_removed: total_mass_removed as f32,
            total_mass_deposited: total_deposited as f32,
            sediment_mass: sediment_mass as f32,
            final_mean_elevation: final_mean,
            final_min_elevation: min_elev,
            final_max_elevation: max_elev,
            mass_residual: residual as f32,
            steps_run: steps,
        };
    }

    0
}
