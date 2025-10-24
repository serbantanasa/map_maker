use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::VecDeque;
use std::f32;
use std::mem;
use std::slice;

const DEFAULT_PLATE_COMPONENTS: usize = 6;

#[repr(C)]
pub struct HotspotEvent {
    pub row: i32,
    pub col: i32,
    pub strength: f32,
    pub plume_factor: f32,
}

#[repr(C)]
pub struct HotspotEventArray {
    pub data: *mut HotspotEvent,
    pub len: usize,
}

#[repr(C)]
pub struct WorldAgeStats {
    pub convective_vigor: f32,
    pub mean_crust_thickness: f32,
    pub std_crust_thickness: f32,
    pub mean_isostatic_offset: f32,
    pub hotspot_count: i32,
    pub uplift_mean: f32,
    pub subsidence_mean: f32,
    pub thermal_decay_factor: f32,
    pub water_fraction: f32,
    pub uplift_sigma_gt1: f32,
    pub uplift_sigma_gt2: f32,
    pub uplift_sigma_gt3: f32,
    pub subsidence_sigma_gt1: f32,
    pub subsidence_sigma_gt2: f32,
    pub subsidence_sigma_gt3: f32,
    pub hotspot_density: f32,
}

fn clamp_unit(value: f32) -> f32 {
    value.max(0.0).min(1.0)
}

fn compute_decay_factor(world_age: f32, half_life: f32) -> f32 {
    let half_life = if half_life <= 0.0 { 1.0 } else { half_life };
    (-world_age / half_life).exp()
}

fn sample_poisson(lambda: f64, rng: &mut ChaCha8Rng) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0f64;
    while k < 64 {
        k += 1;
        let u: f64 = rng.gen();
        p *= u;
        if p <= l {
            return k - 1;
        }
    }
    lambda.max(0.0).round() as usize
}

unsafe fn slice_from_raw<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    slice::from_raw_parts(ptr, len)
}

unsafe fn slice_from_raw_mut<'a>(ptr: *mut f32, len: usize) -> &'a mut [f32] {
    slice::from_raw_parts_mut(ptr, len)
}

#[no_mangle]
pub unsafe extern "C" fn world_age_free_events(array: HotspotEventArray) {
    if !array.data.is_null() && array.len > 0 {
        let _ = Vec::from_raw_parts(array.data, array.len, array.len);
    }
}

#[no_mangle]
pub unsafe extern "C" fn world_age_run(
    height: i32,
    width: i32,
    seed: u64,
    world_age: f32,
    thermal_decay_half_life: f32,
    hotspot_scale: f32,
    isostasy_factor: f32,
    radiogenic_heat_scale: f32,
    plate_components: i32,
    plate_field_ptr: *const f32,
    convergence_ptr: *const f32,
    divergence_ptr: *const f32,
    subduction_ptr: *const f32,
    shear_ptr: *const f32,
    hotspot_ptr: *const f32,
    crust_out_ptr: *mut f32,
    isostasy_out_ptr: *mut f32,
    uplift_out_ptr: *mut f32,
    subsidence_out_ptr: *mut f32,
    compression_out_ptr: *mut f32,
    extension_out_ptr: *mut f32,
    shear_out_ptr: *mut f32,
    coastal_exposure_out_ptr: *mut f32,
    lithosphere_stiffness_out_ptr: *mut f32,
    base_ocean_mask_ptr: *mut f32,
    events_out: *mut HotspotEventArray,
    stats_out: *mut WorldAgeStats,
) -> i32 {
    if height <= 0 || width <= 0 {
        return -1;
    }
    if plate_components <= 0 {
        return -2;
    }
    if plate_field_ptr.is_null()
        || convergence_ptr.is_null()
        || divergence_ptr.is_null()
        || subduction_ptr.is_null()
        || shear_ptr.is_null()
        || hotspot_ptr.is_null()
        || crust_out_ptr.is_null()
        || isostasy_out_ptr.is_null()
        || uplift_out_ptr.is_null()
        || subsidence_out_ptr.is_null()
        || compression_out_ptr.is_null()
        || extension_out_ptr.is_null()
        || shear_out_ptr.is_null()
        || coastal_exposure_out_ptr.is_null()
        || lithosphere_stiffness_out_ptr.is_null()
        || base_ocean_mask_ptr.is_null()
        || events_out.is_null()
    {
        return -3;
    }

    let total = (height as usize) * (width as usize);
    let plate_components_usize = plate_components as usize;
    if plate_components_usize != DEFAULT_PLATE_COMPONENTS {
        // Allow different representations if the caller explicitly requests it, but
        // ensure the buffer is large enough.
        let expected = total * plate_components_usize;
        if expected == 0 {
            return -4;
        }
    }

    let plate_field = slice_from_raw(plate_field_ptr, total * plate_components_usize);
    let convergence = slice_from_raw(convergence_ptr, total);
    let divergence = slice_from_raw(divergence_ptr, total);
    let subduction = slice_from_raw(subduction_ptr, total);
    let shear = slice_from_raw(shear_ptr, total);
    let hotspot = slice_from_raw(hotspot_ptr, total);

    let crust_out = slice_from_raw_mut(crust_out_ptr, total);
    let isostasy_out = slice_from_raw_mut(isostasy_out_ptr, total);
    let uplift_out = slice_from_raw_mut(uplift_out_ptr, total);
    let subsidence_out = slice_from_raw_mut(subsidence_out_ptr, total);
    let compression_out = slice_from_raw_mut(compression_out_ptr, total);
    let extension_out = slice_from_raw_mut(extension_out_ptr, total);
    let shear_field_out = slice_from_raw_mut(shear_out_ptr, total);
    let coastal_exposure_out = slice_from_raw_mut(coastal_exposure_out_ptr, total);
    let lithosphere_stiffness_out = slice_from_raw_mut(lithosphere_stiffness_out_ptr, total);
    let base_ocean_mask_out = slice_from_raw_mut(base_ocean_mask_ptr, total);

    let decay_factor = compute_decay_factor(world_age, thermal_decay_half_life);
    let thermal_drive = clamp_unit(1.0 - decay_factor);
    let heat_scale = if radiogenic_heat_scale <= 0.0 {
        1.0
    } else {
        radiogenic_heat_scale
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC1A0_1D2C_A5E2_5319);

    let mut adjusted_thickness = vec![0.0f32; total];
    let mut sum_thickness = 0.0f64;
    let mut sum_thickness_sq = 0.0f64;
    let mut sum_velocity = 0.0f64;

    for idx in 0..total {
        let base_index = idx * plate_components_usize;
        let is_continental = plate_field[base_index + 1] >= 0.5;
        let base_thickness = plate_field[base_index + 2].max(0.1);
        let vel_u = plate_field[base_index + 4];
        let vel_v = plate_field[base_index + 5];
        let speed = (vel_u * vel_u + vel_v * vel_v).sqrt() as f64;

        let continental_boost = 1.0 + 0.18 * thermal_drive * heat_scale;
        let oceanic_emplacement = 1.0 - 0.28 * thermal_drive;
        let heat_bias = 0.05 * (heat_scale - 1.0);
        let thickness = if is_continental {
            (base_thickness * (continental_boost + heat_bias)).max(2.0)
        } else {
            (base_thickness * (oceanic_emplacement - heat_bias)).max(1.0)
        };

        adjusted_thickness[idx] = thickness;
        sum_thickness += thickness as f64;
        sum_thickness_sq += (thickness as f64) * (thickness as f64);
        sum_velocity += speed;
    }

    let total_f64 = total as f64;
    let mean_thickness = if total > 0 {
        (sum_thickness / total_f64) as f32
    } else {
        0.0
    };
    let variance = if total > 0 {
        (sum_thickness_sq / total_f64) - (mean_thickness as f64 * mean_thickness as f64)
    } else {
        0.0
    };
    let std_thickness = variance.max(0.0).sqrt() as f32;

    let mut offsets = vec![0.0f32; total];
    let mut offset_sum = 0.0f64;
    let mut uplift_sum = 0.0f64;
    let mut subsidence_sum = 0.0f64;
    let mut uplift_sq_sum = 0.0f64;
    let mut subsidence_sq_sum = 0.0f64;
    let mut ocean_cells: usize = 0;
    let mut events: Vec<HotspotEvent> = Vec::new();
    let mut shear_raw = vec![0.0f32; total];
    let mut max_shear_val = 1e-6f32;

    let base_hotspot_rate = (hotspot_scale * thermal_drive * heat_scale).max(0.0);
    let poisson_cap = 6.0f64;

    for idx in 0..total {
        let base_index = idx * plate_components_usize;
        let row = (idx / (width as usize)) as i32;
        let col = (idx % (width as usize)) as i32;
        let is_continental = plate_field[base_index + 1] >= 0.5;
        if is_continental {
            base_ocean_mask_out[idx] = 0.0;
        } else {
            base_ocean_mask_out[idx] = 1.0;
            ocean_cells += 1;
        }
        let thickness = adjusted_thickness[idx];
        crust_out[idx] = thickness;

        let deviation = thickness - mean_thickness;
        let density_factor = if is_continental { 1.0 } else { 0.65 };
        let offset = isostasy_factor * deviation * density_factor;
        offsets[idx] = offset;
        offset_sum += offset as f64;

        let conv = convergence[idx].max(0.0);
        let div = divergence[idx].max(0.0);
        let subd = subduction[idx].max(0.0);
        let hotspot_strength = clamp_unit(hotspot[idx]);

        let compression_val = ((conv + 1.4 * subd) * 0.5).min(1.0);
        compression_out[idx] = compression_val;
        let extension_raw = (div * (1.0 + 0.4 * thermal_drive)) - 0.25 * conv;
        let extension_val = clamp_unit(extension_raw);
        extension_out[idx] = extension_val;
        let shear_val = (subduction[idx].abs() + shear[idx].abs()) * (0.5 + 0.5 * thermal_drive);
        shear_raw[idx] = shear_val;
        if shear_val > max_shear_val {
            max_shear_val = shear_val;
        }

        let uplift = conv * (1.0 + 0.65 * thermal_drive) + subd * (0.35 + 0.25 * heat_scale) + hotspot_strength * 0.25 * heat_scale;
        let subsidence = div * (0.6 + 0.4 * thermal_drive) + deviation.max(0.0) * 0.18 * (1.0 - thermal_drive) + hotspot_strength * 0.05;

        uplift_out[idx] = uplift;
        subsidence_out[idx] = subsidence;
        uplift_sum += uplift as f64;
        subsidence_sum += subsidence as f64;
        uplift_sq_sum += (uplift as f64) * (uplift as f64);
        subsidence_sq_sum += (subsidence as f64) * (subsidence as f64);

        let lambda = (hotspot_strength as f64 * base_hotspot_rate as f64).min(poisson_cap);
        if lambda > 1e-6 {
            let count = sample_poisson(lambda, &mut rng);
            if count > 0 {
                let strength = (hotspot_strength * heat_scale).min(2.0);
                for _ in 0..count {
                    let plume_factor = clamp_unit(thermal_drive + rng.gen::<f32>() * 0.3);
                    events.push(HotspotEvent {
                        row,
                        col,
                        strength,
                        plume_factor,
                    });
                }
            }
        }
    }

    let mean_offset = if total > 0 {
        (offset_sum / total_f64) as f32
    } else {
        0.0
    };
    for idx in 0..total {
        isostasy_out[idx] = offsets[idx] - mean_offset;
    }

    let mean_thickness_safe = mean_thickness.max(1e-3);
    for idx in 0..total {
        let ratio = (adjusted_thickness[idx] / mean_thickness_safe).clamp(0.25, 4.0);
        let stiffness = ((ratio - 1.0) * 0.35 + 0.5).clamp(0.0, 1.0);
        lithosphere_stiffness_out[idx] = stiffness;
    }

    let width_usize = width as usize;
    let height_usize = height as usize;

    let norm_shear = if max_shear_val <= 1e-6 { 1.0 } else { max_shear_val };
    let mut shear_blurred = vec![0.0f32; total];
    for r in 0..height_usize {
        for c in 0..width_usize {
            let mut acc = 0.0f32;
            let mut count = 0.0f32;
            for dr in -1..=1 {
                let nr = r as isize + dr;
                if nr < 0 || nr >= height_usize as isize {
                    continue;
                }
                for dc in -1..=1 {
                    let nc = c as isize + dc;
                    if nc < 0 || nc >= width_usize as isize {
                        continue;
                    }
                    let nidx = (nr as usize) * width_usize + (nc as usize);
                    acc += shear_raw[nidx];
                    count += 1.0;
                }
            }
            let idx = r * width_usize + c;
            shear_blurred[idx] = if count > 0.0 { acc / count } else { shear_raw[idx] };
        }
    }
    for idx in 0..total {
        let normalized = (shear_blurred[idx] / (norm_shear + 1e-6)).clamp(0.0, 1.0);
        shear_field_out[idx] = normalized;
    }

    let mut coastline_queue: VecDeque<usize> = VecDeque::new();
    let mut distance = vec![-1i32; total];
    let neighbor_offsets = [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)];

    for r in 0..height_usize {
        for c in 0..width_usize {
            let idx = r * width_usize + c;
            let current_ocean = base_ocean_mask_out[idx] >= 0.5;
            let mut is_coast = false;
            for (dr, dc) in neighbor_offsets {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                    continue;
                }
                let nidx = (nr as usize) * width_usize + (nc as usize);
                let neighbor_ocean = base_ocean_mask_out[nidx] >= 0.5;
                if neighbor_ocean != current_ocean {
                    is_coast = true;
                    break;
                }
            }
            if is_coast {
                distance[idx] = 0;
                coastline_queue.push_back(idx);
            }
        }
    }

    if coastline_queue.is_empty() {
        for idx in 0..total {
            coastal_exposure_out[idx] = base_ocean_mask_out[idx];
        }
    } else {
        while let Some(idx) = coastline_queue.pop_front() {
            let r = idx / width_usize;
            let c = idx % width_usize;
            let current_distance = distance[idx];
            for (dr, dc) in neighbor_offsets {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= height as i32 || nc < 0 || nc >= width as i32 {
                    continue;
                }
                let nidx = (nr as usize) * width_usize + (nc as usize);
                if distance[nidx] == -1 {
                    distance[nidx] = current_distance + 1;
                    coastline_queue.push_back(nidx);
                }
            }
        }

        let decay = 6.0f32;
        for idx in 0..total {
            let dist = distance[idx];
            let exposure = if dist >= 0 {
                (-(dist as f32) / decay).exp()
            } else {
                0.0
            };
            if base_ocean_mask_out[idx] >= 0.5 {
                coastal_exposure_out[idx] = exposure;
            } else {
                coastal_exposure_out[idx] = exposure * 0.75;
            }
        }
    }

    let uplift_mean = if total > 0 {
        (uplift_sum / total_f64) as f32
    } else {
        0.0
    };
    let subsidence_mean = if total > 0 {
        (subsidence_sum / total_f64) as f32
    } else {
        0.0
    };

    let convective_vigor = if total > 0 {
        ((sum_velocity / total_f64) as f32) * (heat_scale * (0.5 + 0.5 * decay_factor))
    } else {
        0.0
    };

    let uplift_variance = if total > 0 {
        (uplift_sq_sum / total_f64) - (uplift_mean as f64 * uplift_mean as f64)
    } else {
        0.0
    };
    let subsidence_variance = if total > 0 {
        (subsidence_sq_sum / total_f64) - (subsidence_mean as f64 * subsidence_mean as f64)
    } else {
        0.0
    };
    let uplift_std = uplift_variance.max(0.0).sqrt() as f32;
    let subsidence_std = subsidence_variance.max(0.0).sqrt() as f32;

    let sigma_thresholds = [1.0f32, 2.0, 3.0];
    let mut uplift_counts = [0usize; 3];
    let mut subsidence_counts = [0usize; 3];
    if total > 0 {
        let uplift_std_eps = uplift_std.max(1e-6);
        let subsidence_std_eps = subsidence_std.max(1e-6);
        for idx in 0..total {
            let du = (uplift_out[idx] - uplift_mean) / uplift_std_eps;
            let ds = (subsidence_out[idx] - subsidence_mean) / subsidence_std_eps;
            for (i, threshold) in sigma_thresholds.iter().enumerate() {
                if du >= *threshold {
                    uplift_counts[i] += 1;
                }
                if ds >= *threshold {
                    subsidence_counts[i] += 1;
                }
            }
        }
    }

    let mut uplift_sigma_fracs = [0.0f32; 3];
    let mut subsidence_sigma_fracs = [0.0f32; 3];
    if total > 0 {
        let total_f32 = total as f32;
        for i in 0..3 {
            uplift_sigma_fracs[i] = uplift_counts[i] as f32 / total_f32;
            subsidence_sigma_fracs[i] = subsidence_counts[i] as f32 / total_f32;
        }
    }

    let water_fraction = if total > 0 {
        ocean_cells as f32 / total as f32
    } else {
        0.0
    };

    let mut events_opt = Some(events);
    let mut event_len = events_opt.as_ref().map(|v| v.len()).unwrap_or(0);
    if !events_out.is_null() {
        (*events_out).data = std::ptr::null_mut();
        (*events_out).len = 0;
        if event_len > 0 {
            if let Some(mut owned) = events_opt.take() {
                let ptr = owned.as_mut_ptr();
                mem::forget(owned);
                (*events_out).data = ptr;
                (*events_out).len = event_len;
            }
        }
    }
    if !events_out.is_null() {
        event_len = (*events_out).len;
    }

    let hotspot_density = if total > 0 {
        event_len as f32 / total as f32
    } else {
        0.0
    };

    if !stats_out.is_null() {
        (*stats_out) = WorldAgeStats {
            convective_vigor,
            mean_crust_thickness: mean_thickness,
            std_crust_thickness: std_thickness,
            mean_isostatic_offset: mean_offset,
            hotspot_count: event_len.min(i32::MAX as usize) as i32,
            uplift_mean,
            subsidence_mean,
            thermal_decay_factor: decay_factor,
            water_fraction,
            uplift_sigma_gt1: uplift_sigma_fracs[0],
            uplift_sigma_gt2: uplift_sigma_fracs[1],
            uplift_sigma_gt3: uplift_sigma_fracs[2],
            subsidence_sigma_gt1: subsidence_sigma_fracs[0],
            subsidence_sigma_gt2: subsidence_sigma_fracs[1],
            subsidence_sigma_gt3: subsidence_sigma_fracs[2],
            hotspot_density,
        };
    }

    0
}
