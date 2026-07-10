use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::f32;
use std::mem;
use std::slice;

const DEFAULT_PLATE_COMPONENTS: usize = 6;
const SPHERICAL_PLATE_COMPONENTS: usize = 7;
const D4_NEIGHBORS: usize = 4;
const SHEAR_SMOOTHING_ANGULAR_SCALE: f32 = 0.02;
const MARGIN_EFOLD_ANGULAR_SCALE: f32 = 0.075;
const MAX_DIFFUSION_PASSES: usize = 256;

#[no_mangle]
pub extern "C" fn world_age_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_world_age_abi_version() -> u32 {
    1
}

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
pub struct SphericalHotspotEvent {
    pub global_cell_id: i32,
    pub strength: f32,
    pub plume_factor: f32,
}

#[repr(C)]
pub struct SphericalHotspotEventArray {
    pub data: *mut SphericalHotspotEvent,
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
    value.clamp(0.0, 1.0)
}

fn compute_decay_factor(world_age: f32, half_life: f32) -> f32 {
    let half_life = if half_life <= 0.0 { 1.0 } else { half_life };
    (-f32::consts::LN_2 * world_age / half_life).exp()
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
/// Release hotspot events allocated by [`world_age_run`].
///
/// # Safety
///
/// `array` must be returned by a successful `world_age_run` call and must not
/// have been released previously.
pub unsafe extern "C" fn world_age_free_events(array: HotspotEventArray) {
    if !array.data.is_null() && array.len > 0 {
        let slice = std::ptr::slice_from_raw_parts_mut(array.data, array.len);
        drop(Box::from_raw(slice));
    }
}

#[no_mangle]
/// Release spherical hotspot events allocated by
/// [`world_age_run_cubed_sphere`].
///
/// # Safety
///
/// `array` must come from one successful spherical world-age call and must not
/// have been released previously.
pub unsafe extern "C" fn world_age_free_spherical_events(array: SphericalHotspotEventArray) {
    if !array.data.is_null() && array.len > 0 {
        let slice = std::ptr::slice_from_raw_parts_mut(array.data, array.len);
        drop(Box::from_raw(slice));
    }
}

#[no_mangle]
/// Compute age-dependent crustal and tectonic fields.
///
/// # Safety
///
/// Every non-null input and output pointer must reference an aligned buffer of
/// the length implied by `height`, `width`, and `plate_components`. Output
/// buffers must be writable and must not alias inputs or one another.
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

    for (idx, adjusted) in adjusted_thickness.iter_mut().enumerate() {
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

        *adjusted = thickness;
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

        let uplift = conv * (1.0 + 0.65 * thermal_drive)
            + subd * (0.35 + 0.25 * heat_scale)
            + hotspot_strength * 0.25 * heat_scale;
        let subsidence = div * (0.6 + 0.4 * thermal_drive)
            + deviation.max(0.0) * 0.18 * (1.0 - thermal_drive)
            + hotspot_strength * 0.05;

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

    let norm_shear = if max_shear_val <= 1e-6 {
        1.0
    } else {
        max_shear_val
    };
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
            shear_blurred[idx] = if count > 0.0 {
                acc / count
            } else {
                shear_raw[idx]
            };
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
                if nr < 0 || nr >= height || nc < 0 || nc >= width {
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
        coastal_exposure_out[..total].copy_from_slice(&base_ocean_mask_out[..total]);
    } else {
        while let Some(idx) = coastline_queue.pop_front() {
            let r = idx / width_usize;
            let c = idx % width_usize;
            let current_distance = distance[idx];
            for (dr, dc) in neighbor_offsets {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= height || nc < 0 || nc >= width {
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
            if let Some(owned) = events_opt.take() {
                let boxed = owned.into_boxed_slice();
                (*events_out).data = Box::into_raw(boxed) as *mut HotspotEvent;
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

fn valid_ffi_len<T>(len: usize) -> bool {
    len.checked_mul(mem::size_of::<T>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

fn diffuse_d4(field: &mut [f32], neighbors: &[i32], areas: &[f64], angular_scale: f32) {
    let target_variance = angular_scale * angular_scale;
    let min_area = areas.iter().copied().fold(f64::INFINITY, f64::min) as f32;
    let max_diffusion_time = target_variance / min_area.max(f32::MIN_POSITIVE);
    let passes = ((max_diffusion_time / 0.5).ceil() as usize).clamp(1, MAX_DIFFUSION_PASSES);
    let mut scratch = field.to_vec();
    for _ in 0..passes {
        scratch.copy_from_slice(field);
        for (cell, value) in field.iter_mut().enumerate() {
            let adjacent = &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS];
            let mean = adjacent
                .iter()
                .map(|neighbor| scratch[*neighbor as usize])
                .sum::<f32>()
                / D4_NEIGHBORS as f32;
            let blend = (target_variance / areas[cell] as f32 / passes as f32).clamp(0.0, 0.5);
            *value = scratch[cell] * (1.0 - blend) + mean * blend;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DistanceEntry {
    distance: f32,
    cell: usize,
}

impl Eq for DistanceEntry {}

impl Ord for DistanceEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

impl PartialOrd for DistanceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn margin_distances(continental: &[bool], neighbors: &[i32], areas: &[f64]) -> Vec<f32> {
    let mut distance = vec![f32::INFINITY; continental.len()];
    let mut queue = BinaryHeap::new();
    for cell in 0..continental.len() {
        let adjacent = &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS];
        if adjacent
            .iter()
            .any(|neighbor| continental[*neighbor as usize] != continental[cell])
        {
            distance[cell] = 0.0;
            queue.push(DistanceEntry {
                distance: 0.0,
                cell,
            });
        }
    }
    while let Some(entry) = queue.pop() {
        if entry.distance > distance[entry.cell] {
            continue;
        }
        let cell_scale = (areas[entry.cell] as f32).sqrt();
        for &neighbor in &neighbors[entry.cell * D4_NEIGHBORS..(entry.cell + 1) * D4_NEIGHBORS] {
            let adjacent = neighbor as usize;
            let edge_scale = 0.5 * (cell_scale + (areas[adjacent] as f32).sqrt());
            let candidate = entry.distance + edge_scale;
            if candidate < distance[adjacent] {
                distance[adjacent] = candidate;
                queue.push(DistanceEntry {
                    distance: candidate,
                    cell: adjacent,
                });
            }
        }
    }
    distance
}

fn weighted_sample_index(weights: &[f64], total_weight: f64, rng: &mut ChaCha8Rng) -> usize {
    let target = rng.gen::<f64>() * total_weight;
    let mut cumulative = 0.0f64;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if cumulative >= target {
            return index;
        }
    }
    weights.len().saturating_sub(1)
}

#[no_mangle]
/// Initialize age-conditioned crustal state on canonical cubed-sphere cells.
///
/// The proto-ocean output classifies oceanic crust and is not a final water or
/// sea-level solution. Coastal exposure is continental-margin proximity.
///
/// # Safety
///
/// Inputs and outputs must be aligned, correctly sized, non-overlapping
/// buffers. Areas contain one positive `f64` per cell; neighbors contain four
/// valid global IDs per cell; the plate field has seven `f32` components per
/// cell; scalar fields contain one `f32` per cell. Event and stats pointers may
/// be null, though a null event pointer discards generated events.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn world_age_run_cubed_sphere(
    cell_count: i32,
    seed: u64,
    world_age: f32,
    thermal_decay_half_life: f32,
    hotspot_scale: f32,
    isostasy_factor: f32,
    radiogenic_heat_scale: f32,
    plate_components: i32,
    area_ptr: *const f64,
    neighbors_ptr: *const i32,
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
    margin_proximity_out_ptr: *mut f32,
    lithosphere_stiffness_out_ptr: *mut f32,
    proto_ocean_mask_ptr: *mut f32,
    events_out: *mut SphericalHotspotEventArray,
    stats_out: *mut WorldAgeStats,
) -> i32 {
    if cell_count <= 0
        || plate_components < SPHERICAL_PLATE_COMPONENTS as i32
        || !world_age.is_finite()
        || world_age < 0.0
        || !thermal_decay_half_life.is_finite()
        || thermal_decay_half_life <= 0.0
        || !hotspot_scale.is_finite()
        || hotspot_scale < 0.0
        || !isostasy_factor.is_finite()
        || !radiogenic_heat_scale.is_finite()
        || radiogenic_heat_scale <= 0.0
        || area_ptr.is_null()
        || neighbors_ptr.is_null()
        || plate_field_ptr.is_null()
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
        || margin_proximity_out_ptr.is_null()
        || lithosphere_stiffness_out_ptr.is_null()
        || proto_ocean_mask_ptr.is_null()
    {
        return 1;
    }
    let total = cell_count as usize;
    let components = plate_components as usize;
    let Some(plate_len) = total.checked_mul(components) else {
        return 2;
    };
    let Some(neighbor_len) = total.checked_mul(D4_NEIGHBORS) else {
        return 2;
    };
    if !valid_ffi_len::<f64>(total)
        || !valid_ffi_len::<i32>(neighbor_len)
        || !valid_ffi_len::<f32>(plate_len)
        || !valid_ffi_len::<f32>(total)
    {
        return 2;
    }

    let areas = slice::from_raw_parts(area_ptr, total);
    let neighbors = slice::from_raw_parts(neighbors_ptr, neighbor_len);
    let plate_field = slice::from_raw_parts(plate_field_ptr, plate_len);
    let convergence = slice::from_raw_parts(convergence_ptr, total);
    let divergence = slice::from_raw_parts(divergence_ptr, total);
    let subduction = slice::from_raw_parts(subduction_ptr, total);
    let shear = slice::from_raw_parts(shear_ptr, total);
    let hotspot = slice::from_raw_parts(hotspot_ptr, total);
    let crust_out = slice::from_raw_parts_mut(crust_out_ptr, total);
    let isostasy_out = slice::from_raw_parts_mut(isostasy_out_ptr, total);
    let uplift_out = slice::from_raw_parts_mut(uplift_out_ptr, total);
    let subsidence_out = slice::from_raw_parts_mut(subsidence_out_ptr, total);
    let compression_out = slice::from_raw_parts_mut(compression_out_ptr, total);
    let extension_out = slice::from_raw_parts_mut(extension_out_ptr, total);
    let shear_out = slice::from_raw_parts_mut(shear_out_ptr, total);
    let margin_out = slice::from_raw_parts_mut(margin_proximity_out_ptr, total);
    let stiffness_out = slice::from_raw_parts_mut(lithosphere_stiffness_out_ptr, total);
    let proto_ocean_out = slice::from_raw_parts_mut(proto_ocean_mask_ptr, total);

    let mut total_area = 0.0f64;
    for cell in 0..total {
        let area = areas[cell];
        let base = cell * components;
        if !area.is_finite()
            || area <= 0.0
            || plate_field[base..base + SPHERICAL_PLATE_COMPONENTS]
                .iter()
                .any(|value| !value.is_finite())
            || [
                convergence[cell],
                divergence[cell],
                subduction[cell],
                shear[cell],
                hotspot[cell],
            ]
            .iter()
            .any(|value| !value.is_finite())
        {
            return 3;
        }
        let mut unique = [
            neighbors[cell * 4],
            neighbors[cell * 4 + 1],
            neighbors[cell * 4 + 2],
            neighbors[cell * 4 + 3],
        ];
        unique.sort_unstable();
        if unique.windows(2).any(|pair| pair[0] == pair[1])
            || unique
                .iter()
                .any(|neighbor| *neighbor < 0 || *neighbor as usize >= total)
        {
            return 4;
        }
        total_area += area;
    }
    if !total_area.is_finite() || total_area <= 0.0 {
        return 3;
    }

    let residual_heat = compute_decay_factor(world_age, thermal_decay_half_life);
    let heat_scale = radiogenic_heat_scale.clamp(0.1, 4.0);
    let convective_heat = ((0.28 + 0.72 * residual_heat) * heat_scale).clamp(0.05, 2.5);
    let cooling = (1.0 - residual_heat).clamp(0.0, 1.0);
    let mut continental = vec![false; total];
    let mut buoyancy = vec![0.0f32; total];
    let mut shear_smoothed = vec![0.0f32; total];
    let mut thickness_sum = 0.0f64;
    let mut thickness_sq_sum = 0.0f64;
    let mut buoyancy_sum = 0.0f64;
    let mut speed_sum = 0.0f64;
    let mut ocean_area = 0.0f64;
    let mantle_density = 3.30f32;

    for cell in 0..total {
        let base = cell * components;
        continental[cell] = plate_field[base + 1] >= 0.5;
        let base_thickness = plate_field[base + 2].max(0.1);
        let density = plate_field[base + 3].clamp(2.4, 3.5);
        let thickness_factor = if continental[cell] {
            0.98 + 0.08 * convective_heat
        } else {
            0.84 + 0.22 * convective_heat
        };
        let thickness = (base_thickness * thickness_factor).max(0.5);
        crust_out[cell] = thickness;
        let root = thickness * (mantle_density - density).max(-0.2) / mantle_density;
        buoyancy[cell] = root * isostasy_factor;
        shear_smoothed[cell] = (shear[cell].abs() + 0.45 * subduction[cell].max(0.0))
            * (0.65 + 0.35 * convective_heat);

        let area = areas[cell];
        thickness_sum += thickness as f64 * area;
        thickness_sq_sum += thickness as f64 * thickness as f64 * area;
        buoyancy_sum += buoyancy[cell] as f64 * area;
        let vx = plate_field[base + 4] as f64;
        let vy = plate_field[base + 5] as f64;
        let vz = plate_field[base + 6] as f64;
        speed_sum += (vx * vx + vy * vy + vz * vz).sqrt() * area;
        if !continental[cell] {
            proto_ocean_out[cell] = 1.0;
            ocean_area += area;
        } else {
            proto_ocean_out[cell] = 0.0;
        }
    }

    let mean_thickness = (thickness_sum / total_area) as f32;
    let thickness_variance =
        (thickness_sq_sum / total_area - mean_thickness as f64 * mean_thickness as f64).max(0.0);
    let mean_buoyancy = (buoyancy_sum / total_area) as f32;
    let mut uplift_sum = 0.0f64;
    let mut subsidence_sum = 0.0f64;
    let mut uplift_sq_sum = 0.0f64;
    let mut subsidence_sq_sum = 0.0f64;

    for cell in 0..total {
        isostasy_out[cell] = buoyancy[cell] - mean_buoyancy;
        let conv = convergence[cell].max(0.0);
        let div = divergence[cell].max(0.0);
        let subd = subduction[cell].clamp(0.0, 1.0);
        let hot = hotspot[cell].clamp(0.0, 1.0);
        compression_out[cell] = (0.55 * conv + 0.7 * subd).clamp(0.0, 1.0);
        extension_out[cell] = (div * (0.8 + 0.35 * convective_heat) - 0.2 * conv).clamp(0.0, 1.0);
        let stiffness_base = if continental[cell] { 0.58 } else { 0.36 };
        stiffness_out[cell] = (stiffness_base
            + 0.24 * cooling
            + 0.12 * (crust_out[cell] / mean_thickness.max(0.1) - 1.0))
            .clamp(0.0, 1.0);
        let uplift = conv * (0.8 + 0.45 * convective_heat)
            + subd * (0.45 + 0.2 * convective_heat)
            + hot * 0.22 * convective_heat;
        let oceanic_cooling = if continental[cell] {
            0.0
        } else {
            0.08 * cooling
        };
        let subsidence = div * (0.7 + 0.25 * convective_heat) + oceanic_cooling + hot * 0.025;
        uplift_out[cell] = uplift;
        subsidence_out[cell] = subsidence;
        let area = areas[cell];
        uplift_sum += uplift as f64 * area;
        subsidence_sum += subsidence as f64 * area;
        uplift_sq_sum += uplift as f64 * uplift as f64 * area;
        subsidence_sq_sum += subsidence as f64 * subsidence as f64 * area;
    }

    diffuse_d4(
        &mut shear_smoothed,
        neighbors,
        areas,
        SHEAR_SMOOTHING_ANGULAR_SCALE,
    );
    let max_shear = shear_smoothed
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6);
    for (output, value) in shear_out.iter_mut().zip(shear_smoothed.iter()) {
        *output = (*value / max_shear).clamp(0.0, 1.0);
    }

    let distances = margin_distances(&continental, neighbors, areas);
    for cell in 0..total {
        margin_out[cell] = if distances[cell].is_finite() {
            (-distances[cell] / MARGIN_EFOLD_ANGULAR_SCALE).exp()
        } else {
            0.0
        };
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC1A0_1D2C_A5E2_5319);
    let event_weights: Vec<f64> = hotspot
        .iter()
        .zip(areas.iter())
        .map(|(value, area)| value.clamp(0.0, 1.0) as f64 * *area)
        .collect();
    let total_event_weight: f64 = event_weights.iter().sum();
    let mean_hotspot = total_event_weight / total_area;
    let expected_events =
        hotspot_scale as f64 * (4.0 + 28.0 * convective_heat as f64) * (0.5 + 2.0 * mean_hotspot);
    let event_count = if total_event_weight > 0.0 {
        sample_poisson(expected_events, &mut rng)
    } else {
        0
    };
    let mut events = Vec::with_capacity(event_count);
    for _ in 0..event_count {
        let cell = weighted_sample_index(&event_weights, total_event_weight, &mut rng);
        events.push(SphericalHotspotEvent {
            global_cell_id: cell as i32,
            strength: (hotspot[cell].clamp(0.0, 1.0) * (0.7 + 0.6 * convective_heat)).min(2.0),
            plume_factor: clamp_unit(convective_heat + rng.gen::<f32>() * 0.2),
        });
    }

    let uplift_mean = (uplift_sum / total_area) as f32;
    let subsidence_mean = (subsidence_sum / total_area) as f32;
    let uplift_std = (uplift_sq_sum / total_area - uplift_mean as f64 * uplift_mean as f64)
        .max(0.0)
        .sqrt() as f32;
    let subsidence_std = (subsidence_sq_sum / total_area
        - subsidence_mean as f64 * subsidence_mean as f64)
        .max(0.0)
        .sqrt() as f32;
    let mut uplift_sigma_area = [0.0f64; 3];
    let mut subsidence_sigma_area = [0.0f64; 3];
    for cell in 0..total {
        let uplift_z = (uplift_out[cell] - uplift_mean) / uplift_std.max(1e-6);
        let subsidence_z = (subsidence_out[cell] - subsidence_mean) / subsidence_std.max(1e-6);
        for (index, threshold) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            if uplift_z >= *threshold {
                uplift_sigma_area[index] += areas[cell];
            }
            if subsidence_z >= *threshold {
                subsidence_sigma_area[index] += areas[cell];
            }
        }
    }

    let event_len = events.len();
    if !events_out.is_null() {
        (*events_out).data = std::ptr::null_mut();
        (*events_out).len = 0;
        if event_len > 0 {
            let boxed = events.into_boxed_slice();
            (*events_out).data = Box::into_raw(boxed) as *mut SphericalHotspotEvent;
            (*events_out).len = event_len;
        }
    }
    if !stats_out.is_null() {
        (*stats_out) = WorldAgeStats {
            convective_vigor: (speed_sum / total_area) as f32 * convective_heat,
            mean_crust_thickness: mean_thickness,
            std_crust_thickness: thickness_variance.sqrt() as f32,
            mean_isostatic_offset: (isostasy_out
                .iter()
                .zip(areas.iter())
                .map(|(value, area)| *value as f64 * *area)
                .sum::<f64>()
                / total_area) as f32,
            hotspot_count: event_len.min(i32::MAX as usize) as i32,
            uplift_mean,
            subsidence_mean,
            thermal_decay_factor: residual_heat,
            water_fraction: (ocean_area / total_area) as f32,
            uplift_sigma_gt1: (uplift_sigma_area[0] / total_area) as f32,
            uplift_sigma_gt2: (uplift_sigma_area[1] / total_area) as f32,
            uplift_sigma_gt3: (uplift_sigma_area[2] / total_area) as f32,
            subsidence_sigma_gt1: (subsidence_sigma_area[0] / total_area) as f32,
            subsidence_sigma_gt2: (subsidence_sigma_area[1] / total_area) as f32,
            subsidence_sigma_gt3: (subsidence_sigma_area[2] / total_area) as f32,
            hotspot_density: event_len as f32 / total_area as f32,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cyclic_d4_neighbors(cell_count: usize) -> Vec<i32> {
        let mut neighbors = Vec::with_capacity(cell_count * D4_NEIGHBORS);
        for cell in 0..cell_count {
            for offset in 1..=D4_NEIGHBORS {
                neighbors.push(((cell + offset) % cell_count) as i32);
            }
        }
        neighbors
    }

    #[test]
    fn margin_distances_cross_graph_boundaries() {
        let continental = [true, true, true, false, false, false];
        let neighbors = cyclic_d4_neighbors(continental.len());
        let areas = [0.1f64; 6];
        let distances = margin_distances(&continental, &neighbors, &areas);

        assert_eq!(distances.len(), continental.len());
        assert!(distances.iter().all(|distance| *distance == 0.0));
    }

    #[test]
    fn spherical_kernel_closes_area_weighted_isostasy_and_owns_events() {
        let cell_count = 6usize;
        let areas = [0.7f64, 1.1, 1.4, 2.0, 2.3, 3.0];
        let neighbors = cyclic_d4_neighbors(cell_count);
        let mut plate_field = vec![0.0f32; cell_count * SPHERICAL_PLATE_COMPONENTS];
        for cell in 0..cell_count {
            let base = cell * SPHERICAL_PLATE_COMPONENTS;
            let continental = cell < 3;
            plate_field[base] = cell as f32;
            plate_field[base + 1] = if continental { 1.0 } else { 0.0 };
            plate_field[base + 2] = if continental { 35.0 } else { 7.0 };
            plate_field[base + 3] = if continental { 2.75 } else { 3.0 };
            plate_field[base + 4] = 0.02 * cell as f32;
            plate_field[base + 5] = 0.01;
            plate_field[base + 6] = -0.01;
        }
        let convergence = vec![0.2f32; cell_count];
        let divergence = vec![0.1f32; cell_count];
        let subduction = vec![0.15f32; cell_count];
        let shear = vec![0.08f32; cell_count];
        let hotspot = vec![1.0f32; cell_count];
        let mut outputs = vec![vec![0.0f32; cell_count]; 10];
        let mut events = SphericalHotspotEventArray {
            data: std::ptr::null_mut(),
            len: 0,
        };
        let mut stats = WorldAgeStats {
            convective_vigor: 0.0,
            mean_crust_thickness: 0.0,
            std_crust_thickness: 0.0,
            mean_isostatic_offset: 0.0,
            hotspot_count: 0,
            uplift_mean: 0.0,
            subsidence_mean: 0.0,
            thermal_decay_factor: 0.0,
            water_fraction: 0.0,
            uplift_sigma_gt1: 0.0,
            uplift_sigma_gt2: 0.0,
            uplift_sigma_gt3: 0.0,
            subsidence_sigma_gt1: 0.0,
            subsidence_sigma_gt2: 0.0,
            subsidence_sigma_gt3: 0.0,
            hotspot_density: 0.0,
        };

        let result = unsafe {
            world_age_run_cubed_sphere(
                cell_count as i32,
                17,
                4.1,
                1.8,
                20.0,
                0.6,
                1.0,
                SPHERICAL_PLATE_COMPONENTS as i32,
                areas.as_ptr(),
                neighbors.as_ptr(),
                plate_field.as_ptr(),
                convergence.as_ptr(),
                divergence.as_ptr(),
                subduction.as_ptr(),
                shear.as_ptr(),
                hotspot.as_ptr(),
                outputs[0].as_mut_ptr(),
                outputs[1].as_mut_ptr(),
                outputs[2].as_mut_ptr(),
                outputs[3].as_mut_ptr(),
                outputs[4].as_mut_ptr(),
                outputs[5].as_mut_ptr(),
                outputs[6].as_mut_ptr(),
                outputs[7].as_mut_ptr(),
                outputs[8].as_mut_ptr(),
                outputs[9].as_mut_ptr(),
                &mut events,
                &mut stats,
            )
        };

        assert_eq!(result, 0);
        assert!(outputs.iter().flatten().all(|value| value.is_finite()));
        let total_area: f64 = areas.iter().sum();
        let weighted_offset = outputs[1]
            .iter()
            .zip(areas.iter())
            .map(|(value, area)| *value as f64 * *area)
            .sum::<f64>()
            / total_area;
        assert!(weighted_offset.abs() < 1e-6);
        assert!(stats.mean_isostatic_offset.abs() < 1e-6);
        assert_eq!(events.len, stats.hotspot_count as usize);
        assert!(events.len > 0);
        let generated = unsafe { slice::from_raw_parts(events.data, events.len) };
        assert!(generated
            .iter()
            .all(|event| (0..cell_count as i32).contains(&event.global_cell_id)));
        unsafe { world_age_free_spherical_events(events) };
    }

    #[test]
    fn residual_heat_decreases_with_age() {
        assert!((compute_decay_factor(1.8, 1.8) - 0.5).abs() < 1e-6);
        assert!(compute_decay_factor(4.5, 1.8) < compute_decay_factor(1.0, 1.8));
    }
}
