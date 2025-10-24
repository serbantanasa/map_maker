use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::cmp::Ordering;
use std::f32::consts::PI;

const PLATE_COMPONENTS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlateType {
    Continental,
    Oceanic,
}

#[derive(Clone)]
struct PlateSeed {
    position: [f32; 2],
    plate_type: PlateType,
}

#[derive(Default, Clone)]
struct PlateAccum {
    position: [f32; 2],
    count: f32,
}

struct BoundaryData {
    convergence: Vec<f32>,
    divergence: Vec<f32>,
    shear: Vec<f32>,
    subduction: Vec<f32>,
    convergence_sum: f64,
    divergence_sum: f64,
    shear_sum: f64,
    subduction_sum: f64,
}

#[repr(C)]
pub struct TectonicsStats {
    pub plate_count: i32,
    pub continental_fraction: f64,
    pub velocity_mean: f64,
    pub velocity_std: f64,
    pub hotspot_mean: f64,
    pub boundary_metric_mean: f64,
    pub convergence_sum: f64,
    pub divergence_sum: f64,
    pub shear_sum: f64,
    pub subduction_mean: f64,
    pub hotspot_count: i32,
}

fn wrap_distance_sq(
    a: [f32; 2],
    b: [f32; 2],
    width: f32,
    height: f32,
    wrap_x: bool,
    wrap_y: bool,
) -> f32 {
    let mut dx = a[0] - b[0];
    if wrap_x {
        if dx > width * 0.5 {
            dx -= width;
        } else if dx < -width * 0.5 {
            dx += width;
        }
    }
    let mut dy = a[1] - b[1];
    if wrap_y {
        if dy > height * 0.5 {
            dy -= height;
        } else if dy < -height * 0.5 {
            dy += height;
        }
    }
    dx * dx + dy * dy
}

fn poisson_sample(
    height: usize,
    width: usize,
    seed: u64,
    count: usize,
    wrap_x: bool,
    wrap_y: bool,
) -> Vec<PlateSeed> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6C8E_9CF5_1234_5678);
    let total_cells = (height * width).max(1);
    let mut min_dist_sq = (((total_cells as f32) / (count.max(1) as f32)).sqrt() * 0.75).powi(2).max(4.0);
    let mut seeds: Vec<PlateSeed> = Vec::with_capacity(count);
    let mut attempts: usize = 0;
    let width_f = width as f32;
    let height_f = height as f32;

    while seeds.len() < count && attempts < count.max(1) * 6000 {
        attempts += 1;
        let row = rng.gen_range(0..height);
        let col = rng.gen_range(0..width);
        let position = [col as f32 + 0.5, row as f32 + 0.5];
        let ok = seeds.iter().all(|s| {
            wrap_distance_sq(position, s.position, width_f, height_f, wrap_x, wrap_y) >= min_dist_sq
        });
        if ok {
            seeds.push(PlateSeed {
                position,
                plate_type: PlateType::Oceanic,
            });
        }
        if attempts % (count.max(1) * 128) == 0 {
            min_dist_sq = (min_dist_sq * 0.82).max(1.0);
        }
    }

    while seeds.len() < count {
        let row = rng.gen_range(0..height);
        let col = rng.gen_range(0..width);
        seeds.push(PlateSeed {
            position: [col as f32 + 0.5, row as f32 + 0.5],
            plate_type: PlateType::Oceanic,
        });
    }

    seeds
}

fn assign_cells_into(
    seeds: &[PlateSeed],
    assignments: &mut [usize],
    height: usize,
    width: usize,
    wrap_x: bool,
    wrap_y: bool,
) {
    let width_f = width as f32;
    let height_f = height as f32;
    assignments
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, row_assign)| {
            let y = row as f32 + 0.5;
            for col in 0..width {
                let x = col as f32 + 0.5;
                let mut best_idx = 0usize;
                let mut best_dist = f32::MAX;
                for (seed_idx, seed) in seeds.iter().enumerate() {
                    let dist = wrap_distance_sq(
                        [x, y],
                        seed.position,
                        width_f,
                        height_f,
                        wrap_x,
                        wrap_y,
                    );
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = seed_idx;
                    }
                }
                row_assign[col] = best_idx;
            }
        });
}

fn recentre_seeds(
    seeds: &mut [PlateSeed],
    assignments: &[usize],
    height: usize,
    width: usize,
    wrap_x: bool,
    wrap_y: bool,
    rng: &mut ChaCha8Rng,
) -> Vec<PlateAccum> {
    let plate_count = seeds.len();
    let mut accum = vec![PlateAccum::default(); plate_count];
    let width_f = width as f32;
    let height_f = height as f32;

    for (idx, &plate) in assignments.iter().enumerate() {
        if plate >= plate_count {
            continue;
        }
        let row = (idx / width) as f32 + 0.5;
        let col = (idx % width) as f32 + 0.5;
        let entry = &mut accum[plate];
        entry.position[0] += col;
        entry.position[1] += row;
        entry.count += 1.0;
    }

    for (plate_id, seed) in seeds.iter_mut().enumerate() {
        let data = &accum[plate_id];
        if data.count > 1.0 {
            seed.position[0] = data.position[0] / data.count;
            seed.position[1] = data.position[1] / data.count;
            if wrap_x {
                seed.position[0] = (seed.position[0] + width_f).rem_euclid(width_f);
            } else {
                seed.position[0] = seed.position[0].clamp(0.5, width_f - 0.5);
            }
            if wrap_y {
                seed.position[1] = (seed.position[1] + height_f).rem_euclid(height_f);
            } else {
                seed.position[1] = seed.position[1].clamp(0.5, height_f - 0.5);
            }
        } else {
            let row = rng.gen_range(0..height);
            let col = rng.gen_range(0..width);
            seed.position = [col as f32 + 0.5, row as f32 + 0.5];
        }
    }

    accum
}

fn classify_plate_types(accum: &[PlateAccum], target_fraction: f32) -> Vec<PlateType> {
    if accum.is_empty() {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..accum.len()).collect();
    indices.sort_by(|a, b| {
        accum[*b]
            .count
            .partial_cmp(&accum[*a].count)
            .unwrap_or(Ordering::Equal)
    });

    let total_area: f32 = accum.iter().map(|a| a.count).sum::<f32>().max(1.0);
    let mut remaining_area = (target_fraction.clamp(0.0, 1.0) * total_area).max(0.0);
    let mut types = vec![PlateType::Oceanic; accum.len()];

    for &idx in &indices {
        if remaining_area <= 0.0 {
            break;
        }
        types[idx] = PlateType::Continental;
        remaining_area -= accum[idx].count;
    }

    if !types.iter().any(|t| *t == PlateType::Oceanic) {
        if let Some(&idx) = indices.last() {
            types[idx] = PlateType::Oceanic;
        }
    }

    types
}

fn compute_plate_velocities(
    seeds: &[PlateSeed],
    counts: &[f32],
    velocity_scale: f32,
    drift_bias: f32,
    base_seed: u64,
    height: usize,
) -> Vec<[f32; 2]> {
    let mut velocities = vec![[0.0f32; 2]; seeds.len()];
    for (idx, seed) in seeds.iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed ^ ((idx as u64 + 1) * 0x9E37_79B9_7F4A_7C15));
        let angle = rng.gen::<f32>() * 2.0 * PI;
        let magnitude = velocity_scale.max(1e-3) * (0.45 + rng.gen::<f32>() * 0.55);
        let random_component = [angle.cos() * magnitude, angle.sin() * magnitude];

        let lat = if height > 1 {
            let normalized = seed.position[1] / height as f32;
            (0.5 - normalized) * PI
        } else {
            0.0
        };
        let drift_strength = drift_bias * lat.cos();
        let meridional = drift_bias * 0.25 * lat.sin();
        let drift_component = [drift_strength, meridional];

        velocities[idx][0] = random_component[0] + drift_component[0];
        velocities[idx][1] = random_component[1] + drift_component[1];
    }

    let mut total_weight = 0.0f32;
    let mut momentum = [0.0f32; 2];
    for (vel, weight) in velocities.iter().zip(counts.iter()) {
        let w = if *weight > 0.0 { *weight } else { 1.0 };
        total_weight += w;
        momentum[0] += vel[0] * w;
        momentum[1] += vel[1] * w;
    }
    if total_weight > 0.0 {
        let avg = [momentum[0] / total_weight, momentum[1] / total_weight];
        for vel in velocities.iter_mut() {
            vel[0] -= avg[0];
            vel[1] -= avg[1];
        }
    }

    velocities
}

fn smooth_cell_velocities(
    vel_u: &mut [f32],
    vel_v: &mut [f32],
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
    steps: usize,
    dt: f32,
) {
    if steps == 0 || dt <= 0.0 {
        return;
    }
    let smoothing = (0.04 * dt).clamp(0.0, 0.25);
    if smoothing <= 0.0 {
        return;
    }
    let mut scratch_u = vel_u.to_vec();
    let mut scratch_v = vel_v.to_vec();
    let total = width * height;
    for _ in 0..steps {
        scratch_u.copy_from_slice(vel_u);
        scratch_v.copy_from_slice(vel_v);
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let left = if col > 0 {
                    idx - 1
                } else if wrap_x {
                    row * width + (width - 1)
                } else {
                    idx
                };
                let right = if col + 1 < width {
                    idx + 1
                } else if wrap_x {
                    row * width
                } else {
                    idx
                };
                let up = if row > 0 {
                    idx - width
                } else if wrap_y {
                    (height - 1) * width + col
                } else {
                    idx
                };
                let down = if row + 1 < height {
                    idx + width
                } else if wrap_y {
                    col
                } else {
                    idx
                };
                let lap_u = (scratch_u[left] + scratch_u[right] + scratch_u[up] + scratch_u[down]) * 0.25
                    - scratch_u[idx];
                let lap_v = (scratch_v[left] + scratch_v[right] + scratch_v[up] + scratch_v[down]) * 0.25
                    - scratch_v[idx];
                vel_u[idx] = scratch_u[idx] + smoothing * lap_u;
                vel_v[idx] = scratch_v[idx] + smoothing * lap_v;
            }
        }
        if total > 0 {
            let sum_u: f64 = vel_u.iter().map(|v| *v as f64).sum();
            let sum_v: f64 = vel_v.iter().map(|v| *v as f64).sum();
            let avg_u = sum_u / total as f64;
            let avg_v = sum_v / total as f64;
            vel_u.iter_mut().for_each(|v| *v -= avg_u as f32);
            vel_v.iter_mut().for_each(|v| *v -= avg_v as f32);
        }
    }
}

fn compute_subduction_probability(
    facing_type: PlateType,
    opposing_type: PlateType,
    convergence_rate: f32,
    thickness_self: f32,
    thickness_other: f32,
    bias: f32,
) -> f32 {
    if convergence_rate <= 0.0 {
        return 0.0;
    }
    if facing_type != PlateType::Oceanic || opposing_type != PlateType::Continental {
        return 0.0;
    }
    let thickness_factor =
        (thickness_other - thickness_self).max(0.0) / (thickness_other + thickness_self + 1e-6);
    let base = (convergence_rate * 0.6 + bias).clamp(-6.0, 6.0);
    let prob = 1.0 / (1.0 + (-base).exp());
    (prob * (0.6 + 0.4 * thickness_factor)).clamp(0.0, 1.0)
}

fn compute_boundary_fields(
    assignments: &[usize],
    plate_types: &[PlateType],
    thickness: &[f32],
    density: &[f32],
    vel_u: &[f32],
    vel_v: &[f32],
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
    subduction_bias: f32,
) -> BoundaryData {
    let total = width * height;
    let mut convergence_acc = vec![0.0f32; total];
    let mut divergence_acc = vec![0.0f32; total];
    let mut shear_acc = vec![0.0f32; total];
    let mut subduction_acc = vec![0.0f32; total];
    let mut counts = vec![0.0f32; total];

    let mut convergence_sum = 0.0f64;
    let mut divergence_sum = 0.0f64;
    let mut shear_sum = 0.0f64;
    let mut subduction_sum = 0.0f64;

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            let plate_i = assignments[idx];
            let vel_i = [vel_u[idx], vel_v[idx]];
            let plate_type_i = plate_types[plate_i];
            let thickness_i = thickness[plate_i];
            let density_i = density[plate_i];

            // East neighbor
            if col + 1 < width || wrap_x {
                let ncol = if col + 1 < width { col + 1 } else { 0 };
                let nidx = row * width + ncol;
                let plate_j = assignments[nidx];
                if plate_i != plate_j {
                    let vel_j = [vel_u[nidx], vel_v[nidx]];
                    let plate_type_j = plate_types[plate_j];
                    let thickness_j = thickness[plate_j];
                    let density_j = density[plate_j];
                    let rel = [vel_j[0] - vel_i[0], vel_j[1] - vel_i[1]];
                    let normal_component = rel[0];
                    let tangent_component = rel[1];
                    let shear_mag = tangent_component.abs();

                    counts[idx] += 1.0;
                    counts[nidx] += 1.0;
                    shear_acc[idx] += shear_mag;
                    shear_acc[nidx] += shear_mag;
                    shear_sum += shear_mag as f64;

                    if normal_component > 0.0 {
                        divergence_acc[idx] += normal_component;
                        convergence_acc[nidx] += normal_component;
                        divergence_sum += normal_component as f64;
                        convergence_sum += normal_component as f64;
                        let prob = compute_subduction_probability(
                            plate_type_j,
                            plate_type_i,
                            normal_component,
                            thickness_j,
                            thickness_i,
                            subduction_bias,
                        );
                        subduction_acc[nidx] += prob;
                        subduction_sum += prob as f64;
                    } else if normal_component < 0.0 {
                        let convergence = -normal_component;
                        convergence_acc[idx] += convergence;
                        divergence_acc[nidx] += convergence;
                        convergence_sum += convergence as f64;
                        divergence_sum += convergence as f64;
                        let prob = compute_subduction_probability(
                            plate_type_i,
                            plate_type_j,
                            convergence,
                            thickness_i,
                            thickness_j,
                            subduction_bias,
                        );
                        subduction_acc[idx] += prob;
                        subduction_sum += prob as f64;
                    }

                    let density_grad = (density_j - density_i).abs();
                    let shear_bonus = density_grad * 0.05;
                    shear_acc[idx] += shear_bonus;
                    shear_acc[nidx] += shear_bonus;
                    shear_sum += shear_bonus as f64 * 2.0;
                }
            }

            // South neighbor
            if row + 1 < height || wrap_y {
                let nrow = if row + 1 < height { row + 1 } else { 0 };
                let nidx = nrow * width + col;
                let plate_j = assignments[nidx];
                if plate_i != plate_j {
                    let vel_j = [vel_u[nidx], vel_v[nidx]];
                    let plate_type_j = plate_types[plate_j];
                    let thickness_j = thickness[plate_j];
                    let density_j = density[plate_j];
                    let rel = [vel_j[0] - vel_i[0], vel_j[1] - vel_i[1]];
                    let normal_component = rel[1];
                    let tangent_component = -rel[0];
                    let shear_mag = tangent_component.abs();

                    counts[idx] += 1.0;
                    counts[nidx] += 1.0;
                    shear_acc[idx] += shear_mag;
                    shear_acc[nidx] += shear_mag;
                    shear_sum += shear_mag as f64;

                    if normal_component > 0.0 {
                        divergence_acc[idx] += normal_component;
                        convergence_acc[nidx] += normal_component;
                        divergence_sum += normal_component as f64;
                        convergence_sum += normal_component as f64;
                        let prob = compute_subduction_probability(
                            plate_type_j,
                            plate_type_i,
                            normal_component,
                            thickness_j,
                            thickness_i,
                            subduction_bias,
                        );
                        subduction_acc[nidx] += prob;
                        subduction_sum += prob as f64;
                    } else if normal_component < 0.0 {
                        let convergence = -normal_component;
                        convergence_acc[idx] += convergence;
                        divergence_acc[nidx] += convergence;
                        convergence_sum += convergence as f64;
                        divergence_sum += convergence as f64;
                        let prob = compute_subduction_probability(
                            plate_type_i,
                            plate_type_j,
                            convergence,
                            thickness_i,
                            thickness_j,
                            subduction_bias,
                        );
                        subduction_acc[idx] += prob;
                        subduction_sum += prob as f64;
                    }

                    let density_grad = (density_j - density_i).abs();
                    let shear_bonus = density_grad * 0.05;
                    shear_acc[idx] += shear_bonus;
                    shear_acc[nidx] += shear_bonus;
                    shear_sum += shear_bonus as f64 * 2.0;
                }
            }
        }
    }

    let mut convergence = vec![0.0f32; total];
    let mut divergence = vec![0.0f32; total];
    let mut shear = vec![0.0f32; total];
    let mut subduction = vec![0.0f32; total];

    for idx in 0..total {
        if counts[idx] > 0.0 {
            let inv = 1.0 / counts[idx];
            convergence[idx] = (convergence_acc[idx] * inv).max(0.0);
            divergence[idx] = (divergence_acc[idx] * inv).max(0.0);
            shear[idx] = (shear_acc[idx] * inv).max(0.0);
            subduction[idx] = (subduction_acc[idx] * inv).clamp(0.0, 1.0);
        }
    }

    BoundaryData {
        convergence,
        divergence,
        shear,
        subduction,
        convergence_sum,
        divergence_sum,
        shear_sum,
        subduction_sum,
    }
}

fn build_hotspot_map(
    shear: &[f32],
    convergence: &[f32],
    divergence: &[f32],
    subduction: &[f32],
    hotspot_density: f32,
    seed: u64,
) -> (Vec<f32>, f64, i32) {
    let total = shear.len();
    let mut shear_max = shear.iter().copied().fold(0.0f32, f32::max);
    let mut conv_max = convergence.iter().copied().fold(0.0f32, f32::max);
    let mut div_max = divergence.iter().copied().fold(0.0f32, f32::max);
    let mut sub_max = subduction.iter().copied().fold(0.0f32, f32::max);
    if shear_max <= 1e-6 {
        shear_max = 1.0;
    }
    if conv_max <= 1e-6 {
        conv_max = 1.0;
    }
    if div_max <= 1e-6 {
        div_max = 1.0;
    }
    if sub_max <= 1e-6 {
        sub_max = 1.0;
    }

    let mut hotspot = vec![0.0f32; total];
    let mut sum = 0.0f64;
    let mut count_high = 0i32;
    let density_factor = (0.5 + hotspot_density.clamp(0.0, 1.0) * 0.8).clamp(0.3, 1.6);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5F1_C3D2_9E37_79B9);

    for idx in 0..total {
        let shear_score = (shear[idx] / shear_max).clamp(0.0, 1.0);
        let conv_score = (convergence[idx] / conv_max).clamp(0.0, 1.0);
        let div_score = (divergence[idx] / div_max).clamp(0.0, 1.0);
        let sub_score = (subduction[idx] / sub_max).clamp(0.0, 1.0);
        let noise: f32 = rng.gen::<f32>() * 0.35;

        let ridge_component = 0.45 * shear_score + 0.20 * div_score;
        let arc_component = 0.35 * sub_score + 0.15 * conv_score;
        let value = (ridge_component + arc_component + noise) * density_factor;
        let clamped = value.clamp(0.0, 1.0);
        hotspot[idx] = clamped;
        sum += clamped as f64;
        if clamped > 0.65 {
            count_high += 1;
        }
    }

    let mean = if total > 0 { sum / total as f64 } else { 0.0 };
    (hotspot, mean, count_high)
}

#[no_mangle]
pub unsafe extern "C" fn tectonics_run(
    height: i32,
    width: i32,
    seed: u64,
    num_plates: i32,
    continental_fraction_target: f32,
    velocity_scale: f32,
    drift_bias: f32,
    hotspot_density: f32,
    subduction_bias: f32,
    lloyd_iterations: i32,
    time_steps: i32,
    time_step: f32,
    wrap_x: i32,
    wrap_y: i32,
    plate_ptr: *mut f32,
    convergence_ptr: *mut f32,
    divergence_ptr: *mut f32,
    shear_ptr: *mut f32,
    subduction_ptr: *mut f32,
    hotspot_ptr: *mut f32,
    stats_out: *mut TectonicsStats,
) {
    let height = height.max(1) as usize;
    let width = width.max(1) as usize;
    let total = height * width;
    if total == 0 {
        return;
    }

    let plate_field = std::slice::from_raw_parts_mut(plate_ptr, total * PLATE_COMPONENTS);
    let convergence_field = std::slice::from_raw_parts_mut(convergence_ptr, total);
    let divergence_field = std::slice::from_raw_parts_mut(divergence_ptr, total);
    let shear_field = std::slice::from_raw_parts_mut(shear_ptr, total);
    let subduction_field = std::slice::from_raw_parts_mut(subduction_ptr, total);
    let hotspot_field = std::slice::from_raw_parts_mut(hotspot_ptr, total);

    let wrap_x = wrap_x != 0;
    let wrap_y = wrap_y != 0;
    let plate_count = num_plates.max(1) as usize;
    let lloyd_iters = lloyd_iterations.max(0) as usize;
    let integration_steps = time_steps.max(0) as usize;
    let dt = time_step.max(0.0);

    let mut seeds = poisson_sample(height, width, seed, plate_count, wrap_x, wrap_y);

    let mut assignments = vec![0usize; total];
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA17F_98C3_5D21_2F4B);

    for _ in 0..lloyd_iters {
        assign_cells_into(&seeds, &mut assignments, height, width, wrap_x, wrap_y);
        recentre_seeds(
            &mut seeds,
            &assignments,
            height,
            width,
            wrap_x,
            wrap_y,
            &mut rng,
        );
    }

    assign_cells_into(&seeds, &mut assignments, height, width, wrap_x, wrap_y);
    let accum = recentre_seeds(
        &mut seeds,
        &assignments,
        height,
        width,
        wrap_x,
        wrap_y,
        &mut rng,
    );

    let counts: Vec<f32> = accum.iter().map(|a| a.count).collect();
    let plate_types = classify_plate_types(&accum, continental_fraction_target);
    for (seed, plate_type) in seeds.iter_mut().zip(plate_types.iter()) {
        seed.plate_type = *plate_type;
    }

    let thickness: Vec<f32> = plate_types
        .iter()
        .map(|t| if *t == PlateType::Continental { 35.0 } else { 8.0 })
        .collect();
    let density: Vec<f32> = plate_types
        .iter()
        .map(|t| if *t == PlateType::Continental { 2.8 } else { 3.3 })
        .collect();

    let plate_velocities = compute_plate_velocities(
        &seeds,
        &counts,
        velocity_scale,
        drift_bias,
        seed,
        height,
    );

    let mut cell_vel_u = vec![0.0f32; total];
    let mut cell_vel_v = vec![0.0f32; total];
    for (idx, &plate) in assignments.iter().enumerate() {
        cell_vel_u[idx] = plate_velocities[plate][0];
        cell_vel_v[idx] = plate_velocities[plate][1];
    }

    smooth_cell_velocities(
        &mut cell_vel_u,
        &mut cell_vel_v,
        width,
        height,
        wrap_x,
        wrap_y,
        integration_steps,
        dt,
    );

    let boundary = compute_boundary_fields(
        &assignments,
        &plate_types,
        &thickness,
        &density,
        &cell_vel_u,
        &cell_vel_v,
        width,
        height,
        wrap_x,
        wrap_y,
        subduction_bias,
    );

    let (hotspot, hotspot_mean, hotspot_count) = build_hotspot_map(
        &boundary.shear,
        &boundary.convergence,
        &boundary.divergence,
        &boundary.subduction,
        hotspot_density,
        seed,
    );

    plate_field
        .par_chunks_mut(PLATE_COMPONENTS)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let plate = assignments[idx];
            chunk[0] = plate as f32;
            chunk[1] = if plate_types[plate] == PlateType::Continental {
                1.0
            } else {
                0.0
            };
            chunk[2] = thickness[plate];
            chunk[3] = density[plate];
            chunk[4] = cell_vel_u[idx];
            chunk[5] = cell_vel_v[idx];
        });

    convergence_field.copy_from_slice(&boundary.convergence);
    divergence_field.copy_from_slice(&boundary.divergence);
    shear_field.copy_from_slice(&boundary.shear);
    subduction_field.copy_from_slice(&boundary.subduction);
    hotspot_field.copy_from_slice(&hotspot);

    let mut velocity_sum = 0.0f64;
    let mut velocity_sq_sum = 0.0f64;
    for idx in 0..total {
        let speed = (cell_vel_u[idx] * cell_vel_u[idx] + cell_vel_v[idx] * cell_vel_v[idx]).sqrt();
        velocity_sum += speed as f64;
        velocity_sq_sum += (speed as f64) * (speed as f64);
    }
    let total_f64 = total as f64;
    let velocity_mean = if total > 0 {
        velocity_sum / total_f64
    } else {
        0.0
    };
    let velocity_std = if total > 0 {
        let mean_sq = velocity_sq_sum / total_f64;
        (mean_sq - velocity_mean * velocity_mean).max(0.0).sqrt()
    } else {
        0.0
    };

    let continental_cells: f64 = assignments
        .iter()
        .map(|&plate| if plate_types[plate] == PlateType::Continental { 1.0 } else { 0.0 })
        .sum();
    let continental_fraction = continental_cells / total_f64;

    let boundary_total =
        boundary.convergence_sum + boundary.divergence_sum + boundary.shear_sum;
    let boundary_metric_mean = if total > 0 {
        boundary_total / total_f64
    } else {
        0.0
    };
    let subduction_mean = if total > 0 {
        boundary.subduction_sum / total_f64
    } else {
        0.0
    };

    if !stats_out.is_null() {
        *stats_out = TectonicsStats {
            plate_count: plate_count as i32,
            continental_fraction,
            velocity_mean,
            velocity_std,
            hotspot_mean,
            boundary_metric_mean,
            convergence_sum: boundary.convergence_sum,
            divergence_sum: boundary.divergence_sum,
            shear_sum: boundary.shear_sum,
            subduction_mean,
            hotspot_count,
        };
    }
}
