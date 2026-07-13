use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::mem::size_of;

const PLATE_COMPONENTS: usize = 6;
const SPHERICAL_PLATE_COMPONENTS: usize = 7;
const D4_NEIGHBORS: usize = 4;

#[no_mangle]
pub extern "C" fn tectonics_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_tectonics_abi_version() -> u32 {
    1
}

#[derive(Clone, Copy, Debug, Default)]
struct SVec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl SVec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn from_f32(values: &[f32]) -> Self {
        Self::new(values[0] as f64, values[1] as f64, values[2] as f64)
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Self {
        let norm = self.norm().max(f64::EPSILON);
        self * (1.0 / norm)
    }
}

impl std::ops::Add for SVec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::AddAssign for SVec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl std::ops::Sub for SVec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl std::ops::Mul<f64> for SVec3 {
    type Output = Self;

    fn mul(self, scale: f64) -> Self {
        Self::new(self.x * scale, self.y * scale, self.z * scale)
    }
}

#[derive(Clone, Copy)]
struct SphericalPlateSeed {
    position: SVec3,
    angular_velocity: SVec3,
}

#[derive(Clone, Copy)]
struct WarpTerm {
    axis: SVec3,
    frequency: f64,
    phase: f64,
    amplitude: f64,
}

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
    let mut min_dist_sq = (((total_cells as f32) / (count.max(1) as f32)).sqrt() * 0.75)
        .powi(2)
        .max(4.0);
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
    warp_seed: u64,
) {
    let width_f = width as f32;
    let height_f = height as f32;
    assignments
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, row_assign)| {
            let y = row as f32 + 0.5;
            for (col, assignment) in row_assign.iter_mut().enumerate() {
                let x = col as f32 + 0.5;
                let warped_cell = warp_position([x, y], width_f, height_f, warp_seed);
                let mut best_idx = 0usize;
                let mut best_dist = f32::MAX;
                for (seed_idx, seed) in seeds.iter().enumerate() {
                    let warped_seed_position =
                        warp_position(seed.position, width_f, height_f, warp_seed);
                    let dist = wrap_distance_sq(
                        warped_cell,
                        warped_seed_position,
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
                *assignment = best_idx;
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

    if !types.contains(&PlateType::Oceanic) {
        if let Some(&idx) = indices.last() {
            types[idx] = PlateType::Oceanic;
        }
    }

    types
}

fn smoothstep(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn value_noise(
    height: usize,
    width: usize,
    lattice_rows: usize,
    lattice_cols: usize,
    seed: u64,
    wrap_x: bool,
) -> Vec<f32> {
    let rows = lattice_rows.max(2).min(height.max(2));
    let cols = lattice_cols.max(2).min(width.max(2));
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let lattice: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect();
    let mut output = vec![0.0f32; height * width];

    for row in 0..height {
        let y = if height > 1 {
            row as f32 * (rows - 1) as f32 / (height - 1) as f32
        } else {
            0.0
        };
        let y0 = (y.floor() as usize).min(rows - 1);
        let y1 = (y0 + 1).min(rows - 1);
        let fy = smoothstep(y - y0 as f32);
        for col in 0..width {
            let x = col as f32 * cols as f32 / width.max(1) as f32;
            let x_floor = x.floor();
            let x0 = (x_floor as usize).min(cols - 1);
            let x1 = if x0 + 1 < cols {
                x0 + 1
            } else if wrap_x {
                0
            } else {
                x0
            };
            let fx = smoothstep(x - x_floor);
            let top = lattice[y0 * cols + x0] * (1.0 - fx) + lattice[y0 * cols + x1] * fx;
            let bottom = lattice[y1 * cols + x0] * (1.0 - fx) + lattice[y1 * cols + x1] * fx;
            output[row * width + col] = top * (1.0 - fy) + bottom * fy;
        }
    }
    output
}

fn build_crust_provinces(
    height: usize,
    width: usize,
    target_fraction: f32,
    seed: u64,
    wrap_x: bool,
) -> (Vec<bool>, Vec<f32>, Vec<f32>) {
    let total = height * width;
    if total == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let octave_specs = [
        (4usize, 8usize, 0.52f32),
        (8, 16, 0.27),
        (16, 32, 0.14),
        (32, 64, 0.07),
    ];
    let mut score = vec![0.0f32; total];
    for (octave, (rows, cols, weight)) in octave_specs.iter().enumerate() {
        let noise = value_noise(
            height,
            width,
            *rows,
            *cols,
            seed ^ (octave as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93),
            wrap_x,
        );
        for (value, contribution) in score.iter_mut().zip(noise.iter()) {
            *value += contribution * weight;
        }
    }

    let land_count = if total == 1 {
        usize::from(target_fraction >= 0.5)
    } else {
        ((target_fraction.clamp(0.0, 1.0) * total as f32).round() as usize).clamp(1, total - 1)
    };
    let mut ranked: Vec<usize> = (0..total).collect();
    ranked.sort_by(|a, b| score[*b].partial_cmp(&score[*a]).unwrap_or(Ordering::Equal));
    let mut continental = vec![false; total];
    for &idx in ranked.iter().take(land_count) {
        continental[idx] = true;
    }

    let (score_min, score_max) = score
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| {
            (low.min(*value), high.max(*value))
        });
    let score_range = (score_max - score_min).max(1e-6);
    let mut thickness = vec![0.0f32; total];
    let mut density = vec![0.0f32; total];
    for idx in 0..total {
        let normalized = ((score[idx] - score_min) / score_range).clamp(0.0, 1.0);
        if continental[idx] {
            thickness[idx] = 29.0 + normalized * 13.0;
            density[idx] = 2.84 - normalized * 0.16;
        } else {
            thickness[idx] = 6.0 + normalized * 4.0;
            density[idx] = 3.34 - normalized * 0.10;
        }
    }

    (continental, thickness, density)
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
        let stream_seed = (idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed ^ stream_seed);
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

#[allow(clippy::too_many_arguments)]
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
                let lap_u = (scratch_u[left] + scratch_u[right] + scratch_u[up] + scratch_u[down])
                    * 0.25
                    - scratch_u[idx];
                let lap_v = (scratch_v[left] + scratch_v[right] + scratch_v[up] + scratch_v[down])
                    * 0.25
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

fn warp_position(position: [f32; 2], width: f32, height: f32, seed: u64) -> [f32; 2] {
    let folded_seed = (seed ^ (seed >> 32)) & 0xffff;
    let phase = folded_seed as f32 / u16::MAX as f32 * 2.0 * PI;
    let u = position[0] / width.max(1.0) * 2.0 * PI;
    let v = position[1] / height.max(1.0) * PI;
    let dx = width
        * (0.020 * (2.0 * v + 0.65 * (3.0 * u + phase).sin()).sin()
            + 0.007 * (5.0 * v - 2.0 * u + phase).sin());
    let dy = height
        * (0.030 * (2.0 * u + 0.55 * (2.0 * v - phase).sin()).sin()
            + 0.009 * (5.0 * u + v - phase).sin());
    [(position[0] + dx).rem_euclid(width), position[1] + dy]
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

#[allow(clippy::too_many_arguments)]
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

fn diffuse_field(
    field: &mut [f32],
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
    passes: usize,
) {
    let mut scratch = field.to_vec();
    for _ in 0..passes {
        scratch.copy_from_slice(field);
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let left = if col > 0 {
                    idx - 1
                } else if wrap_x {
                    row * width + width - 1
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
                let north = if row > 0 {
                    idx - width
                } else if wrap_y {
                    (height - 1) * width + col
                } else {
                    idx
                };
                let south = if row + 1 < height {
                    idx + width
                } else if wrap_y {
                    col
                } else {
                    idx
                };
                let neighbor_mean =
                    (scratch[left] + scratch[right] + scratch[north] + scratch[south]) * 0.25;
                field[idx] = scratch[idx] * 0.3 + neighbor_mean * 0.7;
            }
        }
    }
}

fn condition_boundary_fields(
    boundary: &mut BoundaryData,
    width: usize,
    height: usize,
    wrap_x: bool,
    wrap_y: bool,
    seed: u64,
) {
    let segmentation = value_noise(height, width, 12, 24, seed ^ 0x3C6E_F372_FE94_F82B, wrap_x);
    for field in [
        &mut boundary.convergence,
        &mut boundary.divergence,
        &mut boundary.shear,
        &mut boundary.subduction,
    ] {
        for (value, segment) in field.iter_mut().zip(segmentation.iter()) {
            let normalized = ((*segment + 1.0) * 0.5).clamp(0.0, 1.0);
            let activation = ((normalized - 0.38) / 0.62).clamp(0.0, 1.0);
            let along_strike = 1.65 * smoothstep(activation);
            *value *= along_strike;
        }
        diffuse_field(field, width, height, wrap_x, wrap_y, 12);
    }

    boundary.convergence_sum = boundary.convergence.iter().map(|value| *value as f64).sum();
    boundary.divergence_sum = boundary.divergence.iter().map(|value| *value as f64).sum();
    boundary.shear_sum = boundary.shear.iter().map(|value| *value as f64).sum();
    boundary.subduction_sum = boundary.subduction.iter().map(|value| *value as f64).sum();
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

fn random_unit_vector(rng: &mut ChaCha8Rng) -> SVec3 {
    let z = rng.gen_range(-1.0f64..1.0f64);
    let azimuth = rng.gen_range(0.0f64..std::f64::consts::TAU);
    let radius = (1.0 - z * z).sqrt();
    SVec3::new(radius * azimuth.cos(), radius * azimuth.sin(), z)
}

fn spherical_warp_terms(seed: u64) -> [WarpTerm; 8] {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x4A39_B70D_91C8_2E65);
    std::array::from_fn(|index| WarpTerm {
        axis: random_unit_vector(&mut rng),
        frequency: 2.2 + index as f64 * 1.35 + rng.gen_range(-0.2..0.2),
        phase: rng.gen_range(0.0..std::f64::consts::TAU),
        amplitude: 0.085 / (1.0 + index as f64 * 0.45),
    })
}

fn warp_spherical_direction(direction: SVec3, terms: &[WarpTerm]) -> SVec3 {
    let mut displacement = SVec3::default();
    for term in terms {
        let projection = direction.dot(term.axis);
        let tangent = term.axis - direction * projection;
        let wave = (term.frequency * projection + term.phase).sin();
        displacement += tangent * (term.amplitude * wave);
    }
    (direction + displacement).normalized()
}

fn sample_spherical_plate_seeds(seed: u64, count: usize) -> Vec<SphericalPlateSeed> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC13F_A9A9_02A6_328F);
    let mut seeds = Vec::with_capacity(count);
    let mut minimum_angle = (4.0 / count.max(1) as f64).sqrt().min(1.2) * 0.62;
    let mut attempts = 0usize;
    while seeds.len() < count && attempts < count.max(1) * 8000 {
        attempts += 1;
        let position = random_unit_vector(&mut rng);
        let maximum_dot = minimum_angle.cos();
        if seeds
            .iter()
            .all(|candidate: &SphericalPlateSeed| candidate.position.dot(position) < maximum_dot)
        {
            seeds.push(SphericalPlateSeed {
                position,
                angular_velocity: SVec3::default(),
            });
        }
        if attempts % (count.max(1) * 256) == 0 {
            minimum_angle *= 0.9;
        }
    }
    while seeds.len() < count {
        seeds.push(SphericalPlateSeed {
            position: random_unit_vector(&mut rng),
            angular_velocity: SVec3::default(),
        });
    }
    seeds
}

fn assign_spherical_cells(
    warped_cells: &[SVec3],
    seeds: &[SphericalPlateSeed],
    warp_terms: &[WarpTerm],
    assignments: &mut [usize],
) {
    let warped_seeds: Vec<SVec3> = seeds
        .iter()
        .map(|seed| warp_spherical_direction(seed.position, warp_terms))
        .collect();
    assignments
        .par_iter_mut()
        .zip(warped_cells.par_iter())
        .for_each(|(assignment, cell)| {
            let mut best_plate = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for (plate, seed) in warped_seeds.iter().enumerate() {
                let score = cell.dot(*seed);
                if score > best_score {
                    best_score = score;
                    best_plate = plate;
                }
            }
            *assignment = best_plate;
        });
}

fn recentre_spherical_seeds(
    seeds: &mut [SphericalPlateSeed],
    cells: &[SVec3],
    areas: &[f64],
    assignments: &[usize],
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let mut centroids = vec![SVec3::default(); seeds.len()];
    let mut plate_areas = vec![0.0f64; seeds.len()];
    for ((cell, area), plate) in cells.iter().zip(areas.iter()).zip(assignments.iter()) {
        centroids[*plate] += *cell * *area;
        plate_areas[*plate] += *area;
    }
    for (plate, seed) in seeds.iter_mut().enumerate() {
        if plate_areas[plate] > 0.0 && centroids[plate].norm() > 1e-12 {
            seed.position = centroids[plate].normalized();
        } else {
            seed.position = random_unit_vector(rng);
        }
    }
    plate_areas
}

fn spherical_plate_areas(plate_count: usize, areas: &[f64], assignments: &[usize]) -> Vec<f64> {
    let mut plate_areas = vec![0.0f64; plate_count];
    for (area, plate) in areas.iter().zip(assignments.iter()) {
        plate_areas[*plate] += *area;
    }
    plate_areas
}

fn ensure_nonempty_spherical_plates(
    cells: &[SVec3],
    warped_cells: &[SVec3],
    seeds: &mut [SphericalPlateSeed],
    warp_terms: &[WarpTerm],
    assignments: &mut [usize],
) -> bool {
    for _ in 0..seeds.len() {
        let mut counts = vec![0usize; seeds.len()];
        for plate in assignments.iter() {
            counts[*plate] += 1;
        }
        let empty: Vec<usize> = counts
            .iter()
            .enumerate()
            .filter_map(|(plate, count)| (*count == 0).then_some(plate))
            .collect();
        if empty.is_empty() {
            return true;
        }
        for empty_plate in empty {
            let Some((donor, _)) = counts
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 1)
                .max_by_key(|(_, count)| **count)
            else {
                return false;
            };
            let candidate = assignments
                .iter()
                .enumerate()
                .filter(|(_, plate)| **plate == donor)
                .min_by(|(first, _), (second, _)| {
                    seeds[donor]
                        .position
                        .dot(cells[*first])
                        .partial_cmp(&seeds[donor].position.dot(cells[*second]))
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(index, _)| index)
                .expect("donor plate has cells");
            seeds[empty_plate].position = cells[candidate];
            counts[donor] -= 1;
            counts[empty_plate] = 1;
        }
        assign_spherical_cells(warped_cells, seeds, warp_terms, assignments);
    }
    let mut present = vec![false; seeds.len()];
    for plate in assignments.iter() {
        present[*plate] = true;
    }
    present.into_iter().all(|value| value)
}

fn repair_plate_connectivity(
    assignments: &mut [usize],
    neighbors: &[i32],
    plate_count: usize,
) -> bool {
    let total = assignments.len();
    let mut visited = vec![false; total];
    let mut largest_components: Vec<Vec<usize>> = vec![Vec::new(); plate_count];

    for start in 0..total {
        if visited[start] {
            continue;
        }
        let plate = assignments[start];
        let mut component = Vec::new();
        let mut pending = vec![start];
        visited[start] = true;
        while let Some(cell) = pending.pop() {
            component.push(cell);
            for &neighbor in &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS] {
                let adjacent = neighbor as usize;
                if !visited[adjacent] && assignments[adjacent] == plate {
                    visited[adjacent] = true;
                    pending.push(adjacent);
                }
            }
        }
        if component.len() > largest_components[plate].len() {
            largest_components[plate] = component;
        }
    }
    if largest_components.iter().any(Vec::is_empty) {
        return false;
    }

    let unassigned = usize::MAX;
    let mut repaired = vec![unassigned; total];
    let mut frontier = VecDeque::new();
    for (plate, component) in largest_components.iter().enumerate() {
        for &cell in component {
            repaired[cell] = plate;
            frontier.push_back(cell);
        }
    }
    while let Some(cell) = frontier.pop_front() {
        let plate = repaired[cell];
        for &neighbor in &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS] {
            let adjacent = neighbor as usize;
            if repaired[adjacent] == unassigned {
                repaired[adjacent] = plate;
                frontier.push_back(adjacent);
            }
        }
    }
    if repaired.contains(&unassigned) {
        return false;
    }
    assignments.copy_from_slice(&repaired);
    true
}

fn spherical_province_score(cells: &[SVec3], seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xD2B7_4407_B1CE_6E93);
    let pole = random_unit_vector(&mut rng);
    let raw_axis = random_unit_vector(&mut rng);
    let first_axis = (raw_axis - pole * raw_axis.dot(pole)).normalized();
    let second_axis = pole.cross(first_axis).normalized();
    let bases = [
        pole,
        pole * -1.0,
        first_axis,
        first_axis * -1.0,
        second_axis,
        second_axis * -1.0,
    ];
    let foci: Vec<(SVec3, SVec3, f64)> = bases
        .iter()
        .map(|base| {
            let raw_tangent = random_unit_vector(&mut rng);
            let tangent = (raw_tangent - *base * raw_tangent.dot(*base)).normalized();
            let elongation = rng.gen_range(0.16..0.46);
            (
                (*base + tangent * elongation).normalized(),
                (*base - tangent * elongation).normalized(),
                rng.gen_range(-0.055..0.055),
            )
        })
        .collect();
    let detail_terms: Vec<(SVec3, f64, f64, f64)> = (0..18)
        .map(|index| {
            let octave = index / 3;
            let frequency = 2.6 * 1.68f64.powi(octave);
            let weight = 1.0 / 1.35f64.powi(octave);
            (
                random_unit_vector(&mut rng),
                frequency + rng.gen_range(-0.18..0.18),
                rng.gen_range(0.0..std::f64::consts::TAU),
                weight,
            )
        })
        .collect();
    let detail_weight: f64 = detail_terms.iter().map(|term| term.3).sum();
    let warp_terms = spherical_warp_terms(seed ^ 0x6E5A_17C9_B42D_308F);
    cells
        .par_iter()
        .map(|cell| {
            let warped = warp_spherical_direction(*cell, &warp_terms);
            let mut best_craton = f64::NEG_INFINITY;
            let mut second_craton = f64::NEG_INFINITY;
            for (first, second, bias) in &foci {
                let score = warped.dot(*first).max(warped.dot(*second)) + *bias;
                if score > best_craton {
                    second_craton = best_craton;
                    best_craton = score;
                } else if score > second_craton {
                    second_craton = score;
                }
            }
            let detail: f64 = detail_terms
                .iter()
                .map(|(axis, frequency, phase, weight)| {
                    (warped.dot(*axis) * *frequency + *phase).sin() * *weight
                })
                .sum();
            best_craton + (best_craton - second_craton) * 0.5 + detail / detail_weight * 0.22
        })
        .collect()
}

fn build_spherical_crust_provinces(
    cells: &[SVec3],
    areas: &[f64],
    target_fraction: f32,
    seed: u64,
) -> (Vec<bool>, Vec<f32>, Vec<f32>, f64) {
    let score = spherical_province_score(cells, seed);
    let mut ranked: Vec<usize> = (0..cells.len()).collect();
    ranked.sort_by(|a, b| score[*b].partial_cmp(&score[*a]).unwrap_or(Ordering::Equal));
    let total_area: f64 = areas.iter().sum();
    let target_area = total_area * target_fraction.clamp(0.0, 1.0) as f64;
    let mut continental = vec![false; cells.len()];
    let mut continental_area = 0.0f64;
    for index in ranked {
        if continental_area >= target_area {
            break;
        }
        continental[index] = true;
        continental_area += areas[index];
    }

    let score_min = score.iter().copied().fold(f64::INFINITY, f64::min);
    let score_max = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let score_range = (score_max - score_min).max(1e-12);
    let mut thickness = vec![0.0f32; cells.len()];
    let mut density = vec![0.0f32; cells.len()];
    for index in 0..cells.len() {
        let normalized = ((score[index] - score_min) / score_range).clamp(0.0, 1.0) as f32;
        if continental[index] {
            thickness[index] = 27.0 + normalized * 16.0;
            density[index] = 2.86 - normalized * 0.19;
        } else {
            thickness[index] = 5.5 + normalized * 5.0;
            density[index] = 3.36 - normalized * 0.13;
        }
    }
    (
        continental,
        thickness,
        density,
        continental_area / total_area.max(f64::EPSILON),
    )
}

fn assign_spherical_angular_velocities(
    seeds: &mut [SphericalPlateSeed],
    plate_areas: &[f64],
    velocity_scale: f32,
    drift_bias: f32,
    seed: u64,
) {
    let mut weighted_mean = SVec3::default();
    let mut total_area = 0.0f64;
    for (index, (plate, area)) in seeds.iter_mut().zip(plate_areas.iter()).enumerate() {
        let stream = (index as u64 + 1).wrapping_mul(0x8CB9_2BA7_2F3D_8DD7);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ stream);
        let axis = random_unit_vector(&mut rng);
        let magnitude = velocity_scale.max(1e-3) as f64 * rng.gen_range(0.45..1.0);
        plate.angular_velocity = axis * magnitude;
        plate.angular_velocity.z += drift_bias as f64 * plate.position.z;
        weighted_mean += plate.angular_velocity * *area;
        total_area += *area;
    }
    if total_area > 0.0 {
        weighted_mean = weighted_mean * (1.0 / total_area);
        for plate in seeds {
            plate.angular_velocity = plate.angular_velocity - weighted_mean;
        }
    }
}

fn diffuse_graph_field(field: &mut [f32], neighbors: &[i32], passes: usize) {
    let mut scratch = field.to_vec();
    for _ in 0..passes {
        scratch.copy_from_slice(field);
        for (index, value) in field.iter_mut().enumerate() {
            let adjacent = &neighbors[index * D4_NEIGHBORS..(index + 1) * D4_NEIGHBORS];
            let mean = adjacent
                .iter()
                .map(|neighbor| scratch[*neighbor as usize])
                .sum::<f32>()
                / D4_NEIGHBORS as f32;
            *value = scratch[index] * 0.45 + mean * 0.55;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_spherical_boundaries(
    cells: &[SVec3],
    neighbors: &[i32],
    assignments: &[usize],
    velocities: &[SVec3],
    continental: &[bool],
    thickness: &[f32],
    subduction_bias: f32,
) -> BoundaryData {
    let total = cells.len();
    let mut convergence = vec![0.0f32; total];
    let mut divergence = vec![0.0f32; total];
    let mut shear = vec![0.0f32; total];
    let mut subduction = vec![0.0f32; total];
    let mut counts = vec![0.0f32; total];

    for index in 0..total {
        for &neighbor in &neighbors[index * D4_NEIGHBORS..(index + 1) * D4_NEIGHBORS] {
            let other = neighbor as usize;
            if other <= index || assignments[index] == assignments[other] {
                continue;
            }
            let first = cells[index];
            let second = cells[other];
            let cosine = first.dot(second).clamp(-1.0, 1.0);
            let first_toward_second = (second - first * cosine).normalized();
            let second_toward_first = (first - second * cosine).normalized();
            let closing_rate = velocities[index].dot(first_toward_second)
                + velocities[other].dot(second_toward_first);
            let boundary_tangent = first.cross(second).normalized();
            let shear_rate = (velocities[index].dot(boundary_tangent)
                - velocities[other].dot(boundary_tangent))
            .abs() as f32;

            counts[index] += 1.0;
            counts[other] += 1.0;
            shear[index] += shear_rate;
            shear[other] += shear_rate;
            if closing_rate > 0.0 {
                let rate = closing_rate as f32;
                convergence[index] += rate;
                convergence[other] += rate;
                let first_type = if continental[index] {
                    PlateType::Continental
                } else {
                    PlateType::Oceanic
                };
                let second_type = if continental[other] {
                    PlateType::Continental
                } else {
                    PlateType::Oceanic
                };
                subduction[index] += compute_subduction_probability(
                    first_type,
                    second_type,
                    rate,
                    thickness[index],
                    thickness[other],
                    subduction_bias,
                );
                subduction[other] += compute_subduction_probability(
                    second_type,
                    first_type,
                    rate,
                    thickness[other],
                    thickness[index],
                    subduction_bias,
                );
            } else if closing_rate < 0.0 {
                let rate = (-closing_rate) as f32;
                divergence[index] += rate;
                divergence[other] += rate;
            }
        }
    }

    for index in 0..total {
        if counts[index] > 0.0 {
            let inverse = 1.0 / counts[index];
            convergence[index] *= inverse;
            divergence[index] *= inverse;
            shear[index] *= inverse;
            subduction[index] = (subduction[index] * inverse).clamp(0.0, 1.0);
        }
    }
    diffuse_graph_field(&mut convergence, neighbors, 3);
    diffuse_graph_field(&mut divergence, neighbors, 3);
    diffuse_graph_field(&mut shear, neighbors, 3);
    diffuse_graph_field(&mut subduction, neighbors, 2);

    BoundaryData {
        convergence_sum: convergence.iter().map(|value| *value as f64).sum(),
        divergence_sum: divergence.iter().map(|value| *value as f64).sum(),
        shear_sum: shear.iter().map(|value| *value as f64).sum(),
        subduction_sum: subduction.iter().map(|value| *value as f64).sum(),
        convergence,
        divergence,
        shear,
        subduction,
    }
}

fn valid_ffi_len<T>(len: usize) -> bool {
    len.checked_mul(size_of::<T>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

#[no_mangle]
/// Generate a seam-free kinematic plate snapshot on canonical cubed-sphere cells.
///
/// Plate components are global plate ID, continental-crust flag, crustal
/// thickness, density, and the three components of a global tangent velocity.
///
/// # Safety
///
/// Input pointers must reference non-overlapping buffers sized from
/// `cell_count`: three `f32` values per XYZ vector, one `f64` area per cell,
/// and four `i32` D4 neighbors per cell. Output pointers must reference seven
/// `f32` values per plate cell and one `f32` per scalar field. Output buffers
/// must not alias inputs or each other. `stats_out` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tectonics_run_cubed_sphere(
    cell_count: i32,
    seed: u64,
    num_plates: i32,
    continental_fraction_target: f32,
    velocity_scale: f32,
    drift_bias: f32,
    hotspot_density: f32,
    subduction_bias: f32,
    lloyd_iterations: i32,
    xyz_ptr: *const f32,
    area_ptr: *const f64,
    neighbors_ptr: *const i32,
    plate_ptr: *mut f32,
    convergence_ptr: *mut f32,
    divergence_ptr: *mut f32,
    shear_ptr: *mut f32,
    subduction_ptr: *mut f32,
    hotspot_ptr: *mut f32,
    stats_out: *mut TectonicsStats,
) -> i32 {
    if cell_count <= 0
        || num_plates < 2
        || num_plates > cell_count
        || num_plates > 4096
        || num_plates as i64 * 4 > cell_count as i64
        || !continental_fraction_target.is_finite()
        || !velocity_scale.is_finite()
        || !drift_bias.is_finite()
        || !hotspot_density.is_finite()
        || !subduction_bias.is_finite()
        || xyz_ptr.is_null()
        || area_ptr.is_null()
        || neighbors_ptr.is_null()
        || plate_ptr.is_null()
        || convergence_ptr.is_null()
        || divergence_ptr.is_null()
        || shear_ptr.is_null()
        || subduction_ptr.is_null()
        || hotspot_ptr.is_null()
    {
        return 1;
    }
    let total = cell_count as usize;
    let Some(xyz_len) = total.checked_mul(3) else {
        return 2;
    };
    let Some(neighbor_len) = total.checked_mul(D4_NEIGHBORS) else {
        return 2;
    };
    let Some(plate_len) = total.checked_mul(SPHERICAL_PLATE_COMPONENTS) else {
        return 2;
    };
    if !valid_ffi_len::<f32>(xyz_len)
        || !valid_ffi_len::<f64>(total)
        || !valid_ffi_len::<i32>(neighbor_len)
        || !valid_ffi_len::<f32>(plate_len)
        || !valid_ffi_len::<f32>(total)
    {
        return 2;
    }

    let xyz = std::slice::from_raw_parts(xyz_ptr, xyz_len);
    let areas = std::slice::from_raw_parts(area_ptr, total);
    let neighbors = std::slice::from_raw_parts(neighbors_ptr, neighbor_len);
    let plate_field = std::slice::from_raw_parts_mut(plate_ptr, plate_len);
    let convergence_field = std::slice::from_raw_parts_mut(convergence_ptr, total);
    let divergence_field = std::slice::from_raw_parts_mut(divergence_ptr, total);
    let shear_field = std::slice::from_raw_parts_mut(shear_ptr, total);
    let subduction_field = std::slice::from_raw_parts_mut(subduction_ptr, total);
    let hotspot_field = std::slice::from_raw_parts_mut(hotspot_ptr, total);

    let mut cells = Vec::with_capacity(total);
    for (index, vector) in xyz.chunks_exact(3).enumerate() {
        let cell = SVec3::from_f32(vector);
        let norm = cell.norm();
        if !norm.is_finite()
            || !(0.99..=1.01).contains(&norm)
            || !areas[index].is_finite()
            || areas[index] <= 0.0
        {
            return 3;
        }
        cells.push(cell.normalized());
    }
    for (index, adjacent) in neighbors.chunks_exact(D4_NEIGHBORS).enumerate() {
        let mut unique = [adjacent[0], adjacent[1], adjacent[2], adjacent[3]];
        unique.sort_unstable();
        if unique.windows(2).any(|pair| pair[0] == pair[1])
            || adjacent.iter().any(|neighbor| {
                *neighbor < 0 || *neighbor as usize >= total || *neighbor as usize == index
            })
        {
            return 4;
        }
    }

    let plate_count = num_plates as usize;
    let warp_terms = spherical_warp_terms(seed);
    let warped_cells: Vec<SVec3> = cells
        .par_iter()
        .map(|cell| warp_spherical_direction(*cell, &warp_terms))
        .collect();
    let mut seeds = sample_spherical_plate_seeds(seed, plate_count);
    let mut assignments = vec![0usize; total];
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA17F_98C3_5D21_2F4B);
    for _ in 0..lloyd_iterations.max(0) as usize {
        assign_spherical_cells(&warped_cells, &seeds, &warp_terms, &mut assignments);
        recentre_spherical_seeds(&mut seeds, &cells, areas, &assignments, &mut rng);
    }
    assign_spherical_cells(&warped_cells, &seeds, &warp_terms, &mut assignments);
    if !ensure_nonempty_spherical_plates(
        &cells,
        &warped_cells,
        &mut seeds,
        &warp_terms,
        &mut assignments,
    ) {
        return 5;
    }
    if !repair_plate_connectivity(&mut assignments, neighbors, plate_count) {
        return 5;
    }
    let plate_areas = spherical_plate_areas(plate_count, areas, &assignments);
    assign_spherical_angular_velocities(&mut seeds, &plate_areas, velocity_scale, drift_bias, seed);

    let velocities: Vec<SVec3> = cells
        .par_iter()
        .zip(assignments.par_iter())
        .map(|(cell, plate)| seeds[*plate].angular_velocity.cross(*cell))
        .collect();
    let (continental, thickness, density, continental_fraction) =
        build_spherical_crust_provinces(&cells, areas, continental_fraction_target, seed);
    let boundary = compute_spherical_boundaries(
        &cells,
        neighbors,
        &assignments,
        &velocities,
        &continental,
        &thickness,
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
        .par_chunks_mut(SPHERICAL_PLATE_COMPONENTS)
        .enumerate()
        .for_each(|(index, output)| {
            output[0] = assignments[index] as f32;
            output[1] = f32::from(continental[index]);
            output[2] = thickness[index];
            output[3] = density[index];
            output[4] = velocities[index].x as f32;
            output[5] = velocities[index].y as f32;
            output[6] = velocities[index].z as f32;
        });
    convergence_field.copy_from_slice(&boundary.convergence);
    divergence_field.copy_from_slice(&boundary.divergence);
    shear_field.copy_from_slice(&boundary.shear);
    subduction_field.copy_from_slice(&boundary.subduction);
    hotspot_field.copy_from_slice(&hotspot);

    if !stats_out.is_null() {
        let total_area: f64 = areas.iter().sum();
        let mut speed_sum = 0.0f64;
        let mut speed_sq_sum = 0.0f64;
        for (velocity, area) in velocities.iter().zip(areas.iter()) {
            let speed = velocity.norm();
            speed_sum += speed * *area;
            speed_sq_sum += speed * speed * *area;
        }
        let velocity_mean = speed_sum / total_area;
        let velocity_std = (speed_sq_sum / total_area - velocity_mean * velocity_mean)
            .max(0.0)
            .sqrt();
        *stats_out = TectonicsStats {
            plate_count: num_plates,
            continental_fraction,
            velocity_mean,
            velocity_std,
            hotspot_mean,
            boundary_metric_mean: (boundary.convergence_sum
                + boundary.divergence_sum
                + boundary.shear_sum)
                / total as f64,
            convergence_sum: boundary.convergence_sum,
            divergence_sum: boundary.divergence_sum,
            shear_sum: boundary.shear_sum,
            subduction_mean: boundary.subduction_sum / total as f64,
            hotspot_count,
        };
    }
    0
}

#[no_mangle]
/// Generate plate assignments and instantaneous boundary fields.
///
/// # Safety
///
/// Every output pointer must reference an aligned writable buffer of the
/// length implied by `height` and `width`. Output buffers must not alias one
/// another. `stats_out` may be null; all field pointers must be valid.
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
        assign_cells_into(
            &seeds,
            &mut assignments,
            height,
            width,
            wrap_x,
            wrap_y,
            seed,
        );
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

    assign_cells_into(
        &seeds,
        &mut assignments,
        height,
        width,
        wrap_x,
        wrap_y,
        seed,
    );
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

    let plate_thickness: Vec<f32> = plate_types
        .iter()
        .map(|t| {
            if *t == PlateType::Continental {
                35.0
            } else {
                8.0
            }
        })
        .collect();
    let plate_density: Vec<f32> = plate_types
        .iter()
        .map(|t| {
            if *t == PlateType::Continental {
                2.8
            } else {
                3.3
            }
        })
        .collect();

    let plate_velocities =
        compute_plate_velocities(&seeds, &counts, velocity_scale, drift_bias, seed, height);

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

    let mut boundary = compute_boundary_fields(
        &assignments,
        &plate_types,
        &plate_thickness,
        &plate_density,
        &cell_vel_u,
        &cell_vel_v,
        width,
        height,
        wrap_x,
        wrap_y,
        subduction_bias,
    );
    condition_boundary_fields(&mut boundary, width, height, wrap_x, wrap_y, seed);

    let (continental_crust, crust_thickness, crust_density) =
        build_crust_provinces(height, width, continental_fraction_target, seed, wrap_x);

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
            chunk[1] = if continental_crust[idx] { 1.0 } else { 0.0 };
            chunk[2] = crust_thickness[idx];
            chunk[3] = crust_density[idx];
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

    let continental_cells: f64 = continental_crust
        .iter()
        .map(|&is_continental| if is_continental { 1.0 } else { 0.0 })
        .sum();
    let continental_fraction = continental_cells / total_f64;

    let boundary_total = boundary.convergence_sum + boundary.divergence_sum + boundary.shear_sum;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crust_provinces_are_deterministic_and_match_target_fraction() {
        let height = 32;
        let width = 64;
        let first = build_crust_provinces(height, width, 0.35, 1234, true);
        let second = build_crust_provinces(height, width, 0.35, 1234, true);

        assert_eq!(first, second);
        let land_count = first.0.iter().filter(|&&value| value).count();
        assert_eq!(
            land_count,
            (0.35 * (height * width) as f32).round() as usize
        );
        assert!(first.1.iter().all(|value| (6.0..=42.0).contains(value)));
        assert!(first.2.iter().all(|value| (2.67..=3.35).contains(value)));
    }

    #[test]
    fn crust_provinces_do_not_copy_plate_boundaries() {
        let height = 24;
        let width = 48;
        let assignments: Vec<usize> = (0..height * width).map(|idx| idx % 2).collect();
        let (continental, _, _) = build_crust_provinces(height, width, 0.4, 9876, true);

        let continental_on_each_plate = [0usize, 1usize].map(|plate| {
            continental
                .iter()
                .zip(assignments.iter())
                .filter(|(is_land, owner)| **is_land && **owner == plate)
                .count()
        });
        assert!(continental_on_each_plate.iter().all(|&count| count > 0));
    }

    #[test]
    fn plate_warp_is_periodic_and_curves_interior_coordinates() {
        let width = 512.0;
        let height = 256.0;
        let seed = 42;
        let seam_a = warp_position([0.0, 100.0], width, height, seed);
        let seam_b = warp_position([width, 100.0], width, height, seed);
        let interior = [180.0, 90.0];
        let warped = warp_position(interior, width, height, seed);

        assert!((seam_a[0] - seam_b[0]).abs() < 1e-4);
        assert!((seam_a[1] - seam_b[1]).abs() < 1e-4);
        assert!((warped[0] - interior[0]).abs() > 0.5 || (warped[1] - interior[1]).abs() > 0.5);
    }

    #[test]
    fn boundary_diffusion_spreads_without_losing_signal() {
        let mut field = vec![0.0f32; 9 * 9];
        field[4 * 9 + 4] = 1.0;
        diffuse_field(&mut field, 9, 9, true, true, 3);

        let sum: f32 = field.iter().sum();
        let nonzero = field.iter().filter(|&&value| value > 0.0).count();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(nonzero > 5);
        assert!(field[4 * 9 + 4] < 1.0);
    }

    #[test]
    fn plate_velocity_stream_seeds_do_not_overflow_in_debug_builds() {
        let seeds: Vec<PlateSeed> = (0..64)
            .map(|index| PlateSeed {
                position: [index as f32, index as f32 * 0.5],
                plate_type: PlateType::Oceanic,
            })
            .collect();
        let counts = vec![1.0f32; seeds.len()];
        let velocities = compute_plate_velocities(&seeds, &counts, 1.0, 0.1, u64::MAX, 128);
        assert_eq!(velocities.len(), seeds.len());
        assert!(velocities
            .iter()
            .flatten()
            .all(|component| component.is_finite()));
    }

    #[test]
    fn spherical_warp_is_deterministic_unit_length_and_nontrivial() {
        let direction = SVec3::new(0.3, -0.4, 0.866_025_403_8).normalized();
        let terms = spherical_warp_terms(42);
        assert_eq!(terms.len(), 8);
        assert!(terms[7].frequency > terms[4].frequency);
        assert!(terms[7].amplitude < terms[0].amplitude);
        let first = warp_spherical_direction(direction, &terms);
        let second = warp_spherical_direction(direction, &terms);
        assert!((first.norm() - 1.0).abs() < 1e-12);
        assert!((first - second).norm() < 1e-15);
        assert!((first - direction).norm() > 1e-4);
    }

    #[test]
    fn spherical_plate_rotation_has_zero_area_weighted_mean() {
        let mut seeds = sample_spherical_plate_seeds(19, 24);
        let areas: Vec<f64> = (0..seeds.len()).map(|index| index as f64 + 1.0).collect();
        assign_spherical_angular_velocities(&mut seeds, &areas, 1.0, 0.2, 19);
        let total_area: f64 = areas.iter().sum();
        let mean = seeds
            .iter()
            .zip(areas.iter())
            .fold(SVec3::default(), |sum, (plate, area)| {
                sum + plate.angular_velocity * *area
            })
            * (1.0 / total_area);
        assert!(mean.norm() < 1e-12);
    }

    #[test]
    fn spherical_drift_bias_tracks_seed_latitude() {
        let mut seeds = sample_spherical_plate_seeds(23, 32);
        let areas = vec![1.0f64; seeds.len()];
        assign_spherical_angular_velocities(&mut seeds, &areas, 0.001, 8.0, 23);
        let covariance: f64 = seeds
            .iter()
            .map(|plate| plate.position.z * plate.angular_velocity.z)
            .sum::<f64>()
            / seeds.len() as f64;
        assert!(covariance > 0.5);
    }

    #[test]
    fn connectivity_repair_absorbs_isolated_fragments() {
        let width = 4usize;
        let height = 4usize;
        let mut neighbors = vec![0i32; width * height * D4_NEIGHBORS];
        for row in 0..height {
            for col in 0..width {
                let index = row * width + col;
                neighbors[index * 4] = (((row + height - 1) % height) * width + col) as i32;
                neighbors[index * 4 + 1] = (((row + 1) % height) * width + col) as i32;
                neighbors[index * 4 + 2] = (row * width + (col + width - 1) % width) as i32;
                neighbors[index * 4 + 3] = (row * width + (col + 1) % width) as i32;
            }
        }
        let mut assignments = vec![1usize; width * height];
        assignments[0] = 0;
        assignments[1] = 0;
        assignments[width] = 0;
        assignments[width + 1] = 0;
        assignments[2 * width + 2] = 0;
        assert!(repair_plate_connectivity(&mut assignments, &neighbors, 2));

        for plate in 0..2 {
            let members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter_map(|(index, owner)| (*owner == plate).then_some(index))
                .collect();
            let mut visited = vec![false; assignments.len()];
            let mut pending = vec![members[0]];
            visited[members[0]] = true;
            while let Some(cell) = pending.pop() {
                for &neighbor in &neighbors[cell * 4..cell * 4 + 4] {
                    let adjacent = neighbor as usize;
                    if assignments[adjacent] == plate && !visited[adjacent] {
                        visited[adjacent] = true;
                        pending.push(adjacent);
                    }
                }
            }
            assert!(members.iter().all(|index| visited[*index]));
        }
    }

    #[test]
    fn spherical_crust_provinces_are_independent_and_area_targeted() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let cells: Vec<SVec3> = (0..4096).map(|_| random_unit_vector(&mut rng)).collect();
        let areas = vec![4.0 * std::f64::consts::PI / cells.len() as f64; cells.len()];
        let first = build_spherical_crust_provinces(&cells, &areas, 0.35, 88);
        let second = build_spherical_crust_provinces(&cells, &areas, 0.35, 88);
        assert_eq!(first.0, second.0);
        assert!((first.3 - 0.35).abs() <= 1.0 / cells.len() as f64);
        assert!(first.1.iter().all(|value| (5.5..=43.0).contains(value)));
        assert!(first.2.iter().all(|value| (2.66..=3.37).contains(value)));
    }
}
