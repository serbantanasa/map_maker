use std::collections::{HashMap, HashSet};

use topology_native::{
    cubed_sphere_cell_area_steradians, cubed_sphere_cell_xyz, cubed_sphere_decode_index,
    cubed_sphere_global_index, cubed_sphere_neighbor_index,
};

const D4_NEIGHBORS: usize = 4;
const TERRAIN_START_FREQUENCY: f64 = 4.0;

#[no_mangle]
pub extern "C" fn l3_terrain_native_abi_version() -> u32 {
    1
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct L3TerrainConfig {
    pub parent_resolution: i32,
    pub factor: i32,
    pub planet_radius_m: f64,
    pub terrain_seed: u64,
    pub relief_realization_fraction: f32,
    pub base_wavelength_m: f32,
    pub octave_count: i32,
    pub persistence: f32,
    pub domain_warp_fraction: f32,
    pub orogenic_ridge_fraction: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct L3TerrainStats {
    pub parent_count: i32,
    pub cell_count: i64,
    pub fine_resolution: i32,
    pub context_neighbor_count: i64,
    pub missing_context_neighbor_count: i64,
    pub selected_area_km2: f64,
    pub maximum_parent_area_relative_error: f64,
    pub maximum_parent_elevation_error_m: f64,
    pub minimum_elevation_m: f32,
    pub maximum_elevation_m: f32,
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn from_slice(values: &[f32], offset: usize) -> Self {
        Self {
            x: values[offset] as f64,
            y: values[offset + 1] as f64,
            z: values[offset + 2] as f64,
        }
    }

    fn from_array(values: [f64; 3]) -> Self {
        Self {
            x: values[0],
            y: values[1],
            z: values[2],
        }
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalized_or(self, fallback: Self) -> Self {
        let norm = self.norm();
        if norm > 1e-12 && norm.is_finite() {
            self.scale(1.0 / norm)
        } else {
            fallback
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }
}

#[derive(Clone, Copy)]
struct Cell {
    cell_id: u64,
    parent_id: i32,
    face: u8,
    row: i32,
    col: i32,
    xyz: [f32; 3],
    area_km2: f64,
    elevation_m: f32,
    offset_m: f32,
    unresolved_relief_m: f32,
}

struct Context<'a> {
    elevation_m: &'a [f32],
    relief_m: &'a [f32],
    rock_strength: &'a [f32],
    orogenic_strength: &'a [f32],
    ridge_direction_xyz: &'a [f32],
    row_by_id: HashMap<i32, usize>,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn lattice_hash(seed: u64, x: i64, y: i64, z: i64) -> f64 {
    let mut bits = seed;
    bits = splitmix64(bits ^ (x as u64).wrapping_mul(0x9e37_79b1_85eb_ca87));
    bits = splitmix64(bits ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f));
    bits = splitmix64(bits ^ (z as u64).wrapping_mul(0x1656_67b1_9e37_79f9));
    let unit = (bits >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    unit * 2.0 - 1.0
}

fn fade(value: f64) -> f64 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn lerp(first: f64, second: f64, amount: f64) -> f64 {
    first + (second - first) * amount
}

fn value_noise(position: Vec3, seed: u64) -> f64 {
    let x0 = position.x.floor() as i64;
    let y0 = position.y.floor() as i64;
    let z0 = position.z.floor() as i64;
    let tx = fade(position.x - x0 as f64);
    let ty = fade(position.y - y0 as f64);
    let tz = fade(position.z - z0 as f64);
    let mut values = [0.0f64; 8];
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                values[dz * 4 + dy * 2 + dx] =
                    lattice_hash(seed, x0 + dx as i64, y0 + dy as i64, z0 + dz as i64);
            }
        }
    }
    let z0_value = lerp(
        lerp(values[0], values[1], tx),
        lerp(values[2], values[3], tx),
        ty,
    );
    let z1_value = lerp(
        lerp(values[4], values[5], tx),
        lerp(values[6], values[7], tx),
        ty,
    );
    lerp(z0_value, z1_value, tz)
}

fn fbm(
    position: Vec3,
    seed: u64,
    octave_count: usize,
    persistence: f64,
    initial_frequency: f64,
) -> f64 {
    let mut frequency = initial_frequency;
    let mut amplitude = 1.0;
    let mut total = 0.0;
    let mut normalization = 0.0;
    for octave in 0..octave_count {
        let octave_seed = splitmix64(seed ^ octave as u64);
        total += value_noise(position.scale(frequency), octave_seed) * amplitude;
        normalization += amplitude;
        frequency *= 2.0;
        amplitude *= persistence;
    }
    total / normalization.max(f64::EPSILON)
}

fn terrain_signal(
    xyz: Vec3,
    ridge_direction: Vec3,
    rock_strength: f64,
    orogenic_strength: f64,
    config: &L3TerrainConfig,
) -> f64 {
    let scale = config.planet_radius_m / config.base_wavelength_m as f64;
    let base = xyz.scale(scale);
    let warp_scale = config.domain_warp_fraction as f64;
    let warp = Vec3 {
        x: value_noise(
            base.scale(0.43).add(Vec3 {
                x: 19.1,
                y: -7.3,
                z: 2.7,
            }),
            config.terrain_seed ^ 0x31a5,
        ),
        y: value_noise(
            base.scale(0.43).add(Vec3 {
                x: -3.8,
                y: 11.7,
                z: 23.2,
            }),
            config.terrain_seed ^ 0xb479,
        ),
        z: value_noise(
            base.scale(0.43).add(Vec3 {
                x: 7.4,
                y: 29.3,
                z: -13.6,
            }),
            config.terrain_seed ^ 0xe213,
        ),
    }
    .scale(warp_scale);
    let warped = base.add(warp);
    let octave_count = config.octave_count as usize;
    let persistence = config.persistence as f64;
    let rolling = fbm(
        warped,
        config.terrain_seed,
        octave_count,
        persistence,
        TERRAIN_START_FREQUENCY,
    );

    let radial = xyz.normalized_or(Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    });
    let tangent = ridge_direction
        .sub(radial.scale(ridge_direction.dot(radial)))
        .normalized_or(Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        });
    let parallel = tangent.scale(warped.dot(tangent));
    let perpendicular = warped.sub(parallel);
    let anisotropic = parallel.scale(0.58).add(perpendicular.scale(1.24));
    let ridge_noise = fbm(
        anisotropic.add(warp.scale(0.55)),
        config.terrain_seed ^ 0x8d58_91a3,
        octave_count.saturating_sub(1).max(2),
        persistence,
        TERRAIN_START_FREQUENCY,
    );
    let ridged = 1.0 - 2.0 * ridge_noise.abs();
    let ridge_weight = (orogenic_strength * config.orogenic_ridge_fraction as f64).clamp(0.0, 0.8);
    let lithology_sharpness = 0.88 + 0.24 * rock_strength;
    ((1.0 - ridge_weight) * rolling + ridge_weight * ridged * lithology_sharpness).clamp(-1.0, 1.0)
}

fn interpolate_scalar(
    center: f64,
    north: f64,
    south: f64,
    west: f64,
    east: f64,
    x: f64,
    y: f64,
) -> f64 {
    let horizontal = if x >= 0.0 {
        (east - center) * x
    } else {
        (center - west) * x
    };
    let vertical = if y >= 0.0 {
        (south - center) * y
    } else {
        (center - north) * y
    };
    center + horizontal + vertical
}

fn context_scalar(values: &[f32], context: &Context<'_>, parent: i32, fallback: f64) -> f64 {
    context
        .row_by_id
        .get(&parent)
        .map_or(fallback, |&row| values[row] as f64)
}

fn context_vector(context: &Context<'_>, parent: i32, fallback: Vec3) -> Vec3 {
    context.row_by_id.get(&parent).map_or(fallback, |&row| {
        Vec3::from_slice(context.ridge_direction_xyz, row * 3)
    })
}

fn neighbor_id(parent_id: i32, slot: usize, resolution: usize) -> Option<i32> {
    cubed_sphere_neighbor_index(parent_id as usize, slot, resolution)
        .and_then(|value| i32::try_from(value).ok())
}

fn generate_parent(
    parent_id: i32,
    context: &Context<'_>,
    config: &L3TerrainConfig,
) -> Result<Vec<Cell>, i32> {
    let coarse = config.parent_resolution as usize;
    let factor = config.factor as usize;
    let fine = coarse.checked_mul(factor).ok_or(2)?;
    let &parent_index = context.row_by_id.get(&parent_id).ok_or(4)?;
    let (face, parent_row, parent_col) =
        cubed_sphere_decode_index(parent_id as usize, coarse).ok_or(4)?;
    let parent_elevation = context.elevation_m[parent_index] as f64;
    let parent_relief = context.relief_m[parent_index] as f64;
    let parent_rock = context.rock_strength[parent_index] as f64;
    let parent_orogenic = context.orogenic_strength[parent_index] as f64;
    let parent_ridge = Vec3::from_slice(context.ridge_direction_xyz, parent_index * 3);
    let neighbor_ids = std::array::from_fn::<_, D4_NEIGHBORS, _>(|slot| {
        neighbor_id(parent_id, slot, coarse).unwrap_or(parent_id)
    });
    let scalar_neighbors = |values: &[f32], center: f64| -> [f64; D4_NEIGHBORS] {
        std::array::from_fn(|slot| context_scalar(values, context, neighbor_ids[slot], center))
    };
    let elevation_neighbors = scalar_neighbors(context.elevation_m, parent_elevation);
    let relief_neighbors = scalar_neighbors(context.relief_m, parent_relief);
    let rock_neighbors = scalar_neighbors(context.rock_strength, parent_rock);
    let orogenic_neighbors = scalar_neighbors(context.orogenic_strength, parent_orogenic);
    let ridge_neighbors = std::array::from_fn::<_, D4_NEIGHBORS, _>(|slot| {
        context_vector(context, neighbor_ids[slot], parent_ridge)
    });
    let radius_squared_km = (config.planet_radius_m / 1_000.0).powi(2);
    let mut cells = Vec::with_capacity(factor * factor);
    let mut raw_offsets = Vec::with_capacity(factor * factor);

    for child_row in 0..factor {
        for child_col in 0..factor {
            let row = parent_row * factor + child_row;
            let col = parent_col * factor + child_col;
            let cell_id = cubed_sphere_global_index(face, row, col, fine).ok_or(4)? as u64;
            let xyz_array = cubed_sphere_cell_xyz(face, row, col, fine).ok_or(4)?;
            let xyz = Vec3::from_array(xyz_array);
            let area = cubed_sphere_cell_area_steradians(face, row, col, fine).ok_or(4)?
                * radius_squared_km;
            let unit_y = (child_row as f64 + 0.5) / factor as f64;
            let unit_x = (child_col as f64 + 0.5) / factor as f64;
            let y = unit_y - 0.5;
            let x = unit_x - 0.5;
            let smooth_elevation = interpolate_scalar(
                parent_elevation,
                elevation_neighbors[0],
                elevation_neighbors[1],
                elevation_neighbors[2],
                elevation_neighbors[3],
                x,
                y,
            );
            let local_relief = interpolate_scalar(
                parent_relief,
                relief_neighbors[0],
                relief_neighbors[1],
                relief_neighbors[2],
                relief_neighbors[3],
                x,
                y,
            )
            .max(0.0);
            let local_rock = interpolate_scalar(
                parent_rock,
                rock_neighbors[0],
                rock_neighbors[1],
                rock_neighbors[2],
                rock_neighbors[3],
                x,
                y,
            )
            .clamp(0.0, 1.0);
            let local_orogenic = interpolate_scalar(
                parent_orogenic,
                orogenic_neighbors[0],
                orogenic_neighbors[1],
                orogenic_neighbors[2],
                orogenic_neighbors[3],
                x,
                y,
            )
            .clamp(0.0, 1.0);
            let ridge = parent_ridge
                .add(ridge_neighbors[0].sub(parent_ridge).scale(-y.min(0.0)))
                .add(ridge_neighbors[1].sub(parent_ridge).scale(y.max(0.0)))
                .add(ridge_neighbors[2].sub(parent_ridge).scale(-x.min(0.0)))
                .add(ridge_neighbors[3].sub(parent_ridge).scale(x.max(0.0)))
                .normalized_or(parent_ridge);
            let realization =
                config.relief_realization_fraction as f64 * (0.78 + 0.22 * local_rock);
            let residual = terrain_signal(xyz, ridge, local_rock, local_orogenic, config)
                * local_relief
                * realization;
            raw_offsets.push(smooth_elevation - parent_elevation + residual);
            cells.push(Cell {
                cell_id,
                parent_id,
                face: face as u8,
                row: row as i32,
                col: col as i32,
                xyz: [xyz.x as f32, xyz.y as f32, xyz.z as f32],
                area_km2: area,
                elevation_m: parent_elevation as f32,
                offset_m: 0.0,
                unresolved_relief_m: (local_relief * (1.0 - realization)).max(0.0) as f32,
            });
        }
    }

    for (cell, offset) in cells.iter_mut().zip(raw_offsets) {
        cell.elevation_m = (parent_elevation + offset) as f32;
        cell.offset_m = cell.elevation_m - parent_elevation as f32;
    }
    Ok(cells)
}

fn validate_config(config: &L3TerrainConfig) -> Result<(usize, usize), i32> {
    let parent_resolution = usize::try_from(config.parent_resolution).map_err(|_| 2)?;
    let factor = usize::try_from(config.factor).map_err(|_| 2)?;
    if parent_resolution == 0
        || factor < 2
        || !config.planet_radius_m.is_finite()
        || config.planet_radius_m <= 0.0
        || !config.relief_realization_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.relief_realization_fraction)
        || !config.base_wavelength_m.is_finite()
        || config.base_wavelength_m <= 0.0
        || !(2..=8).contains(&config.octave_count)
        || !config.persistence.is_finite()
        || !(0.0..1.0).contains(&config.persistence)
        || !config.domain_warp_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.domain_warp_fraction)
        || !config.orogenic_ridge_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.orogenic_ridge_fraction)
    {
        return Err(2);
    }
    let fine = parent_resolution.checked_mul(factor).ok_or(2)?;
    if fine > i32::MAX as usize {
        return Err(2);
    }
    Ok((parent_resolution, fine))
}

#[allow(clippy::too_many_arguments)]
fn run_chunk(
    config: &L3TerrainConfig,
    context_ids: &[i32],
    context_elevation_m: &[f32],
    context_relief_m: &[f32],
    context_rock_strength: &[f32],
    context_orogenic_strength: &[f32],
    context_ridge_direction_xyz: &[f32],
    chunk_parent_ids: &[i32],
    cell_id_out: &mut [u64],
    parent_id_out: &mut [i32],
    face_out: &mut [u8],
    row_out: &mut [i32],
    col_out: &mut [i32],
    xyz_out: &mut [f32],
    area_km2_out: &mut [f64],
    elevation_m_out: &mut [f32],
    offset_m_out: &mut [f32],
    unresolved_relief_m_out: &mut [f32],
    stats: &mut L3TerrainStats,
) -> Result<(), i32> {
    let (_, fine) = validate_config(config)?;
    if context_ids.is_empty() || chunk_parent_ids.is_empty() {
        return Err(1);
    }
    let context_count = context_ids.len();
    if context_elevation_m.len() != context_count
        || context_relief_m.len() != context_count
        || context_rock_strength.len() != context_count
        || context_orogenic_strength.len() != context_count
        || context_ridge_direction_xyz.len() != context_count * 3
    {
        return Err(1);
    }
    let output_count = chunk_parent_ids
        .len()
        .checked_mul(config.factor as usize)
        .and_then(|value| value.checked_mul(config.factor as usize))
        .ok_or(2)?;
    if cell_id_out.len() != output_count
        || parent_id_out.len() != output_count
        || face_out.len() != output_count
        || row_out.len() != output_count
        || col_out.len() != output_count
        || xyz_out.len() != output_count * 3
        || area_km2_out.len() != output_count
        || elevation_m_out.len() != output_count
        || offset_m_out.len() != output_count
        || unresolved_relief_m_out.len() != output_count
    {
        return Err(1);
    }
    let mut row_by_id = HashMap::with_capacity(context_count);
    for (row, &cell_id) in context_ids.iter().enumerate() {
        if cell_id < 0
            || row_by_id.insert(cell_id, row).is_some()
            || !context_elevation_m[row].is_finite()
            || !context_relief_m[row].is_finite()
            || context_relief_m[row] < 0.0
            || !context_rock_strength[row].is_finite()
            || !(0.0..=1.0).contains(&context_rock_strength[row])
            || !context_orogenic_strength[row].is_finite()
            || !(0.0..=1.0).contains(&context_orogenic_strength[row])
            || context_ridge_direction_xyz[row * 3..row * 3 + 3]
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(3);
        }
    }
    let mut chunk_set = HashSet::with_capacity(chunk_parent_ids.len());
    if chunk_parent_ids
        .iter()
        .any(|id| !row_by_id.contains_key(id) || !chunk_set.insert(*id))
    {
        return Err(4);
    }
    let context = Context {
        elevation_m: context_elevation_m,
        relief_m: context_relief_m,
        rock_strength: context_rock_strength,
        orogenic_strength: context_orogenic_strength,
        ridge_direction_xyz: context_ridge_direction_xyz,
        row_by_id,
    };
    let children_per_parent = (config.factor as usize).pow(2);
    let mut selected_area_km2 = 0.0;
    let mut context_neighbor_count = 0i64;
    let mut missing_context_neighbor_count = 0i64;
    let mut maximum_area_error = 0.0f64;
    let mut maximum_elevation_error = 0.0f64;
    let mut minimum_elevation = f32::INFINITY;
    let mut maximum_elevation = f32::NEG_INFINITY;
    for (parent_offset, &parent_id) in chunk_parent_ids.iter().enumerate() {
        for slot in 0..D4_NEIGHBORS {
            context_neighbor_count += 1;
            let present = neighbor_id(parent_id, slot, config.parent_resolution as usize)
                .is_some_and(|neighbor| context.row_by_id.contains_key(&neighbor));
            if !present {
                missing_context_neighbor_count += 1;
            }
        }
        let generated = generate_parent(parent_id, &context, config)?;
        let start = parent_offset * children_per_parent;
        let parent_row = context.row_by_id[&parent_id];
        let parent_area_steradians = {
            let (face, row, col) =
                cubed_sphere_decode_index(parent_id as usize, config.parent_resolution as usize)
                    .ok_or(4)?;
            cubed_sphere_cell_area_steradians(face, row, col, config.parent_resolution as usize)
                .ok_or(4)?
        };
        let parent_area_km2 = parent_area_steradians * (config.planet_radius_m / 1_000.0).powi(2);
        let restricted_area = generated.iter().map(|cell| cell.area_km2).sum::<f64>();
        let restricted_elevation = generated
            .iter()
            .map(|cell| cell.area_km2 * cell.elevation_m as f64)
            .sum::<f64>()
            / restricted_area;
        selected_area_km2 += restricted_area;
        maximum_area_error =
            maximum_area_error.max(((restricted_area - parent_area_km2) / parent_area_km2).abs());
        maximum_elevation_error = maximum_elevation_error
            .max((restricted_elevation - context.elevation_m[parent_row] as f64).abs());
        for (local, cell) in generated.into_iter().enumerate() {
            let output = start + local;
            cell_id_out[output] = cell.cell_id;
            parent_id_out[output] = cell.parent_id;
            face_out[output] = cell.face;
            row_out[output] = cell.row;
            col_out[output] = cell.col;
            xyz_out[output * 3..output * 3 + 3].copy_from_slice(&cell.xyz);
            area_km2_out[output] = cell.area_km2;
            elevation_m_out[output] = cell.elevation_m;
            offset_m_out[output] = cell.offset_m;
            unresolved_relief_m_out[output] = cell.unresolved_relief_m;
            minimum_elevation = minimum_elevation.min(cell.elevation_m);
            maximum_elevation = maximum_elevation.max(cell.elevation_m);
        }
    }
    *stats = L3TerrainStats {
        parent_count: chunk_parent_ids.len() as i32,
        cell_count: output_count as i64,
        fine_resolution: fine as i32,
        context_neighbor_count,
        missing_context_neighbor_count,
        selected_area_km2,
        maximum_parent_area_relative_error: maximum_area_error,
        maximum_parent_elevation_error_m: maximum_elevation_error,
        minimum_elevation_m: minimum_elevation,
        maximum_elevation_m: maximum_elevation,
    };
    Ok(())
}

#[no_mangle]
/// Generate one deterministic parent-aligned L3 terrain chunk.
///
/// # Safety
///
/// Input and output pointers must reference non-overlapping buffers with the
/// lengths implied by the context count, chunk parent count, and factor.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn l3_terrain_generate_chunk(
    config_ptr: *const L3TerrainConfig,
    context_count: i32,
    context_parent_ids_ptr: *const i32,
    context_elevation_m_ptr: *const f32,
    context_relief_m_ptr: *const f32,
    context_rock_strength_ptr: *const f32,
    context_orogenic_strength_ptr: *const f32,
    context_ridge_direction_xyz_ptr: *const f32,
    chunk_parent_count: i32,
    chunk_parent_ids_ptr: *const i32,
    cell_id_out_ptr: *mut u64,
    parent_id_out_ptr: *mut i32,
    face_out_ptr: *mut u8,
    row_out_ptr: *mut i32,
    col_out_ptr: *mut i32,
    xyz_out_ptr: *mut f32,
    area_km2_out_ptr: *mut f64,
    elevation_m_out_ptr: *mut f32,
    offset_m_out_ptr: *mut f32,
    unresolved_relief_m_out_ptr: *mut f32,
    stats_out_ptr: *mut L3TerrainStats,
) -> i32 {
    if config_ptr.is_null()
        || context_count <= 0
        || chunk_parent_count <= 0
        || context_parent_ids_ptr.is_null()
        || context_elevation_m_ptr.is_null()
        || context_relief_m_ptr.is_null()
        || context_rock_strength_ptr.is_null()
        || context_orogenic_strength_ptr.is_null()
        || context_ridge_direction_xyz_ptr.is_null()
        || chunk_parent_ids_ptr.is_null()
        || cell_id_out_ptr.is_null()
        || parent_id_out_ptr.is_null()
        || face_out_ptr.is_null()
        || row_out_ptr.is_null()
        || col_out_ptr.is_null()
        || xyz_out_ptr.is_null()
        || area_km2_out_ptr.is_null()
        || elevation_m_out_ptr.is_null()
        || offset_m_out_ptr.is_null()
        || unresolved_relief_m_out_ptr.is_null()
        || stats_out_ptr.is_null()
    {
        return 1;
    }
    let config = unsafe { &*config_ptr };
    let factor = match usize::try_from(config.factor) {
        Ok(value) if value >= 2 => value,
        _ => return 2,
    };
    let context_len = context_count as usize;
    let parent_len = chunk_parent_count as usize;
    let Some(output_len) = parent_len
        .checked_mul(factor)
        .and_then(|value| value.checked_mul(factor))
    else {
        return 2;
    };
    let context_ids = unsafe { std::slice::from_raw_parts(context_parent_ids_ptr, context_len) };
    let context_elevation_m =
        unsafe { std::slice::from_raw_parts(context_elevation_m_ptr, context_len) };
    let context_relief_m = unsafe { std::slice::from_raw_parts(context_relief_m_ptr, context_len) };
    let context_rock_strength =
        unsafe { std::slice::from_raw_parts(context_rock_strength_ptr, context_len) };
    let context_orogenic_strength =
        unsafe { std::slice::from_raw_parts(context_orogenic_strength_ptr, context_len) };
    let context_ridge_direction_xyz =
        unsafe { std::slice::from_raw_parts(context_ridge_direction_xyz_ptr, context_len * 3) };
    let chunk_parent_ids = unsafe { std::slice::from_raw_parts(chunk_parent_ids_ptr, parent_len) };
    let cell_id_out = unsafe { std::slice::from_raw_parts_mut(cell_id_out_ptr, output_len) };
    let parent_id_out = unsafe { std::slice::from_raw_parts_mut(parent_id_out_ptr, output_len) };
    let face_out = unsafe { std::slice::from_raw_parts_mut(face_out_ptr, output_len) };
    let row_out = unsafe { std::slice::from_raw_parts_mut(row_out_ptr, output_len) };
    let col_out = unsafe { std::slice::from_raw_parts_mut(col_out_ptr, output_len) };
    let xyz_out = unsafe { std::slice::from_raw_parts_mut(xyz_out_ptr, output_len * 3) };
    let area_km2_out = unsafe { std::slice::from_raw_parts_mut(area_km2_out_ptr, output_len) };
    let elevation_m_out =
        unsafe { std::slice::from_raw_parts_mut(elevation_m_out_ptr, output_len) };
    let offset_m_out = unsafe { std::slice::from_raw_parts_mut(offset_m_out_ptr, output_len) };
    let unresolved_relief_m_out =
        unsafe { std::slice::from_raw_parts_mut(unresolved_relief_m_out_ptr, output_len) };
    let stats_out = unsafe { &mut *stats_out_ptr };
    match run_chunk(
        config,
        context_ids,
        context_elevation_m,
        context_relief_m,
        context_rock_strength,
        context_orogenic_strength,
        context_ridge_direction_xyz,
        chunk_parent_ids,
        cell_id_out,
        parent_id_out,
        face_out,
        row_out,
        col_out,
        xyz_out,
        area_km2_out,
        elevation_m_out,
        offset_m_out,
        unresolved_relief_m_out,
        stats_out,
    ) {
        Ok(()) => 0,
        Err(status) => status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (
        L3TerrainConfig,
        Vec<i32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let resolution = 2_048usize;
        let face = 4usize;
        let ids = (1_000..=1_003)
            .flat_map(|row| {
                (1_000..=1_003).map(move |col| {
                    cubed_sphere_global_index(face, row, col, resolution).unwrap() as i32
                })
            })
            .collect::<Vec<_>>();
        let elevation = ids
            .iter()
            .map(|id| {
                let (_, row, col) = cubed_sphere_decode_index(*id as usize, resolution).unwrap();
                250.0 + row as f32 * 18.0 + col as f32 * 9.0
            })
            .collect::<Vec<_>>();
        let relief = vec![180.0; ids.len()];
        let rock = vec![0.7; ids.len()];
        let orogenic = vec![0.55; ids.len()];
        let ridge = ids
            .iter()
            .flat_map(|_| [0.0f32, 1.0, 0.0])
            .collect::<Vec<_>>();
        let config = L3TerrainConfig {
            parent_resolution: resolution as i32,
            factor: 22,
            planet_radius_m: 6_371_000.0,
            terrain_seed: 42,
            relief_realization_fraction: 0.42,
            base_wavelength_m: 16_000.0,
            octave_count: 4,
            persistence: 0.52,
            domain_warp_fraction: 0.2,
            orogenic_ridge_fraction: 0.3,
        };
        (config, ids, elevation, relief, rock, orogenic, ridge)
    }

    fn generated(chunk: &[i32]) -> (Vec<u64>, Vec<f32>, L3TerrainStats) {
        let (config, ids, elevation, relief, rock, orogenic, ridge) = fixture();
        let count = chunk.len() * config.factor as usize * config.factor as usize;
        let mut cell_ids = vec![0; count];
        let mut parents = vec![0; count];
        let mut faces = vec![0; count];
        let mut rows = vec![0; count];
        let mut cols = vec![0; count];
        let mut xyz = vec![0.0; count * 3];
        let mut areas = vec![0.0; count];
        let mut terrain = vec![0.0; count];
        let mut offsets = vec![0.0; count];
        let mut unresolved = vec![0.0; count];
        let mut stats = L3TerrainStats::default();
        run_chunk(
            &config,
            &ids,
            &elevation,
            &relief,
            &rock,
            &orogenic,
            &ridge,
            chunk,
            &mut cell_ids,
            &mut parents,
            &mut faces,
            &mut rows,
            &mut cols,
            &mut xyz,
            &mut areas,
            &mut terrain,
            &mut offsets,
            &mut unresolved,
            &mut stats,
        )
        .unwrap();
        (cell_ids, terrain, stats)
    }

    #[test]
    fn terrain_is_parent_conditioned_and_uses_64_bit_global_ids() {
        let (_, ids, ..) = fixture();
        let chunk = [ids[5], ids[6]];
        let (cell_ids, terrain, stats) = generated(&chunk);
        assert_eq!(cell_ids.len(), 968);
        assert!(cell_ids.iter().all(|value| *value > u32::MAX as u64));
        assert!(terrain.iter().all(|value| value.is_finite()));
        assert!(stats.maximum_parent_area_relative_error < 1e-9);
        assert!(stats.maximum_parent_elevation_error_m < 25.0);
    }

    #[test]
    fn output_is_invariant_to_chunk_partitioning() {
        let (_, ids, ..) = fixture();
        let first = [ids[5], ids[6]];
        let (joint_ids, joint_terrain, _) = generated(&first);
        let (first_ids, first_terrain, _) = generated(&first[..1]);
        let (second_ids, second_terrain, _) = generated(&first[1..]);
        assert_eq!(joint_ids, [first_ids, second_ids].concat());
        assert_eq!(joint_terrain, [first_terrain, second_terrain].concat());
    }
}
