use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use topology_native::{
    cubed_sphere_cell_area_steradians, cubed_sphere_cell_xyz, cubed_sphere_decode_index,
    cubed_sphere_global_index, cubed_sphere_neighbor_index,
};

const D4_NEIGHBORS: usize = 4;

#[no_mangle]
pub extern "C" fn refinement_native_abi_version() -> u32 {
    2
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RefinementConfig {
    pub coarse_resolution: i32,
    pub factor: i32,
    pub planet_radius_m: f64,
    pub terrain_seed: u64,
    pub terrain_noise_fraction: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RefinedCellRecord {
    pub fine_cell_id: i32,
    pub parent_cell_id: i32,
    pub face: i32,
    pub row: i32,
    pub col: i32,
    pub xyz: [f32; 3],
    pub area_km2: f64,
    pub terrain_elevation_m: f32,
    pub terrain_offset_m: f32,
    pub parent_relief_m: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RefinedReachRecord {
    pub reach_id: i32,
    pub path_offset: i32,
    pub path_count: i32,
    pub entry_fine_cell: i32,
    pub exit_fine_cell: i32,
    pub path_length_m: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReachCellRecord {
    pub reach_id: i32,
    pub fine_cell_id: i32,
    pub parent_cell_id: i32,
    pub path_order: i32,
    pub reach_length_m: f64,
    pub channel_fraction: f32,
    pub valley_fraction: f32,
    pub floodplain_fraction: f32,
    pub potential_incised_volume_m3: f64,
}

#[repr(C)]
pub struct RefinedCellArray {
    pub data: *mut RefinedCellRecord,
    pub len: usize,
}

#[repr(C)]
pub struct RefinedReachArray {
    pub data: *mut RefinedReachRecord,
    pub len: usize,
}

#[repr(C)]
pub struct ReachCellArray {
    pub data: *mut ReachCellRecord,
    pub len: usize,
}

#[repr(C)]
pub struct Int32Array {
    pub data: *mut i32,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RefinementStats {
    pub parent_count: i32,
    pub child_count: i32,
    pub reach_count: i32,
    pub reach_cell_count: i32,
    pub fine_resolution: i32,
    pub path_topology_valid: i32,
    pub selected_area_km2: f64,
    pub maximum_parent_area_relative_error: f64,
    pub maximum_parent_elevation_error_m: f64,
    pub total_reach_length_km: f64,
    pub represented_channel_area_km2: f64,
    pub represented_valley_area_km2: f64,
    pub represented_floodplain_area_km2: f64,
    pub total_potential_incised_volume_km3: f64,
}

struct Inputs<'a> {
    config: RefinementConfig,
    parent_ids: &'a [i32],
    parent_elevation_m: &'a [f32],
    parent_relief_m: &'a [f32],
    parent_area_steradians: &'a [f64],
    reach_ids: &'a [i32],
    reach_from_nodes: &'a [i32],
    reach_to_nodes: &'a [i32],
    reach_offsets: &'a [i32],
    reach_parent_cells: &'a [i32],
    channel_width_m: &'a [f32],
    valley_width_m: &'a [f32],
    floodplain_width_m: &'a [f32],
    incision_m: &'a [f32],
}

struct Outcome {
    cells: Vec<RefinedCellRecord>,
    reaches: Vec<RefinedReachRecord>,
    path_cells: Vec<i32>,
    memberships: Vec<ReachCellRecord>,
    stats: RefinementStats,
}

#[derive(Clone, Copy)]
struct QueueState {
    cost: f64,
    cell: i32,
}

impl PartialEq for QueueState {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell && self.cost.to_bits() == other.cost.to_bits()
    }
}

impl Eq for QueueState {}

impl PartialOrd for QueueState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn signed_noise(seed: u64, cell: i32) -> f64 {
    let bits = splitmix64(seed ^ cell as u32 as u64);
    let unit = (bits >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
    unit * 2.0 - 1.0
}

fn angular_distance(first: [f32; 3], second: [f32; 3]) -> f64 {
    let dot = (first[0] as f64 * second[0] as f64
        + first[1] as f64 * second[1] as f64
        + first[2] as f64 * second[2] as f64)
        .clamp(-1.0, 1.0);
    dot.acos()
}

fn validate_inputs(inputs: &Inputs<'_>) -> Result<(usize, usize), i32> {
    let coarse = usize::try_from(inputs.config.coarse_resolution).map_err(|_| 2)?;
    let factor = usize::try_from(inputs.config.factor).map_err(|_| 2)?;
    if coarse == 0
        || factor <= 1
        || !inputs.config.planet_radius_m.is_finite()
        || inputs.config.planet_radius_m <= 0.0
        || !inputs.config.terrain_noise_fraction.is_finite()
        || !(0.0..=1.0).contains(&inputs.config.terrain_noise_fraction)
    {
        return Err(2);
    }
    let fine = coarse.checked_mul(factor).ok_or(2)?;
    let fine_cells = 6usize
        .checked_mul(fine)
        .and_then(|value| value.checked_mul(fine))
        .ok_or(2)?;
    if fine_cells > i32::MAX as usize || inputs.parent_ids.is_empty() || inputs.reach_ids.is_empty()
    {
        return Err(2);
    }
    let parent_count = inputs.parent_ids.len();
    if inputs.parent_elevation_m.len() != parent_count
        || inputs.parent_relief_m.len() != parent_count
        || inputs.parent_area_steradians.len() != parent_count
    {
        return Err(1);
    }
    let reach_count = inputs.reach_ids.len();
    if inputs.reach_from_nodes.len() != reach_count
        || inputs.reach_to_nodes.len() != reach_count
        || inputs.reach_offsets.len() != reach_count + 1
        || inputs.channel_width_m.len() != reach_count
        || inputs.valley_width_m.len() != reach_count
        || inputs.floodplain_width_m.len() != reach_count
        || inputs.incision_m.len() != reach_count
    {
        return Err(1);
    }
    if inputs.reach_offsets.first() != Some(&0)
        || inputs.reach_offsets.last().copied() != Some(inputs.reach_parent_cells.len() as i32)
    {
        return Err(4);
    }
    let mut parent_set = HashSet::with_capacity(parent_count);
    let coarse_cells = 6usize * coarse * coarse;
    for index in 0..parent_count {
        let parent = inputs.parent_ids[index];
        if parent < 0
            || parent as usize >= coarse_cells
            || !parent_set.insert(parent)
            || !inputs.parent_elevation_m[index].is_finite()
            || !inputs.parent_relief_m[index].is_finite()
            || inputs.parent_relief_m[index] < 0.0
            || !inputs.parent_area_steradians[index].is_finite()
            || inputs.parent_area_steradians[index] <= 0.0
        {
            return Err(3);
        }
    }
    let mut reach_set = HashSet::with_capacity(reach_count);
    for reach_index in 0..reach_count {
        if !reach_set.insert(inputs.reach_ids[reach_index])
            || !inputs.channel_width_m[reach_index].is_finite()
            || inputs.channel_width_m[reach_index] <= 0.0
            || !inputs.valley_width_m[reach_index].is_finite()
            || inputs.valley_width_m[reach_index] < inputs.channel_width_m[reach_index]
            || !inputs.floodplain_width_m[reach_index].is_finite()
            || inputs.floodplain_width_m[reach_index] < 0.0
            || !inputs.incision_m[reach_index].is_finite()
            || inputs.incision_m[reach_index] < 0.0
        {
            return Err(4);
        }
        let start_raw = inputs.reach_offsets[reach_index];
        let end_raw = inputs.reach_offsets[reach_index + 1];
        let Some(path_length) = end_raw.checked_sub(start_raw) else {
            return Err(4);
        };
        if start_raw < 0 || end_raw < 0 || path_length < 2 {
            return Err(4);
        }
        let start = start_raw as usize;
        let end = end_raw as usize;
        if end > inputs.reach_parent_cells.len() {
            return Err(4);
        }
        let path = &inputs.reach_parent_cells[start..end];
        if path.first() != Some(&inputs.reach_from_nodes[reach_index])
            || path.last() != Some(&inputs.reach_to_nodes[reach_index])
            || path.iter().any(|cell| !parent_set.contains(cell))
        {
            return Err(4);
        }
        for pair in path.windows(2) {
            let source = pair[0] as usize;
            let target = pair[1] as usize;
            let adjacent = (0..D4_NEIGHBORS)
                .any(|slot| cubed_sphere_neighbor_index(source, slot, coarse) == Some(target));
            if !adjacent {
                return Err(4);
            }
        }
    }
    Ok((coarse, fine))
}

type ParentRanges = HashMap<i32, (usize, usize, usize)>;
type ChildGeneration = (Vec<RefinedCellRecord>, HashMap<i32, usize>, ParentRanges);
type ReachGeneration = (Vec<RefinedReachRecord>, Vec<i32>, Vec<ReachCellRecord>);

fn generate_children(
    inputs: &Inputs<'_>,
    coarse: usize,
    fine: usize,
) -> Result<ChildGeneration, i32> {
    let factor = inputs.config.factor as usize;
    let children_per_parent = factor * factor;
    let radius_squared_km = (inputs.config.planet_radius_m / 1_000.0).powi(2);
    let mut cells = Vec::with_capacity(inputs.parent_ids.len() * children_per_parent);
    let mut local_by_id = HashMap::with_capacity(cells.capacity());
    let mut ranges = HashMap::with_capacity(inputs.parent_ids.len());
    let parent_input_by_id = inputs
        .parent_ids
        .iter()
        .enumerate()
        .map(|(index, &parent)| (parent, index))
        .collect::<HashMap<_, _>>();

    for (parent_index, &parent_id) in inputs.parent_ids.iter().enumerate() {
        let (face, parent_row, parent_col) =
            cubed_sphere_decode_index(parent_id as usize, coarse).ok_or(3)?;
        let start = cells.len();
        for child_row in 0..factor {
            for child_col in 0..factor {
                let row = parent_row * factor + child_row;
                let col = parent_col * factor + child_col;
                let fine_id = cubed_sphere_global_index(face, row, col, fine).ok_or(3)?;
                let xyz = cubed_sphere_cell_xyz(face, row, col, fine).ok_or(3)?;
                let area = cubed_sphere_cell_area_steradians(face, row, col, fine).ok_or(3)?;
                let record = RefinedCellRecord {
                    fine_cell_id: fine_id as i32,
                    parent_cell_id: parent_id,
                    face: face as i32,
                    row: row as i32,
                    col: col as i32,
                    xyz: [xyz[0] as f32, xyz[1] as f32, xyz[2] as f32],
                    area_km2: area * radius_squared_km,
                    terrain_elevation_m: inputs.parent_elevation_m[parent_index],
                    terrain_offset_m: 0.0,
                    parent_relief_m: inputs.parent_relief_m[parent_index],
                };
                local_by_id.insert(record.fine_cell_id, cells.len());
                cells.push(record);
            }
        }
        ranges.insert(parent_id, (start, cells.len(), parent_index));
    }

    let mut noise = cells
        .iter()
        .map(|cell| signed_noise(inputs.config.terrain_seed, cell.fine_cell_id))
        .collect::<Vec<_>>();
    for _ in 0..3 {
        let previous = noise.clone();
        for &(start, end, _) in ranges.values() {
            for local in start..end {
                let cell = cells[local];
                let mut sum = previous[local] * 2.0;
                let mut weight = 2.0;
                for slot in 0..D4_NEIGHBORS {
                    let Some(neighbor_id) =
                        cubed_sphere_neighbor_index(cell.fine_cell_id as usize, slot, fine)
                    else {
                        continue;
                    };
                    let Some(&neighbor_local) = local_by_id.get(&(neighbor_id as i32)) else {
                        continue;
                    };
                    if cells[neighbor_local].parent_cell_id == cell.parent_cell_id {
                        sum += previous[neighbor_local];
                        weight += 1.0;
                    }
                }
                noise[local] = sum / weight;
            }
        }
    }

    for &(start, end, parent_index) in ranges.values() {
        let total_area = cells[start..end]
            .iter()
            .map(|cell| cell.area_km2)
            .sum::<f64>();
        let mean = cells[start..end]
            .iter()
            .zip(&noise[start..end])
            .map(|(cell, value)| cell.area_km2 * value)
            .sum::<f64>()
            / total_area;
        for value in &mut noise[start..end] {
            *value -= mean;
        }
        let maximum = noise[start..end]
            .iter()
            .map(|value| value.abs())
            .fold(0.0f64, f64::max)
            .max(f64::EPSILON);
        let amplitude = inputs.parent_relief_m[parent_index] as f64
            * inputs.config.terrain_noise_fraction as f64;
        let parent_id = inputs.parent_ids[parent_index];
        let parent_elevation = inputs.parent_elevation_m[parent_index] as f64;
        let neighbor_elevation = |slot: usize| -> f64 {
            cubed_sphere_neighbor_index(parent_id as usize, slot, coarse)
                .and_then(|neighbor| parent_input_by_id.get(&(neighbor as i32)).copied())
                .map_or(parent_elevation, |index| {
                    inputs.parent_elevation_m[index] as f64
                })
        };
        let north = neighbor_elevation(0);
        let south = neighbor_elevation(1);
        let west = neighbor_elevation(2);
        let east = neighbor_elevation(3);
        let mut offsets = Vec::with_capacity(end - start);
        for local in start..end {
            let local_row = cells[local].row as usize % factor;
            let local_col = cells[local].col as usize % factor;
            let y = (local_row as f64 + 0.5) / factor as f64 - 0.5;
            let x = (local_col as f64 + 0.5) / factor as f64 - 0.5;
            let horizontal = if x >= 0.0 {
                (east - parent_elevation) * x
            } else {
                (parent_elevation - west) * x
            };
            let vertical = if y >= 0.0 {
                (south - parent_elevation) * y
            } else {
                (parent_elevation - north) * y
            };
            offsets.push(horizontal + vertical + noise[local] / maximum * amplitude);
        }
        let offset_mean = cells[start..end]
            .iter()
            .zip(&offsets)
            .map(|(cell, offset)| cell.area_km2 * offset)
            .sum::<f64>()
            / total_area;
        for (cell, offset) in cells[start..end].iter_mut().zip(offsets) {
            let corrected_offset = offset - offset_mean;
            cell.terrain_elevation_m = (parent_elevation + corrected_offset) as f32;
            cell.terrain_offset_m = cell.terrain_elevation_m - parent_elevation as f32;
        }
    }

    Ok((cells, local_by_id, ranges))
}

fn choose_anchor(
    parent: i32,
    cells: &[RefinedCellRecord],
    ranges: &ParentRanges,
    factor: usize,
) -> Result<i32, i32> {
    let &(start, end, _) = ranges.get(&parent).ok_or(4)?;
    let mut best = None::<(f64, i32)>;
    let center = (factor as f64 - 1.0) * 0.5;
    for cell in &cells[start..end] {
        let local_row = cell.row as usize % factor;
        let local_col = cell.col as usize % factor;
        let center_distance = ((local_row as f64 - center).powi(2)
            + (local_col as f64 - center).powi(2))
            / (factor as f64 * factor as f64);
        let score =
            cell.terrain_elevation_m as f64 + cell.parent_relief_m as f64 * 0.2 * center_distance;
        let candidate = (score, cell.fine_cell_id);
        if best.map_or(true, |current| candidate < current) {
            best = Some(candidate);
        }
    }
    best.map(|(_, cell)| cell).ok_or(5)
}

struct RoutingContext<'a> {
    fine: usize,
    cells: &'a [RefinedCellRecord],
    local_by_id: &'a HashMap<i32, usize>,
    ranges: &'a ParentRanges,
    used_edges: &'a HashSet<(i32, i32)>,
    used_nodes: &'a HashSet<i32>,
}

fn boundary_pair(
    source_parent: i32,
    target_parent: i32,
    context: &RoutingContext<'_>,
) -> Result<(i32, i32), i32> {
    let &(start, end, _) = context.ranges.get(&source_parent).ok_or(4)?;
    let mut best = None::<(f64, i32, i32)>;
    for source in &context.cells[start..end] {
        for slot in 0..D4_NEIGHBORS {
            let Some(target) =
                cubed_sphere_neighbor_index(source.fine_cell_id as usize, slot, context.fine)
            else {
                continue;
            };
            let Some(&target_local) = context.local_by_id.get(&(target as i32)) else {
                continue;
            };
            if context.cells[target_local].parent_cell_id != target_parent {
                continue;
            }
            if context.used_nodes.contains(&source.fine_cell_id)
                || context.used_nodes.contains(&(target as i32))
            {
                continue;
            }
            if context
                .used_edges
                .contains(&(target as i32, source.fine_cell_id))
            {
                continue;
            }
            let score = source.terrain_elevation_m as f64
                + context.cells[target_local].terrain_elevation_m as f64;
            let candidate = (score, source.fine_cell_id, target as i32);
            if best.map_or(true, |current| candidate < current) {
                best = Some(candidate);
            }
        }
    }
    best.map(|(_, source, target)| (source, target)).ok_or(5)
}

fn route_inside_parent(
    parent: i32,
    source: i32,
    targets: &[i32],
    context: &RoutingContext<'_>,
) -> Result<Vec<i32>, i32> {
    if targets.contains(&source) {
        return Ok(vec![source]);
    }
    if targets.is_empty() {
        return Err(5);
    }
    let target_set = targets.iter().copied().collect::<HashSet<_>>();
    let &(start, end, _) = context.ranges.get(&parent).ok_or(4)?;
    let minimum_elevation = context.cells[start..end]
        .iter()
        .map(|cell| cell.terrain_elevation_m as f64)
        .fold(f64::INFINITY, f64::min);
    let relief = context.cells[start].parent_relief_m as f64;
    let scale = relief.max(1.0);
    let mut queue = BinaryHeap::new();
    let mut distance = HashMap::<i32, f64>::new();
    let mut next = HashMap::<i32, i32>::new();
    for &target in targets {
        distance.insert(target, 0.0);
        queue.push(QueueState {
            cost: 0.0,
            cell: target,
        });
    }

    while let Some(state) = queue.pop() {
        if state.cell == source {
            break;
        }
        if state.cost > *distance.get(&state.cell).unwrap_or(&f64::INFINITY) {
            continue;
        }
        let downstream_local = *context.local_by_id.get(&state.cell).ok_or(5)?;
        for slot in 0..D4_NEIGHBORS {
            let Some(neighbor) =
                cubed_sphere_neighbor_index(state.cell as usize, slot, context.fine)
            else {
                continue;
            };
            let neighbor = neighbor as i32;
            let Some(&neighbor_local) = context.local_by_id.get(&neighbor) else {
                continue;
            };
            if context.cells[neighbor_local].parent_cell_id != parent {
                continue;
            }
            if context.used_nodes.contains(&neighbor) && !target_set.contains(&neighbor) {
                continue;
            }
            if context.used_edges.contains(&(state.cell, neighbor)) {
                continue;
            }
            let edge_angle = angular_distance(
                context.cells[neighbor_local].xyz,
                context.cells[downstream_local].xyz,
            );
            let normalized_lowland = (context.cells[downstream_local].terrain_elevation_m as f64
                - minimum_elevation)
                / scale;
            let uphill = (context.cells[downstream_local].terrain_elevation_m as f64
                - context.cells[neighbor_local].terrain_elevation_m as f64)
                .max(0.0)
                / scale;
            let edge_cost = edge_angle * (1.0 + 0.35 * normalized_lowland + 1.5 * uphill);
            let candidate = state.cost + edge_cost;
            if candidate + 1e-15 < *distance.get(&neighbor).unwrap_or(&f64::INFINITY) {
                distance.insert(neighbor, candidate);
                next.insert(neighbor, state.cell);
                queue.push(QueueState {
                    cost: candidate,
                    cell: neighbor,
                });
            }
        }
    }
    if !distance.contains_key(&source) {
        return Err(5);
    }
    let mut path = vec![source];
    while !target_set.contains(path.last().unwrap_or(&source)) {
        let downstream = *next.get(path.last().unwrap_or(&source)).ok_or(5)?;
        path.push(downstream);
    }
    Ok(path)
}

fn append_segment(path: &mut Vec<i32>, segment: &[i32]) {
    let start = usize::from(
        path.last()
            .is_some_and(|last| segment.first() == Some(last)),
    );
    path.extend_from_slice(&segment[start..]);
}

fn bounded_fraction(numerator: f64, denominator: f64) -> f32 {
    let exact = (numerator / denominator).clamp(0.0, 1.0);
    let rounded = exact as f32;
    if rounded as f64 > exact && rounded > 0.0 {
        f32::from_bits(rounded.to_bits() - 1)
    } else {
        rounded
    }
}

fn refine_reaches(
    inputs: &Inputs<'_>,
    fine: usize,
    cells: &[RefinedCellRecord],
    local_by_id: &HashMap<i32, usize>,
    ranges: &ParentRanges,
) -> Result<ReachGeneration, i32> {
    let factor = inputs.config.factor as usize;
    let mut anchors = HashMap::<i32, i32>::new();
    for &node in inputs.reach_from_nodes.iter().chain(inputs.reach_to_nodes) {
        if let std::collections::hash_map::Entry::Vacant(entry) = anchors.entry(node) {
            entry.insert(choose_anchor(node, cells, ranges, factor)?);
        }
    }
    let mut boundary_cache = HashMap::<(i32, i32), (i32, i32)>::new();
    let mut used_edges = HashSet::<(i32, i32)>::new();
    let mut used_nodes = HashSet::<i32>::new();
    let reach_by_from_node = inputs
        .reach_from_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect::<HashMap<_, _>>();
    let mut generated_paths = vec![Vec::<i32>::new(); inputs.reach_ids.len()];
    let mut reach_records = Vec::with_capacity(inputs.reach_ids.len());
    let mut all_path_cells = Vec::<i32>::new();
    let mut memberships = Vec::<ReachCellRecord>::new();

    for reach_index in 0..inputs.reach_ids.len() {
        let routing_context = RoutingContext {
            fine,
            cells,
            local_by_id,
            ranges,
            used_edges: &used_edges,
            used_nodes: &used_nodes,
        };
        let coarse_start = inputs.reach_offsets[reach_index] as usize;
        let coarse_end = inputs.reach_offsets[reach_index + 1] as usize;
        let coarse_path = &inputs.reach_parent_cells[coarse_start..coarse_end];
        let mut transitions = Vec::with_capacity(coarse_path.len() - 1);
        for pair in coarse_path.windows(2) {
            let key = (pair[0], pair[1]);
            let boundary = if let Some(boundary) = boundary_cache.get(&key).filter(|boundary| {
                !used_edges.contains(&(boundary.1, boundary.0))
                    && !used_nodes.contains(&boundary.0)
                    && !used_nodes.contains(&boundary.1)
            }) {
                *boundary
            } else {
                let boundary = boundary_pair(pair[0], pair[1], &routing_context)?;
                boundary_cache.insert(key, boundary);
                boundary
            };
            transitions.push(boundary);
        }

        let mut path = Vec::new();
        let mut current = *anchors
            .get(&inputs.reach_from_nodes[reach_index])
            .ok_or(5)?;
        for (parent_index, &parent) in coarse_path.iter().enumerate() {
            let final_parent = parent_index + 1 == coarse_path.len();
            let targets = if final_parent {
                if let Some(&downstream_index) =
                    reach_by_from_node.get(&inputs.reach_to_nodes[reach_index])
                {
                    let downstream_targets = generated_paths[downstream_index]
                        .iter()
                        .copied()
                        .filter(|cell| {
                            local_by_id
                                .get(cell)
                                .is_some_and(|index| cells[*index].parent_cell_id == parent)
                        })
                        .collect::<Vec<_>>();
                    if downstream_targets.is_empty() {
                        return Err(5);
                    }
                    downstream_targets
                } else {
                    vec![*anchors.get(&inputs.reach_to_nodes[reach_index]).ok_or(5)?]
                }
            } else {
                vec![transitions[parent_index].0]
            };
            let segment = route_inside_parent(parent, current, &targets, &routing_context)?;
            append_segment(&mut path, &segment);
            if parent_index + 1 < coarse_path.len() {
                let next = transitions[parent_index].1;
                if path.last() != Some(&next) {
                    path.push(next);
                }
                current = next;
            }
        }
        if path.len() < 2 {
            return Err(5);
        }
        for pair in path.windows(2) {
            let adjacent = (0..D4_NEIGHBORS).any(|slot| {
                cubed_sphere_neighbor_index(pair[0] as usize, slot, fine) == Some(pair[1] as usize)
            });
            if !adjacent {
                return Err(5);
            }
            if used_edges.contains(&(pair[1], pair[0])) {
                return Err(5);
            }
        }
        for pair in path.windows(2) {
            used_edges.insert((pair[0], pair[1]));
        }
        used_nodes.extend(path.iter().copied());
        generated_paths[reach_index] = path.clone();

        let mut cell_lengths = vec![0.0f64; path.len()];
        for edge in 0..path.len() - 1 {
            let first = cells[*local_by_id.get(&path[edge]).ok_or(5)?].xyz;
            let second = cells[*local_by_id.get(&path[edge + 1]).ok_or(5)?].xyz;
            let length = angular_distance(first, second) * inputs.config.planet_radius_m;
            cell_lengths[edge] += 0.5 * length;
            cell_lengths[edge + 1] += 0.5 * length;
        }
        let path_length_m = cell_lengths.iter().sum::<f64>();
        let path_offset = i32::try_from(all_path_cells.len()).map_err(|_| 5)?;
        let path_count = i32::try_from(path.len()).map_err(|_| 5)?;
        all_path_cells.extend_from_slice(&path);
        reach_records.push(RefinedReachRecord {
            reach_id: inputs.reach_ids[reach_index],
            path_offset,
            path_count,
            entry_fine_cell: path[0],
            exit_fine_cell: *path.last().ok_or(5)?,
            path_length_m,
        });

        let channel_width = inputs.channel_width_m[reach_index] as f64;
        let valley_width = inputs.valley_width_m[reach_index] as f64;
        let floodplain_width = inputs.floodplain_width_m[reach_index] as f64;
        let incision = inputs.incision_m[reach_index] as f64;
        for (path_order, (&fine_cell, &length)) in path.iter().zip(&cell_lengths).enumerate() {
            let local = *local_by_id.get(&fine_cell).ok_or(5)?;
            let cell = cells[local];
            let area_m2 = cell.area_km2 * 1_000_000.0;
            memberships.push(ReachCellRecord {
                reach_id: inputs.reach_ids[reach_index],
                fine_cell_id: fine_cell,
                parent_cell_id: cell.parent_cell_id,
                path_order: path_order as i32,
                reach_length_m: length,
                channel_fraction: bounded_fraction(channel_width * length, area_m2),
                valley_fraction: bounded_fraction(valley_width * length, area_m2),
                floodplain_fraction: bounded_fraction(floodplain_width * length, area_m2),
                potential_incised_volume_m3: channel_width * length * incision,
            });
        }
    }
    Ok((reach_records, all_path_cells, memberships))
}

fn compute_stats(
    inputs: &Inputs<'_>,
    fine: usize,
    cells: &[RefinedCellRecord],
    ranges: &ParentRanges,
    reaches: &[RefinedReachRecord],
    memberships: &[ReachCellRecord],
) -> RefinementStats {
    let radius_squared_km = (inputs.config.planet_radius_m / 1_000.0).powi(2);
    let mut maximum_area_error = 0.0f64;
    let mut maximum_elevation_error = 0.0f64;
    for &(start, end, parent_index) in ranges.values() {
        let child_area = cells[start..end]
            .iter()
            .map(|cell| cell.area_km2)
            .sum::<f64>();
        let parent_area = inputs.parent_area_steradians[parent_index] * radius_squared_km;
        maximum_area_error = maximum_area_error.max((child_area - parent_area).abs() / parent_area);
        let mean_elevation = cells[start..end]
            .iter()
            .map(|cell| cell.terrain_elevation_m as f64 * cell.area_km2)
            .sum::<f64>()
            / child_area;
        maximum_elevation_error = maximum_elevation_error
            .max((mean_elevation - inputs.parent_elevation_m[parent_index] as f64).abs());
    }
    let selected_area_km2 = cells.iter().map(|cell| cell.area_km2).sum::<f64>();
    let area_by_cell = cells
        .iter()
        .map(|cell| (cell.fine_cell_id, cell.area_km2))
        .collect::<HashMap<_, _>>();
    let represented_area = |fraction: fn(&ReachCellRecord) -> f32| {
        memberships
            .iter()
            .map(|record| {
                fraction(record) as f64
                    * area_by_cell
                        .get(&record.fine_cell_id)
                        .copied()
                        .unwrap_or(0.0)
            })
            .sum::<f64>()
    };
    let represented_channel_area_km2 = represented_area(|record| record.channel_fraction);
    let represented_valley_area_km2 = represented_area(|record| record.valley_fraction);
    let represented_floodplain_area_km2 = represented_area(|record| record.floodplain_fraction);
    RefinementStats {
        parent_count: inputs.parent_ids.len() as i32,
        child_count: cells.len() as i32,
        reach_count: reaches.len() as i32,
        reach_cell_count: memberships.len() as i32,
        fine_resolution: fine as i32,
        path_topology_valid: 1,
        selected_area_km2,
        maximum_parent_area_relative_error: maximum_area_error,
        maximum_parent_elevation_error_m: maximum_elevation_error,
        total_reach_length_km: reaches
            .iter()
            .map(|reach| reach.path_length_m / 1_000.0)
            .sum(),
        represented_channel_area_km2,
        represented_valley_area_km2,
        represented_floodplain_area_km2,
        total_potential_incised_volume_km3: memberships
            .iter()
            .map(|record| record.potential_incised_volume_m3 / 1e9)
            .sum(),
    }
}

fn run_refinement(inputs: &Inputs<'_>) -> Result<Outcome, i32> {
    let (coarse, fine) = validate_inputs(inputs)?;
    let (cells, local_by_id, ranges) = generate_children(inputs, coarse, fine)?;
    let (reaches, path_cells, memberships) =
        refine_reaches(inputs, fine, &cells, &local_by_id, &ranges)?;
    let stats = compute_stats(inputs, fine, &cells, &ranges, &reaches, &memberships);
    Ok(Outcome {
        cells,
        reaches,
        path_cells,
        memberships,
        stats,
    })
}

fn boxed_array<T>(values: Vec<T>) -> (*mut T, usize) {
    let mut values = values.into_boxed_slice();
    let result = (values.as_mut_ptr(), values.len());
    std::mem::forget(values);
    result
}

unsafe fn free_boxed_array<T>(data: *mut T, len: usize) {
    if !data.is_null() {
        let slice = std::ptr::slice_from_raw_parts_mut(data, len);
        unsafe {
            drop(Box::from_raw(slice));
        }
    }
}

#[no_mangle]
/// Refine one complete coarse drainage basin into sparse fine cells and inherited reaches.
///
/// # Safety
///
/// Every input pointer must reference the number of contiguous values implied by
/// its count or offset array. Output pointers must be writable and non-aliasing.
pub unsafe extern "C" fn refinement_run_basin(
    config: *const RefinementConfig,
    parent_count: i32,
    parent_ids: *const i32,
    parent_elevation_m: *const f32,
    parent_relief_m: *const f32,
    parent_area_steradians: *const f64,
    reach_count: i32,
    reach_ids: *const i32,
    reach_from_nodes: *const i32,
    reach_to_nodes: *const i32,
    reach_offsets: *const i32,
    reach_parent_cell_count: i32,
    reach_parent_cells: *const i32,
    channel_width_m: *const f32,
    valley_width_m: *const f32,
    floodplain_width_m: *const f32,
    incision_m: *const f32,
    cells_out: *mut RefinedCellArray,
    reaches_out: *mut RefinedReachArray,
    path_cells_out: *mut Int32Array,
    memberships_out: *mut ReachCellArray,
    stats_out: *mut RefinementStats,
) -> i32 {
    if config.is_null()
        || parent_count <= 0
        || parent_ids.is_null()
        || parent_elevation_m.is_null()
        || parent_relief_m.is_null()
        || parent_area_steradians.is_null()
        || reach_count <= 0
        || reach_ids.is_null()
        || reach_from_nodes.is_null()
        || reach_to_nodes.is_null()
        || reach_offsets.is_null()
        || reach_parent_cell_count <= 0
        || reach_parent_cells.is_null()
        || channel_width_m.is_null()
        || valley_width_m.is_null()
        || floodplain_width_m.is_null()
        || incision_m.is_null()
        || cells_out.is_null()
        || reaches_out.is_null()
        || path_cells_out.is_null()
        || memberships_out.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let parent_count = parent_count as usize;
    let reach_count = reach_count as usize;
    let reach_parent_cell_count = reach_parent_cell_count as usize;
    let inputs = unsafe {
        Inputs {
            config: *config,
            parent_ids: std::slice::from_raw_parts(parent_ids, parent_count),
            parent_elevation_m: std::slice::from_raw_parts(parent_elevation_m, parent_count),
            parent_relief_m: std::slice::from_raw_parts(parent_relief_m, parent_count),
            parent_area_steradians: std::slice::from_raw_parts(
                parent_area_steradians,
                parent_count,
            ),
            reach_ids: std::slice::from_raw_parts(reach_ids, reach_count),
            reach_from_nodes: std::slice::from_raw_parts(reach_from_nodes, reach_count),
            reach_to_nodes: std::slice::from_raw_parts(reach_to_nodes, reach_count),
            reach_offsets: std::slice::from_raw_parts(reach_offsets, reach_count + 1),
            reach_parent_cells: std::slice::from_raw_parts(
                reach_parent_cells,
                reach_parent_cell_count,
            ),
            channel_width_m: std::slice::from_raw_parts(channel_width_m, reach_count),
            valley_width_m: std::slice::from_raw_parts(valley_width_m, reach_count),
            floodplain_width_m: std::slice::from_raw_parts(floodplain_width_m, reach_count),
            incision_m: std::slice::from_raw_parts(incision_m, reach_count),
        }
    };
    let outcome = match run_refinement(&inputs) {
        Ok(outcome) => outcome,
        Err(status) => return status,
    };
    let (cell_data, cell_len) = boxed_array(outcome.cells);
    let (reach_data, reach_len) = boxed_array(outcome.reaches);
    let (path_data, path_len) = boxed_array(outcome.path_cells);
    let (membership_data, membership_len) = boxed_array(outcome.memberships);
    unsafe {
        *cells_out = RefinedCellArray {
            data: cell_data,
            len: cell_len,
        };
        *reaches_out = RefinedReachArray {
            data: reach_data,
            len: reach_len,
        };
        *path_cells_out = Int32Array {
            data: path_data,
            len: path_len,
        };
        *memberships_out = ReachCellArray {
            data: membership_data,
            len: membership_len,
        };
        *stats_out = outcome.stats;
    }
    0
}

#[no_mangle]
/// # Safety
///
/// `array` must have been returned by [`refinement_run_basin`] and not freed before.
pub unsafe extern "C" fn refinement_free_cells(array: RefinedCellArray) {
    unsafe {
        free_boxed_array(array.data, array.len);
    }
}

#[no_mangle]
/// # Safety
///
/// `array` must have been returned by [`refinement_run_basin`] and not freed before.
pub unsafe extern "C" fn refinement_free_reaches(array: RefinedReachArray) {
    unsafe {
        free_boxed_array(array.data, array.len);
    }
}

#[no_mangle]
/// # Safety
///
/// `array` must have been returned by [`refinement_run_basin`] and not freed before.
pub unsafe extern "C" fn refinement_free_i32(array: Int32Array) {
    unsafe {
        free_boxed_array(array.data, array.len);
    }
}

#[no_mangle]
/// # Safety
///
/// `array` must have been returned by [`refinement_run_basin`] and not freed before.
pub unsafe extern "C" fn refinement_free_memberships(array: ReachCellArray) {
    unsafe {
        free_boxed_array(array.data, array.len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent_area(cell: i32, resolution: usize) -> f64 {
        let (face, row, col) = cubed_sphere_decode_index(cell as usize, resolution).unwrap();
        cubed_sphere_cell_area_steradians(face, row, col, resolution).unwrap()
    }

    #[test]
    fn sparse_refinement_conserves_parents_and_routes_reach() {
        let coarse = 4usize;
        let first = cubed_sphere_global_index(0, 1, 1, coarse).unwrap() as i32;
        let second = cubed_sphere_global_index(0, 1, 2, coarse).unwrap() as i32;
        let parent_ids = [first, second];
        let elevations = [120.0f32, 80.0];
        let relief = [90.0f32, 60.0];
        let areas = [parent_area(first, coarse), parent_area(second, coarse)];
        let reach_ids = [7i32];
        let from = [first];
        let to = [second];
        let offsets = [0i32, 2];
        let path = [first, second];
        let channel = [40.0f32];
        let valley = [1_200.0f32];
        let floodplain = [600.0f32];
        let incision = [25.0f32];
        let inputs = Inputs {
            config: RefinementConfig {
                coarse_resolution: coarse as i32,
                factor: 4,
                planet_radius_m: 6_371_000.0,
                terrain_seed: 42,
                terrain_noise_fraction: 0.45,
            },
            parent_ids: &parent_ids,
            parent_elevation_m: &elevations,
            parent_relief_m: &relief,
            parent_area_steradians: &areas,
            reach_ids: &reach_ids,
            reach_from_nodes: &from,
            reach_to_nodes: &to,
            reach_offsets: &offsets,
            reach_parent_cells: &path,
            channel_width_m: &channel,
            valley_width_m: &valley,
            floodplain_width_m: &floodplain,
            incision_m: &incision,
        };
        let result = run_refinement(&inputs).unwrap();
        assert_eq!(result.cells.len(), 32);
        assert_eq!(result.reaches.len(), 1);
        assert!(result.reaches[0].path_count >= 2);
        assert!(result.reaches[0].path_length_m > 0.0);
        assert_eq!(result.stats.path_topology_valid, 1);
        assert!(result.stats.maximum_parent_area_relative_error < 1e-12);
        assert!(result.stats.maximum_parent_elevation_error_m < 1e-3);
        assert!(result
            .memberships
            .iter()
            .all(|record| record.channel_fraction < record.valley_fraction));
        for pair in result.path_cells.windows(2) {
            assert!((0..D4_NEIGHBORS).any(|slot| {
                cubed_sphere_neighbor_index(pair[0] as usize, slot, coarse * 4)
                    == Some(pair[1] as usize)
            }));
        }
    }

    #[test]
    fn sparse_refinement_routes_across_cube_face_edge() {
        let coarse = 4usize;
        let first = cubed_sphere_global_index(0, 1, coarse - 1, coarse).unwrap() as i32;
        let second = cubed_sphere_neighbor_index(first as usize, 3, coarse).unwrap() as i32;
        assert_ne!(
            cubed_sphere_decode_index(first as usize, coarse).unwrap().0,
            cubed_sphere_decode_index(second as usize, coarse)
                .unwrap()
                .0
        );
        let parent_ids = [first, second];
        let elevations = [40.0f32, 35.0];
        let relief = [30.0f32, 30.0];
        let areas = [parent_area(first, coarse), parent_area(second, coarse)];
        let reach_ids = [3i32];
        let from = [first];
        let to = [second];
        let offsets = [0i32, 2];
        let path = [first, second];
        let channel = [25.0f32];
        let valley = [800.0f32];
        let floodplain = [500.0f32];
        let incision = [12.0f32];
        let inputs = Inputs {
            config: RefinementConfig {
                coarse_resolution: coarse as i32,
                factor: 4,
                planet_radius_m: 6_371_000.0,
                terrain_seed: 91,
                terrain_noise_fraction: 0.35,
            },
            parent_ids: &parent_ids,
            parent_elevation_m: &elevations,
            parent_relief_m: &relief,
            parent_area_steradians: &areas,
            reach_ids: &reach_ids,
            reach_from_nodes: &from,
            reach_to_nodes: &to,
            reach_offsets: &offsets,
            reach_parent_cells: &path,
            channel_width_m: &channel,
            valley_width_m: &valley,
            floodplain_width_m: &floodplain,
            incision_m: &incision,
        };
        let result = run_refinement(&inputs).unwrap();
        let fine = coarse * 4;
        assert!(result.path_cells.windows(2).all(|pair| {
            (0..D4_NEIGHBORS).any(|slot| {
                cubed_sphere_neighbor_index(pair[0] as usize, slot, fine) == Some(pair[1] as usize)
            })
        }));
        let mut collapsed = Vec::new();
        for &fine_cell in &result.path_cells {
            let (face, row, col) = cubed_sphere_decode_index(fine_cell as usize, fine).unwrap();
            let parent = cubed_sphere_global_index(face, row / 4, col / 4, coarse).unwrap() as i32;
            if collapsed.last() != Some(&parent) {
                collapsed.push(parent);
            }
        }
        assert_eq!(collapsed, path);
    }
}
