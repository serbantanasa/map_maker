use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::slice;

use topology_native::cubed_sphere_neighbor_index;

const D4_NEIGHBORS: usize = 4;
const ANCHOR_NORMAL: u8 = 0;
const ANCHOR_CHANNEL: u8 = 1;
const ANCHOR_EXCLUDED: u8 = 2;
const ANCHOR_OUTSIDE: u8 = 3;
const TERMINAL_OCEAN: i32 = -1;
const TERMINAL_HANDOFF: i32 = -2;
const NO_FIXED_RECEIVER: i32 = i32::MIN;

#[no_mangle]
pub extern "C" fn hydrology_pass2_native_abi_version() -> u32 {
    1
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HydrologyPass2Config {
    pub fine_resolution: i32,
    pub minimum_depression_depth_m: f64,
    pub planet_radius_m: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct StabilizedCellRecord {
    pub fine_cell_id: i32,
    pub baseline_receiver_id: i32,
    pub stabilized_receiver_id: i32,
    pub baseline_anchor_cell_id: i32,
    pub stabilized_anchor_cell_id: i32,
    pub baseline_depression_id: i32,
    pub stabilized_depression_id: i32,
    pub anchor_kind: u8,
    pub receiver_changed: u8,
    pub depression_changed: u8,
    pub terminal_kind: u8,
    pub baseline_hydrologic_elevation_m: f64,
    pub stabilized_hydrologic_elevation_m: f64,
    pub baseline_fill_depth_m: f64,
    pub stabilized_fill_depth_m: f64,
    pub stabilized_flow_slope: f64,
    pub contributing_area_km2: f64,
    pub flow_direction_xyz: [f32; 3],
}

#[repr(C)]
pub struct StabilizedCellArray {
    pub data: *mut StabilizedCellRecord,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HydrologyPass2Stats {
    pub cell_count: i32,
    pub active_cell_count: i32,
    pub channel_cell_count: i32,
    pub excluded_cell_count: i32,
    pub outside_cell_count: i32,
    pub physical_trunk_edge_count: i32,
    pub baseline_uncovered_cell_count: i32,
    pub stabilized_uncovered_cell_count: i32,
    pub baseline_depression_count: i32,
    pub stabilized_depression_count: i32,
    pub receiver_changed_cell_count: i32,
    pub depression_changed_cell_count: i32,
    pub graph_valid: i32,
    pub trunk_preserved_valid: i32,
    pub process_exclusion_valid: i32,
    pub active_area_km2: f64,
    pub receiver_changed_area_km2: f64,
    pub receiver_changed_area_fraction: f64,
    pub terminal_accumulated_area_km2: f64,
    pub contributing_area_residual_km2: f64,
    pub baseline_depression_area_km2: f64,
    pub stabilized_depression_area_km2: f64,
    pub baseline_depression_volume_km3: f64,
    pub stabilized_depression_volume_km3: f64,
    pub maximum_baseline_fill_depth_m: f64,
    pub maximum_stabilized_fill_depth_m: f64,
}

struct Inputs<'a> {
    cell_ids: &'a [i32],
    terrain_before_m: &'a [f64],
    routing_surface_after_m: &'a [f64],
    cell_areas_km2: &'a [f64],
    cell_xyz: &'a [f32],
    anchor_kinds: &'a [u8],
    source_active: &'a [u8],
    fixed_receiver_ids: &'a [i32],
}

struct Route {
    receiver: Vec<i32>,
    anchor: Vec<i32>,
    hydrologic_elevation_m: Vec<f64>,
    fill_depth_m: Vec<f64>,
    depression_id: Vec<i32>,
    depression_count: usize,
    uncovered_count: usize,
    depression_area_km2: f64,
    depression_volume_km3: f64,
    maximum_fill_depth_m: f64,
}

struct Outcome {
    cells: Vec<StabilizedCellRecord>,
    stats: HydrologyPass2Stats,
}

struct RoutingDomain {
    row_by_id: HashMap<i32, usize>,
    adjacency: Vec<[i32; D4_NEIGHBORS]>,
}

#[derive(Clone, Copy)]
struct QueueState {
    elevation_m: f64,
    cell_id: i32,
    row: usize,
}

impl PartialEq for QueueState {
    fn eq(&self, other: &Self) -> bool {
        self.cell_id == other.cell_id && self.elevation_m.to_bits() == other.elevation_m.to_bits()
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
            .elevation_m
            .total_cmp(&self.elevation_m)
            .then_with(|| other.cell_id.cmp(&self.cell_id))
    }
}

fn build_adjacency(
    resolution: usize,
    cell_ids: &[i32],
    row_by_id: &HashMap<i32, usize>,
) -> Vec<[i32; D4_NEIGHBORS]> {
    cell_ids
        .iter()
        .map(|&cell_id| {
            let mut neighbors = [-1i32; D4_NEIGHBORS];
            for (slot, target) in neighbors.iter_mut().enumerate() {
                if let Some(neighbor) =
                    cubed_sphere_neighbor_index(cell_id as usize, slot, resolution)
                {
                    let neighbor = neighbor as i32;
                    if row_by_id.contains_key(&neighbor) {
                        *target = neighbor;
                    }
                }
            }
            neighbors
        })
        .collect()
}

fn validate_inputs(
    config: HydrologyPass2Config,
    inputs: &Inputs<'_>,
) -> Result<RoutingDomain, i32> {
    let resolution = usize::try_from(config.fine_resolution).map_err(|_| 1)?;
    let cell_count = inputs.cell_ids.len();
    if resolution == 0
        || cell_count == 0
        || !config.minimum_depression_depth_m.is_finite()
        || config.minimum_depression_depth_m <= 0.0
        || !config.planet_radius_m.is_finite()
        || config.planet_radius_m <= 0.0
        || inputs.terrain_before_m.len() != cell_count
        || inputs.routing_surface_after_m.len() != cell_count
        || inputs.cell_areas_km2.len() != cell_count
        || inputs.cell_xyz.len() != cell_count * 3
        || inputs.anchor_kinds.len() != cell_count
        || inputs.source_active.len() != cell_count
        || inputs.fixed_receiver_ids.len() != cell_count
    {
        return Err(1);
    }
    let global_count = 6usize
        .checked_mul(resolution)
        .and_then(|value| value.checked_mul(resolution))
        .filter(|value| *value <= i32::MAX as usize)
        .ok_or(1)?;
    let mut row_by_id = HashMap::with_capacity(cell_count);
    for row in 0..cell_count {
        let cell_id = inputs.cell_ids[row];
        if cell_id < 0
            || cell_id as usize >= global_count
            || row_by_id.insert(cell_id, row).is_some()
            || !inputs.terrain_before_m[row].is_finite()
            || !inputs.routing_surface_after_m[row].is_finite()
            || !inputs.cell_areas_km2[row].is_finite()
            || inputs.cell_areas_km2[row] <= 0.0
            || inputs.cell_xyz[row * 3..row * 3 + 3]
                .iter()
                .any(|value| !value.is_finite())
            || inputs.anchor_kinds[row] > ANCHOR_OUTSIDE
            || inputs.source_active[row] > 1
        {
            return Err(2);
        }
        match inputs.anchor_kinds[row] {
            ANCHOR_CHANNEL => {
                if inputs.fixed_receiver_ids[row] < TERMINAL_HANDOFF {
                    return Err(3);
                }
            }
            ANCHOR_NORMAL => {
                if inputs.fixed_receiver_ids[row] != NO_FIXED_RECEIVER
                    && inputs.fixed_receiver_ids[row] < TERMINAL_HANDOFF
                {
                    return Err(3);
                }
            }
            _ if inputs.fixed_receiver_ids[row] != NO_FIXED_RECEIVER => return Err(3),
            _ => {}
        }
    }
    if !inputs
        .anchor_kinds
        .iter()
        .any(|kind| *kind != ANCHOR_NORMAL)
    {
        return Err(3);
    }
    let adjacency = build_adjacency(resolution, inputs.cell_ids, &row_by_id);
    for (row, neighbors) in adjacency.iter().enumerate() {
        if inputs.fixed_receiver_ids[row] < 0 {
            continue;
        }
        let target_id = inputs.fixed_receiver_ids[row];
        if !row_by_id.contains_key(&target_id) {
            return Err(3);
        }
        if !neighbors.contains(&target_id) {
            return Err(3);
        }
        if inputs.anchor_kinds[row] == ANCHOR_CHANNEL {
            let target_row = row_by_id[&target_id];
            if inputs.anchor_kinds[target_row] != ANCHOR_CHANNEL {
                return Err(3);
            }
        }
    }
    Ok(RoutingDomain {
        row_by_id,
        adjacency,
    })
}

fn route_surface(
    config: HydrologyPass2Config,
    inputs: &Inputs<'_>,
    surface: &[f64],
    row_by_id: &HashMap<i32, usize>,
    adjacency: &[[i32; D4_NEIGHBORS]],
) -> Route {
    let cell_count = inputs.cell_ids.len();
    let mut receiver = vec![NO_FIXED_RECEIVER; cell_count];
    let mut anchor = vec![-1i32; cell_count];
    let mut hydrologic_elevation_m = surface.to_vec();
    let mut visited = vec![false; cell_count];
    let mut heap = BinaryHeap::new();
    for row in 0..cell_count {
        if inputs.anchor_kinds[row] == ANCHOR_NORMAL {
            continue;
        }
        visited[row] = true;
        anchor[row] = inputs.cell_ids[row];
        receiver[row] = match inputs.anchor_kinds[row] {
            ANCHOR_CHANNEL => inputs.fixed_receiver_ids[row],
            ANCHOR_EXCLUDED => TERMINAL_HANDOFF,
            ANCHOR_OUTSIDE => TERMINAL_OCEAN,
            _ => NO_FIXED_RECEIVER,
        };
        heap.push(QueueState {
            elevation_m: surface[row],
            cell_id: inputs.cell_ids[row],
            row,
        });
    }

    while let Some(state) = heap.pop() {
        for &neighbor_id in &adjacency[state.row] {
            if neighbor_id < 0 {
                continue;
            }
            let neighbor = row_by_id[&neighbor_id];
            if visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;
            receiver[neighbor] = state.cell_id;
            anchor[neighbor] = anchor[state.row];
            hydrologic_elevation_m[neighbor] = surface[neighbor].max(state.elevation_m);
            heap.push(QueueState {
                elevation_m: hydrologic_elevation_m[neighbor],
                cell_id: neighbor_id,
                row: neighbor,
            });
        }
    }

    for row in 0..cell_count {
        if inputs.anchor_kinds[row] == ANCHOR_NORMAL
            && inputs.fixed_receiver_ids[row] != NO_FIXED_RECEIVER
        {
            receiver[row] = inputs.fixed_receiver_ids[row];
        }
    }
    let mut upstream_count = vec![0usize; cell_count];
    for &receiver_id in &receiver {
        if let Some(&target) = row_by_id.get(&receiver_id) {
            upstream_count[target] += 1;
        }
    }
    let mut ready = (0..cell_count)
        .filter(|row| upstream_count[*row] == 0)
        .collect::<VecDeque<_>>();
    let mut order = Vec::with_capacity(cell_count);
    while let Some(row) = ready.pop_front() {
        order.push(row);
        if let Some(&target) = row_by_id.get(&receiver[row]) {
            upstream_count[target] -= 1;
            if upstream_count[target] == 0 {
                ready.push_back(target);
            }
        }
    }
    if order.len() == cell_count {
        for &row in order.iter().rev() {
            if let Some(&target) = row_by_id.get(&receiver[row]) {
                hydrologic_elevation_m[row] = surface[row].max(hydrologic_elevation_m[target]);
            } else {
                hydrologic_elevation_m[row] = surface[row];
            }
        }
    }

    let uncovered_count = visited.iter().filter(|value| !**value).count();
    let fill_depth_m = hydrologic_elevation_m
        .iter()
        .zip(surface)
        .map(|(filled, raw)| (filled - raw).max(0.0))
        .collect::<Vec<_>>();
    let mut depression_id = vec![-1i32; cell_count];
    let eligible = (0..cell_count)
        .map(|row| {
            visited[row]
                && inputs.anchor_kinds[row] == ANCHOR_NORMAL
                && inputs.fixed_receiver_ids[row] == NO_FIXED_RECEIVER
                && fill_depth_m[row] >= config.minimum_depression_depth_m
        })
        .collect::<Vec<_>>();
    let mut depression_count = 0usize;
    let mut depression_area_km2 = 0.0;
    let mut depression_volume_km3 = 0.0;
    for start in 0..cell_count {
        if !eligible[start] || depression_id[start] >= 0 {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        depression_id[start] = -2;
        let mut identifier = inputs.cell_ids[start];
        while let Some(row) = queue.pop_front() {
            component.push(row);
            identifier = identifier.min(inputs.cell_ids[row]);
            for &neighbor_id in &adjacency[row] {
                let Some(&neighbor) = row_by_id.get(&neighbor_id) else {
                    continue;
                };
                if eligible[neighbor] && depression_id[neighbor] == -1 {
                    depression_id[neighbor] = -2;
                    queue.push_back(neighbor);
                }
            }
        }
        for row in component {
            depression_id[row] = identifier;
            depression_area_km2 += inputs.cell_areas_km2[row];
            depression_volume_km3 += fill_depth_m[row] * inputs.cell_areas_km2[row] / 1_000.0;
        }
        depression_count += 1;
    }
    Route {
        receiver,
        anchor,
        hydrologic_elevation_m,
        fill_depth_m: fill_depth_m.clone(),
        depression_id,
        depression_count,
        uncovered_count,
        depression_area_km2,
        depression_volume_km3,
        maximum_fill_depth_m: fill_depth_m.iter().copied().fold(0.0, f64::max),
    }
}

fn angular_distance(first: &[f32], second: &[f32]) -> f64 {
    let dot = (first[0] as f64 * second[0] as f64
        + first[1] as f64 * second[1] as f64
        + first[2] as f64 * second[2] as f64)
        .clamp(-1.0, 1.0);
    dot.acos()
}

fn flow_direction(first: &[f32], second: &[f32]) -> [f32; 3] {
    let source = [first[0] as f64, first[1] as f64, first[2] as f64];
    let delta = [
        second[0] as f64 - source[0],
        second[1] as f64 - source[1],
        second[2] as f64 - source[2],
    ];
    let radial = delta[0] * source[0] + delta[1] * source[1] + delta[2] * source[2];
    let tangent = [
        delta[0] - radial * source[0],
        delta[1] - radial * source[1],
        delta[2] - radial * source[2],
    ];
    let norm = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    if norm <= f64::EPSILON {
        [0.0; 3]
    } else {
        [
            (tangent[0] / norm) as f32,
            (tangent[1] / norm) as f32,
            (tangent[2] / norm) as f32,
        ]
    }
}

fn accumulate_area(
    inputs: &Inputs<'_>,
    route: &Route,
    row_by_id: &HashMap<i32, usize>,
) -> (Vec<f64>, bool, f64, f64) {
    let cell_count = inputs.cell_ids.len();
    let mut upstream_count = vec![0usize; cell_count];
    for &receiver_id in &route.receiver {
        if let Some(&target) = row_by_id.get(&receiver_id) {
            upstream_count[target] += 1;
        }
    }
    let mut ready = BinaryHeap::new();
    for (row, &count) in upstream_count.iter().enumerate() {
        if count == 0 {
            ready.push(std::cmp::Reverse((inputs.cell_ids[row], row)));
        }
    }
    let mut contributing_area = (0..cell_count)
        .map(|row| {
            if inputs.source_active[row] == 0 {
                0.0
            } else {
                inputs.cell_areas_km2[row]
            }
        })
        .collect::<Vec<_>>();
    let active_area = contributing_area.iter().sum::<f64>();
    let mut processed = 0usize;
    while let Some(std::cmp::Reverse((_, row))) = ready.pop() {
        processed += 1;
        if let Some(&target) = row_by_id.get(&route.receiver[row]) {
            contributing_area[target] += contributing_area[row];
            upstream_count[target] -= 1;
            if upstream_count[target] == 0 {
                ready.push(std::cmp::Reverse((inputs.cell_ids[target], target)));
            }
        }
    }
    let graph_valid = processed == cell_count;
    let terminal_area = route
        .receiver
        .iter()
        .enumerate()
        .filter(|(_, receiver)| **receiver < 0)
        .map(|(row, _)| contributing_area[row])
        .sum::<f64>();
    (contributing_area, graph_valid, active_area, terminal_area)
}

fn run_pass(config: HydrologyPass2Config, inputs: &Inputs<'_>) -> Result<Outcome, i32> {
    let domain = validate_inputs(config, inputs)?;
    let row_by_id = &domain.row_by_id;
    let adjacency = &domain.adjacency;
    let baseline = route_surface(
        config,
        inputs,
        inputs.terrain_before_m,
        row_by_id,
        adjacency,
    );
    let stabilized = route_surface(
        config,
        inputs,
        inputs.routing_surface_after_m,
        row_by_id,
        adjacency,
    );
    let (contributing_area, graph_valid, active_area, terminal_area) =
        accumulate_area(inputs, &stabilized, row_by_id);
    let mut receiver_changed_count = 0usize;
    let mut receiver_changed_area = 0.0;
    let mut depression_changed_count = 0usize;
    let mut physical_trunk_edge_count = 0usize;
    let mut trunk_preserved_valid = true;
    let mut process_exclusion_valid = true;
    let mut cells = Vec::with_capacity(inputs.cell_ids.len());
    for (row, &cell_contributing_area) in contributing_area.iter().enumerate() {
        let receiver_changed = baseline.receiver[row] != stabilized.receiver[row];
        let depression_changed = baseline.depression_id[row] != stabilized.depression_id[row];
        if receiver_changed && inputs.source_active[row] != 0 {
            receiver_changed_count += 1;
            receiver_changed_area += inputs.cell_areas_km2[row];
        }
        if depression_changed && inputs.source_active[row] != 0 {
            depression_changed_count += 1;
        }
        if inputs.anchor_kinds[row] == ANCHOR_CHANNEL {
            trunk_preserved_valid &= baseline.receiver[row] == inputs.fixed_receiver_ids[row]
                && stabilized.receiver[row] == inputs.fixed_receiver_ids[row];
            physical_trunk_edge_count += usize::from(inputs.fixed_receiver_ids[row] >= 0);
        }
        if inputs.anchor_kinds[row] == ANCHOR_EXCLUDED {
            process_exclusion_valid &= baseline.receiver[row] == TERMINAL_HANDOFF
                && stabilized.receiver[row] == TERMINAL_HANDOFF
                && baseline.depression_id[row] < 0
                && stabilized.depression_id[row] < 0;
        }
        let receiver_id = stabilized.receiver[row];
        let (flow_slope, direction) = if let Some(&target) = row_by_id.get(&receiver_id) {
            let length_m = angular_distance(
                &inputs.cell_xyz[row * 3..row * 3 + 3],
                &inputs.cell_xyz[target * 3..target * 3 + 3],
            ) * config.planet_radius_m;
            let slope = if length_m > 0.0 {
                ((stabilized.hydrologic_elevation_m[row]
                    - stabilized.hydrologic_elevation_m[target])
                    / length_m)
                    .max(0.0)
            } else {
                0.0
            };
            (
                slope,
                flow_direction(
                    &inputs.cell_xyz[row * 3..row * 3 + 3],
                    &inputs.cell_xyz[target * 3..target * 3 + 3],
                ),
            )
        } else {
            (0.0, [0.0; 3])
        };
        cells.push(StabilizedCellRecord {
            fine_cell_id: inputs.cell_ids[row],
            baseline_receiver_id: baseline.receiver[row],
            stabilized_receiver_id: receiver_id,
            baseline_anchor_cell_id: baseline.anchor[row],
            stabilized_anchor_cell_id: stabilized.anchor[row],
            baseline_depression_id: baseline.depression_id[row],
            stabilized_depression_id: stabilized.depression_id[row],
            anchor_kind: inputs.anchor_kinds[row],
            receiver_changed: u8::from(receiver_changed),
            depression_changed: u8::from(depression_changed),
            terminal_kind: match receiver_id {
                TERMINAL_OCEAN => 1,
                TERMINAL_HANDOFF => 2,
                _ => 0,
            },
            baseline_hydrologic_elevation_m: baseline.hydrologic_elevation_m[row],
            stabilized_hydrologic_elevation_m: stabilized.hydrologic_elevation_m[row],
            baseline_fill_depth_m: baseline.fill_depth_m[row],
            stabilized_fill_depth_m: stabilized.fill_depth_m[row],
            stabilized_flow_slope: flow_slope,
            contributing_area_km2: cell_contributing_area,
            flow_direction_xyz: direction,
        });
    }
    let active_cell_count = inputs
        .source_active
        .iter()
        .filter(|value| **value != 0)
        .count();
    let area_residual = terminal_area - active_area;
    let area_tolerance = 1e-9 * active_area.max(1.0);
    let graph_valid = graph_valid
        && baseline.uncovered_count == 0
        && stabilized.uncovered_count == 0
        && area_residual.abs() <= area_tolerance;
    let stats = HydrologyPass2Stats {
        cell_count: i32::try_from(inputs.cell_ids.len()).map_err(|_| 1)?,
        active_cell_count: i32::try_from(active_cell_count).map_err(|_| 1)?,
        channel_cell_count: i32::try_from(
            inputs
                .anchor_kinds
                .iter()
                .filter(|kind| **kind == ANCHOR_CHANNEL)
                .count(),
        )
        .map_err(|_| 1)?,
        excluded_cell_count: i32::try_from(
            inputs
                .anchor_kinds
                .iter()
                .filter(|kind| **kind == ANCHOR_EXCLUDED)
                .count(),
        )
        .map_err(|_| 1)?,
        outside_cell_count: i32::try_from(
            inputs
                .anchor_kinds
                .iter()
                .filter(|kind| **kind == ANCHOR_OUTSIDE)
                .count(),
        )
        .map_err(|_| 1)?,
        physical_trunk_edge_count: i32::try_from(physical_trunk_edge_count).map_err(|_| 1)?,
        baseline_uncovered_cell_count: i32::try_from(baseline.uncovered_count).map_err(|_| 1)?,
        stabilized_uncovered_cell_count: i32::try_from(stabilized.uncovered_count)
            .map_err(|_| 1)?,
        baseline_depression_count: i32::try_from(baseline.depression_count).map_err(|_| 1)?,
        stabilized_depression_count: i32::try_from(stabilized.depression_count).map_err(|_| 1)?,
        receiver_changed_cell_count: i32::try_from(receiver_changed_count).map_err(|_| 1)?,
        depression_changed_cell_count: i32::try_from(depression_changed_count).map_err(|_| 1)?,
        graph_valid: i32::from(graph_valid),
        trunk_preserved_valid: i32::from(trunk_preserved_valid),
        process_exclusion_valid: i32::from(process_exclusion_valid),
        active_area_km2: active_area,
        receiver_changed_area_km2: receiver_changed_area,
        receiver_changed_area_fraction: receiver_changed_area / active_area.max(f64::EPSILON),
        terminal_accumulated_area_km2: terminal_area,
        contributing_area_residual_km2: area_residual,
        baseline_depression_area_km2: baseline.depression_area_km2,
        stabilized_depression_area_km2: stabilized.depression_area_km2,
        baseline_depression_volume_km3: baseline.depression_volume_km3,
        stabilized_depression_volume_km3: stabilized.depression_volume_km3,
        maximum_baseline_fill_depth_m: baseline.maximum_fill_depth_m,
        maximum_stabilized_fill_depth_m: stabilized.maximum_fill_depth_m,
    };
    Ok(Outcome { cells, stats })
}

fn into_raw_array<T>(values: Vec<T>) -> (*mut T, usize) {
    let mut values = values.into_boxed_slice();
    let result = (values.as_mut_ptr(), values.len());
    let _ = Box::leak(values);
    result
}

fn free_raw_array<T>(data: *mut T, len: usize) {
    if !data.is_null() {
        unsafe {
            let values = std::ptr::slice_from_raw_parts_mut(data, len);
            drop(Box::from_raw(values));
        }
    }
}

unsafe fn read_slice<'a, T>(pointer: *const T, len: usize) -> &'a [T] {
    unsafe { slice::from_raw_parts(pointer, len) }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// Stabilize one sparse refined basin after erosion.
///
/// # Safety
///
/// Every input pointer must reference a contiguous buffer of the declared
/// length. Output pointers must be valid and must not alias input buffers.
pub unsafe extern "C" fn hydrology_pass2_run(
    config: HydrologyPass2Config,
    cell_count: usize,
    cell_ids: *const i32,
    terrain_before_m: *const f64,
    routing_surface_after_m: *const f64,
    cell_areas_km2: *const f64,
    cell_xyz: *const f32,
    anchor_kinds: *const u8,
    source_active: *const u8,
    fixed_receiver_ids: *const i32,
    cells_out: *mut StabilizedCellArray,
    stats_out: *mut HydrologyPass2Stats,
) -> i32 {
    if cell_count == 0
        || cell_ids.is_null()
        || terrain_before_m.is_null()
        || routing_surface_after_m.is_null()
        || cell_areas_km2.is_null()
        || cell_xyz.is_null()
        || anchor_kinds.is_null()
        || source_active.is_null()
        || fixed_receiver_ids.is_null()
        || cells_out.is_null()
        || stats_out.is_null()
    {
        return 4;
    }
    let inputs = Inputs {
        cell_ids: unsafe { read_slice(cell_ids, cell_count) },
        terrain_before_m: unsafe { read_slice(terrain_before_m, cell_count) },
        routing_surface_after_m: unsafe { read_slice(routing_surface_after_m, cell_count) },
        cell_areas_km2: unsafe { read_slice(cell_areas_km2, cell_count) },
        cell_xyz: unsafe { read_slice(cell_xyz, cell_count * 3) },
        anchor_kinds: unsafe { read_slice(anchor_kinds, cell_count) },
        source_active: unsafe { read_slice(source_active, cell_count) },
        fixed_receiver_ids: unsafe { read_slice(fixed_receiver_ids, cell_count) },
    };
    match run_pass(config, &inputs) {
        Ok(outcome) => {
            let (cells, len) = into_raw_array(outcome.cells);
            unsafe {
                *cells_out = StabilizedCellArray { data: cells, len };
                *stats_out = outcome.stats;
            }
            0
        }
        Err(status) => status,
    }
}

#[no_mangle]
pub extern "C" fn hydrology_pass2_free_cells(array: StabilizedCellArray) {
    free_raw_array(array.data, array.len);
}

#[cfg(test)]
mod tests {
    use super::*;
    use topology_native::{cubed_sphere_cell_xyz, cubed_sphere_global_index};

    fn grid_fixture() -> (Vec<i32>, Vec<f32>) {
        let resolution = 4;
        let mut ids = Vec::new();
        let mut xyz = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                ids.push(cubed_sphere_global_index(0, row, col, resolution).unwrap() as i32);
                xyz.extend(
                    cubed_sphere_cell_xyz(0, row, col, resolution)
                        .unwrap()
                        .map(|value| value as f32),
                );
            }
        }
        (ids, xyz)
    }

    #[test]
    fn stabilizes_sparse_basin_and_conserves_contributing_area() {
        let (ids, xyz) = grid_fixture();
        let terrain_before = vec![6.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0, 3.0, 6.0];
        let terrain_after = vec![6.0, 5.0, 6.0, 5.0, 2.0, 5.0, 6.0, 1.0, 6.0];
        let mut anchors = vec![ANCHOR_NORMAL; 9];
        let mut fixed = vec![NO_FIXED_RECEIVER; 9];
        for row in [1usize, 4, 7] {
            anchors[row] = ANCHOR_CHANNEL;
        }
        fixed[1] = ids[4];
        fixed[4] = ids[7];
        fixed[7] = TERMINAL_OCEAN;
        let inputs = Inputs {
            cell_ids: &ids,
            terrain_before_m: &terrain_before,
            routing_surface_after_m: &terrain_after,
            cell_areas_km2: &[1.0; 9],
            cell_xyz: &xyz,
            anchor_kinds: &anchors,
            source_active: &[1; 9],
            fixed_receiver_ids: &fixed,
        };
        let outcome = run_pass(
            HydrologyPass2Config {
                fine_resolution: 4,
                minimum_depression_depth_m: 0.5,
                planet_radius_m: 6_371_000.0,
            },
            &inputs,
        )
        .expect("valid stabilization");
        assert_eq!(outcome.stats.graph_valid, 1);
        assert_eq!(outcome.stats.trunk_preserved_valid, 1);
        assert_eq!(outcome.stats.physical_trunk_edge_count, 2);
        assert!((outcome.stats.terminal_accumulated_area_km2 - 9.0).abs() < 1e-12);
        assert_eq!(outcome.cells[1].stabilized_receiver_id, ids[4]);
        assert_eq!(outcome.cells[4].stabilized_receiver_id, ids[7]);
        assert_eq!(outcome.cells[7].stabilized_receiver_id, TERMINAL_OCEAN);
    }

    #[test]
    fn constrained_subgrid_outlet_may_discharge_to_an_ordinary_cell() {
        let (ids, xyz) = grid_fixture();
        let terrain = vec![6.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0, 3.0, 6.0];
        let mut anchors = vec![ANCHOR_NORMAL; 9];
        let mut fixed = vec![NO_FIXED_RECEIVER; 9];
        anchors[7] = ANCHOR_CHANNEL;
        fixed[1] = ids[4];
        fixed[7] = TERMINAL_OCEAN;
        let inputs = Inputs {
            cell_ids: &ids,
            terrain_before_m: &terrain,
            routing_surface_after_m: &terrain,
            cell_areas_km2: &[1.0; 9],
            cell_xyz: &xyz,
            anchor_kinds: &anchors,
            source_active: &[1; 9],
            fixed_receiver_ids: &fixed,
        };
        let outcome = run_pass(
            HydrologyPass2Config {
                fine_resolution: 4,
                minimum_depression_depth_m: 0.5,
                planet_radius_m: 6_371_000.0,
            },
            &inputs,
        )
        .expect("valid outlet-to-terrain handoff");
        assert_eq!(outcome.stats.graph_valid, 1);
        assert_eq!(outcome.cells[1].stabilized_receiver_id, ids[4]);
        assert_eq!(outcome.cells[1].anchor_kind, ANCHOR_NORMAL);
        assert_eq!(outcome.cells[4].anchor_kind, ANCHOR_NORMAL);
        assert!(outcome.cells[1].stabilized_depression_id < 0);
    }

    #[test]
    fn preserved_depression_is_a_nonphysical_handoff() {
        let (ids, xyz) = grid_fixture();
        let terrain = vec![5.0; 9];
        let mut anchors = vec![ANCHOR_NORMAL; 9];
        anchors[4] = ANCHOR_EXCLUDED;
        let inputs = Inputs {
            cell_ids: &ids,
            terrain_before_m: &terrain,
            routing_surface_after_m: &terrain,
            cell_areas_km2: &[1.0; 9],
            cell_xyz: &xyz,
            anchor_kinds: &anchors,
            source_active: &[1, 1, 1, 1, 0, 1, 1, 1, 1],
            fixed_receiver_ids: &[NO_FIXED_RECEIVER; 9],
        };
        let outcome = run_pass(
            HydrologyPass2Config {
                fine_resolution: 4,
                minimum_depression_depth_m: 0.5,
                planet_radius_m: 6_371_000.0,
            },
            &inputs,
        )
        .expect("valid excluded handoff");
        assert_eq!(outcome.stats.graph_valid, 1);
        assert_eq!(outcome.stats.process_exclusion_valid, 1);
        assert_eq!(outcome.cells[4].stabilized_receiver_id, TERMINAL_HANDOFF);
        assert_eq!(outcome.cells[4].contributing_area_km2, 8.0);
    }
}
