use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::slice;

const REACH_CHANNEL: u8 = 1;
const REACH_CONNECTOR: u8 = 2;
const TERMINAL_NONE: u8 = 0;
const TERMINAL_OCEAN: u8 = 1;
const TERMINAL_SINK: u8 = 2;

#[no_mangle]
pub extern "C" fn fluvial_native_abi_version() -> u32 {
    4
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FluvialConfig {
    pub planet_radius_m: f64,
    pub minimum_bed_slope: f64,
    pub maximum_deposition_fraction: f64,
    pub deposition_slope_scale: f64,
    pub maximum_deposition_depth_m: f64,
    /// Bank/valley carve depth as a fraction of local channel incision depth.
    /// Zero disables bank carving (channel prism only).
    pub bank_incision_fraction: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BedProfileRecord {
    pub reach_id: i32,
    pub fine_cell_id: i32,
    pub parent_cell_id: i32,
    pub path_order: i32,
    pub terrain_elevation_m: f64,
    pub bed_elevation_m: f64,
    pub incision_depth_m: f64,
    pub reach_length_m: f64,
    pub eroded_volume_m3: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ReachBudgetRecord {
    pub reach_id: i32,
    pub has_physical_bed: u8,
    pub entry_bed_elevation_m: f64,
    pub exit_bed_elevation_m: f64,
    pub minimum_realized_slope: f64,
    pub maximum_incision_depth_m: f64,
    pub upstream_input_volume_m3: f64,
    pub local_erosion_volume_m3: f64,
    pub bank_eroded_volume_m3: f64,
    pub available_sediment_volume_m3: f64,
    pub floodplain_deposition_volume_m3: f64,
    pub downstream_transfer_volume_m3: f64,
    pub terminal_deposition_volume_m3: f64,
    pub exported_sediment_volume_m3: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CellBudgetRecord {
    pub fine_cell_id: i32,
    pub parent_cell_id: i32,
    pub eroded_volume_m3: f64,
    pub deposited_volume_m3: f64,
    pub maximum_incision_depth_m: f64,
}

#[repr(C)]
pub struct BedProfileArray {
    pub data: *mut BedProfileRecord,
    pub len: usize,
}

#[repr(C)]
pub struct ReachBudgetArray {
    pub data: *mut ReachBudgetRecord,
    pub len: usize,
}

#[repr(C)]
pub struct CellBudgetArray {
    pub data: *mut CellBudgetRecord,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FluvialStats {
    pub physical_node_count: i32,
    pub physical_edge_count: i32,
    pub physical_component_count: i32,
    pub profile_record_count: i32,
    pub reach_count: i32,
    pub connector_reach_count: i32,
    pub maximum_incision_depth_m: f64,
    pub minimum_realized_slope: f64,
    pub total_eroded_volume_m3: f64,
    pub total_channel_eroded_volume_m3: f64,
    pub total_bank_eroded_volume_m3: f64,
    pub total_floodplain_deposition_volume_m3: f64,
    pub total_terminal_deposition_volume_m3: f64,
    pub total_exported_sediment_volume_m3: f64,
    pub sediment_conservation_residual_m3: f64,
    pub maximum_junction_bed_error_m: f64,
    pub bed_profile_valid: i32,
    pub sediment_conservation_valid: i32,
}

struct Inputs<'a> {
    cell_ids: &'a [i32],
    cell_parent_ids: &'a [i32],
    cell_terrain_m: &'a [f32],
    cell_areas_km2: &'a [f64],
    cell_xyz: &'a [f32],
    reach_ids: &'a [i32],
    downstream_reach_ids: &'a [i32],
    reach_kinds: &'a [u8],
    terminal_kinds: &'a [u8],
    channel_width_m: &'a [f32],
    reach_slope: &'a [f32],
    membership_reach_ids: &'a [i32],
    membership_cell_ids: &'a [i32],
    membership_parent_ids: &'a [i32],
    membership_path_order: &'a [i32],
    membership_reach_length_m: &'a [f64],
    membership_channel_fraction: &'a [f32],
    membership_valley_fraction: &'a [f32],
    membership_floodplain_fraction: &'a [f32],
}

struct Outcome {
    profiles: Vec<BedProfileRecord>,
    reaches: Vec<ReachBudgetRecord>,
    cells: Vec<CellBudgetRecord>,
    stats: FluvialStats,
}

#[derive(Clone, Copy, Default)]
struct CellAccumulator {
    parent_cell_id: i32,
    eroded_volume_m3: f64,
    deposited_volume_m3: f64,
    maximum_incision_depth_m: f64,
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn angular_distance(first: [f32; 3], second: [f32; 3]) -> f64 {
    let dot = first[0] as f64 * second[0] as f64
        + first[1] as f64 * second[1] as f64
        + first[2] as f64 * second[2] as f64;
    dot.clamp(-1.0, 1.0).acos()
}

fn topological_order(adjacency: &[Vec<usize>], indegree: &[usize]) -> Result<Vec<usize>, i32> {
    let mut degree = indegree.to_vec();
    let mut ready = BinaryHeap::new();
    for (node, &value) in degree.iter().enumerate() {
        if value == 0 {
            ready.push(Reverse(node));
        }
    }
    let mut order = Vec::with_capacity(adjacency.len());
    while let Some(Reverse(node)) = ready.pop() {
        order.push(node);
        for &target in &adjacency[node] {
            degree[target] -= 1;
            if degree[target] == 0 {
                ready.push(Reverse(target));
            }
        }
    }
    if order.len() != adjacency.len() {
        return Err(5);
    }
    Ok(order)
}

fn validate_inputs(config: FluvialConfig, inputs: &Inputs<'_>) -> Result<(), i32> {
    if !config.planet_radius_m.is_finite()
        || config.planet_radius_m <= 0.0
        || !finite_nonnegative(config.minimum_bed_slope)
        || !finite_nonnegative(config.maximum_deposition_fraction)
        || config.maximum_deposition_fraction > 1.0
        || !config.deposition_slope_scale.is_finite()
        || config.deposition_slope_scale <= 0.0
        || !finite_nonnegative(config.maximum_deposition_depth_m)
        || !finite_nonnegative(config.bank_incision_fraction)
        || config.bank_incision_fraction > 1.0
    {
        return Err(1);
    }
    let cell_count = inputs.cell_ids.len();
    let reach_count = inputs.reach_ids.len();
    let membership_count = inputs.membership_reach_ids.len();
    if cell_count == 0
        || reach_count == 0
        || inputs.cell_parent_ids.len() != cell_count
        || inputs.cell_terrain_m.len() != cell_count
        || inputs.cell_areas_km2.len() != cell_count
        || inputs.cell_xyz.len() != cell_count * 3
        || inputs.downstream_reach_ids.len() != reach_count
        || inputs.reach_kinds.len() != reach_count
        || inputs.terminal_kinds.len() != reach_count
        || inputs.channel_width_m.len() != reach_count
        || inputs.reach_slope.len() != reach_count
        || inputs.membership_cell_ids.len() != membership_count
        || inputs.membership_parent_ids.len() != membership_count
        || inputs.membership_path_order.len() != membership_count
        || inputs.membership_reach_length_m.len() != membership_count
        || inputs.membership_channel_fraction.len() != membership_count
        || inputs.membership_valley_fraction.len() != membership_count
        || inputs.membership_floodplain_fraction.len() != membership_count
    {
        return Err(1);
    }
    if inputs.cell_terrain_m.iter().any(|value| !value.is_finite())
        || inputs
            .cell_areas_km2
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || inputs.cell_xyz.iter().any(|value| !value.is_finite())
        || inputs
            .channel_width_m
            .iter()
            .chain(inputs.reach_slope)
            .any(|value| !value.is_finite() || *value < 0.0)
        || inputs
            .membership_reach_length_m
            .iter()
            .any(|value| !finite_nonnegative(*value))
        || inputs
            .membership_channel_fraction
            .iter()
            .chain(inputs.membership_valley_fraction)
            .chain(inputs.membership_floodplain_fraction)
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(6);
    }
    Ok(())
}

fn run_fluvial(config: FluvialConfig, inputs: &Inputs<'_>) -> Result<Outcome, i32> {
    validate_inputs(config, inputs)?;
    let cell_row_by_id = inputs
        .cell_ids
        .iter()
        .enumerate()
        .map(|(row, &cell_id)| (cell_id, row))
        .collect::<HashMap<_, _>>();
    if cell_row_by_id.len() != inputs.cell_ids.len() {
        return Err(3);
    }
    let reach_row_by_id = inputs
        .reach_ids
        .iter()
        .enumerate()
        .map(|(row, &reach_id)| (reach_id, row))
        .collect::<HashMap<_, _>>();
    if reach_row_by_id.len() != inputs.reach_ids.len() {
        return Err(3);
    }

    let mut centerline_by_reach = vec![Vec::<usize>::new(); inputs.reach_ids.len()];
    let mut support_by_reach = vec![Vec::<usize>::new(); inputs.reach_ids.len()];
    let mut physical_cell_ids = BTreeSet::new();
    for reach in 0..inputs.reach_ids.len() {
        if !matches!(inputs.reach_kinds[reach], REACH_CHANNEL | REACH_CONNECTOR)
            || (inputs.reach_kinds[reach] == REACH_CONNECTOR
                && inputs.channel_width_m[reach] != 0.0)
        {
            return Err(4);
        }
    }
    for membership in 0..inputs.membership_reach_ids.len() {
        let reach = *reach_row_by_id
            .get(&inputs.membership_reach_ids[membership])
            .ok_or(3)?;
        let cell = *cell_row_by_id
            .get(&inputs.membership_cell_ids[membership])
            .ok_or(3)?;
        if inputs.cell_parent_ids[cell] != inputs.membership_parent_ids[membership] {
            return Err(3);
        }
        if inputs.reach_kinds[reach] == REACH_CONNECTOR {
            return Err(4);
        }
        support_by_reach[reach].push(membership);
        if inputs.membership_reach_length_m[membership] > 0.0 {
            if inputs.reach_kinds[reach] != REACH_CHANNEL
                || inputs.channel_width_m[reach] <= 0.0
                || inputs.membership_channel_fraction[membership] <= 0.0
            {
                return Err(4);
            }
            centerline_by_reach[reach].push(membership);
            physical_cell_ids.insert(inputs.membership_cell_ids[membership]);
        }
    }
    for centerline in &mut centerline_by_reach {
        centerline.sort_by_key(|&membership| {
            (
                inputs.membership_path_order[membership],
                inputs.membership_cell_ids[membership],
            )
        });
        if centerline.windows(2).any(|pair| {
            inputs.membership_path_order[pair[0]] == inputs.membership_path_order[pair[1]]
        }) {
            return Err(3);
        }
    }

    let node_cell_ids = physical_cell_ids.into_iter().collect::<Vec<_>>();
    let node_by_cell = node_cell_ids
        .iter()
        .enumerate()
        .map(|(node, &cell_id)| (cell_id, node))
        .collect::<HashMap<_, _>>();
    let mut node_terrain = Vec::with_capacity(node_cell_ids.len());
    let mut node_xyz = Vec::with_capacity(node_cell_ids.len());
    for &cell_id in &node_cell_ids {
        let row = *cell_row_by_id.get(&cell_id).ok_or(3)?;
        node_terrain.push(inputs.cell_terrain_m[row] as f64);
        node_xyz.push([
            inputs.cell_xyz[row * 3],
            inputs.cell_xyz[row * 3 + 1],
            inputs.cell_xyz[row * 3 + 2],
        ]);
    }

    let mut edge_set = HashSet::new();
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); node_cell_ids.len()];
    let mut undirected = vec![Vec::<usize>::new(); node_cell_ids.len()];
    let mut indegree = vec![0usize; node_cell_ids.len()];
    for centerline in &centerline_by_reach {
        for pair in centerline.windows(2) {
            let first_order = inputs.membership_path_order[pair[0]];
            let second_order = inputs.membership_path_order[pair[1]];
            if second_order != first_order + 1 {
                continue;
            }
            let source = *node_by_cell
                .get(&inputs.membership_cell_ids[pair[0]])
                .ok_or(3)?;
            let target = *node_by_cell
                .get(&inputs.membership_cell_ids[pair[1]])
                .ok_or(3)?;
            if source == target || edge_set.contains(&(target, source)) {
                return Err(5);
            }
            if edge_set.insert((source, target)) {
                let length_m =
                    angular_distance(node_xyz[source], node_xyz[target]) * config.planet_radius_m;
                if !length_m.is_finite() || length_m <= 0.0 {
                    return Err(6);
                }
                adjacency[source].push((target, length_m));
                undirected[source].push(target);
                undirected[target].push(source);
                indegree[target] += 1;
            }
        }
    }
    for targets in &mut adjacency {
        targets.sort_by_key(|(target, _)| *target);
    }
    let adjacency_nodes = adjacency
        .iter()
        .map(|targets| {
            targets
                .iter()
                .map(|(target, _)| *target)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let node_order = topological_order(&adjacency_nodes, &indegree)?;
    let mut bed = node_terrain.clone();
    for &source in &node_order {
        for &(target, length_m) in &adjacency[source] {
            let maximum_target = bed[source] - config.minimum_bed_slope * length_m;
            bed[target] = bed[target].min(maximum_target);
        }
    }

    let mut component_count = 0usize;
    let mut seen = vec![false; node_cell_ids.len()];
    for start in 0..node_cell_ids.len() {
        if seen[start] {
            continue;
        }
        component_count += 1;
        seen[start] = true;
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            for &neighbor in &undirected[node] {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }

    let mut cell_accumulators = BTreeMap::<i32, CellAccumulator>::new();
    let mut profiles = Vec::new();
    let mut local_erosion = vec![0.0f64; inputs.reach_ids.len()];
    let mut maximum_reach_incision = vec![0.0f64; inputs.reach_ids.len()];
    let mut entry_bed = vec![f64::NAN; inputs.reach_ids.len()];
    let mut exit_bed = vec![f64::NAN; inputs.reach_ids.len()];
    let mut minimum_reach_slope = vec![f64::NAN; inputs.reach_ids.len()];
    let mut maximum_incision_depth = 0.0f64;
    let mut minimum_realized_slope = f64::INFINITY;
    for (reach, centerline) in centerline_by_reach.iter().enumerate() {
        if !centerline.is_empty() {
            let first_node = *node_by_cell
                .get(&inputs.membership_cell_ids[centerline[0]])
                .ok_or(3)?;
            let last_node = *node_by_cell
                .get(&inputs.membership_cell_ids[*centerline.last().ok_or(3)?])
                .ok_or(3)?;
            entry_bed[reach] = bed[first_node];
            exit_bed[reach] = bed[last_node];
        }
        let mut reach_minimum_slope = f64::INFINITY;
        for pair in centerline.windows(2) {
            if inputs.membership_path_order[pair[1]] != inputs.membership_path_order[pair[0]] + 1 {
                continue;
            }
            let source = *node_by_cell
                .get(&inputs.membership_cell_ids[pair[0]])
                .ok_or(3)?;
            let target = *node_by_cell
                .get(&inputs.membership_cell_ids[pair[1]])
                .ok_or(3)?;
            let length_m =
                angular_distance(node_xyz[source], node_xyz[target]) * config.planet_radius_m;
            let realized = (bed[source] - bed[target]) / length_m;
            reach_minimum_slope = reach_minimum_slope.min(realized);
            minimum_realized_slope = minimum_realized_slope.min(realized);
        }
        if reach_minimum_slope.is_finite() {
            minimum_reach_slope[reach] = reach_minimum_slope;
        }
        for &membership in centerline {
            let cell_id = inputs.membership_cell_ids[membership];
            let cell_row = *cell_row_by_id.get(&cell_id).ok_or(3)?;
            let node = *node_by_cell.get(&cell_id).ok_or(3)?;
            let incision_depth_m = (node_terrain[node] - bed[node]).max(0.0);
            let eroded_volume_m3 = incision_depth_m
                * inputs.channel_width_m[reach] as f64
                * inputs.membership_reach_length_m[membership];
            local_erosion[reach] += eroded_volume_m3;
            maximum_reach_incision[reach] = maximum_reach_incision[reach].max(incision_depth_m);
            maximum_incision_depth = maximum_incision_depth.max(incision_depth_m);
            let accumulator = cell_accumulators.entry(cell_id).or_insert(CellAccumulator {
                parent_cell_id: inputs.cell_parent_ids[cell_row],
                ..CellAccumulator::default()
            });
            accumulator.eroded_volume_m3 += eroded_volume_m3;
            accumulator.maximum_incision_depth_m =
                accumulator.maximum_incision_depth_m.max(incision_depth_m);
            profiles.push(BedProfileRecord {
                reach_id: inputs.reach_ids[reach],
                fine_cell_id: cell_id,
                parent_cell_id: inputs.membership_parent_ids[membership],
                path_order: inputs.membership_path_order[membership],
                terrain_elevation_m: node_terrain[node],
                bed_elevation_m: bed[node],
                incision_depth_m,
                reach_length_m: inputs.membership_reach_length_m[membership],
                eroded_volume_m3,
            });
        }
    }
    profiles.sort_by_key(|record| (record.reach_id, record.path_order, record.fine_cell_id));

    // Bank / valley carve: lower valley-support area (including lateral memberships)
    // by a fraction of local channel incision. This is additional solid volume beyond
    // the channel prism and feeds the same source-to-sink sediment budget.
    let mut bank_erosion = vec![0.0f64; inputs.reach_ids.len()];
    let mut channel_incision_by_cell = HashMap::<i32, f64>::new();
    for record in &profiles {
        channel_incision_by_cell
            .entry(record.fine_cell_id)
            .and_modify(|depth| *depth = depth.max(record.incision_depth_m))
            .or_insert(record.incision_depth_m);
    }
    if config.bank_incision_fraction > 0.0 {
        for (reach, support) in support_by_reach.iter().enumerate() {
            if inputs.reach_kinds[reach] != REACH_CHANNEL || maximum_reach_incision[reach] <= 0.0 {
                continue;
            }
            let mut path_incision = HashMap::<i32, f64>::new();
            for &membership in &centerline_by_reach[reach] {
                let cell_id = inputs.membership_cell_ids[membership];
                let depth = *channel_incision_by_cell.get(&cell_id).unwrap_or(&0.0);
                let path_order = inputs.membership_path_order[membership];
                path_incision
                    .entry(path_order)
                    .and_modify(|value| *value = value.max(depth))
                    .or_insert(depth);
            }
            let reach_max_incision = maximum_reach_incision[reach];
            for &membership in support {
                let valley = inputs.membership_valley_fraction[membership] as f64;
                let channel = inputs.membership_channel_fraction[membership] as f64;
                let bank_fraction = (valley - channel).max(0.0);
                if bank_fraction <= 0.0 {
                    continue;
                }
                let cell_id = inputs.membership_cell_ids[membership];
                let cell = *cell_row_by_id.get(&cell_id).ok_or(3)?;
                let associated_incision = if inputs.membership_reach_length_m[membership] > 0.0 {
                    *channel_incision_by_cell.get(&cell_id).unwrap_or(&0.0)
                } else {
                    // Lateral valley support inherits the same-path centerline cut when
                    // positive; otherwise the reach-max incision so banks still carve
                    // around any incised reach segment.
                    let path_depth = path_incision
                        .get(&inputs.membership_path_order[membership])
                        .copied()
                        .unwrap_or(0.0);
                    if path_depth > 0.0 {
                        path_depth
                    } else {
                        reach_max_incision
                    }
                };
                if associated_incision <= 0.0 {
                    continue;
                }
                let bank_depth_m = config.bank_incision_fraction * associated_incision;
                let area_m2 = inputs.cell_areas_km2[cell] * 1_000_000.0;
                let bank_volume_m3 = bank_depth_m * bank_fraction * area_m2;
                if bank_volume_m3 <= 0.0 {
                    continue;
                }
                bank_erosion[reach] += bank_volume_m3;
                local_erosion[reach] += bank_volume_m3;
                let accumulator = cell_accumulators.entry(cell_id).or_insert(CellAccumulator {
                    parent_cell_id: inputs.cell_parent_ids[cell],
                    ..CellAccumulator::default()
                });
                accumulator.eroded_volume_m3 += bank_volume_m3;
                // Bank carve is not channel bed incision; do not raise maximum_channel_incision.
            }
        }
    }

    let mut reach_adjacency = vec![Vec::<usize>::new(); inputs.reach_ids.len()];
    let mut reach_indegree = vec![0usize; inputs.reach_ids.len()];
    for (reach, downstream_targets) in reach_adjacency.iter_mut().enumerate() {
        let downstream_id = inputs.downstream_reach_ids[reach];
        if downstream_id >= 0 {
            let downstream = *reach_row_by_id.get(&downstream_id).ok_or(3)?;
            downstream_targets.push(downstream);
            reach_indegree[downstream] += 1;
            if inputs.terminal_kinds[reach] != TERMINAL_NONE {
                return Err(3);
            }
        } else if !matches!(inputs.terminal_kinds[reach], TERMINAL_OCEAN | TERMINAL_SINK) {
            return Err(3);
        }
    }
    let reach_order = topological_order(&reach_adjacency, &reach_indegree)?;
    let mut upstream_input = vec![0.0f64; inputs.reach_ids.len()];
    let mut reaches = vec![ReachBudgetRecord::default(); inputs.reach_ids.len()];
    let mut total_floodplain_deposition = 0.0f64;
    let mut total_terminal_deposition = 0.0f64;
    let mut total_exported = 0.0f64;
    for &reach in &reach_order {
        let available = upstream_input[reach] + local_erosion[reach];
        let mut floodplain_area_m2 = 0.0f64;
        let mut valley_area_m2 = 0.0f64;
        for &membership in &support_by_reach[reach] {
            let cell = *cell_row_by_id
                .get(&inputs.membership_cell_ids[membership])
                .ok_or(3)?;
            let area_m2 = inputs.cell_areas_km2[cell] * 1_000_000.0;
            floodplain_area_m2 +=
                inputs.membership_floodplain_fraction[membership] as f64 * area_m2;
            valley_area_m2 += inputs.membership_valley_fraction[membership] as f64 * area_m2;
        }
        let deposition_fraction = if inputs.reach_kinds[reach] == REACH_CHANNEL
            && floodplain_area_m2 > 0.0
            && valley_area_m2 > 0.0
        {
            let support_ratio = (floodplain_area_m2 / valley_area_m2).clamp(0.0, 1.0);
            let slope_factor = config.deposition_slope_scale
                / (inputs.reach_slope[reach] as f64 + config.deposition_slope_scale);
            (config.maximum_deposition_fraction * support_ratio * slope_factor).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let deposition_capacity = floodplain_area_m2 * config.maximum_deposition_depth_m;
        let deposition = (available * deposition_fraction).min(deposition_capacity);
        if deposition > 0.0 {
            for &membership in &support_by_reach[reach] {
                let cell_id = inputs.membership_cell_ids[membership];
                let cell = *cell_row_by_id.get(&cell_id).ok_or(3)?;
                let support_area_m2 = inputs.membership_floodplain_fraction[membership] as f64
                    * inputs.cell_areas_km2[cell]
                    * 1_000_000.0;
                if support_area_m2 <= 0.0 {
                    continue;
                }
                let allocated = deposition * support_area_m2 / floodplain_area_m2;
                let accumulator = cell_accumulators.entry(cell_id).or_insert(CellAccumulator {
                    parent_cell_id: inputs.cell_parent_ids[cell],
                    ..CellAccumulator::default()
                });
                accumulator.deposited_volume_m3 += allocated;
            }
        }
        total_floodplain_deposition += deposition;
        let remainder = (available - deposition).max(0.0);
        let downstream_id = inputs.downstream_reach_ids[reach];
        let mut downstream_transfer = 0.0;
        let mut terminal_deposition = 0.0;
        let mut exported = 0.0;
        if downstream_id >= 0 {
            let downstream = *reach_row_by_id.get(&downstream_id).ok_or(3)?;
            upstream_input[downstream] += remainder;
            downstream_transfer = remainder;
        } else if inputs.terminal_kinds[reach] == TERMINAL_SINK {
            terminal_deposition = remainder;
            total_terminal_deposition += remainder;
        } else {
            exported = remainder;
            total_exported += remainder;
        }
        reaches[reach] = ReachBudgetRecord {
            reach_id: inputs.reach_ids[reach],
            has_physical_bed: u8::from(!centerline_by_reach[reach].is_empty()),
            entry_bed_elevation_m: entry_bed[reach],
            exit_bed_elevation_m: exit_bed[reach],
            minimum_realized_slope: minimum_reach_slope[reach],
            maximum_incision_depth_m: maximum_reach_incision[reach],
            upstream_input_volume_m3: upstream_input[reach],
            local_erosion_volume_m3: local_erosion[reach],
            bank_eroded_volume_m3: bank_erosion[reach],
            available_sediment_volume_m3: available,
            floodplain_deposition_volume_m3: deposition,
            downstream_transfer_volume_m3: downstream_transfer,
            terminal_deposition_volume_m3: terminal_deposition,
            exported_sediment_volume_m3: exported,
        };
    }

    let cells = cell_accumulators
        .into_iter()
        .map(|(fine_cell_id, value)| CellBudgetRecord {
            fine_cell_id,
            parent_cell_id: value.parent_cell_id,
            eroded_volume_m3: value.eroded_volume_m3,
            deposited_volume_m3: value.deposited_volume_m3,
            maximum_incision_depth_m: value.maximum_incision_depth_m,
        })
        .collect::<Vec<_>>();
    let total_eroded = local_erosion.iter().sum::<f64>();
    let total_bank_eroded = bank_erosion.iter().sum::<f64>();
    let total_channel_eroded = (total_eroded - total_bank_eroded).max(0.0);
    let conservation_residual =
        total_eroded - total_floodplain_deposition - total_terminal_deposition - total_exported;
    let conservation_tolerance = 1e-9 * total_eroded.max(1.0);
    let bed_profile_valid = bed
        .iter()
        .enumerate()
        .all(|(node, value)| value.is_finite() && *value <= node_terrain[node] + 1e-6)
        && adjacency.iter().enumerate().all(|(source, targets)| {
            targets.iter().all(|(target, length_m)| {
                bed[source] + 1e-6 >= bed[*target] + config.minimum_bed_slope * *length_m
            })
        });
    let stats = FluvialStats {
        physical_node_count: i32::try_from(node_cell_ids.len()).map_err(|_| 1)?,
        physical_edge_count: i32::try_from(edge_set.len()).map_err(|_| 1)?,
        physical_component_count: i32::try_from(component_count).map_err(|_| 1)?,
        profile_record_count: i32::try_from(profiles.len()).map_err(|_| 1)?,
        reach_count: i32::try_from(inputs.reach_ids.len()).map_err(|_| 1)?,
        connector_reach_count: i32::try_from(
            inputs
                .reach_kinds
                .iter()
                .filter(|kind| **kind == REACH_CONNECTOR)
                .count(),
        )
        .map_err(|_| 1)?,
        maximum_incision_depth_m: maximum_incision_depth,
        minimum_realized_slope: if minimum_realized_slope.is_finite() {
            minimum_realized_slope
        } else {
            0.0
        },
        total_eroded_volume_m3: total_eroded,
        total_channel_eroded_volume_m3: total_channel_eroded,
        total_bank_eroded_volume_m3: total_bank_eroded,
        total_floodplain_deposition_volume_m3: total_floodplain_deposition,
        total_terminal_deposition_volume_m3: total_terminal_deposition,
        total_exported_sediment_volume_m3: total_exported,
        sediment_conservation_residual_m3: conservation_residual,
        maximum_junction_bed_error_m: 0.0,
        bed_profile_valid: i32::from(bed_profile_valid),
        sediment_conservation_valid: i32::from(
            conservation_residual.abs() <= conservation_tolerance,
        ),
    };
    Ok(Outcome {
        profiles,
        reaches,
        cells,
        stats,
    })
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
/// Solve sparse channel-bed profiles and route their eroded sediment.
///
/// # Safety
///
/// Every pointer must reference a contiguous buffer of the declared length.
/// Output pointers must be valid and must not alias input buffers.
pub unsafe extern "C" fn fluvial_run(
    config: FluvialConfig,
    cell_count: usize,
    reach_count: usize,
    membership_count: usize,
    cell_ids: *const i32,
    cell_parent_ids: *const i32,
    cell_terrain_m: *const f32,
    cell_areas_km2: *const f64,
    cell_xyz: *const f32,
    reach_ids: *const i32,
    downstream_reach_ids: *const i32,
    reach_kinds: *const u8,
    terminal_kinds: *const u8,
    channel_width_m: *const f32,
    reach_slope: *const f32,
    membership_reach_ids: *const i32,
    membership_cell_ids: *const i32,
    membership_parent_ids: *const i32,
    membership_path_order: *const i32,
    membership_reach_length_m: *const f64,
    membership_channel_fraction: *const f32,
    membership_valley_fraction: *const f32,
    membership_floodplain_fraction: *const f32,
    profiles_out: *mut BedProfileArray,
    reaches_out: *mut ReachBudgetArray,
    cells_out: *mut CellBudgetArray,
    stats_out: *mut FluvialStats,
) -> i32 {
    if cell_count == 0
        || reach_count == 0
        || cell_ids.is_null()
        || cell_parent_ids.is_null()
        || cell_terrain_m.is_null()
        || cell_areas_km2.is_null()
        || cell_xyz.is_null()
        || reach_ids.is_null()
        || downstream_reach_ids.is_null()
        || reach_kinds.is_null()
        || terminal_kinds.is_null()
        || channel_width_m.is_null()
        || reach_slope.is_null()
        || (membership_count > 0
            && (membership_reach_ids.is_null()
                || membership_cell_ids.is_null()
                || membership_parent_ids.is_null()
                || membership_path_order.is_null()
                || membership_reach_length_m.is_null()
                || membership_channel_fraction.is_null()
                || membership_valley_fraction.is_null()
                || membership_floodplain_fraction.is_null()))
        || profiles_out.is_null()
        || reaches_out.is_null()
        || cells_out.is_null()
        || stats_out.is_null()
    {
        return 2;
    }
    let inputs = Inputs {
        cell_ids: unsafe { read_slice(cell_ids, cell_count) },
        cell_parent_ids: unsafe { read_slice(cell_parent_ids, cell_count) },
        cell_terrain_m: unsafe { read_slice(cell_terrain_m, cell_count) },
        cell_areas_km2: unsafe { read_slice(cell_areas_km2, cell_count) },
        cell_xyz: unsafe { read_slice(cell_xyz, cell_count * 3) },
        reach_ids: unsafe { read_slice(reach_ids, reach_count) },
        downstream_reach_ids: unsafe { read_slice(downstream_reach_ids, reach_count) },
        reach_kinds: unsafe { read_slice(reach_kinds, reach_count) },
        terminal_kinds: unsafe { read_slice(terminal_kinds, reach_count) },
        channel_width_m: unsafe { read_slice(channel_width_m, reach_count) },
        reach_slope: unsafe { read_slice(reach_slope, reach_count) },
        membership_reach_ids: unsafe { read_slice(membership_reach_ids, membership_count) },
        membership_cell_ids: unsafe { read_slice(membership_cell_ids, membership_count) },
        membership_parent_ids: unsafe { read_slice(membership_parent_ids, membership_count) },
        membership_path_order: unsafe { read_slice(membership_path_order, membership_count) },
        membership_reach_length_m: unsafe {
            read_slice(membership_reach_length_m, membership_count)
        },
        membership_channel_fraction: unsafe {
            read_slice(membership_channel_fraction, membership_count)
        },
        membership_valley_fraction: unsafe {
            read_slice(membership_valley_fraction, membership_count)
        },
        membership_floodplain_fraction: unsafe {
            read_slice(membership_floodplain_fraction, membership_count)
        },
    };
    match run_fluvial(config, &inputs) {
        Ok(outcome) => {
            let (profiles, profile_count) = into_raw_array(outcome.profiles);
            let (reaches, output_reach_count) = into_raw_array(outcome.reaches);
            let (cells, output_cell_count) = into_raw_array(outcome.cells);
            unsafe {
                *profiles_out = BedProfileArray {
                    data: profiles,
                    len: profile_count,
                };
                *reaches_out = ReachBudgetArray {
                    data: reaches,
                    len: output_reach_count,
                };
                *cells_out = CellBudgetArray {
                    data: cells,
                    len: output_cell_count,
                };
                *stats_out = outcome.stats;
            }
            0
        }
        Err(status) => status,
    }
}

#[no_mangle]
pub extern "C" fn fluvial_free_profiles(array: BedProfileArray) {
    free_raw_array(array.data, array.len);
}

#[no_mangle]
pub extern "C" fn fluvial_free_reaches(array: ReachBudgetArray) {
    free_raw_array(array.data, array.len);
}

#[no_mangle]
pub extern "C" fn fluvial_free_cells(array: CellBudgetArray) {
    free_raw_array(array.data, array.len);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FluvialConfig {
        FluvialConfig {
            planet_radius_m: 1.0,
            minimum_bed_slope: 0.0,
            maximum_deposition_fraction: 0.25,
            deposition_slope_scale: 0.001,
            maximum_deposition_depth_m: 10.0,
            bank_incision_fraction: 0.0,
        }
    }

    #[test]
    fn profile_cuts_downstream_rise_and_conserves_sediment() {
        let inputs = Inputs {
            cell_ids: &[10, 11, 12],
            cell_parent_ids: &[1, 1, 1],
            cell_terrain_m: &[10.0, 8.0, 9.0],
            cell_areas_km2: &[1.0, 1.0, 1.0],
            cell_xyz: &[1.0, 0.0, 0.0, 0.999, 0.04, 0.0, 0.996, 0.08, 0.0],
            reach_ids: &[1],
            downstream_reach_ids: &[-1],
            reach_kinds: &[REACH_CHANNEL],
            terminal_kinds: &[TERMINAL_OCEAN],
            channel_width_m: &[10.0],
            reach_slope: &[0.01],
            membership_reach_ids: &[1, 1, 1],
            membership_cell_ids: &[10, 11, 12],
            membership_parent_ids: &[1, 1, 1],
            membership_path_order: &[0, 1, 2],
            membership_reach_length_m: &[50.0, 100.0, 50.0],
            membership_channel_fraction: &[0.0005, 0.001, 0.0005],
            membership_valley_fraction: &[0.01, 0.01, 0.01],
            membership_floodplain_fraction: &[0.005, 0.005, 0.005],
        };
        let outcome = run_fluvial(config(), &inputs).expect("valid fluvial result");
        let beds = outcome
            .profiles
            .iter()
            .map(|record| record.bed_elevation_m)
            .collect::<Vec<_>>();
        assert_eq!(beds, vec![10.0, 8.0, 8.0]);
        assert_eq!(outcome.profiles[2].incision_depth_m, 1.0);
        assert_eq!(outcome.stats.bed_profile_valid, 1);
        assert_eq!(outcome.stats.sediment_conservation_valid, 1);
        assert!(outcome.stats.total_eroded_volume_m3 > 0.0);
        assert!(outcome.stats.total_exported_sediment_volume_m3 > 0.0);
        assert_eq!(outcome.stats.total_bank_eroded_volume_m3, 0.0);
    }

    #[test]
    fn bank_carve_adds_valley_volume_and_still_conserves() {
        let mut controls = config();
        controls.bank_incision_fraction = 0.4;
        let inputs = Inputs {
            cell_ids: &[10, 11, 12, 99],
            cell_parent_ids: &[1, 1, 1, 1],
            cell_terrain_m: &[10.0, 8.0, 9.0, 9.5],
            cell_areas_km2: &[1.0, 1.0, 1.0, 1.0],
            cell_xyz: &[
                1.0, 0.0, 0.0, 0.999, 0.04, 0.0, 0.996, 0.08, 0.0, 0.999, 0.0, 0.04,
            ],
            reach_ids: &[1],
            downstream_reach_ids: &[-1],
            reach_kinds: &[REACH_CHANNEL],
            terminal_kinds: &[TERMINAL_OCEAN],
            channel_width_m: &[10.0],
            reach_slope: &[0.01],
            membership_reach_ids: &[1, 1, 1, 1],
            membership_cell_ids: &[10, 11, 12, 99],
            membership_parent_ids: &[1, 1, 1, 1],
            membership_path_order: &[0, 1, 2, 1],
            membership_reach_length_m: &[50.0, 100.0, 50.0, 0.0],
            membership_channel_fraction: &[0.0005, 0.001, 0.0005, 0.0],
            membership_valley_fraction: &[0.01, 0.01, 0.01, 0.02],
            membership_floodplain_fraction: &[0.005, 0.005, 0.005, 0.01],
        };
        let outcome = run_fluvial(controls, &inputs).expect("bank carve result");
        assert!(outcome.stats.total_bank_eroded_volume_m3 > 0.0);
        assert!(
            outcome.stats.total_eroded_volume_m3
                > outcome.stats.total_channel_eroded_volume_m3 + 1e-9
        );
        assert_eq!(outcome.stats.sediment_conservation_valid, 1);
        assert!(outcome.reaches[0].bank_eroded_volume_m3 > 0.0);
        let lateral = outcome
            .cells
            .iter()
            .find(|cell| cell.fine_cell_id == 99)
            .expect("lateral bank cell");
        assert!(lateral.eroded_volume_m3 > 0.0);
    }

    #[test]
    fn connector_transfers_without_physical_process_support() {
        let inputs = Inputs {
            cell_ids: &[10, 11, 12, 13],
            cell_parent_ids: &[1, 1, 2, 2],
            cell_terrain_m: &[10.0, 8.0, 7.0, 8.0],
            cell_areas_km2: &[1.0; 4],
            cell_xyz: &[
                1.0, 0.0, 0.0, 0.999, 0.04, 0.0, 0.996, 0.08, 0.0, 0.992, 0.12, 0.0,
            ],
            reach_ids: &[1, 2, 3],
            downstream_reach_ids: &[2, 3, -1],
            reach_kinds: &[REACH_CHANNEL, REACH_CONNECTOR, REACH_CHANNEL],
            terminal_kinds: &[TERMINAL_NONE, TERMINAL_NONE, TERMINAL_OCEAN],
            channel_width_m: &[10.0, 0.0, 20.0],
            reach_slope: &[0.01, 0.0, 0.001],
            membership_reach_ids: &[1, 1, 3, 3],
            membership_cell_ids: &[10, 11, 12, 13],
            membership_parent_ids: &[1, 1, 2, 2],
            membership_path_order: &[0, 1, 0, 1],
            membership_reach_length_m: &[50.0; 4],
            membership_channel_fraction: &[0.0005, 0.0005, 0.001, 0.001],
            membership_valley_fraction: &[0.01; 4],
            membership_floodplain_fraction: &[0.005; 4],
        };
        let outcome = run_fluvial(config(), &inputs).expect("valid connector routing");
        let connector = &outcome.reaches[1];
        assert_eq!(connector.has_physical_bed, 0);
        assert_eq!(connector.local_erosion_volume_m3, 0.0);
        assert_eq!(connector.floodplain_deposition_volume_m3, 0.0);
        assert_eq!(
            connector.upstream_input_volume_m3,
            outcome.reaches[0].downstream_transfer_volume_m3
        );
        assert_eq!(
            connector.downstream_transfer_volume_m3,
            connector.upstream_input_volume_m3
        );
        assert_eq!(outcome.stats.sediment_conservation_valid, 1);
    }

    #[test]
    fn confluences_share_one_junction_bed() {
        let inputs = Inputs {
            cell_ids: &[10, 20, 12, 13],
            cell_parent_ids: &[1, 2, 3, 3],
            cell_terrain_m: &[10.0, 9.0, 8.0, 9.0],
            cell_areas_km2: &[1.0; 4],
            cell_xyz: &[
                1.0, 0.0, 0.0, 0.999, -0.04, 0.0, 0.999, 0.04, 0.0, 0.996, 0.08, 0.0,
            ],
            reach_ids: &[1, 2, 3],
            downstream_reach_ids: &[3, 3, -1],
            reach_kinds: &[REACH_CHANNEL; 3],
            terminal_kinds: &[TERMINAL_NONE, TERMINAL_NONE, TERMINAL_OCEAN],
            channel_width_m: &[10.0, 10.0, 20.0],
            reach_slope: &[0.01, 0.01, 0.001],
            membership_reach_ids: &[1, 1, 2, 2, 3, 3],
            membership_cell_ids: &[10, 12, 20, 12, 12, 13],
            membership_parent_ids: &[1, 3, 2, 3, 3, 3],
            membership_path_order: &[0, 1, 0, 1, 0, 1],
            membership_reach_length_m: &[50.0; 6],
            membership_channel_fraction: &[0.0005, 0.0005, 0.0005, 0.0005, 0.001, 0.001],
            membership_valley_fraction: &[0.01; 6],
            membership_floodplain_fraction: &[0.005; 6],
        };
        let outcome = run_fluvial(config(), &inputs).expect("valid confluence profile");
        let junction_beds = outcome
            .profiles
            .iter()
            .filter(|record| record.fine_cell_id == 12)
            .map(|record| record.bed_elevation_m)
            .collect::<Vec<_>>();
        assert_eq!(junction_beds, vec![8.0, 8.0, 8.0]);
        assert_eq!(outcome.reaches[0].exit_bed_elevation_m, 8.0);
        assert_eq!(outcome.reaches[1].exit_bed_elevation_m, 8.0);
        assert_eq!(outcome.reaches[2].entry_bed_elevation_m, 8.0);
        assert_eq!(outcome.stats.maximum_junction_bed_error_m, 0.0);
        assert_eq!(outcome.stats.sediment_conservation_valid, 1);
    }
}
