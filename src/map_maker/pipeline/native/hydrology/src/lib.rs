use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::slice;

const D4_NEIGHBORS: usize = 4;
const MONTHS: usize = 12;
const SECONDS_PER_YEAR: f64 = 365.2425 * 86_400.0;
const SECONDS_PER_MONTH: f64 = SECONDS_PER_YEAR / MONTHS as f64;
const WATER_DENSITY_KG_M3: f32 = 1_000.0;
const GRAVITY_M_S2: f32 = 9.80665;

const WATER_NONE: u8 = 0;
const WATER_WETLAND: u8 = 1;
const WATER_ENDORHEIC: u8 = 2;
const WATER_STABLE_LAKE: u8 = 3;
const WATER_OVERFLOW_LAKE: u8 = 4;
const WATER_BREACHED: u8 = 5;

const SINK_NONE: u8 = 0;
const SINK_OCEAN: u8 = 1;
const SINK_WETLAND: u8 = 2;
const SINK_ENDORHEIC: u8 = 3;
const SINK_STABLE_LAKE: u8 = 4;
const WATER_DOMINATED_CELL_FRACTION: f32 = 0.50;

fn waterbody_dominates_cell(fraction: f32) -> bool {
    fraction >= WATER_DOMINATED_CELL_FRACTION
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HydrologyConfig {
    pub planet_radius_m: f64,
    pub minimum_depression_depth_m: f32,
    pub wetland_mean_depth_m: f32,
    pub endorheic_aridity_threshold: f32,
    pub maximum_fill_time_years: f32,
    pub lake_seepage_mm_year: f32,
    pub subgrid_relief_scale: f32,
    pub subgrid_connected_basin_fraction: f32,
    pub breach_score_threshold: f32,
    pub maximum_breach_incision_m: f32,
    pub breach_length_cells: i32,
    pub river_discharge_threshold_m3s: f32,
    pub river_contributing_area_threshold_km2: f64,
    pub river_minimum_discharge_m3s: f32,
}

#[repr(C)]
pub struct LakeRecord {
    pub depression_id: i32,
    pub lake_id: i32,
    pub class_code: i32,
    pub sink_cell: i32,
    pub outlet_cell: i32,
    pub outlet_receiver: i32,
    pub cell_count: i32,
    pub water_cell_count: i32,
    pub open_outlet: i32,
    pub area_km2: f64,
    pub water_area_km2: f64,
    pub volume_km3: f64,
    pub surface_elevation_m: f32,
    pub maximum_depth_m: f32,
    pub mean_depth_m: f32,
    pub catchment_area_km2: f64,
    pub mean_inflow_m3s: f32,
    pub annual_inflow_km3: f64,
    pub annual_evaporation_km3: f64,
    pub annual_seepage_km3: f64,
    pub annual_balance_km3: f64,
    pub fill_time_years: f32,
    pub salinity_index: f32,
    pub mean_aridity_index: f32,
    pub breach_incision_m: f32,
}

#[repr(C)]
pub struct LakeRecordArray {
    pub data: *mut LakeRecord,
    pub len: usize,
}

#[repr(C)]
pub struct BreachRecord {
    pub breach_id: i32,
    pub depression_id: i32,
    pub outlet_cell: i32,
    pub downstream_cell: i32,
    pub pre_breach_spill_elevation_m: f32,
    pub post_breach_outlet_elevation_m: f32,
    pub incision_m: f32,
    pub gorge_length_km: f32,
    pub sediment_pulse_km3: f64,
    pub trigger_score: f32,
}

#[repr(C)]
pub struct BreachRecordArray {
    pub data: *mut BreachRecord,
    pub len: usize,
}

#[repr(C)]
pub struct RiverReachRecord {
    pub reach_id: i32,
    pub from_node: i32,
    pub to_node: i32,
    pub downstream_reach_id: i32,
    pub basin_id: i32,
    pub vertex_offset: i32,
    pub vertex_count: i32,
    pub strahler_order: i32,
    pub morphology_code: i32,
    pub bed_material_code: i32,
    pub flow_direction_vector: [f32; 3],
    pub slope: f32,
    pub discharge_mean: f32,
    pub discharge_seasonal: [f32; MONTHS],
    pub velocity_mean: f32,
    pub velocity_seasonal: [f32; MONTHS],
    pub stream_power: f32,
    pub channel_width_m: f32,
    pub channel_depth_m: f32,
    pub valley_width_m: f32,
    pub floodplain_width_m: f32,
    pub meander_index: f32,
    pub braiding_index: f32,
    pub incision_m: f32,
    pub sediment_load_kg_s: f32,
}

#[repr(C)]
pub struct RiverReachRecordArray {
    pub data: *mut RiverReachRecord,
    pub len: usize,
}

#[repr(C)]
pub struct Int32Array {
    pub data: *mut i32,
    pub len: usize,
}

#[repr(C)]
pub struct HydrologyStats {
    pub depression_count: i32,
    pub lake_count: i32,
    pub breach_count: i32,
    pub basin_count: i32,
    pub reach_count: i32,
    pub wetland_count: i32,
    pub endorheic_count: i32,
    pub stable_lake_count: i32,
    pub overflow_lake_count: i32,
    pub land_cell_count: i32,
    pub river_cell_count: i32,
    pub closed_sink_count: i32,
    pub topology_valid: i32,
    pub maximum_contributing_area_km2: f64,
    pub maximum_discharge_m3s: f32,
    pub annual_runoff_km3: f64,
    pub lake_volume_km3: f64,
    pub breach_sediment_pulse_km3: f64,
    pub annual_open_water_loss_km3: f64,
    pub conservation_relative_error: f64,
}

#[derive(Clone, Copy)]
struct FloodEntry {
    level: f32,
    cell: usize,
}

impl PartialEq for FloodEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell && self.level.to_bits() == other.level.to_bits()
    }
}

impl Eq for FloodEntry {}

impl PartialOrd for FloodEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloodEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .level
            .total_cmp(&self.level)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

struct FloodResult {
    filled: Vec<f32>,
    receiver: Vec<i32>,
    order: Vec<usize>,
}

struct RiverProperties {
    river: Vec<bool>,
    velocity: Vec<f32>,
    stream_power: Vec<f32>,
    corridor: Vec<f32>,
    floodplain: Vec<f32>,
}

#[derive(Clone)]
struct Depression {
    id: i32,
    cells: Vec<usize>,
    water_cells: Vec<usize>,
    water_fractions: Vec<f32>,
    potential_fractions: Vec<f32>,
    sink_cell: usize,
    outlet_cell: usize,
    outlet_receiver: usize,
    hydraulic_control_cell: usize,
    hydraulic_control_receiver: usize,
    spill_elevation_m: f32,
    maximum_depth_m: f32,
    mean_depth_m: f32,
    area_km2: f64,
    potential_water_area_km2: f64,
    volume_km3: f64,
    water_area_km2: f64,
    water_volume_km3: f64,
    water_surface_elevation_m: f32,
    catchment_area_km2: f64,
    monthly_inflow_m3s: [f64; MONTHS],
    mean_aridity: f32,
    class_code: u8,
    open_outlet: bool,
    lake_id: i32,
    fill_time_years: f32,
    annual_inflow_km3: f64,
    annual_evaporation_km3: f64,
    annual_seepage_km3: f64,
    annual_balance_km3: f64,
    salinity_index: f32,
    breach_static_score: f32,
    breach_score: f32,
    breach_incision_m: f32,
}

#[no_mangle]
pub extern "C" fn hydrology_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_hydrology_abi_version() -> u32 {
    3
}

#[no_mangle]
pub extern "C" fn regional_hydrology_abi_version() -> u32 {
    1
}

fn priority_flood(
    elevation: &[f32],
    ocean: &[u8],
    neighbors: &[i32],
    neighbor_count: usize,
) -> Option<FloodResult> {
    let total = elevation.len();
    let mut filled = vec![f32::INFINITY; total];
    let mut receiver = vec![-1i32; total];
    let mut visited = vec![false; total];
    let mut queue = BinaryHeap::new();
    for cell in 0..total {
        if ocean[cell] != 0 {
            filled[cell] = elevation[cell];
            visited[cell] = true;
            queue.push(FloodEntry {
                level: elevation[cell],
                cell,
            });
        }
    }
    if queue.is_empty() {
        return None;
    }

    let mut order = Vec::with_capacity(total);
    while let Some(entry) = queue.pop() {
        if entry.level.to_bits() != filled[entry.cell].to_bits() {
            continue;
        }
        order.push(entry.cell);
        for adjacent in &neighbors[entry.cell * neighbor_count..(entry.cell + 1) * neighbor_count] {
            let next = *adjacent as usize;
            if visited[next] {
                continue;
            }
            visited[next] = true;
            filled[next] = elevation[next].max(entry.level);
            receiver[next] = entry.cell as i32;
            queue.push(FloodEntry {
                level: filled[next],
                cell: next,
            });
        }
    }
    if order.len() != total {
        return None;
    }
    Some(FloodResult {
        filled,
        receiver,
        order,
    })
}

fn edge_distance_m(first: usize, second: usize, xyz: &[f32], radius_m: f64) -> f64 {
    let dot = (0..3)
        .map(|axis| xyz[first * 3 + axis] as f64 * xyz[second * 3 + axis] as f64)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    dot.acos().max(1e-9) * radius_m
}

fn subgrid_inundation(
    mean_elevation_m: f32,
    relief_m: f32,
    water_surface_m: f32,
    minimum_depth_m: f32,
    relief_scale: f32,
    connected_fraction: f32,
) -> (f32, f64) {
    let span_m = (relief_m.max(0.0) * relief_scale.max(0.01)).max(2.0 * minimum_depth_m.max(0.1));
    let minimum_elevation_m = mean_elevation_m - 0.5 * span_m;
    let surface_above_minimum_m = water_surface_m - minimum_elevation_m;
    let raw_fraction = (surface_above_minimum_m / span_m).clamp(0.0, 1.0);
    let connected_fraction = connected_fraction.clamp(0.0, 1.0);
    let fraction = raw_fraction * connected_fraction;
    // Priority-flood head establishes coarse connectivity; it is not bathymetry.
    // Storage deeper than the unresolved relief span belongs to regional refinement.
    let storage_surface_above_minimum_m = surface_above_minimum_m.clamp(0.0, span_m);
    let equivalent_depth_m = ((raw_fraction as f64 * storage_surface_above_minimum_m as f64
        - 0.5 * span_m as f64 * raw_fraction as f64 * raw_fraction as f64)
        * connected_fraction as f64)
        .max(0.0);
    (fraction, equivalent_depth_m)
}

fn find_depressions(
    flood: &FloodResult,
    elevation: &[f32],
    relief: &[f32],
    ocean: &[u8],
    neighbors: &[i32],
    neighbor_count: usize,
    area_km2: &[f64],
    config: &HydrologyConfig,
) -> (Vec<i32>, Vec<Depression>) {
    let total = elevation.len();
    let candidate: Vec<bool> = (0..total)
        .map(|cell| {
            ocean[cell] == 0
                && flood.filled[cell] - elevation[cell] >= config.minimum_depression_depth_m
        })
        .collect();
    let mut depression_id = vec![-1i32; total];
    let mut depressions = Vec::new();
    let mut queue = VecDeque::new();

    for start in 0..total {
        if !candidate[start] || depression_id[start] >= 0 {
            continue;
        }
        let id = depressions.len() as i32;
        depression_id[start] = id;
        queue.push_back(start);
        let mut cells = Vec::new();
        while let Some(cell) = queue.pop_front() {
            cells.push(cell);
            for adjacent in &neighbors[cell * neighbor_count..(cell + 1) * neighbor_count] {
                let next = *adjacent as usize;
                if candidate[next] && depression_id[next] < 0 {
                    depression_id[next] = id;
                    queue.push_back(next);
                }
            }
        }

        let sink_cell = *cells
            .iter()
            .min_by(|first, second| {
                elevation[**first]
                    .total_cmp(&elevation[**second])
                    .then_with(|| first.cmp(second))
            })
            .expect("component is non-empty");
        let mut outlet_cell = cells[0];
        let mut outlet_receiver = flood.receiver[outlet_cell].max(0) as usize;
        let mut component_saddle_m = f32::INFINITY;
        for &cell in &cells {
            for adjacent in &neighbors[cell * neighbor_count..(cell + 1) * neighbor_count] {
                let next = *adjacent as usize;
                if depression_id[next] == id {
                    continue;
                }
                let saddle = elevation[cell].max(elevation[next]);
                if saddle < component_saddle_m
                    || (saddle.to_bits() == component_saddle_m.to_bits()
                        && (cell, next) < (outlet_cell, outlet_receiver))
                {
                    component_saddle_m = saddle;
                    outlet_cell = cell;
                    outlet_receiver = next;
                }
            }
        }
        let spill_elevation_m = flood.filled[sink_cell];
        let mut hydraulic_control_cell = outlet_cell;
        let mut hydraulic_control_receiver = outlet_receiver;
        let mut current = sink_cell;
        for _ in 0..total {
            let downstream = flood.receiver[current];
            if downstream < 0 {
                break;
            }
            let downstream = downstream as usize;
            if flood.filled[downstream].total_cmp(&spill_elevation_m) == Ordering::Less {
                hydraulic_control_cell = current;
                hydraulic_control_receiver = downstream;
                break;
            }
            current = downstream;
        }
        let area = cells.iter().map(|cell| area_km2[*cell]).sum::<f64>();
        let connected_fraction = config.subgrid_connected_basin_fraction.clamp(0.0, 1.0);
        let mut potential_fractions = Vec::with_capacity(cells.len());
        let mut potential_water_area_km2 = 0.0f64;
        let mut maximum_depth_m = 0.0f32;
        let mut volume_km3 = 0.0f64;
        for &cell in &cells {
            let (fraction, equivalent_depth_m) = subgrid_inundation(
                elevation[cell],
                relief[cell],
                spill_elevation_m,
                config.minimum_depression_depth_m,
                config.subgrid_relief_scale,
                connected_fraction,
            );
            potential_fractions.push(fraction);
            potential_water_area_km2 += fraction as f64 * area_km2[cell];
            if fraction > 0.0 {
                let span_m = (relief[cell].max(0.0) * config.subgrid_relief_scale.max(0.01))
                    .max(2.0 * config.minimum_depression_depth_m.max(0.1));
                maximum_depth_m = maximum_depth_m
                    .max((spill_elevation_m - (elevation[cell] - 0.5 * span_m)).clamp(0.0, span_m));
                volume_km3 += equivalent_depth_m * area_km2[cell] / 1_000.0;
            }
        }
        let mean_depth_m = if potential_water_area_km2 > 0.0 {
            (volume_km3 * 1_000.0 / potential_water_area_km2) as f32
        } else {
            0.0
        };
        depressions.push(Depression {
            id,
            cells,
            water_cells: Vec::new(),
            water_fractions: Vec::new(),
            potential_fractions,
            sink_cell,
            outlet_cell,
            outlet_receiver,
            hydraulic_control_cell,
            hydraulic_control_receiver,
            spill_elevation_m,
            maximum_depth_m,
            mean_depth_m,
            area_km2: area,
            potential_water_area_km2,
            volume_km3,
            water_area_km2: 0.0,
            water_volume_km3: 0.0,
            water_surface_elevation_m: elevation[sink_cell],
            catchment_area_km2: 0.0,
            monthly_inflow_m3s: [0.0; MONTHS],
            mean_aridity: 0.0,
            class_code: WATER_NONE,
            open_outlet: false,
            lake_id: -1,
            fill_time_years: f32::INFINITY,
            annual_inflow_km3: 0.0,
            annual_evaporation_km3: 0.0,
            annual_seepage_km3: 0.0,
            annual_balance_km3: 0.0,
            salinity_index: 0.0,
            breach_static_score: 0.0,
            breach_score: 0.0,
            breach_incision_m: 0.0,
        });
    }
    (depression_id, depressions)
}

fn depression_catchments(flood: &FloodResult, depression_id: &[i32], ocean: &[u8]) -> Vec<i32> {
    let mut owner = vec![-1i32; depression_id.len()];
    for &cell in &flood.order {
        if ocean[cell] != 0 {
            continue;
        }
        if depression_id[cell] >= 0 {
            owner[cell] = depression_id[cell];
        } else if flood.receiver[cell] >= 0 {
            owner[cell] = owner[flood.receiver[cell] as usize];
        }
    }
    owner
}

#[allow(clippy::too_many_arguments)]
fn classify_depressions(
    depressions: &mut [Depression],
    owner: &[i32],
    area_km2: &[f64],
    local_runoff: &[f64],
    evaporation: &[f32],
    aridity: &[f32],
    rock_strength: &[f32],
    accommodation: &[f32],
    config: &HydrologyConfig,
) {
    let mut catchment_area = vec![0.0f64; depressions.len()];
    let mut catchment_runoff = vec![[0.0f64; MONTHS]; depressions.len()];
    for (cell, owned) in owner.iter().enumerate() {
        if *owned < 0 {
            continue;
        }
        let index = *owned as usize;
        catchment_area[index] += area_km2[cell];
        for month in 0..MONTHS {
            catchment_runoff[index][month] += local_runoff[month * owner.len() + cell];
        }
    }

    for depression in depressions {
        depression.catchment_area_km2 = catchment_area[depression.id as usize];
        depression.monthly_inflow_m3s = catchment_runoff[depression.id as usize];
        depression.annual_inflow_km3 =
            depression.monthly_inflow_m3s.iter().sum::<f64>() * SECONDS_PER_MONTH / 1e9;
        let mut evaporation_volume_m3 = 0.0f64;
        let mut aridity_weighted = 0.0f64;
        let mut rock_weighted = 0.0f64;
        let mut accommodation_weighted = 0.0f64;
        for (&cell, &fraction) in depression.cells.iter().zip(&depression.potential_fractions) {
            let water_area_km2 = fraction as f64 * area_km2[cell];
            let annual_evaporation_mm = (0..MONTHS)
                .map(|month| evaporation[month * owner.len() + cell].max(0.0) as f64)
                .sum::<f64>();
            evaporation_volume_m3 += annual_evaporation_mm * water_area_km2 * 1_000.0;
            aridity_weighted += aridity[cell].max(0.0) as f64 * water_area_km2;
            rock_weighted += rock_strength[cell].clamp(0.0, 1.0) as f64 * water_area_km2;
            accommodation_weighted += accommodation[cell].clamp(0.0, 1.0) as f64 * water_area_km2;
        }
        let inverse_area = 1.0 / depression.potential_water_area_km2.max(1e-12);
        depression.mean_aridity = (aridity_weighted * inverse_area) as f32;
        let mean_rock = (rock_weighted * inverse_area) as f32;
        let mean_accommodation = (accommodation_weighted * inverse_area) as f32;
        depression.annual_evaporation_km3 = evaporation_volume_m3 / 1e9;
        depression.annual_seepage_km3 = config.lake_seepage_mm_year.max(0.0) as f64
            * depression.potential_water_area_km2
            * 1_000.0
            * (1.0 - 0.65 * mean_accommodation as f64)
            / 1e9;
        depression.annual_balance_km3 = depression.annual_inflow_km3
            - depression.annual_evaporation_km3
            - depression.annual_seepage_km3;
        if depression.annual_balance_km3 > 1e-12 {
            depression.fill_time_years =
                (depression.volume_km3 / depression.annual_balance_km3) as f32;
        }
        depression.salinity_index = (depression.annual_evaporation_km3
            / depression.annual_inflow_km3.max(1e-9))
        .clamp(0.0, 10.0) as f32;

        let mean_inflow =
            (depression.monthly_inflow_m3s.iter().sum::<f64>() / MONTHS as f64) as f32;
        let depth_score = (depression.maximum_depth_m / 600.0).clamp(0.0, 1.0);
        let discharge_score = ((1.0 + mean_inflow.max(0.0)).ln() / 8.0).clamp(0.0, 1.0);
        depression.breach_static_score =
            0.42 * depth_score + 0.20 * (1.0 - mean_rock) + 0.10 * (1.0 - mean_accommodation);
        depression.breach_score = depression.breach_static_score + 0.28 * discharge_score;

        let overflows = depression.annual_balance_km3 > 0.0
            && depression.fill_time_years <= config.maximum_fill_time_years;
        depression.class_code = if depression.mean_depth_m <= config.wetland_mean_depth_m {
            WATER_WETLAND
        } else if depression.mean_aridity <= config.endorheic_aridity_threshold
            && depression.annual_balance_km3 <= 0.0
        {
            WATER_ENDORHEIC
        } else if overflows && depression.breach_score >= config.breach_score_threshold {
            WATER_BREACHED
        } else if overflows {
            WATER_OVERFLOW_LAKE
        } else if depression.annual_balance_km3 <= 0.0
            && depression.mean_aridity <= config.endorheic_aridity_threshold * 1.25
        {
            WATER_ENDORHEIC
        } else {
            WATER_STABLE_LAKE
        };
        depression.open_outlet = overflows && depression.class_code != WATER_BREACHED;
    }
}

fn refresh_depression_classification(depression: &mut Depression, config: &HydrologyConfig) {
    depression.annual_inflow_km3 =
        depression.monthly_inflow_m3s.iter().sum::<f64>() * SECONDS_PER_MONTH / 1e9;
    depression.annual_balance_km3 = depression.annual_inflow_km3
        - depression.annual_evaporation_km3
        - depression.annual_seepage_km3;
    depression.fill_time_years = if depression.annual_balance_km3 > 1e-12 {
        (depression.volume_km3 / depression.annual_balance_km3) as f32
    } else {
        f32::INFINITY
    };
    depression.salinity_index = (depression.annual_evaporation_km3
        / depression.annual_inflow_km3.max(1e-9))
    .clamp(0.0, 10.0) as f32;
    let mean_inflow = (depression.monthly_inflow_m3s.iter().sum::<f64>() / MONTHS as f64) as f32;
    let discharge_score = ((1.0 + mean_inflow.max(0.0)).ln() / 8.0).clamp(0.0, 1.0);
    depression.breach_score = depression.breach_static_score + 0.28 * discharge_score;
    let overflows = depression.annual_balance_km3 > 0.0
        && depression.fill_time_years <= config.maximum_fill_time_years;
    depression.class_code = if depression.mean_depth_m <= config.wetland_mean_depth_m {
        WATER_WETLAND
    } else if depression.mean_aridity <= config.endorheic_aridity_threshold
        && depression.annual_balance_km3 <= 0.0
    {
        WATER_ENDORHEIC
    } else if overflows && depression.breach_score >= config.breach_score_threshold {
        WATER_BREACHED
    } else if overflows {
        WATER_OVERFLOW_LAKE
    } else if depression.annual_balance_km3 <= 0.0
        && depression.mean_aridity <= config.endorheic_aridity_threshold * 1.25
    {
        WATER_ENDORHEIC
    } else {
        WATER_STABLE_LAKE
    };
    depression.open_outlet = overflows && depression.class_code != WATER_BREACHED;
}

fn propagate_depression_overflow(
    depressions: &mut [Depression],
    flood: &FloodResult,
    depression_id: &[i32],
    config: &HydrologyConfig,
) {
    let mut flood_rank = vec![usize::MAX; depression_id.len()];
    for (rank, cell) in flood.order.iter().enumerate() {
        flood_rank[*cell] = rank;
    }
    let mut downstream = vec![-1i32; depressions.len()];
    for depression in depressions.iter() {
        let mut cell = depression.hydraulic_control_receiver as i32;
        let mut steps = 0usize;
        while cell >= 0 && steps < depression_id.len() {
            let candidate = depression_id[cell as usize];
            if candidate >= 0 && candidate != depression.id {
                downstream[depression.id as usize] = candidate;
                break;
            }
            cell = flood.receiver[cell as usize];
            steps += 1;
        }
    }
    let mut order: Vec<usize> = (0..depressions.len()).collect();
    order.sort_by_key(|index| std::cmp::Reverse(flood_rank[depressions[*index].sink_cell]));
    for index in order {
        refresh_depression_classification(&mut depressions[index], config);
        let routes_downstream =
            depressions[index].open_outlet || depressions[index].class_code == WATER_BREACHED;
        let target = downstream[index];
        if !routes_downstream || target < 0 {
            continue;
        }
        let loss_rate_m3s = (depressions[index].annual_evaporation_km3
            + depressions[index].annual_seepage_km3)
            * 1e9
            / SECONDS_PER_YEAR;
        let catchment_area = depressions[index].catchment_area_km2;
        let outflow: [f64; MONTHS] = std::array::from_fn(|month| {
            (depressions[index].monthly_inflow_m3s[month] - loss_rate_m3s).max(0.0)
        });
        let target = target as usize;
        depressions[target].catchment_area_km2 += catchment_area;
        for (inflow, routed) in depressions[target]
            .monthly_inflow_m3s
            .iter_mut()
            .zip(outflow)
        {
            *inflow += routed;
        }
    }
}

fn assign_water_extents(
    depressions: &mut [Depression],
    elevation: &[f32],
    relief: &[f32],
    area_km2: &[f64],
    config: &HydrologyConfig,
) {
    for depression in depressions {
        depression.water_cells.clear();
        depression.water_fractions.clear();
        if depression.class_code == WATER_BREACHED {
            depression.water_area_km2 = 0.0;
            depression.water_volume_km3 = 0.0;
            depression.water_surface_elevation_m = depression.spill_elevation_m;
            depression.maximum_depth_m = 0.0;
            depression.mean_depth_m = 0.0;
            depression.annual_evaporation_km3 = 0.0;
            depression.annual_seepage_km3 = 0.0;
            depression.annual_balance_km3 = depression.annual_inflow_km3;
            continue;
        }

        let full_loss = depression.annual_evaporation_km3 + depression.annual_seepage_km3;
        let equilibrium_fraction = if depression.open_outlet {
            1.0
        } else if full_loss > 1e-12 {
            (depression.annual_inflow_km3 / full_loss).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let spill_limited_area_km2 = depression
            .cells
            .iter()
            .map(|cell| {
                let (fraction, _) = subgrid_inundation(
                    elevation[*cell],
                    relief[*cell],
                    depression.spill_elevation_m,
                    config.minimum_depression_depth_m,
                    config.subgrid_relief_scale,
                    config.subgrid_connected_basin_fraction,
                );
                fraction as f64 * area_km2[*cell]
            })
            .sum::<f64>();
        let target_area = if depression.open_outlet {
            spill_limited_area_km2
        } else {
            depression.potential_water_area_km2 * equilibrium_fraction
        };
        let mut lower_surface = depression
            .cells
            .iter()
            .map(|cell| {
                let span = (relief[*cell].max(0.0) * config.subgrid_relief_scale.max(0.01))
                    .max(2.0 * config.minimum_depression_depth_m.max(0.1));
                elevation[*cell] - 0.5 * span
            })
            .fold(depression.spill_elevation_m, f32::min);
        let mut upper_surface = depression.spill_elevation_m;
        if !depression.open_outlet {
            for _ in 0..36 {
                let trial_surface = 0.5 * (lower_surface + upper_surface);
                let trial_area = depression
                    .cells
                    .iter()
                    .map(|cell| {
                        let (fraction, _) = subgrid_inundation(
                            elevation[*cell],
                            relief[*cell],
                            trial_surface,
                            config.minimum_depression_depth_m,
                            config.subgrid_relief_scale,
                            config.subgrid_connected_basin_fraction,
                        );
                        fraction as f64 * area_km2[*cell]
                    })
                    .sum::<f64>();
                if trial_area < target_area {
                    lower_surface = trial_surface;
                } else {
                    upper_surface = trial_surface;
                }
            }
        }
        let water_surface = if depression.open_outlet {
            depression.spill_elevation_m
        } else {
            0.5 * (lower_surface + upper_surface)
        };
        let mut water_area_km2 = 0.0f64;
        let mut water_volume_km3 = 0.0f64;
        let mut maximum_depth_m = 0.0f32;
        for &cell in &depression.cells {
            let (fraction, equivalent_depth_m) = subgrid_inundation(
                elevation[cell],
                relief[cell],
                water_surface,
                config.minimum_depression_depth_m,
                config.subgrid_relief_scale,
                config.subgrid_connected_basin_fraction,
            );
            if fraction <= 1e-6 {
                continue;
            }
            depression.water_cells.push(cell);
            depression.water_fractions.push(fraction);
            water_area_km2 += fraction as f64 * area_km2[cell];
            water_volume_km3 += equivalent_depth_m * area_km2[cell] / 1_000.0;
            let span_m = (relief[cell].max(0.0) * config.subgrid_relief_scale.max(0.01))
                .max(2.0 * config.minimum_depression_depth_m.max(0.1));
            maximum_depth_m = maximum_depth_m
                .max((water_surface - (elevation[cell] - 0.5 * span_m)).clamp(0.0, span_m));
        }
        depression.water_surface_elevation_m = water_surface;
        depression.water_area_km2 = water_area_km2;
        depression.water_volume_km3 = water_volume_km3;
        depression.maximum_depth_m = maximum_depth_m;
        depression.mean_depth_m = if water_area_km2 > 0.0 {
            (water_volume_km3 * 1_000.0 / water_area_km2) as f32
        } else {
            0.0
        };
        let water_fraction =
            depression.water_area_km2 / depression.potential_water_area_km2.max(1e-12);
        depression.annual_evaporation_km3 *= water_fraction;
        depression.annual_seepage_km3 *= water_fraction;
        depression.annual_balance_km3 = depression.annual_inflow_km3
            - depression.annual_evaporation_km3
            - depression.annual_seepage_km3;
        debug_assert!(
            (depression.water_area_km2 - target_area).abs()
                <= depression.potential_water_area_km2.max(1.0) * 1e-5
        );
    }
}

fn finalize_waterbody_classes(depressions: &mut [Depression], config: &HydrologyConfig) {
    for depression in depressions.iter_mut() {
        if depression.class_code != WATER_BREACHED
            && depression.mean_depth_m <= config.wetland_mean_depth_m
        {
            depression.class_code = WATER_WETLAND;
        }
        depression.lake_id = -1;
    }
    let mut next_lake_id = 0i32;
    for depression in depressions {
        if depression.class_code != WATER_BREACHED && depression.class_code != WATER_WETLAND {
            depression.lake_id = next_lake_id;
            next_lake_id += 1;
        }
    }
}

fn apply_breaches(
    depressions: &mut [Depression],
    flood: &FloodResult,
    elevation: &[f32],
    relief: &[f32],
    area_km2: &[f64],
    config: &HydrologyConfig,
) -> (Vec<f32>, Vec<f32>, Vec<BreachRecord>) {
    let mut hydrologic_elevation = elevation.to_vec();
    let mut incision = vec![0.0f32; elevation.len()];
    let mut records = Vec::new();
    let length = config.breach_length_cells.max(1) as usize;
    for depression in depressions {
        if depression.class_code != WATER_BREACHED {
            continue;
        }
        let target_incision = (depression.maximum_depth_m
            * (0.30 + 0.55 * depression.breach_score))
            .clamp(1.0, config.maximum_breach_incision_m);
        depression.breach_incision_m = target_incision;
        let mut current = depression.hydraulic_control_cell;
        let mut gorge_length_m = 0.0f64;
        let mut sediment_km3 = 0.0f64;
        let mean_inflow_m3s = depression.monthly_inflow_m3s.iter().sum::<f64>() / MONTHS as f64;
        let gorge_width_km = (0.10 + 0.012 * mean_inflow_m3s.sqrt()).clamp(0.10, 2.0);
        for step in 0..length {
            let fraction = 1.0 - 0.72 * step as f32 / length.max(1) as f32;
            let local_incision = target_incision
                * fraction
                * (0.75 + 0.25 * (relief[current] / 1_000.0).clamp(0.0, 1.0));
            incision[current] = incision[current].max(local_incision);
            hydrologic_elevation[current] = elevation[current] - incision[current];
            let segment_length_km = area_km2[current].sqrt();
            sediment_km3 += local_incision as f64 / 1_000.0 * gorge_width_km * segment_length_km;
            let next = flood.receiver[current];
            if next < 0 || next as usize == current {
                break;
            }
            gorge_length_m += area_km2[current].sqrt() * 1_000.0;
            current = next as usize;
        }
        records.push(BreachRecord {
            breach_id: records.len() as i32,
            depression_id: depression.id,
            outlet_cell: depression.hydraulic_control_cell as i32,
            downstream_cell: depression.hydraulic_control_receiver as i32,
            pre_breach_spill_elevation_m: depression.spill_elevation_m,
            post_breach_outlet_elevation_m: hydrologic_elevation[depression.hydraulic_control_cell],
            incision_m: target_incision,
            gorge_length_km: (gorge_length_m / 1_000.0) as f32,
            sediment_pulse_km3: sediment_km3,
            trigger_score: depression.breach_score,
        });
    }
    (hydrologic_elevation, incision, records)
}

fn retain_post_breach_lakes(
    depressions: &mut [Depression],
    flood: &FloodResult,
    elevation: &[f32],
    relief: &[f32],
    area_km2: &[f64],
    config: &HydrologyConfig,
) {
    for depression in depressions {
        if depression.class_code != WATER_BREACHED {
            continue;
        }

        let control_surface = flood.filled[depression.sink_cell].min(depression.spill_elevation_m);
        depression.spill_elevation_m = control_surface;
        depression.water_surface_elevation_m = control_surface;
        let residual_area_km2 = depression
            .cells
            .iter()
            .map(|cell| {
                let (fraction, _) = subgrid_inundation(
                    elevation[*cell],
                    relief[*cell],
                    control_surface,
                    config.minimum_depression_depth_m,
                    config.subgrid_relief_scale,
                    config.subgrid_connected_basin_fraction,
                );
                fraction as f64 * area_km2[*cell]
            })
            .sum::<f64>();
        let retained_area_tolerance_km2 = depression.potential_water_area_km2.max(1.0) * 1e-9;
        let residual_head_m = control_surface - elevation[depression.sink_cell];
        if residual_head_m >= config.minimum_depression_depth_m
            && residual_area_km2 > retained_area_tolerance_km2
        {
            depression.class_code = WATER_OVERFLOW_LAKE;
            depression.open_outlet = true;
        }
    }
}

fn route_preserved_water(
    depressions: &mut [Depression],
    depression_id: &[i32],
    neighbors: &[i32],
    neighbor_count: usize,
    flood_order: &[usize],
    receiver: &mut [i32],
) {
    let mut queue = VecDeque::new();
    let mut visited = vec![false; receiver.len()];
    let mut flood_rank = vec![usize::MAX; receiver.len()];
    for (rank, cell) in flood_order.iter().enumerate() {
        flood_rank[*cell] = rank;
    }
    for depression in depressions {
        if depression.class_code == WATER_BREACHED {
            continue;
        }
        let (root, external_receiver) = if depression.open_outlet {
            let outlet = depression
                .cells
                .iter()
                .filter_map(|cell| {
                    let downstream = receiver[*cell];
                    (downstream >= 0 && depression_id[downstream as usize] != depression.id)
                        .then_some((*cell, downstream as usize))
                })
                .min_by_key(|(cell, downstream)| (flood_rank[*cell], *cell, *downstream))
                .unwrap_or((depression.outlet_cell, depression.outlet_receiver));
            depression.outlet_cell = outlet.0;
            depression.outlet_receiver = outlet.1;
            (outlet.0, Some(outlet.1))
        } else {
            (depression.sink_cell, None)
        };
        for &cell in &depression.cells {
            visited[cell] = false;
        }
        visited[root] = true;
        queue.push_back(root);
        while let Some(cell) = queue.pop_front() {
            for adjacent in &neighbors[cell * neighbor_count..(cell + 1) * neighbor_count] {
                let next = *adjacent as usize;
                if depression_id[next] == depression.id && !visited[next] {
                    visited[next] = true;
                    receiver[next] = cell as i32;
                    queue.push_back(next);
                }
            }
        }
        if depression.open_outlet {
            receiver[root] = external_receiver.map_or(-1, |cell| cell as i32);
        } else {
            receiver[root] = -1;
        }
    }
}

fn topological_order(receiver: &[i32], ocean: &[u8]) -> Option<Vec<usize>> {
    let mut indegree = vec![0usize; receiver.len()];
    let land_count = ocean.iter().filter(|value| **value == 0).count();
    for cell in 0..receiver.len() {
        if ocean[cell] != 0 || receiver[cell] < 0 {
            continue;
        }
        let downstream = receiver[cell] as usize;
        if ocean[downstream] == 0 {
            indegree[downstream] += 1;
        }
    }
    let mut queue = BinaryHeap::new();
    for cell in 0..receiver.len() {
        if ocean[cell] == 0 && indegree[cell] == 0 {
            queue.push(std::cmp::Reverse(cell));
        }
    }
    let mut order = Vec::with_capacity(land_count);
    while let Some(std::cmp::Reverse(cell)) = queue.pop() {
        order.push(cell);
        let downstream = receiver[cell];
        if downstream >= 0 && ocean[downstream as usize] == 0 {
            let next = downstream as usize;
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push(std::cmp::Reverse(next));
            }
        }
    }
    (order.len() == land_count).then_some(order)
}

fn assign_basins(
    order: &[usize],
    receiver: &[i32],
    ocean: &[u8],
    water_class: &[u8],
) -> (Vec<i32>, Vec<u8>, i32) {
    let mut basin_id = vec![-1i32; receiver.len()];
    let mut sink_type = vec![SINK_NONE; receiver.len()];
    let mut basin_count = 0i32;
    for &cell in order.iter().rev() {
        let downstream = receiver[cell];
        if downstream < 0 {
            basin_id[cell] = basin_count;
            sink_type[cell] = match water_class[cell] {
                WATER_WETLAND => SINK_WETLAND,
                WATER_ENDORHEIC => SINK_ENDORHEIC,
                WATER_STABLE_LAKE => SINK_STABLE_LAKE,
                _ => SINK_ENDORHEIC,
            };
            basin_count += 1;
        } else if ocean[downstream as usize] != 0 {
            basin_id[cell] = basin_count;
            sink_type[cell] = SINK_OCEAN;
            basin_count += 1;
        } else {
            basin_id[cell] = basin_id[downstream as usize];
            sink_type[cell] = sink_type[downstream as usize];
        }
    }
    (basin_id, sink_type, basin_count)
}

fn flow_geometry(
    receiver: &[i32],
    ocean: &[u8],
    elevation: &[f32],
    xyz: &[f32],
    radius_m: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut direction = vec![0.0f32; receiver.len() * 3];
    let mut slope = vec![0.0f32; receiver.len()];
    for cell in 0..receiver.len() {
        if ocean[cell] != 0 || receiver[cell] < 0 {
            continue;
        }
        let downstream = receiver[cell] as usize;
        let distance = edge_distance_m(cell, downstream, xyz, radius_m);
        slope[cell] = ((elevation[cell] - elevation[downstream]).max(0.0) as f64 / distance)
            .clamp(0.0, 1.0) as f32;
        let dot = (0..3)
            .map(|axis| xyz[cell * 3 + axis] * xyz[downstream * 3 + axis])
            .sum::<f32>();
        let mut norm = 0.0f32;
        for axis in 0..3 {
            let value = xyz[downstream * 3 + axis] - dot * xyz[cell * 3 + axis];
            direction[cell * 3 + axis] = value;
            norm += value * value;
        }
        norm = norm.sqrt().max(1e-12);
        for axis in 0..3 {
            direction[cell * 3 + axis] /= norm;
        }
    }
    (direction, slope)
}

#[allow(clippy::too_many_arguments)]
fn river_properties(
    receiver: &[i32],
    ocean: &[u8],
    preserved_waterbody_support: &[bool],
    area_accumulation: &[f64],
    discharge: &[f64],
    slope: &[f32],
    relief: &[f32],
    config: &HydrologyConfig,
) -> RiverProperties {
    let total = receiver.len();
    let mut river = vec![false; total];
    let mut velocity = vec![0.0f32; total];
    let mut stream_power = vec![0.0f32; total];
    let mut corridor = vec![0.0f32; total];
    let mut floodplain = vec![0.0f32; total];
    let log_reference = (config.river_discharge_threshold_m3s.max(1.0) * 100.0).ln_1p();
    for cell in 0..total {
        if ocean[cell] != 0 || preserved_waterbody_support[cell] || receiver[cell] < 0 {
            continue;
        }
        let mean_q = (0..MONTHS)
            .map(|month| discharge[month * total + cell])
            .sum::<f64>()
            / MONTHS as f64;
        river[cell] = mean_q >= config.river_minimum_discharge_m3s as f64
            && (mean_q >= config.river_discharge_threshold_m3s as f64
                || area_accumulation[cell] >= config.river_contributing_area_threshold_km2);
        if !river[cell] {
            continue;
        }
        let q = mean_q.max(0.01) as f32;
        velocity[cell] = (0.22 + 2.8 * slope[cell].sqrt() * q.powf(0.14)).clamp(0.12, 8.0);
        stream_power[cell] = WATER_DENSITY_KG_M3 * GRAVITY_M_S2 * q * slope[cell];
        corridor[cell] = (q.ln_1p() / log_reference).clamp(0.0, 1.0);
        let plainness = (1.0 - relief[cell] / 900.0).clamp(0.0, 1.0);
        let low_gradient = (1.0 - slope[cell] / 0.004).clamp(0.0, 1.0);
        floodplain[cell] =
            (plainness * low_gradient * (0.25 + 0.75 * corridor[cell])).clamp(0.0, 1.0);
    }
    RiverProperties {
        river,
        velocity,
        stream_power,
        corridor,
        floodplain,
    }
}

fn strahler_orders(order: &[usize], receiver: &[i32], river: &[bool]) -> Vec<i32> {
    let mut result = vec![0i32; river.len()];
    let mut maximum_child = vec![0i32; river.len()];
    let mut maximum_count = vec![0i32; river.len()];
    for &cell in order {
        if !river[cell] {
            continue;
        }
        result[cell] = if maximum_child[cell] == 0 {
            1
        } else if maximum_count[cell] >= 2 {
            maximum_child[cell] + 1
        } else {
            maximum_child[cell]
        };
        let downstream = receiver[cell];
        if downstream >= 0 && river[downstream as usize] {
            let next = downstream as usize;
            if result[cell] > maximum_child[next] {
                maximum_child[next] = result[cell];
                maximum_count[next] = 1;
            } else if result[cell] == maximum_child[next] {
                maximum_count[next] += 1;
            }
        }
    }
    result
}

fn extend_reach_support(
    receiver: &[i32],
    ocean: &[u8],
    river: &[bool],
    preserved_waterbody_support: &[bool],
) -> Vec<bool> {
    let mut support = river.to_vec();
    for cell in 0..river.len() {
        if !river[cell] || receiver[cell] < 0 {
            continue;
        }
        let mut downstream = receiver[cell] as usize;
        while ocean[downstream] == 0 && !river[downstream] {
            if !preserved_waterbody_support[downstream] {
                break;
            }
            if support[downstream] {
                break;
            }
            support[downstream] = true;
            if receiver[downstream] < 0 {
                break;
            }
            downstream = receiver[downstream] as usize;
        }
    }
    support
}

#[allow(clippy::too_many_arguments)]
fn extract_reaches(
    order: &[usize],
    receiver: &[i32],
    ocean: &[u8],
    basin_id: &[i32],
    sink_type: &[u8],
    river: &[bool],
    reach_support: &[bool],
    discharge: &[f64],
    direction: &[f32],
    slope: &[f32],
    velocity: &[f32],
    stream_power: &[f32],
    floodplain: &[f32],
    relief: &[f32],
    hydrologic_elevation: &[f32],
) -> (Vec<RiverReachRecord>, Vec<i32>) {
    let total = receiver.len();
    let mut upstream_count = vec![0usize; total];
    let mut transition_entry = vec![false; total];
    for cell in 0..total {
        if reach_support[cell] && receiver[cell] >= 0 && reach_support[receiver[cell] as usize] {
            let downstream = receiver[cell] as usize;
            upstream_count[downstream] += 1;
            transition_entry[downstream] |= river[cell] != river[downstream];
        }
    }
    let starts: Vec<usize> = order
        .iter()
        .copied()
        .filter(|cell| {
            reach_support[*cell] && (upstream_count[*cell] != 1 || transition_entry[*cell])
        })
        .collect();
    let mut reach_by_start = vec![-1i32; total];
    for (reach_id, start) in starts.iter().enumerate() {
        reach_by_start[*start] = reach_id as i32;
    }
    let cell_order = strahler_orders(order, receiver, reach_support);
    let mut records = Vec::with_capacity(starts.len());
    let mut vertices = Vec::new();
    for (reach_index, start) in starts.iter().enumerate() {
        let vertex_offset = vertices.len();
        let mut current = *start;
        let mut cells = Vec::new();
        loop {
            vertices.push(current as i32);
            cells.push(current);
            let downstream = receiver[current];
            if downstream < 0 {
                break;
            }
            let next = downstream as usize;
            if ocean[next] != 0 {
                vertices.push(next as i32);
                break;
            }
            if !reach_support[next] {
                vertices.push(next as i32);
                break;
            }
            if river[current] != river[next] {
                vertices.push(next as i32);
                break;
            }
            if upstream_count[next] != 1 {
                vertices.push(next as i32);
                break;
            }
            current = next;
        }
        let to_node = *vertices.last().expect("reach has a vertex") as usize;
        let downstream_reach_id = if to_node < total {
            let value = reach_by_start[to_node];
            if value == reach_index as i32 {
                -1
            } else {
                value
            }
        } else {
            -1
        };
        let is_connector = !river[*start];
        let mut monthly = [0.0f32; MONTHS];
        let mut monthly_velocity = [0.0f32; MONTHS];
        for month in 0..MONTHS {
            monthly[month] = discharge[month * total + *start].max(0.0) as f32;
            let q_ratio = monthly[month]
                / ((0..MONTHS)
                    .map(|index| discharge[index * total + *start])
                    .sum::<f64>()
                    / MONTHS as f64)
                    .max(0.01) as f32;
            monthly_velocity[month] = if is_connector {
                0.0
            } else {
                (velocity[*start] * q_ratio.powf(0.16)).clamp(0.08, 10.0)
            };
        }
        let discharge_mean = monthly.iter().sum::<f32>() / MONTHS as f32;
        let slope_mean =
            cells.iter().map(|cell| slope[*cell]).sum::<f32>() / cells.len().max(1) as f32;
        let relief_mean =
            cells.iter().map(|cell| relief[*cell]).sum::<f32>() / cells.len().max(1) as f32;
        let floodplain_mean =
            cells.iter().map(|cell| floodplain[*cell]).sum::<f32>() / cells.len().max(1) as f32;
        let velocity_mean =
            cells.iter().map(|cell| velocity[*cell]).sum::<f32>() / cells.len().max(1) as f32;
        let power_mean =
            cells.iter().map(|cell| stream_power[*cell]).sum::<f32>() / cells.len().max(1) as f32;
        let mut flow_direction_vector = [0.0f32; 3];
        for cell in &cells {
            for axis in 0..3 {
                flow_direction_vector[axis] += direction[*cell * 3 + axis];
            }
        }
        let norm = flow_direction_vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for value in &mut flow_direction_vector {
            *value /= norm;
        }
        let maximum_q = monthly.iter().copied().fold(0.0f32, f32::max);
        let minimum_q = monthly.iter().copied().fold(f32::INFINITY, f32::min);
        let seasonality = ((maximum_q - minimum_q) / discharge_mean.max(0.01)).clamp(0.0, 5.0);
        let channel_width_m = (4.2 * discharge_mean.max(0.01).powf(0.45)).max(1.5);
        let channel_depth_m = (0.38 * discharge_mean.max(0.01).powf(0.28)).max(0.25);
        let floodplain_width_m = channel_width_m * (1.0 + 45.0 * floodplain_mean);
        let valley_width_m =
            floodplain_width_m + channel_width_m * (1.0 + (relief_mean / 500.0).clamp(0.0, 5.0));
        let meander_index =
            1.0 + 1.2 * floodplain_mean * (1.0 - (slope_mean / 0.003).clamp(0.0, 1.0));
        let braiding_index = (seasonality / 2.5
            * (0.35 + 0.65 * (relief_mean / 1_200.0).clamp(0.0, 1.0)))
        .clamp(0.0, 1.0);
        let downstream_is_ocean = receiver[*cells.last().expect("reach cell")] >= 0
            && ocean[receiver[*cells.last().expect("reach cell")] as usize] != 0;
        let morphology_code = if is_connector {
            7
        } else if sink_type[*start] == SINK_ENDORHEIC {
            6
        } else if downstream_is_ocean && floodplain_mean > 0.45 {
            5
        } else if slope_mean > 0.015 {
            1
        } else if slope_mean > 0.003 {
            2
        } else if braiding_index > 0.55 {
            4
        } else {
            3
        };
        let bed_material_code = if is_connector {
            0
        } else if slope_mean > 0.012 {
            1
        } else if slope_mean > 0.003 {
            2
        } else if velocity_mean > 0.8 {
            3
        } else {
            4
        };
        let incision_m = if is_connector {
            0.0
        } else {
            (hydrologic_elevation[*start] - hydrologic_elevation[to_node])
                .max(0.0)
                .min(relief_mean.max(0.0))
        };
        let sediment_load_kg_s = discharge_mean
            * (0.002 + 0.035 * (relief_mean / 1_000.0).clamp(0.0, 3.0))
            * (1.0 + 10.0 * slope_mean).clamp(1.0, 3.0);
        records.push(RiverReachRecord {
            reach_id: reach_index as i32,
            from_node: *start as i32,
            to_node: to_node as i32,
            downstream_reach_id,
            basin_id: basin_id[*start],
            vertex_offset: vertex_offset as i32,
            vertex_count: (vertices.len() - vertex_offset) as i32,
            strahler_order: cell_order[*cells.last().expect("reach cell")],
            morphology_code,
            bed_material_code,
            flow_direction_vector,
            slope: slope_mean,
            discharge_mean,
            discharge_seasonal: monthly,
            velocity_mean: if is_connector { 0.0 } else { velocity_mean },
            velocity_seasonal: monthly_velocity,
            stream_power: if is_connector { 0.0 } else { power_mean },
            channel_width_m: if is_connector { 0.0 } else { channel_width_m },
            channel_depth_m: if is_connector { 0.0 } else { channel_depth_m },
            valley_width_m: if is_connector { 0.0 } else { valley_width_m },
            floodplain_width_m: if is_connector {
                0.0
            } else {
                floodplain_width_m
            },
            meander_index: if is_connector { 1.0 } else { meander_index },
            braiding_index: if is_connector { 0.0 } else { braiding_index },
            incision_m,
            sediment_load_kg_s,
        });
    }
    (records, vertices)
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

#[allow(clippy::too_many_arguments)]
fn run_hydrology(
    total: usize,
    neighbor_count: usize,
    config: &HydrologyConfig,
    area_steradians: &[f64],
    neighbors: &[i32],
    xyz: &[f32],
    elevation: &[f32],
    relief: &[f32],
    rock_strength: &[f32],
    accommodation: &[f32],
    ocean: &[u8],
    runoff: &[f32],
    evaporation: &[f32],
    aridity: &[f32],
    depression_id_out: &mut [i32],
    lake_id_out: &mut [i32],
    water_class_out: &mut [u8],
    lake_fraction_out: &mut [f32],
    wetland_fraction_out: &mut [f32],
    fill_depth_out: &mut [f32],
    hydrologic_elevation_out: &mut [f32],
    breach_incision_out: &mut [f32],
    receiver_out: &mut [i32],
    flow_direction_out: &mut [f32],
    flow_slope_out: &mut [f32],
    contributing_area_out: &mut [f64],
    monthly_discharge_out: &mut [f32],
    mean_discharge_out: &mut [f32],
    velocity_out: &mut [f32],
    stream_power_out: &mut [f32],
    basin_id_out: &mut [i32],
    sink_type_out: &mut [u8],
    river_corridor_out: &mut [f32],
    floodplain_out: &mut [f32],
    lake_records_out: &mut LakeRecordArray,
    breach_records_out: &mut BreachRecordArray,
    reach_records_out: &mut RiverReachRecordArray,
    reach_vertices_out: &mut Int32Array,
    stats_out: &mut HydrologyStats,
) -> i32 {
    if total == 0
        || !(D4_NEIGHBORS..=8).contains(&neighbor_count)
        || neighbors.len() != total * neighbor_count
        || config.planet_radius_m <= 0.0
        || !config.planet_radius_m.is_finite()
        || config.subgrid_relief_scale <= 0.0
        || !config.subgrid_relief_scale.is_finite()
        || !(0.0 < config.subgrid_connected_basin_fraction
            && config.subgrid_connected_basin_fraction <= 1.0)
        || !config.subgrid_connected_basin_fraction.is_finite()
        || area_steradians
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || neighbors
            .iter()
            .any(|value| *value < 0 || *value as usize >= total)
        || xyz.iter().any(|value| !value.is_finite())
        || elevation.iter().any(|value| !value.is_finite())
        || relief.iter().any(|value| !value.is_finite())
        || rock_strength.iter().any(|value| !value.is_finite())
        || accommodation.iter().any(|value| !value.is_finite())
        || runoff
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || evaporation
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || aridity.iter().any(|value| !value.is_finite())
    {
        return 2;
    }
    let land_count = ocean.iter().filter(|value| **value == 0).count();
    if land_count == 0 || land_count == total {
        return 3;
    }
    let radius_km = config.planet_radius_m / 1_000.0;
    let area_km2: Vec<f64> = area_steradians
        .iter()
        .map(|area| *area * radius_km * radius_km)
        .collect();
    let mut local_runoff = vec![0.0f64; MONTHS * total];
    for month in 0..MONTHS {
        for cell in 0..total {
            if ocean[cell] == 0 {
                local_runoff[month * total + cell] =
                    runoff[month * total + cell] as f64 * area_km2[cell] * 1_000.0
                        / SECONDS_PER_MONTH;
            }
        }
    }

    let Some(initial_flood) = priority_flood(elevation, ocean, neighbors, neighbor_count) else {
        return 4;
    };
    let (depression_id, mut depressions) = find_depressions(
        &initial_flood,
        elevation,
        relief,
        ocean,
        neighbors,
        neighbor_count,
        &area_km2,
        config,
    );
    let owner = depression_catchments(&initial_flood, &depression_id, ocean);
    classify_depressions(
        &mut depressions,
        &owner,
        &area_km2,
        &local_runoff,
        evaporation,
        aridity,
        rock_strength,
        accommodation,
        config,
    );
    propagate_depression_overflow(&mut depressions, &initial_flood, &depression_id, config);
    let (hydrologic_elevation, breach_incision, breach_records) = apply_breaches(
        &mut depressions,
        &initial_flood,
        elevation,
        relief,
        &area_km2,
        config,
    );
    let Some(final_flood) = priority_flood(&hydrologic_elevation, ocean, neighbors, neighbor_count)
    else {
        return 4;
    };
    retain_post_breach_lakes(
        &mut depressions,
        &final_flood,
        elevation,
        relief,
        &area_km2,
        config,
    );
    assign_water_extents(&mut depressions, elevation, relief, &area_km2, config);
    finalize_waterbody_classes(&mut depressions, config);
    let mut receiver = final_flood.receiver.clone();
    route_preserved_water(
        &mut depressions,
        &depression_id,
        neighbors,
        neighbor_count,
        &final_flood.order,
        &mut receiver,
    );
    let Some(order) = topological_order(&receiver, ocean) else {
        return 5;
    };

    let mut water_class = vec![WATER_NONE; total];
    let mut lake_id = vec![-1i32; total];
    let mut lake_fraction = vec![0.0f32; total];
    let mut wetland_fraction = vec![0.0f32; total];
    let mut nonphysical_hydraulic_support = vec![false; total];
    for depression in &depressions {
        for (&cell, &fraction) in depression
            .water_cells
            .iter()
            .zip(&depression.water_fractions)
        {
            // Fractional water does not erase the canonical vector channel from
            // the whole coarse cell. Regional refinement resolves its shoreline.
            nonphysical_hydraulic_support[cell] = waterbody_dominates_cell(fraction);
            water_class[cell] = depression.class_code;
            lake_id[cell] = depression.lake_id;
            if depression.class_code == WATER_WETLAND {
                wetland_fraction[cell] = fraction;
            } else {
                lake_fraction[cell] = fraction;
            }
        }
        if depression.hydraulic_control_cell < total
            && receiver[depression.hydraulic_control_cell] >= 0
        {
            // A single registered control handoff breaks the physical bed
            // profile across a lake/spill system without classifying the
            // depression's entire drainage path as non-river.
            nonphysical_hydraulic_support[depression.hydraulic_control_cell] = true;
        }
    }
    for cell in 0..total {
        let downstream = receiver[cell];
        if downstream < 0 {
            continue;
        }
        let downstream = downstream as usize;
        let unresolved_fill = final_flood.filled[cell]
            > hydrologic_elevation[cell] + config.minimum_depression_depth_m;
        let downstream_unresolved_fill = final_flood.filled[downstream]
            > hydrologic_elevation[downstream] + config.minimum_depression_depth_m;
        let exits_unresolved_fill = unresolved_fill && !downstream_unresolved_fill;
        let leaves_hydraulic_control =
            depression_id[cell] >= 0 && depression_id[cell] != depression_id[downstream];
        let unresolved_ascent = ocean[downstream] == 0
            && hydrologic_elevation[downstream]
                > hydrologic_elevation[cell] + config.minimum_depression_depth_m;
        if exits_unresolved_fill || leaves_hydraulic_control || unresolved_ascent {
            // Priority-flood topology may cross an unresolved lake, spill sill,
            // or subgrid outlet. Mark the routed fill boundary, not its entire
            // interior, so the graph carries water without propagating a deep
            // basin floor into the downstream physical bed profile.
            nonphysical_hydraulic_support[cell] = true;
        }
    }
    let (basin_id, sink_type, basin_count) = assign_basins(&order, &receiver, ocean, &water_class);
    let (flow_direction, flow_slope) = flow_geometry(
        &receiver,
        ocean,
        &hydrologic_elevation,
        xyz,
        config.planet_radius_m,
    );
    let mut contributing_area: Vec<f64> = area_km2
        .iter()
        .zip(ocean)
        .map(|(area, is_ocean)| if *is_ocean == 0 { *area } else { 0.0 })
        .collect();
    let mut discharge = local_runoff.clone();
    let mut outlet_loss_rate = vec![0.0f64; total];
    for depression in &depressions {
        if depression.open_outlet {
            outlet_loss_rate[depression.outlet_cell] +=
                (depression.annual_evaporation_km3 + depression.annual_seepage_km3) * 1e9
                    / SECONDS_PER_YEAR;
        }
    }
    let mut annual_open_water_loss_km3 = 0.0f64;
    for &cell in &order {
        if outlet_loss_rate[cell] > 0.0 {
            for month in 0..MONTHS {
                let offset = month * total + cell;
                let applied_loss = discharge[offset].min(outlet_loss_rate[cell]);
                discharge[offset] -= applied_loss;
                annual_open_water_loss_km3 += applied_loss * SECONDS_PER_MONTH / 1e9;
            }
        }
        let downstream = receiver[cell];
        if downstream >= 0 && ocean[downstream as usize] == 0 {
            let next = downstream as usize;
            contributing_area[next] += contributing_area[cell];
            for month in 0..MONTHS {
                discharge[month * total + next] += discharge[month * total + cell];
            }
        }
    }
    let river_properties = river_properties(
        &receiver,
        ocean,
        &nonphysical_hydraulic_support,
        &contributing_area,
        &discharge,
        &flow_slope,
        relief,
        config,
    );
    let reach_support = extend_reach_support(
        &receiver,
        ocean,
        &river_properties.river,
        &nonphysical_hydraulic_support,
    );
    let (reach_records, reach_vertices) = extract_reaches(
        &order,
        &receiver,
        ocean,
        &basin_id,
        &sink_type,
        &river_properties.river,
        &reach_support,
        &discharge,
        &flow_direction,
        &flow_slope,
        &river_properties.velocity,
        &river_properties.stream_power,
        &river_properties.floodplain,
        relief,
        &hydrologic_elevation,
    );

    let mut lake_records = Vec::with_capacity(depressions.len());
    for depression in &depressions {
        lake_records.push(LakeRecord {
            depression_id: depression.id,
            lake_id: depression.lake_id,
            class_code: depression.class_code as i32,
            sink_cell: depression.sink_cell as i32,
            outlet_cell: depression.outlet_cell as i32,
            outlet_receiver: depression.outlet_receiver as i32,
            cell_count: depression.cells.len() as i32,
            water_cell_count: depression.water_cells.len() as i32,
            open_outlet: i32::from(depression.open_outlet),
            area_km2: depression.area_km2,
            water_area_km2: depression.water_area_km2,
            volume_km3: depression.water_volume_km3,
            surface_elevation_m: depression.water_surface_elevation_m,
            maximum_depth_m: depression.maximum_depth_m,
            mean_depth_m: depression.mean_depth_m,
            catchment_area_km2: depression.catchment_area_km2,
            mean_inflow_m3s: (depression.monthly_inflow_m3s.iter().sum::<f64>() / MONTHS as f64)
                as f32,
            annual_inflow_km3: depression.annual_inflow_km3,
            annual_evaporation_km3: depression.annual_evaporation_km3,
            annual_seepage_km3: depression.annual_seepage_km3,
            annual_balance_km3: depression.annual_balance_km3,
            fill_time_years: depression.fill_time_years,
            salinity_index: depression.salinity_index,
            mean_aridity_index: depression.mean_aridity,
            breach_incision_m: depression.breach_incision_m,
        });
    }

    depression_id_out.copy_from_slice(&depression_id);
    lake_id_out.copy_from_slice(&lake_id);
    water_class_out.copy_from_slice(&water_class);
    lake_fraction_out.copy_from_slice(&lake_fraction);
    wetland_fraction_out.copy_from_slice(&wetland_fraction);
    for cell in 0..total {
        fill_depth_out[cell] = if ocean[cell] == 0 {
            (initial_flood.filled[cell] - elevation[cell]).max(0.0)
        } else {
            0.0
        };
        mean_discharge_out[cell] = ((0..MONTHS)
            .map(|month| discharge[month * total + cell])
            .sum::<f64>()
            / MONTHS as f64) as f32;
    }
    hydrologic_elevation_out.copy_from_slice(&hydrologic_elevation);
    breach_incision_out.copy_from_slice(&breach_incision);
    receiver_out.copy_from_slice(&receiver);
    flow_direction_out.copy_from_slice(&flow_direction);
    flow_slope_out.copy_from_slice(&flow_slope);
    contributing_area_out.copy_from_slice(&contributing_area);
    for (target, source) in monthly_discharge_out.iter_mut().zip(discharge.iter()) {
        *target = source.max(0.0) as f32;
    }
    velocity_out.copy_from_slice(&river_properties.velocity);
    stream_power_out.copy_from_slice(&river_properties.stream_power);
    basin_id_out.copy_from_slice(&basin_id);
    sink_type_out.copy_from_slice(&sink_type);
    river_corridor_out.copy_from_slice(&river_properties.corridor);
    floodplain_out.copy_from_slice(&river_properties.floodplain);

    let annual_runoff_km3 = local_runoff.iter().sum::<f64>() * SECONDS_PER_MONTH / 1e9;
    let mut terminal_runoff = 0.0f64;
    for cell in 0..total {
        if ocean[cell] == 0 && (receiver[cell] < 0 || ocean[receiver[cell] as usize] != 0) {
            for month in 0..MONTHS {
                terminal_runoff += discharge[month * total + cell] * SECONDS_PER_MONTH / 1e9;
            }
        }
    }
    let conservation_relative_error =
        (terminal_runoff + annual_open_water_loss_km3 - annual_runoff_km3).abs()
            / annual_runoff_km3.max(1e-12);
    let lake_count = depressions
        .iter()
        .filter(|depression| depression.lake_id >= 0 && depression.class_code != WATER_WETLAND)
        .count();
    let closed_sink_count = order.iter().filter(|cell| receiver[**cell] < 0).count();
    let lake_volume_km3 = depressions
        .iter()
        .filter(|depression| depression.lake_id >= 0 && depression.class_code != WATER_WETLAND)
        .map(|depression| depression.water_volume_km3)
        .sum();
    let breach_sediment_pulse_km3 = breach_records
        .iter()
        .map(|record| record.sediment_pulse_km3)
        .sum();
    *stats_out = HydrologyStats {
        depression_count: depressions.len() as i32,
        lake_count: lake_count as i32,
        breach_count: breach_records.len() as i32,
        basin_count,
        reach_count: reach_records.len() as i32,
        wetland_count: depressions
            .iter()
            .filter(|depression| depression.class_code == WATER_WETLAND)
            .count() as i32,
        endorheic_count: depressions
            .iter()
            .filter(|depression| depression.class_code == WATER_ENDORHEIC)
            .count() as i32,
        stable_lake_count: depressions
            .iter()
            .filter(|depression| depression.class_code == WATER_STABLE_LAKE)
            .count() as i32,
        overflow_lake_count: depressions
            .iter()
            .filter(|depression| depression.class_code == WATER_OVERFLOW_LAKE)
            .count() as i32,
        land_cell_count: land_count as i32,
        river_cell_count: river_properties
            .river
            .iter()
            .filter(|value| **value)
            .count() as i32,
        closed_sink_count: closed_sink_count as i32,
        topology_valid: 1,
        maximum_contributing_area_km2: contributing_area.iter().copied().fold(0.0, f64::max),
        maximum_discharge_m3s: mean_discharge_out.iter().copied().fold(0.0, f32::max),
        annual_runoff_km3,
        lake_volume_km3,
        breach_sediment_pulse_km3,
        annual_open_water_loss_km3,
        conservation_relative_error,
    };

    let (data, len) = into_raw_array(lake_records);
    *lake_records_out = LakeRecordArray { data, len };
    let (data, len) = into_raw_array(breach_records);
    *breach_records_out = BreachRecordArray { data, len };
    let (data, len) = into_raw_array(reach_records);
    *reach_records_out = RiverReachRecordArray { data, len };
    let (data, len) = into_raw_array(reach_vertices);
    *reach_vertices_out = Int32Array { data, len };
    0
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)]
/// Build the global hydrology graph and its lake, breach, and river catalogs.
///
/// # Safety
///
/// Every pointer must be aligned, non-null, correctly sized for `cell_count`,
/// and non-overlapping. Neighbor IDs must reference valid cells. Returned
/// record arrays must each be released exactly once with their matching free
/// function.
pub unsafe extern "C" fn hydrology_run_cubed_sphere(
    cell_count: i32,
    config: *const HydrologyConfig,
    area_steradians: *const f64,
    neighbors: *const i32,
    xyz: *const f32,
    elevation: *const f32,
    relief: *const f32,
    rock_strength: *const f32,
    accommodation: *const f32,
    ocean: *const u8,
    runoff: *const f32,
    evaporation: *const f32,
    aridity: *const f32,
    depression_id_out: *mut i32,
    lake_id_out: *mut i32,
    water_class_out: *mut u8,
    lake_fraction_out: *mut f32,
    wetland_fraction_out: *mut f32,
    fill_depth_out: *mut f32,
    hydrologic_elevation_out: *mut f32,
    breach_incision_out: *mut f32,
    receiver_out: *mut i32,
    flow_direction_out: *mut f32,
    flow_slope_out: *mut f32,
    contributing_area_out: *mut f64,
    monthly_discharge_out: *mut f32,
    mean_discharge_out: *mut f32,
    velocity_out: *mut f32,
    stream_power_out: *mut f32,
    basin_id_out: *mut i32,
    sink_type_out: *mut u8,
    river_corridor_out: *mut f32,
    floodplain_out: *mut f32,
    lake_records_out: *mut LakeRecordArray,
    breach_records_out: *mut BreachRecordArray,
    reach_records_out: *mut RiverReachRecordArray,
    reach_vertices_out: *mut Int32Array,
    stats_out: *mut HydrologyStats,
) -> i32 {
    if cell_count <= 0
        || config.is_null()
        || area_steradians.is_null()
        || neighbors.is_null()
        || xyz.is_null()
        || elevation.is_null()
        || relief.is_null()
        || rock_strength.is_null()
        || accommodation.is_null()
        || ocean.is_null()
        || runoff.is_null()
        || evaporation.is_null()
        || aridity.is_null()
        || depression_id_out.is_null()
        || lake_id_out.is_null()
        || water_class_out.is_null()
        || lake_fraction_out.is_null()
        || wetland_fraction_out.is_null()
        || fill_depth_out.is_null()
        || hydrologic_elevation_out.is_null()
        || breach_incision_out.is_null()
        || receiver_out.is_null()
        || flow_direction_out.is_null()
        || flow_slope_out.is_null()
        || contributing_area_out.is_null()
        || monthly_discharge_out.is_null()
        || mean_discharge_out.is_null()
        || velocity_out.is_null()
        || stream_power_out.is_null()
        || basin_id_out.is_null()
        || sink_type_out.is_null()
        || river_corridor_out.is_null()
        || floodplain_out.is_null()
        || lake_records_out.is_null()
        || breach_records_out.is_null()
        || reach_records_out.is_null()
        || reach_vertices_out.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let total = cell_count as usize;
    run_hydrology(
        total,
        D4_NEIGHBORS,
        &*config,
        slice::from_raw_parts(area_steradians, total),
        slice::from_raw_parts(neighbors, total * D4_NEIGHBORS),
        slice::from_raw_parts(xyz, total * 3),
        slice::from_raw_parts(elevation, total),
        slice::from_raw_parts(relief, total),
        slice::from_raw_parts(rock_strength, total),
        slice::from_raw_parts(accommodation, total),
        slice::from_raw_parts(ocean, total),
        slice::from_raw_parts(runoff, total * MONTHS),
        slice::from_raw_parts(evaporation, total * MONTHS),
        slice::from_raw_parts(aridity, total),
        slice::from_raw_parts_mut(depression_id_out, total),
        slice::from_raw_parts_mut(lake_id_out, total),
        slice::from_raw_parts_mut(water_class_out, total),
        slice::from_raw_parts_mut(lake_fraction_out, total),
        slice::from_raw_parts_mut(wetland_fraction_out, total),
        slice::from_raw_parts_mut(fill_depth_out, total),
        slice::from_raw_parts_mut(hydrologic_elevation_out, total),
        slice::from_raw_parts_mut(breach_incision_out, total),
        slice::from_raw_parts_mut(receiver_out, total),
        slice::from_raw_parts_mut(flow_direction_out, total * 3),
        slice::from_raw_parts_mut(flow_slope_out, total),
        slice::from_raw_parts_mut(contributing_area_out, total),
        slice::from_raw_parts_mut(monthly_discharge_out, total * MONTHS),
        slice::from_raw_parts_mut(mean_discharge_out, total),
        slice::from_raw_parts_mut(velocity_out, total),
        slice::from_raw_parts_mut(stream_power_out, total),
        slice::from_raw_parts_mut(basin_id_out, total),
        slice::from_raw_parts_mut(sink_type_out, total),
        slice::from_raw_parts_mut(river_corridor_out, total),
        slice::from_raw_parts_mut(floodplain_out, total),
        &mut *lake_records_out,
        &mut *breach_records_out,
        &mut *reach_records_out,
        &mut *reach_vertices_out,
        &mut *stats_out,
    )
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)]
/// Build hydrology on a bounded regional graph with four or eight neighbors.
///
/// The `ocean` input is the terminal mask for this generic graph. Callers may
/// include explicit open-boundary and registered-outlet cells, then retain a
/// separate physical-ocean mask for publication.
///
/// # Safety
///
/// Every pointer must be aligned, non-null, correctly sized for `cell_count`,
/// and non-overlapping. Neighbor IDs must reference valid local cells. Returned
/// record arrays must each be released exactly once with their matching free
/// function.
pub unsafe extern "C" fn hydrology_run_regional_graph(
    cell_count: i32,
    neighbor_count: i32,
    config: *const HydrologyConfig,
    area_steradians: *const f64,
    neighbors: *const i32,
    xyz: *const f32,
    elevation: *const f32,
    relief: *const f32,
    rock_strength: *const f32,
    accommodation: *const f32,
    terminal: *const u8,
    runoff: *const f32,
    evaporation: *const f32,
    aridity: *const f32,
    depression_id_out: *mut i32,
    lake_id_out: *mut i32,
    water_class_out: *mut u8,
    lake_fraction_out: *mut f32,
    wetland_fraction_out: *mut f32,
    fill_depth_out: *mut f32,
    hydrologic_elevation_out: *mut f32,
    breach_incision_out: *mut f32,
    receiver_out: *mut i32,
    flow_direction_out: *mut f32,
    flow_slope_out: *mut f32,
    contributing_area_out: *mut f64,
    monthly_discharge_out: *mut f32,
    mean_discharge_out: *mut f32,
    velocity_out: *mut f32,
    stream_power_out: *mut f32,
    basin_id_out: *mut i32,
    sink_type_out: *mut u8,
    river_corridor_out: *mut f32,
    floodplain_out: *mut f32,
    lake_records_out: *mut LakeRecordArray,
    breach_records_out: *mut BreachRecordArray,
    reach_records_out: *mut RiverReachRecordArray,
    reach_vertices_out: *mut Int32Array,
    stats_out: *mut HydrologyStats,
) -> i32 {
    if cell_count <= 0
        || !matches!(neighbor_count, 4 | 8)
        || config.is_null()
        || area_steradians.is_null()
        || neighbors.is_null()
        || xyz.is_null()
        || elevation.is_null()
        || relief.is_null()
        || rock_strength.is_null()
        || accommodation.is_null()
        || terminal.is_null()
        || runoff.is_null()
        || evaporation.is_null()
        || aridity.is_null()
        || depression_id_out.is_null()
        || lake_id_out.is_null()
        || water_class_out.is_null()
        || lake_fraction_out.is_null()
        || wetland_fraction_out.is_null()
        || fill_depth_out.is_null()
        || hydrologic_elevation_out.is_null()
        || breach_incision_out.is_null()
        || receiver_out.is_null()
        || flow_direction_out.is_null()
        || flow_slope_out.is_null()
        || contributing_area_out.is_null()
        || monthly_discharge_out.is_null()
        || mean_discharge_out.is_null()
        || velocity_out.is_null()
        || stream_power_out.is_null()
        || basin_id_out.is_null()
        || sink_type_out.is_null()
        || river_corridor_out.is_null()
        || floodplain_out.is_null()
        || lake_records_out.is_null()
        || breach_records_out.is_null()
        || reach_records_out.is_null()
        || reach_vertices_out.is_null()
        || stats_out.is_null()
    {
        return 1;
    }
    let total = cell_count as usize;
    let neighbor_count = neighbor_count as usize;
    run_hydrology(
        total,
        neighbor_count,
        &*config,
        slice::from_raw_parts(area_steradians, total),
        slice::from_raw_parts(neighbors, total * neighbor_count),
        slice::from_raw_parts(xyz, total * 3),
        slice::from_raw_parts(elevation, total),
        slice::from_raw_parts(relief, total),
        slice::from_raw_parts(rock_strength, total),
        slice::from_raw_parts(accommodation, total),
        slice::from_raw_parts(terminal, total),
        slice::from_raw_parts(runoff, total * MONTHS),
        slice::from_raw_parts(evaporation, total * MONTHS),
        slice::from_raw_parts(aridity, total),
        slice::from_raw_parts_mut(depression_id_out, total),
        slice::from_raw_parts_mut(lake_id_out, total),
        slice::from_raw_parts_mut(water_class_out, total),
        slice::from_raw_parts_mut(lake_fraction_out, total),
        slice::from_raw_parts_mut(wetland_fraction_out, total),
        slice::from_raw_parts_mut(fill_depth_out, total),
        slice::from_raw_parts_mut(hydrologic_elevation_out, total),
        slice::from_raw_parts_mut(breach_incision_out, total),
        slice::from_raw_parts_mut(receiver_out, total),
        slice::from_raw_parts_mut(flow_direction_out, total * 3),
        slice::from_raw_parts_mut(flow_slope_out, total),
        slice::from_raw_parts_mut(contributing_area_out, total),
        slice::from_raw_parts_mut(monthly_discharge_out, total * MONTHS),
        slice::from_raw_parts_mut(mean_discharge_out, total),
        slice::from_raw_parts_mut(velocity_out, total),
        slice::from_raw_parts_mut(stream_power_out, total),
        slice::from_raw_parts_mut(basin_id_out, total),
        slice::from_raw_parts_mut(sink_type_out, total),
        slice::from_raw_parts_mut(river_corridor_out, total),
        slice::from_raw_parts_mut(floodplain_out, total),
        &mut *lake_records_out,
        &mut *breach_records_out,
        &mut *reach_records_out,
        &mut *reach_vertices_out,
        &mut *stats_out,
    )
}

#[no_mangle]
pub extern "C" fn hydrology_free_lakes(array: LakeRecordArray) {
    free_raw_array(array.data, array.len);
}

#[no_mangle]
pub extern "C" fn hydrology_free_breaches(array: BreachRecordArray) {
    free_raw_array(array.data, array.len);
}

#[no_mangle]
pub extern "C" fn hydrology_free_reaches(array: RiverReachRecordArray) {
    free_raw_array(array.data, array.len);
}

#[no_mangle]
pub extern "C" fn hydrology_free_i32(array: Int32Array) {
    free_raw_array(array.data, array.len);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> HydrologyConfig {
        HydrologyConfig {
            planet_radius_m: 6_371_000.0,
            minimum_depression_depth_m: 20.0,
            wetland_mean_depth_m: 35.0,
            endorheic_aridity_threshold: 0.35,
            maximum_fill_time_years: 50_000.0,
            lake_seepage_mm_year: 30.0,
            subgrid_relief_scale: 1.0,
            subgrid_connected_basin_fraction: 0.50,
            breach_score_threshold: 1.0,
            maximum_breach_incision_m: 800.0,
            breach_length_cells: 2,
            river_discharge_threshold_m3s: 300.0,
            river_contributing_area_threshold_km2: 200_000.0,
            river_minimum_discharge_m3s: 25.0,
        }
    }

    fn line_neighbors(total: usize) -> Vec<i32> {
        let mut result = Vec::with_capacity(total * D4_NEIGHBORS);
        for cell in 0..total {
            let before = cell.saturating_sub(1) as i32;
            let after = (cell + 1).min(total - 1) as i32;
            result.extend_from_slice(&[before, after, before, after]);
        }
        result
    }

    fn d8_grid_neighbors(height: usize, width: usize) -> Vec<i32> {
        let offsets = [
            (-1isize, 0isize),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ];
        let mut result = Vec::with_capacity(height * width * 8);
        for row in 0..height {
            for column in 0..width {
                for (row_offset, column_offset) in offsets {
                    let next_row = row as isize + row_offset;
                    let next_column = column as isize + column_offset;
                    let cell = if next_row >= 0
                        && next_row < height as isize
                        && next_column >= 0
                        && next_column < width as isize
                    {
                        next_row as usize * width + next_column as usize
                    } else {
                        row * width + column
                    };
                    result.push(cell as i32);
                }
            }
        }
        result
    }

    #[test]
    fn subgrid_hypsometry_integrates_fraction_and_volume() {
        let (half_fraction, equivalent_depth_m) =
            subgrid_inundation(100.0, 200.0, 100.0, 20.0, 1.0, 1.0);
        assert!((half_fraction - 0.5).abs() < 1e-6);
        assert!((equivalent_depth_m - 25.0).abs() < 1e-9);

        let (full_fraction, equivalent_depth_m) =
            subgrid_inundation(100.0, 200.0, 350.0, 20.0, 1.0, 1.0);
        assert!((full_fraction - 1.0).abs() < 1e-6);
        assert!((equivalent_depth_m - 100.0).abs() < 1e-9);

        let (connected_fraction, equivalent_depth_m) =
            subgrid_inundation(100.0, 200.0, 100.0, 20.0, 1.0, 0.25);
        assert!((connected_fraction - 0.125).abs() < 1e-6);
        assert!((equivalent_depth_m - 6.25).abs() < 1e-9);
        assert!((equivalent_depth_m / connected_fraction as f64 - 50.0).abs() < 1e-9);
    }

    #[test]
    fn priority_flood_routes_a_closed_depression_to_ocean() {
        let elevation = [0.0, 8.0, 2.0, 3.0, 9.0];
        let ocean = [1, 0, 0, 0, 0];
        let flood = priority_flood(&elevation, &ocean, &line_neighbors(5), D4_NEIGHBORS).unwrap();
        assert_eq!(flood.receiver[2], 1);
        assert_eq!(flood.filled[2], 8.0);
        assert_eq!(flood.filled[3], 8.0);
        assert_eq!(flood.order.len(), elevation.len());
    }

    #[test]
    fn regional_priority_flood_uses_diagonal_neighbors() {
        let elevation = [0.0, 20.0, 20.0, 20.0, 5.0, 20.0, 20.0, 20.0, 30.0];
        let ocean = [1, 0, 0, 0, 0, 0, 0, 0, 0];
        let flood = priority_flood(&elevation, &ocean, &d8_grid_neighbors(3, 3), 8).unwrap();
        assert_eq!(flood.receiver[4], 0);
        assert_eq!(flood.filled[4], 5.0);
    }

    #[test]
    fn deep_basin_retains_an_open_lake_after_partial_breach() {
        let elevation = [-100.0, 100.0, -1_000.0, 200.0];
        let ocean = [1, 0, 0, 0];
        let neighbors = line_neighbors(elevation.len());
        let area_km2 = [100.0; 4];
        let relief = [100.0; 4];
        let config = test_config();
        let initial_flood = priority_flood(&elevation, &ocean, &neighbors, D4_NEIGHBORS).unwrap();
        let (_, mut depressions) = find_depressions(
            &initial_flood,
            &elevation,
            &relief,
            &ocean,
            &neighbors,
            D4_NEIGHBORS,
            &area_km2,
            &config,
        );
        assert_eq!(depressions.len(), 1);
        depressions[0].class_code = WATER_BREACHED;

        let (hydrologic_elevation, _, records) = apply_breaches(
            &mut depressions,
            &initial_flood,
            &elevation,
            &relief,
            &area_km2,
            &config,
        );
        assert_eq!(records.len(), 1);
        let final_flood =
            priority_flood(&hydrologic_elevation, &ocean, &neighbors, D4_NEIGHBORS).unwrap();
        retain_post_breach_lakes(
            &mut depressions,
            &final_flood,
            &elevation,
            &relief,
            &area_km2,
            &config,
        );
        assign_water_extents(&mut depressions, &elevation, &relief, &area_km2, &config);
        finalize_waterbody_classes(&mut depressions, &config);

        assert_eq!(depressions[0].class_code, WATER_OVERFLOW_LAKE);
        assert!(depressions[0].open_outlet);
        assert_eq!(depressions[0].lake_id, 0);
        assert!(depressions[0].water_area_km2 > 0.0);
        assert!(depressions[0].water_volume_km3 > 0.0);
        assert!(depressions[0].spill_elevation_m < records[0].pre_breach_spill_elevation_m);
        assert!(depressions[0].maximum_depth_m > config.minimum_depression_depth_m);
    }

    #[test]
    fn topological_order_rejects_cycles() {
        let ocean = [1, 0, 0];
        assert!(topological_order(&[-1, 2, 1], &ocean).is_none());
        assert!(topological_order(&[-1, 0, 1], &ocean).is_some());
    }

    #[test]
    fn reach_support_bridges_nonriver_depression_cells() {
        let receiver = [1, 2, 3, 4, 5, -1, 5];
        let ocean = [0, 0, 0, 0, 0, 1, 0];
        let river = [true, true, false, false, true, false, false];
        let preserved = [false, false, true, true, false, false, false];
        let support = extend_reach_support(&receiver, &ocean, &river, &preserved);
        assert_eq!(support, vec![true, true, true, true, true, false, false]);
        assert_eq!(
            extend_reach_support(&receiver, &ocean, &river, &[false; 7]),
            river
        );

        let branched_receiver = [2, 2, 3, -1];
        let branched_support = [true; 4];
        assert_eq!(
            strahler_orders(&[0, 1, 2, 3], &branched_receiver, &branched_support),
            vec![1, 1, 2, 2]
        );
    }

    #[test]
    fn water_balance_opens_wet_basins_and_preserves_dry_terminal_lakes() {
        let elevation = [-10.0, 100.0, 0.0, 20.0, 220.0];
        let ocean = [1, 0, 0, 0, 0];
        let neighbors = line_neighbors(elevation.len());
        let area_km2 = [100.0; 5];
        let flood = priority_flood(&elevation, &ocean, &neighbors, D4_NEIGHBORS).unwrap();
        let relief = [100.0; 5];
        let (depression_id, template) = find_depressions(
            &flood,
            &elevation,
            &relief,
            &ocean,
            &neighbors,
            D4_NEIGHBORS,
            &area_km2,
            &test_config(),
        );
        assert_eq!(template.len(), 1);
        let owner = depression_catchments(&flood, &depression_id, &ocean);
        let rock = [0.8; 5];
        let accommodation = [0.5; 5];

        assert!(template[0].maximum_depth_m > template[0].mean_depth_m);
        let mut shallow = template.clone();
        let mut wetland_config = test_config();
        wetland_config.wetland_mean_depth_m =
            0.5 * (template[0].maximum_depth_m + template[0].mean_depth_m);
        let wet_runoff = vec![10.0; MONTHS * elevation.len()];
        let wet_evaporation = vec![5.0; MONTHS * elevation.len()];
        classify_depressions(
            &mut shallow,
            &owner,
            &area_km2,
            &wet_runoff,
            &wet_evaporation,
            &[1.2; 5],
            &rock,
            &accommodation,
            &wetland_config,
        );
        assert_eq!(shallow[0].class_code, WATER_WETLAND);

        let mut wet = template;
        classify_depressions(
            &mut wet,
            &owner,
            &area_km2,
            &wet_runoff,
            &wet_evaporation,
            &[1.2; 5],
            &rock,
            &accommodation,
            &test_config(),
        );
        assert_eq!(wet[0].class_code, WATER_OVERFLOW_LAKE);
        assert!(wet[0].open_outlet);

        let (depression_id, mut dry) = find_depressions(
            &flood,
            &elevation,
            &relief,
            &ocean,
            &neighbors,
            D4_NEIGHBORS,
            &area_km2,
            &test_config(),
        );
        let owner = depression_catchments(&flood, &depression_id, &ocean);
        let dry_runoff = vec![0.001; MONTHS * elevation.len()];
        let dry_evaporation = vec![500.0; MONTHS * elevation.len()];
        classify_depressions(
            &mut dry,
            &owner,
            &area_km2,
            &dry_runoff,
            &dry_evaporation,
            &[0.15; 5],
            &rock,
            &accommodation,
            &test_config(),
        );
        assign_water_extents(&mut dry, &elevation, &relief, &area_km2, &test_config());
        assert_eq!(dry[0].class_code, WATER_ENDORHEIC);
        finalize_waterbody_classes(&mut dry, &test_config());
        assert_eq!(dry[0].class_code, WATER_WETLAND);
        assert_eq!(dry[0].lake_id, -1);
        assert!(!dry[0].open_outlet);
        assert_eq!(dry[0].water_cells.len(), 1);
        assert!(dry[0].water_fractions[0] < 1.0);
        assert!(dry[0].water_area_km2 < dry[0].area_km2 + 1e-9);
        assert!(dry[0].maximum_depth_m >= dry[0].mean_depth_m);

        let (depression_id, mut breached) = find_depressions(
            &flood,
            &elevation,
            &relief,
            &ocean,
            &neighbors,
            D4_NEIGHBORS,
            &area_km2,
            &test_config(),
        );
        let owner = depression_catchments(&flood, &depression_id, &ocean);
        let mut breach_config = test_config();
        breach_config.breach_score_threshold = 0.0;
        classify_depressions(
            &mut breached,
            &owner,
            &area_km2,
            &wet_runoff,
            &wet_evaporation,
            &[1.2; 5],
            &[0.0; 5],
            &[0.0; 5],
            &breach_config,
        );
        assert_eq!(breached[0].class_code, WATER_BREACHED);
        let (_, incision, records) = apply_breaches(
            &mut breached,
            &flood,
            &elevation,
            &[100.0; 5],
            &area_km2,
            &breach_config,
        );
        assert_eq!(records.len(), 1);
        assert!(incision.iter().any(|value| *value > 0.0));
        assert!(records[0].sediment_pulse_km3 < breached[0].volume_km3);
    }

    #[test]
    fn upstream_overflow_contributes_to_downstream_lake_balance() {
        let elevation = [-10.0, 100.0, 0.0, 200.0, 20.0, 220.0];
        let ocean = [1, 0, 0, 0, 0, 0];
        let neighbors = line_neighbors(elevation.len());
        let area_km2 = [100.0; 6];
        let flood = priority_flood(&elevation, &ocean, &neighbors, D4_NEIGHBORS).unwrap();
        let relief = [100.0; 6];
        let (depression_id, mut depressions) = find_depressions(
            &flood,
            &elevation,
            &relief,
            &ocean,
            &neighbors,
            D4_NEIGHBORS,
            &area_km2,
            &test_config(),
        );
        assert_eq!(depressions.len(), 2);
        let owner = depression_catchments(&flood, &depression_id, &ocean);
        let mut runoff = vec![0.001; MONTHS * elevation.len()];
        let mut evaporation = vec![500.0; MONTHS * elevation.len()];
        for month in 0..MONTHS {
            for cell in 4..6 {
                runoff[month * elevation.len() + cell] = 10.0;
                evaporation[month * elevation.len() + cell] = 5.0;
            }
        }
        classify_depressions(
            &mut depressions,
            &owner,
            &area_km2,
            &runoff,
            &evaporation,
            &[1.0; 6],
            &[0.8; 6],
            &[0.5; 6],
            &test_config(),
        );
        assert!(!depressions[0].open_outlet);
        assert!(depressions[1].open_outlet);
        propagate_depression_overflow(&mut depressions, &flood, &depression_id, &test_config());
        assert!(depressions[0].open_outlet);
        assert!(depressions[0].catchment_area_km2 > 200.0);
        assert!(depressions[0].annual_inflow_km3 > depressions[0].annual_evaporation_km3);
    }
}
