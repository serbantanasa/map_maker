use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::mem;
use std::slice;

const MONTHS: usize = 12;
const HYPSOMETRY_BINS: usize = 65;

const CLASS_DRY: i32 = 0;
const CLASS_TRANSIENT: i32 = 1;
const CLASS_SEASONAL_LAKE: i32 = 2;
const CLASS_PERMANENT_LAKE: i32 = 3;
const CLASS_HYDROLOGIC_WETLAND: i32 = 4;

#[no_mangle]
pub extern "C" fn surface_water_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn surface_water_native_struct_size(kind: u32) -> usize {
    match kind {
        0 => mem::size_of::<SurfaceWaterConfig>(),
        1 => mem::size_of::<SurfaceWaterCandidateRecord>(),
        2 => mem::size_of::<SurfaceWaterCellRecord>(),
        3 => mem::size_of::<SurfaceWaterStats>(),
        _ => 0,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SurfaceWaterConfig {
    pub refinement_factor: i32,
    pub minimum_solver_iterations: i32,
    pub maximum_solver_iterations: i32,
    pub transient_max_months: i32,
    pub permanent_min_months: i32,
    pub convergence_tolerance_fraction: f64,
    pub open_water_evaporation_factor: f64,
    pub seepage_mm_year: f64,
    pub subgrid_relief_scale: f64,
    pub minimum_subgrid_relief_m: f64,
    pub maximum_connected_inundation_fraction: f64,
    pub minimum_wet_area_fraction: f64,
    pub wetland_max_mean_depth_m: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SurfaceWaterCandidateRecord {
    pub depression_id: i32,
    pub downstream_depression_id: i32,
    pub class_code: i32,
    pub cell_count: i32,
    pub catchment_cell_count: i32,
    pub wet_month_count: i32,
    pub solver_iterations: i32,
    pub converged: i32,
    pub open_outlet: i32,
    pub catchment_area_km2: f64,
    pub potential_water_area_km2: f64,
    pub storage_capacity_km3: f64,
    pub annual_direct_inflow_km3: f64,
    pub annual_upstream_inflow_km3: f64,
    pub annual_total_inflow_km3: f64,
    pub annual_evaporation_km3: f64,
    pub annual_seepage_km3: f64,
    pub annual_overflow_km3: f64,
    pub annual_terminal_overflow_km3: f64,
    pub annual_storage_change_km3: f64,
    pub water_balance_residual_km3: f64,
    pub hydroperiod_fraction: f64,
    pub minimum_water_area_km2: f64,
    pub mean_water_area_km2: f64,
    pub maximum_water_area_km2: f64,
    pub mean_wetted_depth_m: f64,
    pub maximum_mean_depth_m: f64,
    pub salinity_index: f64,
    pub monthly_direct_inflow_km3: [f64; MONTHS],
    pub monthly_upstream_inflow_km3: [f64; MONTHS],
    pub monthly_total_inflow_km3: [f64; MONTHS],
    pub monthly_evaporation_km3: [f64; MONTHS],
    pub monthly_seepage_km3: [f64; MONTHS],
    pub monthly_overflow_km3: [f64; MONTHS],
    pub monthly_storage_km3: [f64; MONTHS],
    pub monthly_water_area_km2: [f64; MONTHS],
}

#[repr(C)]
pub struct SurfaceWaterCandidateArray {
    pub data: *mut SurfaceWaterCandidateRecord,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SurfaceWaterCellRecord {
    pub fine_cell_id: i32,
    pub depression_id: i32,
    pub class_code: i32,
    pub potential_inundation_fraction: f32,
    pub minimum_inundation_fraction: f32,
    pub mean_inundation_fraction: f32,
    pub maximum_inundation_fraction: f32,
    pub monthly_inundation_fraction: [f32; MONTHS],
}

#[repr(C)]
pub struct SurfaceWaterCellArray {
    pub data: *mut SurfaceWaterCellRecord,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SurfaceWaterStats {
    pub cell_count: i32,
    pub candidate_count: i32,
    pub candidate_cell_count: i32,
    pub owned_source_cell_count: i32,
    pub dry_count: i32,
    pub transient_count: i32,
    pub seasonal_lake_count: i32,
    pub permanent_lake_count: i32,
    pub hydrologic_wetland_count: i32,
    pub graph_valid: i32,
    pub convergence_valid: i32,
    pub fraction_valid: i32,
    pub storage_valid: i32,
    pub direct_catchment_valid: i32,
    pub maximum_solver_iterations_used: i32,
    pub active_source_area_km2: f64,
    pub owned_catchment_area_km2: f64,
    pub potential_water_area_km2: f64,
    pub storage_capacity_km3: f64,
    pub annual_direct_inflow_km3: f64,
    pub annual_evaporation_km3: f64,
    pub annual_seepage_km3: f64,
    pub annual_terminal_overflow_km3: f64,
    pub annual_storage_change_km3: f64,
    pub water_balance_residual_km3: f64,
    pub water_balance_relative_error: f64,
    pub minimum_inundation_fraction: f64,
    pub maximum_inundation_fraction: f64,
    pub dry_mean_water_area_km2: f64,
    pub transient_mean_water_area_km2: f64,
    pub seasonal_lake_mean_water_area_km2: f64,
    pub permanent_lake_mean_water_area_km2: f64,
    pub hydrologic_wetland_mean_water_area_km2: f64,
}

struct Inputs<'a> {
    cell_ids: &'a [i32],
    receiver_ids: &'a [i32],
    depression_ids: &'a [i32],
    source_active: &'a [u8],
    area_km2: &'a [f64],
    terrain_elevation_m: &'a [f64],
    hydrologic_elevation_m: &'a [f64],
    parent_relief_m: &'a [f32],
    monthly_runoff_mm: &'a [f32],
    monthly_evaporation_mm: &'a [f32],
    sediment_accommodation: &'a [f32],
    candidate_ids: &'a [i32],
    spill_receiver_ids: &'a [i32],
}

struct RoutingDomain {
    candidate_by_id: HashMap<i32, usize>,
    owner: Vec<i32>,
    downstream: Vec<i32>,
    candidate_order: Vec<usize>,
}

struct Hypsometry {
    bottom_m: f64,
    cap_m: f64,
    volume_km3: [f64; HYPSOMETRY_BINS],
    area_km2: [f64; HYPSOMETRY_BINS],
    evaporation_area: [[f64; MONTHS]; HYPSOMETRY_BINS],
    seepage_area_km2: [f64; HYPSOMETRY_BINS],
    potential_fraction_by_member: Vec<f32>,
}

struct CandidateState {
    members: Vec<usize>,
    hypsometry: Hypsometry,
    direct_inflow_km3: [f64; MONTHS],
    catchment_cell_count: usize,
    catchment_area_km2: f64,
    monthly_upstream_inflow_km3: [f64; MONTHS],
    monthly_total_inflow_km3: [f64; MONTHS],
    monthly_evaporation_km3: [f64; MONTHS],
    monthly_seepage_km3: [f64; MONTHS],
    monthly_overflow_km3: [f64; MONTHS],
    monthly_storage_km3: [f64; MONTHS],
    monthly_water_area_km2: [f64; MONTHS],
    periodic_start_storage_km3: f64,
    solver_iterations: usize,
}

struct Outcome {
    candidates: Vec<SurfaceWaterCandidateRecord>,
    cells: Vec<SurfaceWaterCellRecord>,
    stats: SurfaceWaterStats,
}

fn validate_config(config: SurfaceWaterConfig) -> Result<(), i32> {
    if config.refinement_factor < 2
        || config.refinement_factor > 32
        || config.refinement_factor & (config.refinement_factor - 1) != 0
        || config.minimum_solver_iterations < 1
        || config.maximum_solver_iterations < config.minimum_solver_iterations
        || config.maximum_solver_iterations > 256
        || !(0..=11).contains(&config.transient_max_months)
        || config.permanent_min_months != MONTHS as i32
        || config.transient_max_months >= config.permanent_min_months
    {
        return Err(1);
    }
    for value in [
        config.convergence_tolerance_fraction,
        config.open_water_evaporation_factor,
        config.seepage_mm_year,
        config.subgrid_relief_scale,
        config.minimum_subgrid_relief_m,
        config.maximum_connected_inundation_fraction,
        config.minimum_wet_area_fraction,
        config.wetland_max_mean_depth_m,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(1);
        }
    }
    if config.convergence_tolerance_fraction <= 0.0
        || config.open_water_evaporation_factor > 10.0
        || config.seepage_mm_year > 10_000.0
        || config.subgrid_relief_scale <= 0.0
        || config.minimum_subgrid_relief_m <= 0.0
        || config.maximum_connected_inundation_fraction <= 0.0
        || config.maximum_connected_inundation_fraction > 1.0
        || config.minimum_wet_area_fraction <= 0.0
        || config.minimum_wet_area_fraction > 1.0
        || config.wetland_max_mean_depth_m <= 0.0
    {
        return Err(1);
    }
    Ok(())
}

fn validate_inputs(config: SurfaceWaterConfig, inputs: &Inputs<'_>) -> Result<RoutingDomain, i32> {
    validate_config(config)?;
    let cell_count = inputs.cell_ids.len();
    let candidate_count = inputs.candidate_ids.len();
    if cell_count == 0
        || candidate_count == 0
        || inputs.receiver_ids.len() != cell_count
        || inputs.depression_ids.len() != cell_count
        || inputs.source_active.len() != cell_count
        || inputs.area_km2.len() != cell_count
        || inputs.terrain_elevation_m.len() != cell_count
        || inputs.hydrologic_elevation_m.len() != cell_count
        || inputs.parent_relief_m.len() != cell_count
        || inputs.monthly_runoff_mm.len() != cell_count * MONTHS
        || inputs.monthly_evaporation_mm.len() != cell_count * MONTHS
        || inputs.sediment_accommodation.len() != cell_count
        || inputs.spill_receiver_ids.len() != candidate_count
    {
        return Err(1);
    }

    let mut row_by_cell_id = HashMap::with_capacity(cell_count);
    for row in 0..cell_count {
        if inputs.cell_ids[row] < 0
            || row_by_cell_id.insert(inputs.cell_ids[row], row).is_some()
            || inputs.receiver_ids[row] < -2
            || inputs.source_active[row] > 1
            || !inputs.area_km2[row].is_finite()
            || inputs.area_km2[row] <= 0.0
            || !inputs.terrain_elevation_m[row].is_finite()
            || !inputs.hydrologic_elevation_m[row].is_finite()
            || !inputs.parent_relief_m[row].is_finite()
            || inputs.parent_relief_m[row] < 0.0
            || !inputs.sediment_accommodation[row].is_finite()
            || !(0.0..=1.0).contains(&inputs.sediment_accommodation[row])
        {
            return Err(2);
        }
        for month in 0..MONTHS {
            let runoff = inputs.monthly_runoff_mm[month * cell_count + row];
            let evaporation = inputs.monthly_evaporation_mm[month * cell_count + row];
            if !runoff.is_finite() || runoff < 0.0 || !evaporation.is_finite() || evaporation < 0.0
            {
                return Err(2);
            }
        }
    }
    for &receiver in inputs.receiver_ids {
        if receiver >= 0 && !row_by_cell_id.contains_key(&receiver) {
            return Err(2);
        }
    }

    let mut candidate_by_id = HashMap::with_capacity(candidate_count);
    for (index, &candidate_id) in inputs.candidate_ids.iter().enumerate() {
        if candidate_id < 0 || candidate_by_id.insert(candidate_id, index).is_some() {
            return Err(3);
        }
        let spill_receiver = inputs.spill_receiver_ids[index];
        if spill_receiver < -2
            || (spill_receiver >= 0 && !row_by_cell_id.contains_key(&spill_receiver))
        {
            return Err(3);
        }
    }
    let mut membership_count = vec![0usize; candidate_count];
    for row in 0..cell_count {
        let depression_id = inputs.depression_ids[row];
        if depression_id < 0 {
            continue;
        }
        let Some(&candidate) = candidate_by_id.get(&depression_id) else {
            return Err(3);
        };
        if inputs.source_active[row] == 0
            || inputs.hydrologic_elevation_m[row] <= inputs.terrain_elevation_m[row]
        {
            return Err(3);
        }
        membership_count[candidate] += 1;
    }
    if membership_count.contains(&0) {
        return Err(3);
    }

    let mut upstream_count = vec![0usize; cell_count];
    for &receiver in inputs.receiver_ids {
        if receiver >= 0 {
            upstream_count[row_by_cell_id[&receiver]] += 1;
        }
    }
    let mut ready = BinaryHeap::new();
    for (row, &count) in upstream_count.iter().enumerate() {
        if count == 0 {
            ready.push(Reverse((inputs.cell_ids[row], row)));
        }
    }
    let mut cell_order = Vec::with_capacity(cell_count);
    while let Some(Reverse((_, row))) = ready.pop() {
        cell_order.push(row);
        let receiver = inputs.receiver_ids[row];
        if receiver >= 0 {
            let target = row_by_cell_id[&receiver];
            upstream_count[target] -= 1;
            if upstream_count[target] == 0 {
                ready.push(Reverse((inputs.cell_ids[target], target)));
            }
        }
    }
    if cell_order.len() != cell_count {
        return Err(4);
    }

    let mut owner = vec![-1i32; cell_count];
    for &row in cell_order.iter().rev() {
        let depression_id = inputs.depression_ids[row];
        if depression_id >= 0 {
            owner[row] = candidate_by_id[&depression_id] as i32;
        } else if inputs.receiver_ids[row] >= 0 {
            owner[row] = owner[row_by_cell_id[&inputs.receiver_ids[row]]];
        }
    }

    let mut downstream = vec![-1i32; candidate_count];
    let mut candidate_upstream_count = vec![0usize; candidate_count];
    for candidate in 0..candidate_count {
        let spill_receiver = inputs.spill_receiver_ids[candidate];
        if spill_receiver >= 0 {
            downstream[candidate] = owner[row_by_cell_id[&spill_receiver]];
        }
        if downstream[candidate] == candidate as i32 {
            return Err(4);
        }
        if downstream[candidate] >= 0 {
            candidate_upstream_count[downstream[candidate] as usize] += 1;
        }
    }
    let mut candidate_ready = BinaryHeap::new();
    for (candidate, &count) in candidate_upstream_count.iter().enumerate() {
        if count == 0 {
            candidate_ready.push(Reverse((inputs.candidate_ids[candidate], candidate)));
        }
    }
    let mut candidate_order = Vec::with_capacity(candidate_count);
    while let Some(Reverse((_, candidate))) = candidate_ready.pop() {
        candidate_order.push(candidate);
        let target = downstream[candidate];
        if target >= 0 {
            let target = target as usize;
            candidate_upstream_count[target] -= 1;
            if candidate_upstream_count[target] == 0 {
                candidate_ready.push(Reverse((inputs.candidate_ids[target], target)));
            }
        }
    }
    if candidate_order.len() != candidate_count {
        return Err(4);
    }

    Ok(RoutingDomain {
        candidate_by_id,
        owner,
        downstream,
        candidate_order,
    })
}

fn inundation(mean_m: f64, span_m: f64, surface_m: f64, maximum_fraction: f64) -> (f64, f64) {
    let bottom_m = mean_m - 0.5 * span_m;
    let height_m = (surface_m - bottom_m).clamp(0.0, span_m);
    let fraction = (height_m / span_m).min(maximum_fraction);
    let equivalent_depth_m = (fraction * height_m - 0.5 * span_m * fraction * fraction).max(0.0);
    (fraction, equivalent_depth_m)
}

fn build_hypsometry(
    config: SurfaceWaterConfig,
    inputs: &Inputs<'_>,
    members: &[usize],
) -> Result<Hypsometry, i32> {
    let relief_divisor = (config.refinement_factor as f64).sqrt();
    let spans = members
        .iter()
        .map(|&row| {
            (inputs.parent_relief_m[row] as f64 / relief_divisor * config.subgrid_relief_scale)
                .max(config.minimum_subgrid_relief_m)
        })
        .collect::<Vec<_>>();
    let cap_m = inputs.hydrologic_elevation_m[members[0]];
    if members
        .iter()
        .any(|&row| (inputs.hydrologic_elevation_m[row] - cap_m).abs() > 1e-6)
    {
        return Err(3);
    }
    let bottom_m = members
        .iter()
        .zip(&spans)
        .map(|(&row, &span)| inputs.terrain_elevation_m[row] - 0.5 * span)
        .fold(cap_m, f64::min);
    if cap_m <= bottom_m {
        return Err(3);
    }

    let mut volume_km3 = [0.0; HYPSOMETRY_BINS];
    let mut area_km2 = [0.0; HYPSOMETRY_BINS];
    let mut evaporation_area = [[0.0; MONTHS]; HYPSOMETRY_BINS];
    let mut seepage_area_km2 = [0.0; HYPSOMETRY_BINS];
    for bin in 0..HYPSOMETRY_BINS {
        let fraction = bin as f64 / (HYPSOMETRY_BINS - 1) as f64;
        let level_m = bottom_m + fraction * (cap_m - bottom_m);
        for (member_index, &row) in members.iter().enumerate() {
            let (cell_fraction, equivalent_depth_m) = inundation(
                inputs.terrain_elevation_m[row],
                spans[member_index],
                level_m,
                config.maximum_connected_inundation_fraction,
            );
            let covered_area = cell_fraction * inputs.area_km2[row];
            area_km2[bin] += covered_area;
            volume_km3[bin] += equivalent_depth_m * inputs.area_km2[row] / 1_000.0;
            let seepage_factor = 1.0 - 0.65 * inputs.sediment_accommodation[row] as f64;
            seepage_area_km2[bin] += covered_area * seepage_factor;
            for (month, value) in evaporation_area[bin].iter_mut().enumerate() {
                *value += covered_area
                    * inputs.monthly_evaporation_mm[month * inputs.cell_ids.len() + row] as f64;
            }
        }
    }
    if volume_km3[HYPSOMETRY_BINS - 1] <= 0.0
        || area_km2[HYPSOMETRY_BINS - 1] <= 0.0
        || volume_km3.windows(2).any(|pair| pair[1] < pair[0])
        || area_km2.windows(2).any(|pair| pair[1] < pair[0])
    {
        return Err(3);
    }
    let potential_fraction_by_member = members
        .iter()
        .enumerate()
        .map(|(member_index, &row)| {
            inundation(
                inputs.terrain_elevation_m[row],
                spans[member_index],
                cap_m,
                config.maximum_connected_inundation_fraction,
            )
            .0 as f32
        })
        .collect();
    Ok(Hypsometry {
        bottom_m,
        cap_m,
        volume_km3,
        area_km2,
        evaporation_area,
        seepage_area_km2,
        potential_fraction_by_member,
    })
}

fn curve_position(curve: &Hypsometry, storage_km3: f64) -> (usize, f64) {
    let capacity = curve.volume_km3[HYPSOMETRY_BINS - 1];
    let storage = storage_km3.clamp(0.0, capacity);
    let upper = curve
        .volume_km3
        .partition_point(|value| *value < storage)
        .clamp(1, HYPSOMETRY_BINS - 1);
    let lower = upper - 1;
    let width = curve.volume_km3[upper] - curve.volume_km3[lower];
    let fraction = if width <= f64::EPSILON {
        0.0
    } else {
        (storage - curve.volume_km3[lower]) / width
    };
    (lower, fraction.clamp(0.0, 1.0))
}

fn interpolate(values: &[f64; HYPSOMETRY_BINS], lower: usize, fraction: f64) -> f64 {
    values[lower] + fraction * (values[lower + 1] - values[lower])
}

fn curve_properties(curve: &Hypsometry, storage_km3: f64, month: usize) -> (f64, f64, f64) {
    if storage_km3 <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let (lower, fraction) = curve_position(curve, storage_km3);
    let evaporation_area = curve.evaporation_area[lower][month]
        + fraction
            * (curve.evaporation_area[lower + 1][month] - curve.evaporation_area[lower][month]);
    (
        interpolate(&curve.area_km2, lower, fraction),
        evaporation_area,
        interpolate(&curve.seepage_area_km2, lower, fraction),
    )
}

fn curve_level(curve: &Hypsometry, storage_km3: f64) -> f64 {
    if storage_km3 <= 0.0 {
        return curve.bottom_m;
    }
    let (lower, fraction) = curve_position(curve, storage_km3);
    let lower_fraction = lower as f64 / (HYPSOMETRY_BINS - 1) as f64;
    let upper_fraction = (lower + 1) as f64 / (HYPSOMETRY_BINS - 1) as f64;
    curve.bottom_m
        + (lower_fraction + fraction * (upper_fraction - lower_fraction))
            * (curve.cap_m - curve.bottom_m)
}

fn refresh_exact_monthly_areas(
    config: SurfaceWaterConfig,
    inputs: &Inputs<'_>,
    states: &mut [CandidateState],
) {
    let relief_divisor = (config.refinement_factor as f64).sqrt();
    for state in states {
        for month in 0..MONTHS {
            let level = curve_level(&state.hypsometry, state.monthly_storage_km3[month]);
            state.monthly_water_area_km2[month] = state
                .members
                .iter()
                .map(|&row| {
                    let span = (inputs.parent_relief_m[row] as f64 / relief_divisor
                        * config.subgrid_relief_scale)
                        .max(config.minimum_subgrid_relief_m);
                    inundation(
                        inputs.terrain_elevation_m[row],
                        span,
                        level,
                        config.maximum_connected_inundation_fraction,
                    )
                    .0 * inputs.area_km2[row]
                })
                .sum();
        }
    }
}

fn initialize_states(
    config: SurfaceWaterConfig,
    inputs: &Inputs<'_>,
    domain: &RoutingDomain,
) -> Result<Vec<CandidateState>, i32> {
    let candidate_count = inputs.candidate_ids.len();
    let mut members = vec![Vec::new(); candidate_count];
    for (row, &depression_id) in inputs.depression_ids.iter().enumerate() {
        if depression_id >= 0 {
            members[domain.candidate_by_id[&depression_id]].push(row);
        }
    }
    let mut states = Vec::with_capacity(candidate_count);
    for candidate_members in members {
        let hypsometry = build_hypsometry(config, inputs, &candidate_members)?;
        states.push(CandidateState {
            members: candidate_members,
            hypsometry,
            direct_inflow_km3: [0.0; MONTHS],
            catchment_cell_count: 0,
            catchment_area_km2: 0.0,
            monthly_upstream_inflow_km3: [0.0; MONTHS],
            monthly_total_inflow_km3: [0.0; MONTHS],
            monthly_evaporation_km3: [0.0; MONTHS],
            monthly_seepage_km3: [0.0; MONTHS],
            monthly_overflow_km3: [0.0; MONTHS],
            monthly_storage_km3: [0.0; MONTHS],
            monthly_water_area_km2: [0.0; MONTHS],
            periodic_start_storage_km3: 0.0,
            solver_iterations: 0,
        });
    }

    for row in 0..inputs.cell_ids.len() {
        if inputs.source_active[row] == 0 || domain.owner[row] < 0 {
            continue;
        }
        let candidate = domain.owner[row] as usize;
        states[candidate].catchment_cell_count += 1;
        states[candidate].catchment_area_km2 += inputs.area_km2[row];
        for month in 0..MONTHS {
            states[candidate].direct_inflow_km3[month] +=
                inputs.monthly_runoff_mm[month * inputs.cell_ids.len() + row] as f64
                    * inputs.area_km2[row]
                    / 1_000_000.0;
        }
    }

    Ok(states)
}

fn simulate_candidate_year(
    config: SurfaceWaterConfig,
    state: &mut CandidateState,
    upstream_inflow: &[f64; MONTHS],
    initial_storage_km3: f64,
    persist: bool,
) -> f64 {
    if persist {
        state.periodic_start_storage_km3 = initial_storage_km3;
        state.monthly_upstream_inflow_km3 = [0.0; MONTHS];
        state.monthly_total_inflow_km3 = [0.0; MONTHS];
        state.monthly_evaporation_km3 = [0.0; MONTHS];
        state.monthly_seepage_km3 = [0.0; MONTHS];
        state.monthly_overflow_km3 = [0.0; MONTHS];
        state.monthly_storage_km3 = [0.0; MONTHS];
        state.monthly_water_area_km2 = [0.0; MONTHS];
    }
    let mut storage_km3 = initial_storage_km3;
    for (month, &upstream) in upstream_inflow.iter().enumerate() {
        let direct = state.direct_inflow_km3[month];
        let total_inflow = direct + upstream;
        let available = storage_km3 + total_inflow;
        let capacity = state.hypsometry.volume_km3[HYPSOMETRY_BINS - 1];
        let provisional_storage = available.min(capacity);
        let loss_storage = 0.5 * (storage_km3 + provisional_storage);
        let (_, evaporation_area, seepage_area) =
            curve_properties(&state.hypsometry, loss_storage, month);
        let mut evaporation = evaporation_area * config.open_water_evaporation_factor / 1_000_000.0;
        let mut seepage = seepage_area * config.seepage_mm_year / MONTHS as f64 / 1_000_000.0;
        let demanded_loss = evaporation + seepage;
        if demanded_loss > available && demanded_loss > 0.0 {
            let scale = available / demanded_loss;
            evaporation *= scale;
            seepage *= scale;
        }
        let remaining = (available - evaporation - seepage).max(0.0);
        let overflow = (remaining - capacity).max(0.0);
        storage_km3 = remaining.min(capacity);
        if persist {
            let (water_area, _, _) = curve_properties(&state.hypsometry, storage_km3, month);
            state.monthly_upstream_inflow_km3[month] = upstream;
            state.monthly_total_inflow_km3[month] = total_inflow;
            state.monthly_evaporation_km3[month] = evaporation;
            state.monthly_seepage_km3[month] = seepage;
            state.monthly_overflow_km3[month] = overflow;
            state.monthly_storage_km3[month] = storage_km3;
            state.monthly_water_area_km2[month] = water_area;
        }
    }
    storage_km3
}

fn solve_periodic_states(
    config: SurfaceWaterConfig,
    states: &mut [CandidateState],
    downstream: &[i32],
    candidate_order: &[usize],
) -> bool {
    let mut upstream_inflow = vec![[0.0f64; MONTHS]; states.len()];
    let mut all_converged = true;
    for &candidate in candidate_order {
        let state = &mut states[candidate];
        let capacity = state.hypsometry.volume_km3[HYPSOMETRY_BINS - 1];
        let tolerance = config.convergence_tolerance_fraction * capacity.max(1e-12);
        let low_end =
            simulate_candidate_year(config, state, &upstream_inflow[candidate], 0.0, false);
        let high_end =
            simulate_candidate_year(config, state, &upstream_inflow[candidate], capacity, false);
        let initial_storage = if low_end.abs() <= tolerance {
            0.0
        } else if (high_end - capacity).abs() <= tolerance {
            capacity
        } else {
            let mut low = 0.0;
            let mut high = capacity;
            let mut solution = 0.5 * capacity;
            for iteration in 1..=config.maximum_solver_iterations as usize {
                let midpoint = 0.5 * (low + high);
                let end = simulate_candidate_year(
                    config,
                    state,
                    &upstream_inflow[candidate],
                    midpoint,
                    false,
                );
                let residual = end - midpoint;
                solution = midpoint;
                state.solver_iterations = iteration;
                if iteration >= config.minimum_solver_iterations as usize
                    && residual.abs() <= tolerance
                {
                    break;
                }
                if residual > 0.0 {
                    low = midpoint;
                } else {
                    high = midpoint;
                }
            }
            solution
        };
        if state.solver_iterations == 0 {
            state.solver_iterations = 1;
        }
        let final_storage = simulate_candidate_year(
            config,
            state,
            &upstream_inflow[candidate],
            initial_storage,
            true,
        );
        let residual = final_storage - initial_storage;
        if residual.abs() > tolerance {
            all_converged = false;
        }
        let target = downstream[candidate];
        if target >= 0 {
            for (target_inflow, overflow) in upstream_inflow[target as usize]
                .iter_mut()
                .zip(state.monthly_overflow_km3)
            {
                *target_inflow += overflow;
            }
        }
    }
    all_converged
}

fn classify_candidate(config: SurfaceWaterConfig, state: &CandidateState) -> (i32, usize) {
    let potential_area = state.hypsometry.area_km2[HYPSOMETRY_BINS - 1];
    let wet_month_count = state
        .monthly_water_area_km2
        .iter()
        .filter(|area| **area / potential_area.max(1e-12) >= config.minimum_wet_area_fraction)
        .count();
    let total_area = state.monthly_water_area_km2.iter().sum::<f64>();
    let mean_depth_m = if total_area > 0.0 {
        state.monthly_storage_km3.iter().sum::<f64>() * 1_000.0 / total_area
    } else {
        0.0
    };
    let class_code = if wet_month_count == 0 {
        CLASS_DRY
    } else if wet_month_count <= config.transient_max_months as usize {
        CLASS_TRANSIENT
    } else if mean_depth_m <= config.wetland_max_mean_depth_m {
        CLASS_HYDROLOGIC_WETLAND
    } else if wet_month_count >= config.permanent_min_months as usize {
        CLASS_PERMANENT_LAKE
    } else {
        CLASS_SEASONAL_LAKE
    };
    (class_code, wet_month_count)
}

fn run_balance(config: SurfaceWaterConfig, inputs: &Inputs<'_>) -> Result<Outcome, i32> {
    let domain = validate_inputs(config, inputs)?;
    let mut states = initialize_states(config, inputs, &domain)?;
    let convergence_valid = solve_periodic_states(
        config,
        &mut states,
        &domain.downstream,
        &domain.candidate_order,
    );
    refresh_exact_monthly_areas(config, inputs, &mut states);

    let mut candidates = Vec::with_capacity(states.len());
    let mut class_counts = [0usize; 5];
    let mut class_areas = [0.0f64; 5];
    let mut annual_direct_total = 0.0;
    let mut annual_evaporation_total = 0.0;
    let mut annual_seepage_total = 0.0;
    let mut annual_terminal_overflow_total = 0.0;
    let mut annual_storage_change_total = 0.0;
    for (candidate, state) in states.iter().enumerate() {
        let (class_code, wet_month_count) = classify_candidate(config, state);
        class_counts[class_code as usize] += 1;
        let annual_direct = state.direct_inflow_km3.iter().sum::<f64>();
        let annual_upstream = state.monthly_upstream_inflow_km3.iter().sum::<f64>();
        let annual_total = state.monthly_total_inflow_km3.iter().sum::<f64>();
        let annual_evaporation = state.monthly_evaporation_km3.iter().sum::<f64>();
        let annual_seepage = state.monthly_seepage_km3.iter().sum::<f64>();
        let annual_overflow = state.monthly_overflow_km3.iter().sum::<f64>();
        let annual_terminal_overflow = if domain.downstream[candidate] < 0 {
            annual_overflow
        } else {
            0.0
        };
        let annual_storage_change =
            state.monthly_storage_km3[MONTHS - 1] - state.periodic_start_storage_km3;
        let residual = annual_total
            - annual_evaporation
            - annual_seepage
            - annual_overflow
            - annual_storage_change;
        let minimum_area = state
            .monthly_water_area_km2
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let maximum_area = state
            .monthly_water_area_km2
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let mean_area = state.monthly_water_area_km2.iter().sum::<f64>() / MONTHS as f64;
        class_areas[class_code as usize] += mean_area;
        let total_monthly_area = state.monthly_water_area_km2.iter().sum::<f64>();
        let mean_wetted_depth = if total_monthly_area > 0.0 {
            state.monthly_storage_km3.iter().sum::<f64>() * 1_000.0 / total_monthly_area
        } else {
            0.0
        };
        let maximum_mean_depth = (0..MONTHS)
            .map(|month| {
                if state.monthly_water_area_km2[month] > 0.0 {
                    state.monthly_storage_km3[month] * 1_000.0 / state.monthly_water_area_km2[month]
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max);
        let converged = (annual_storage_change.abs()
            / state.hypsometry.volume_km3[HYPSOMETRY_BINS - 1].max(1e-12))
            <= config.convergence_tolerance_fraction;
        candidates.push(SurfaceWaterCandidateRecord {
            depression_id: inputs.candidate_ids[candidate],
            downstream_depression_id: if domain.downstream[candidate] >= 0 {
                inputs.candidate_ids[domain.downstream[candidate] as usize]
            } else {
                -1
            },
            class_code,
            cell_count: state.members.len() as i32,
            catchment_cell_count: state.catchment_cell_count as i32,
            wet_month_count: wet_month_count as i32,
            solver_iterations: state.solver_iterations as i32,
            converged: i32::from(converged),
            open_outlet: i32::from(annual_overflow > 1e-12),
            catchment_area_km2: state.catchment_area_km2,
            potential_water_area_km2: state.hypsometry.area_km2[HYPSOMETRY_BINS - 1],
            storage_capacity_km3: state.hypsometry.volume_km3[HYPSOMETRY_BINS - 1],
            annual_direct_inflow_km3: annual_direct,
            annual_upstream_inflow_km3: annual_upstream,
            annual_total_inflow_km3: annual_total,
            annual_evaporation_km3: annual_evaporation,
            annual_seepage_km3: annual_seepage,
            annual_overflow_km3: annual_overflow,
            annual_terminal_overflow_km3: annual_terminal_overflow,
            annual_storage_change_km3: annual_storage_change,
            water_balance_residual_km3: residual,
            hydroperiod_fraction: wet_month_count as f64 / MONTHS as f64,
            minimum_water_area_km2: minimum_area,
            mean_water_area_km2: mean_area,
            maximum_water_area_km2: maximum_area,
            mean_wetted_depth_m: mean_wetted_depth,
            maximum_mean_depth_m: maximum_mean_depth,
            salinity_index: ((annual_evaporation + annual_seepage) / annual_total.max(1e-12))
                .clamp(0.0, 10.0),
            monthly_direct_inflow_km3: state.direct_inflow_km3,
            monthly_upstream_inflow_km3: state.monthly_upstream_inflow_km3,
            monthly_total_inflow_km3: state.monthly_total_inflow_km3,
            monthly_evaporation_km3: state.monthly_evaporation_km3,
            monthly_seepage_km3: state.monthly_seepage_km3,
            monthly_overflow_km3: state.monthly_overflow_km3,
            monthly_storage_km3: state.monthly_storage_km3,
            monthly_water_area_km2: state.monthly_water_area_km2,
        });
        annual_direct_total += annual_direct;
        annual_evaporation_total += annual_evaporation;
        annual_seepage_total += annual_seepage;
        annual_terminal_overflow_total += annual_terminal_overflow;
        annual_storage_change_total += annual_storage_change;
    }

    let mut cells = Vec::new();
    let relief_divisor = (config.refinement_factor as f64).sqrt();
    let mut minimum_fraction = 1.0f64;
    let mut maximum_fraction = 0.0f64;
    let mut fraction_valid = true;
    for (candidate, state) in states.iter().enumerate() {
        let class_code = candidates[candidate].class_code;
        for (member_index, &row) in state.members.iter().enumerate() {
            let span = (inputs.parent_relief_m[row] as f64 / relief_divisor
                * config.subgrid_relief_scale)
                .max(config.minimum_subgrid_relief_m);
            let mut monthly = [0.0f32; MONTHS];
            for (month, value) in monthly.iter_mut().enumerate() {
                let level = curve_level(&state.hypsometry, state.monthly_storage_km3[month]);
                *value = inundation(
                    inputs.terrain_elevation_m[row],
                    span,
                    level,
                    config.maximum_connected_inundation_fraction,
                )
                .0 as f32;
                minimum_fraction = minimum_fraction.min(*value as f64);
                maximum_fraction = maximum_fraction.max(*value as f64);
                fraction_valid &= value.is_finite() && (0.0..=1.0).contains(value);
            }
            let minimum = monthly.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = monthly.iter().copied().fold(0.0f32, f32::max);
            let mean =
                monthly.iter().map(|value| *value as f64).sum::<f64>() as f32 / MONTHS as f32;
            cells.push(SurfaceWaterCellRecord {
                fine_cell_id: inputs.cell_ids[row],
                depression_id: inputs.candidate_ids[candidate],
                class_code,
                potential_inundation_fraction: state.hypsometry.potential_fraction_by_member
                    [member_index],
                minimum_inundation_fraction: minimum,
                mean_inundation_fraction: mean,
                maximum_inundation_fraction: maximum,
                monthly_inundation_fraction: monthly,
            });
        }
    }
    cells.sort_by_key(|record| record.fine_cell_id);

    let active_source_area = (0..inputs.cell_ids.len())
        .filter(|&row| inputs.source_active[row] != 0)
        .map(|row| inputs.area_km2[row])
        .sum::<f64>();
    let owned_source_cell_count = (0..inputs.cell_ids.len())
        .filter(|&row| inputs.source_active[row] != 0 && domain.owner[row] >= 0)
        .count();
    let owned_catchment_area = states
        .iter()
        .map(|state| state.catchment_area_km2)
        .sum::<f64>();
    let potential_water_area = states
        .iter()
        .map(|state| state.hypsometry.area_km2[HYPSOMETRY_BINS - 1])
        .sum::<f64>();
    let storage_capacity = states
        .iter()
        .map(|state| state.hypsometry.volume_km3[HYPSOMETRY_BINS - 1])
        .sum::<f64>();
    let residual = annual_direct_total
        - annual_evaporation_total
        - annual_seepage_total
        - annual_terminal_overflow_total
        - annual_storage_change_total;
    let scale = annual_direct_total
        .max(annual_evaporation_total + annual_seepage_total + annual_terminal_overflow_total)
        .max(1e-12);
    let storage_valid = states.iter().all(|state| {
        let capacity = state.hypsometry.volume_km3[HYPSOMETRY_BINS - 1];
        state.monthly_storage_km3.iter().all(|storage| {
            storage.is_finite() && *storage >= -1e-12 && *storage <= capacity + 1e-12
        })
    });
    let stats = SurfaceWaterStats {
        cell_count: inputs.cell_ids.len() as i32,
        candidate_count: states.len() as i32,
        candidate_cell_count: cells.len() as i32,
        owned_source_cell_count: owned_source_cell_count as i32,
        dry_count: class_counts[CLASS_DRY as usize] as i32,
        transient_count: class_counts[CLASS_TRANSIENT as usize] as i32,
        seasonal_lake_count: class_counts[CLASS_SEASONAL_LAKE as usize] as i32,
        permanent_lake_count: class_counts[CLASS_PERMANENT_LAKE as usize] as i32,
        hydrologic_wetland_count: class_counts[CLASS_HYDROLOGIC_WETLAND as usize] as i32,
        graph_valid: 1,
        convergence_valid: i32::from(convergence_valid),
        fraction_valid: i32::from(fraction_valid),
        storage_valid: i32::from(storage_valid),
        direct_catchment_valid: i32::from(
            states
                .iter()
                .map(|state| state.catchment_cell_count)
                .sum::<usize>()
                == owned_source_cell_count,
        ),
        maximum_solver_iterations_used: states
            .iter()
            .map(|state| state.solver_iterations)
            .max()
            .unwrap_or(0) as i32,
        active_source_area_km2: active_source_area,
        owned_catchment_area_km2: owned_catchment_area,
        potential_water_area_km2: potential_water_area,
        storage_capacity_km3: storage_capacity,
        annual_direct_inflow_km3: annual_direct_total,
        annual_evaporation_km3: annual_evaporation_total,
        annual_seepage_km3: annual_seepage_total,
        annual_terminal_overflow_km3: annual_terminal_overflow_total,
        annual_storage_change_km3: annual_storage_change_total,
        water_balance_residual_km3: residual,
        water_balance_relative_error: residual.abs() / scale,
        minimum_inundation_fraction: minimum_fraction,
        maximum_inundation_fraction: maximum_fraction,
        dry_mean_water_area_km2: class_areas[CLASS_DRY as usize],
        transient_mean_water_area_km2: class_areas[CLASS_TRANSIENT as usize],
        seasonal_lake_mean_water_area_km2: class_areas[CLASS_SEASONAL_LAKE as usize],
        permanent_lake_mean_water_area_km2: class_areas[CLASS_PERMANENT_LAKE as usize],
        hydrologic_wetland_mean_water_area_km2: class_areas[CLASS_HYDROLOGIC_WETLAND as usize],
    };
    Ok(Outcome {
        candidates,
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
/// Run the refined monthly surface-water balance.
///
/// # Safety
///
/// Every input pointer must reference a contiguous buffer of the declared
/// length. Output pointers must be valid and may not alias input buffers.
pub unsafe extern "C" fn surface_water_run(
    config: SurfaceWaterConfig,
    cell_count: usize,
    candidate_count: usize,
    cell_ids: *const i32,
    receiver_ids: *const i32,
    depression_ids: *const i32,
    source_active: *const u8,
    area_km2: *const f64,
    terrain_elevation_m: *const f64,
    hydrologic_elevation_m: *const f64,
    parent_relief_m: *const f32,
    monthly_runoff_mm: *const f32,
    monthly_evaporation_mm: *const f32,
    sediment_accommodation: *const f32,
    candidate_ids: *const i32,
    spill_receiver_ids: *const i32,
    candidates_out: *mut SurfaceWaterCandidateArray,
    cells_out: *mut SurfaceWaterCellArray,
    stats_out: *mut SurfaceWaterStats,
) -> i32 {
    if cell_count == 0
        || candidate_count == 0
        || cell_ids.is_null()
        || receiver_ids.is_null()
        || depression_ids.is_null()
        || source_active.is_null()
        || area_km2.is_null()
        || terrain_elevation_m.is_null()
        || hydrologic_elevation_m.is_null()
        || parent_relief_m.is_null()
        || monthly_runoff_mm.is_null()
        || monthly_evaporation_mm.is_null()
        || sediment_accommodation.is_null()
        || candidate_ids.is_null()
        || spill_receiver_ids.is_null()
        || candidates_out.is_null()
        || cells_out.is_null()
        || stats_out.is_null()
    {
        return 5;
    }
    let inputs = Inputs {
        cell_ids: unsafe { read_slice(cell_ids, cell_count) },
        receiver_ids: unsafe { read_slice(receiver_ids, cell_count) },
        depression_ids: unsafe { read_slice(depression_ids, cell_count) },
        source_active: unsafe { read_slice(source_active, cell_count) },
        area_km2: unsafe { read_slice(area_km2, cell_count) },
        terrain_elevation_m: unsafe { read_slice(terrain_elevation_m, cell_count) },
        hydrologic_elevation_m: unsafe { read_slice(hydrologic_elevation_m, cell_count) },
        parent_relief_m: unsafe { read_slice(parent_relief_m, cell_count) },
        monthly_runoff_mm: unsafe { read_slice(monthly_runoff_mm, cell_count * MONTHS) },
        monthly_evaporation_mm: unsafe { read_slice(monthly_evaporation_mm, cell_count * MONTHS) },
        sediment_accommodation: unsafe { read_slice(sediment_accommodation, cell_count) },
        candidate_ids: unsafe { read_slice(candidate_ids, candidate_count) },
        spill_receiver_ids: unsafe { read_slice(spill_receiver_ids, candidate_count) },
    };
    match run_balance(config, &inputs) {
        Ok(outcome) => {
            let (candidate_data, candidate_len) = into_raw_array(outcome.candidates);
            let (cell_data, cell_len) = into_raw_array(outcome.cells);
            unsafe {
                *candidates_out = SurfaceWaterCandidateArray {
                    data: candidate_data,
                    len: candidate_len,
                };
                *cells_out = SurfaceWaterCellArray {
                    data: cell_data,
                    len: cell_len,
                };
                *stats_out = outcome.stats;
            }
            0
        }
        Err(status) => status,
    }
}

#[no_mangle]
pub extern "C" fn surface_water_free_candidates(array: SurfaceWaterCandidateArray) {
    free_raw_array(array.data, array.len);
}

#[no_mangle]
pub extern "C" fn surface_water_free_cells(array: SurfaceWaterCellArray) {
    free_raw_array(array.data, array.len);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SurfaceWaterConfig {
        SurfaceWaterConfig {
            refinement_factor: 4,
            minimum_solver_iterations: 4,
            maximum_solver_iterations: 64,
            transient_max_months: 3,
            permanent_min_months: 12,
            convergence_tolerance_fraction: 1e-8,
            open_water_evaporation_factor: 1.0,
            seepage_mm_year: 0.0,
            subgrid_relief_scale: 1.0,
            minimum_subgrid_relief_m: 10.0,
            maximum_connected_inundation_fraction: 1.0,
            minimum_wet_area_fraction: 0.01,
            wetland_max_mean_depth_m: 3.0,
        }
    }

    #[test]
    fn rejects_ambiguous_classification_thresholds() {
        let mut invalid_wet_area = config();
        invalid_wet_area.minimum_wet_area_fraction = 0.0;
        assert_eq!(validate_config(invalid_wet_area), Err(1));

        let mut invalid_permanence = config();
        invalid_permanence.permanent_min_months = 11;
        assert_eq!(validate_config(invalid_permanence), Err(1));
    }

    #[test]
    fn exported_struct_sizes_match_rust_layouts() {
        assert_eq!(
            surface_water_native_struct_size(0),
            mem::size_of::<SurfaceWaterConfig>()
        );
        assert_eq!(
            surface_water_native_struct_size(1),
            mem::size_of::<SurfaceWaterCandidateRecord>()
        );
        assert_eq!(
            surface_water_native_struct_size(2),
            mem::size_of::<SurfaceWaterCellRecord>()
        );
        assert_eq!(
            surface_water_native_struct_size(3),
            mem::size_of::<SurfaceWaterStats>()
        );
        assert_eq!(surface_water_native_struct_size(99), 0);
    }

    #[test]
    fn wet_candidate_fills_spills_and_conserves_water() {
        let cell_ids = [0, 1, 2];
        let receiver_ids = [1, 2, -1];
        let depression_ids = [-1, 1, -1];
        let runoff = [100.0f32; MONTHS * 3];
        let evaporation = [0.0f32; MONTHS * 3];
        let inputs = Inputs {
            cell_ids: &cell_ids,
            receiver_ids: &receiver_ids,
            depression_ids: &depression_ids,
            source_active: &[1, 1, 1],
            area_km2: &[1.0; 3],
            terrain_elevation_m: &[2.0, -10.0, 2.0],
            hydrologic_elevation_m: &[2.0, 0.0, 2.0],
            parent_relief_m: &[20.0; 3],
            monthly_runoff_mm: &runoff,
            monthly_evaporation_mm: &evaporation,
            sediment_accommodation: &[0.5; 3],
            candidate_ids: &[1],
            spill_receiver_ids: &[2],
        };
        let outcome = run_balance(config(), &inputs).expect("valid balance");
        assert_eq!(outcome.stats.graph_valid, 1);
        assert_eq!(outcome.stats.convergence_valid, 1);
        assert_eq!(outcome.candidates[0].class_code, CLASS_PERMANENT_LAKE);
        assert!(outcome.candidates[0].annual_overflow_km3 > 0.0);
        assert!(outcome.stats.water_balance_relative_error < 1e-12);
        assert_eq!(outcome.cells.len(), 1);
        assert!(outcome.cells[0].mean_inundation_fraction > 0.0);
    }

    #[test]
    fn candidate_without_runoff_is_dry() {
        let cell_ids = [0, 1];
        let receiver_ids = [1, -1];
        let depression_ids = [5, -1];
        let runoff = [0.0f32; MONTHS * 2];
        let evaporation = [100.0f32; MONTHS * 2];
        let inputs = Inputs {
            cell_ids: &cell_ids,
            receiver_ids: &receiver_ids,
            depression_ids: &depression_ids,
            source_active: &[1, 1],
            area_km2: &[1.0; 2],
            terrain_elevation_m: &[-5.0, 2.0],
            hydrologic_elevation_m: &[0.0, 2.0],
            parent_relief_m: &[20.0; 2],
            monthly_runoff_mm: &runoff,
            monthly_evaporation_mm: &evaporation,
            sediment_accommodation: &[0.0; 2],
            candidate_ids: &[5],
            spill_receiver_ids: &[1],
        };
        let outcome = run_balance(config(), &inputs).expect("valid dry balance");
        assert_eq!(outcome.candidates[0].class_code, CLASS_DRY);
        assert_eq!(outcome.candidates[0].mean_water_area_km2, 0.0);
    }

    #[test]
    fn upstream_overflow_enters_downstream_candidate_once() {
        let cell_ids = [0, 1, 2, 3, 4];
        let receiver_ids = [1, 2, 3, 4, -1];
        let depression_ids = [-1, 11, -1, 33, -1];
        let mut runoff = [0.0f32; MONTHS * 5];
        for month in 0..MONTHS {
            runoff[month * 5] = 1_000.0;
        }
        let evaporation = [0.0f32; MONTHS * 5];
        let inputs = Inputs {
            cell_ids: &cell_ids,
            receiver_ids: &receiver_ids,
            depression_ids: &depression_ids,
            source_active: &[1; 5],
            area_km2: &[1.0; 5],
            terrain_elevation_m: &[2.0, -10.0, 2.0, -10.0, 2.0],
            hydrologic_elevation_m: &[2.0, 0.0, 2.0, 0.0, 2.0],
            parent_relief_m: &[20.0; 5],
            monthly_runoff_mm: &runoff,
            monthly_evaporation_mm: &evaporation,
            sediment_accommodation: &[0.5; 5],
            candidate_ids: &[11, 33],
            spill_receiver_ids: &[2, 4],
        };
        let outcome = run_balance(config(), &inputs).expect("valid candidate chain");
        assert_eq!(outcome.candidates[0].downstream_depression_id, 33);
        assert!(outcome.candidates[1].annual_upstream_inflow_km3 > 0.0);
        assert!(outcome.stats.water_balance_relative_error < 1e-12);
    }
}
