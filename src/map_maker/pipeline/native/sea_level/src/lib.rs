use std::cmp::Ordering;
use std::collections::HashSet;
use std::slice;

const D4_NEIGHBORS: usize = 4;

#[repr(C)]
pub struct SeaLevelStats {
    pub sea_level_m: f32,
    pub target_ocean_area_fraction: f64,
    pub ocean_mask_area_fraction: f64,
    pub ocean_fractional_area_fraction: f64,
    pub emerged_land_area_fraction: f64,
    pub coastal_cell_area_fraction: f64,
    pub continental_shelf_area_fraction: f64,
    pub inland_below_sea_level_area_fraction: f64,
    pub largest_inland_basin_area_fraction: f64,
    pub largest_land_component_area_fraction: f64,
    pub largest_land_component_share: f64,
    pub largest_land_component_coastline_complexity: f64,
    pub maximum_ocean_depth_m: f32,
    pub maximum_land_elevation_m: f32,
    pub below_level_component_count: i32,
    pub land_component_count: i32,
    pub significant_land_component_count: i32,
    pub coastline_edge_count: u64,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
    area: Vec<f64>,
}

impl UnionFind {
    fn new(cell_count: usize) -> Self {
        Self {
            parent: vec![usize::MAX; cell_count],
            rank: vec![0; cell_count],
            area: vec![0.0; cell_count],
        }
    }

    fn activate(&mut self, cell: usize, area: f64) {
        self.parent[cell] = cell;
        self.area[cell] = area;
    }

    fn is_active(&self, cell: usize) -> bool {
        self.parent[cell] != usize::MAX
    }

    fn find(&mut self, cell: usize) -> usize {
        let parent = self.parent[cell];
        if parent != cell {
            let root = self.find(parent);
            self.parent[cell] = root;
        }
        self.parent[cell]
    }

    fn union(&mut self, first: usize, second: usize) -> usize {
        let mut first_root = self.find(first);
        let mut second_root = self.find(second);
        if first_root == second_root {
            return first_root;
        }
        if self.rank[first_root] < self.rank[second_root] {
            std::mem::swap(&mut first_root, &mut second_root);
        }
        self.parent[second_root] = first_root;
        self.area[first_root] += self.area[second_root];
        if self.rank[first_root] == self.rank[second_root] {
            self.rank[first_root] += 1;
        }
        first_root
    }

    fn component_area(&mut self, cell: usize) -> f64 {
        let root = self.find(cell);
        self.area[root]
    }
}

#[derive(Debug)]
struct OceanSolution {
    sea_level_m: f32,
    ocean: Vec<bool>,
    below_level_component_count: usize,
    largest_inland_area: f64,
}

fn solve_connected_ocean(
    elevation: &[f32],
    areas: &[f64],
    neighbors: &[i32],
    target_area: f64,
) -> OceanSolution {
    let mut ranked: Vec<usize> = (0..elevation.len()).collect();
    ranked.sort_by(|first, second| {
        elevation[*first]
            .total_cmp(&elevation[*second])
            .then_with(|| first.cmp(second))
    });
    let mut union_find = UnionFind::new(elevation.len());
    let mut largest_root = ranked[0];
    let mut largest_area = 0.0f64;
    let mut selected_level = elevation[ranked[0]];
    let mut offset = 0usize;

    while offset < ranked.len() {
        let group_start = offset;
        selected_level = elevation[ranked[offset]];
        while offset < ranked.len()
            && elevation[ranked[offset]].total_cmp(&selected_level) == Ordering::Equal
        {
            let cell = ranked[offset];
            union_find.activate(cell, areas[cell]);
            if areas[cell] > largest_area {
                largest_area = areas[cell];
                largest_root = cell;
            }
            offset += 1;
        }
        for &cell in &ranked[group_start..offset] {
            for &neighbor in &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS] {
                let adjacent = neighbor as usize;
                if union_find.is_active(adjacent) {
                    let root = union_find.union(cell, adjacent);
                    let area = union_find.component_area(root);
                    if area > largest_area {
                        largest_area = area;
                        largest_root = root;
                    }
                }
            }
        }
        if largest_area >= target_area {
            break;
        }
    }

    let selected_root = union_find.find(largest_root);
    let mut ocean = vec![false; elevation.len()];
    let mut roots = HashSet::new();
    let mut largest_inland_area = 0.0f64;
    for cell in 0..elevation.len() {
        if !union_find.is_active(cell) {
            continue;
        }
        let root = union_find.find(cell);
        roots.insert(root);
        if root == selected_root {
            ocean[cell] = true;
        } else {
            largest_inland_area = largest_inland_area.max(union_find.component_area(root));
        }
    }
    OceanSolution {
        sea_level_m: selected_level,
        ocean,
        below_level_component_count: roots.len(),
        largest_inland_area,
    }
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn fractional_ocean_coverage(
    solution: &OceanSolution,
    elevation: &[f32],
    relief: &[f32],
    areas: &[f64],
    neighbors: &[i32],
    target_area: f64,
    minimum_coastal_relief_m: f32,
    coastal_relief_scale: f32,
) -> (Vec<f32>, Vec<bool>) {
    let mut coastal = vec![false; elevation.len()];
    let mut interior_ocean_area = 0.0f64;
    let mut coastal_area = 0.0f64;
    let mut coastal_logits = vec![0.0f64; elevation.len()];
    for cell in 0..elevation.len() {
        coastal[cell] = neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS]
            .iter()
            .any(|neighbor| solution.ocean[*neighbor as usize] != solution.ocean[cell]);
        if coastal[cell] {
            coastal_area += areas[cell];
            let sigma = (minimum_coastal_relief_m + coastal_relief_scale * relief[cell].max(0.0))
                .max(1.0) as f64;
            coastal_logits[cell] = (solution.sea_level_m - elevation[cell]) as f64 / sigma;
        } else if solution.ocean[cell] {
            interior_ocean_area += areas[cell];
        }
    }

    let minimum_coastal_area: f64 = (0..elevation.len())
        .filter(|cell| coastal[*cell] && solution.ocean[*cell])
        .map(|cell| 0.5 * areas[cell])
        .sum();
    let maximum_coastal_area = minimum_coastal_area + 0.5 * coastal_area;
    let desired_coastal_area =
        (target_area - interior_ocean_area).clamp(minimum_coastal_area, maximum_coastal_area);
    let weighted_area = |shift: f64| -> f64 {
        (0..elevation.len())
            .filter(|cell| coastal[*cell])
            .map(|cell| {
                let response = 0.5 * logistic(coastal_logits[cell] + shift);
                let fraction = if solution.ocean[cell] {
                    0.5 + response
                } else {
                    response
                };
                areas[cell] * fraction
            })
            .sum()
    };
    let mut low = -48.0f64;
    let mut high = 48.0f64;
    for _ in 0..72 {
        let middle = 0.5 * (low + high);
        if weighted_area(middle) < desired_coastal_area {
            low = middle;
        } else {
            high = middle;
        }
    }
    let shift = 0.5 * (low + high);
    let mut fractions = vec![0.0f32; elevation.len()];
    for cell in 0..elevation.len() {
        fractions[cell] = if coastal[cell] {
            let response = 0.5 * logistic(coastal_logits[cell] + shift);
            if solution.ocean[cell] {
                (0.5 + response).max(0.500_001) as f32
            } else {
                response.min(0.499_999) as f32
            }
        } else if solution.ocean[cell] {
            1.0
        } else {
            0.0
        };
    }
    (fractions, coastal)
}

#[derive(Default)]
struct LandComponentStats {
    component_count: usize,
    significant_component_count: usize,
    largest_area: f64,
    largest_coastline_complexity: f64,
}

fn land_component_stats(
    ocean: &[bool],
    areas: &[f64],
    neighbors: &[i32],
    total_area: f64,
) -> LandComponentStats {
    let mut visited = vec![false; ocean.len()];
    let mut stats = LandComponentStats::default();
    for start in 0..ocean.len() {
        if ocean[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut pending = vec![start];
        let mut component_area = 0.0f64;
        let mut component_coastline = 0.0f64;
        while let Some(cell) = pending.pop() {
            component_area += areas[cell];
            for &neighbor in &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS] {
                let adjacent = neighbor as usize;
                if ocean[adjacent] {
                    component_coastline += 0.5 * (areas[cell].sqrt() + areas[adjacent].sqrt());
                } else if !visited[adjacent] {
                    visited[adjacent] = true;
                    pending.push(adjacent);
                }
            }
        }
        stats.component_count += 1;
        if component_area > stats.largest_area {
            let area_fraction = component_area / total_area;
            let planet_radius = (total_area / (4.0 * std::f64::consts::PI)).sqrt();
            let minimum_spherical_perimeter = 4.0
                * std::f64::consts::PI
                * planet_radius
                * (area_fraction * (1.0 - area_fraction)).sqrt();
            stats.largest_area = component_area;
            stats.largest_coastline_complexity =
                component_coastline / minimum_spherical_perimeter.max(f64::EPSILON);
        }
        if component_area / total_area >= 0.001 {
            stats.significant_component_count += 1;
        }
    }
    stats
}

#[allow(clippy::too_many_arguments)]
fn run_sea_level(
    target_ocean_area_fraction: f64,
    shelf_depth_m: f32,
    minimum_coastal_relief_m: f32,
    coastal_relief_scale: f32,
    areas: &[f64],
    neighbors: &[i32],
    elevation: &[f32],
    relief: &[f32],
    ocean_mask_out: &mut [f32],
    ocean_fraction_out: &mut [f32],
    surface_elevation_out: &mut [f32],
    ocean_depth_out: &mut [f32],
    shelf_fraction_out: &mut [f32],
    coastal_mask_out: &mut [f32],
    inland_below_sea_level_out: &mut [f32],
) -> SeaLevelStats {
    let total_area: f64 = areas.iter().sum();
    let target_area = target_ocean_area_fraction * total_area;
    let solution = solve_connected_ocean(elevation, areas, neighbors, target_area);
    let (ocean_fraction, coastal) = fractional_ocean_coverage(
        &solution,
        elevation,
        relief,
        areas,
        neighbors,
        target_area,
        minimum_coastal_relief_m,
        coastal_relief_scale,
    );
    let mut mask_area = 0.0f64;
    let mut fractional_area = 0.0f64;
    let mut coastal_area = 0.0f64;
    let mut shelf_area = 0.0f64;
    let mut inland_area = 0.0f64;
    let mut maximum_ocean_depth = 0.0f32;
    let mut maximum_land_elevation = 0.0f32;
    let mut coastline_edges = 0u64;

    for cell in 0..elevation.len() {
        let is_ocean = solution.ocean[cell];
        let relative_elevation = elevation[cell] - solution.sea_level_m;
        let depth = (-relative_elevation).max(0.0);
        let inland_below = !is_ocean && relative_elevation <= 0.0;
        let sigma = (minimum_coastal_relief_m + coastal_relief_scale * relief[cell].max(0.0))
            .max(1.0) as f64;
        let shelf_likelihood = logistic((shelf_depth_m - depth) as f64 / sigma) as f32;
        let shelf_fraction = ocean_fraction[cell] * shelf_likelihood;

        ocean_mask_out[cell] = f32::from(is_ocean);
        ocean_fraction_out[cell] = ocean_fraction[cell];
        surface_elevation_out[cell] = relative_elevation;
        ocean_depth_out[cell] = if is_ocean { depth } else { 0.0 };
        shelf_fraction_out[cell] = shelf_fraction;
        coastal_mask_out[cell] = f32::from(coastal[cell]);
        inland_below_sea_level_out[cell] = f32::from(inland_below);

        mask_area += f64::from(is_ocean) * areas[cell];
        fractional_area += ocean_fraction[cell] as f64 * areas[cell];
        coastal_area += f64::from(coastal[cell]) * areas[cell];
        shelf_area += shelf_fraction as f64 * areas[cell];
        inland_area += f64::from(inland_below) * areas[cell];
        if is_ocean {
            maximum_ocean_depth = maximum_ocean_depth.max(depth);
        } else {
            maximum_land_elevation = maximum_land_elevation.max(relative_elevation);
            coastline_edges += neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS]
                .iter()
                .filter(|neighbor| solution.ocean[**neighbor as usize])
                .count() as u64;
        }
    }

    let land_stats = land_component_stats(&solution.ocean, areas, neighbors, total_area);
    let land_area = total_area - mask_area;
    SeaLevelStats {
        sea_level_m: solution.sea_level_m,
        target_ocean_area_fraction,
        ocean_mask_area_fraction: mask_area / total_area,
        ocean_fractional_area_fraction: fractional_area / total_area,
        emerged_land_area_fraction: 1.0 - fractional_area / total_area,
        coastal_cell_area_fraction: coastal_area / total_area,
        continental_shelf_area_fraction: shelf_area / total_area,
        inland_below_sea_level_area_fraction: inland_area / total_area,
        largest_inland_basin_area_fraction: solution.largest_inland_area / total_area,
        largest_land_component_area_fraction: land_stats.largest_area / total_area,
        largest_land_component_share: land_stats.largest_area / land_area.max(f64::EPSILON),
        largest_land_component_coastline_complexity: land_stats.largest_coastline_complexity,
        maximum_ocean_depth_m: maximum_ocean_depth,
        maximum_land_elevation_m: maximum_land_elevation,
        below_level_component_count: solution.below_level_component_count as i32,
        land_component_count: land_stats.component_count as i32,
        significant_land_component_count: land_stats.significant_component_count as i32,
        coastline_edge_count: coastline_edges,
    }
}

fn finite_slice(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[no_mangle]
pub extern "C" fn sea_level_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_sea_level_abi_version() -> u32 {
    2
}

/// Solve connected ocean extent and fractional coastal coverage.
///
/// Returns 0 on success, 1 for invalid dimensions or pointers, 2 for invalid
/// parameters, 3 for non-finite inputs, and 4 for invalid topology.
///
/// # Safety
///
/// Input pointers must reference readable buffers of the lengths implied by
/// `cell_count`. Output pointers must reference distinct writable buffers of
/// `cell_count` floats and may not alias any input or other output.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sea_level_run_cubed_sphere(
    cell_count: i32,
    target_ocean_area_fraction: f64,
    shelf_depth_m: f32,
    minimum_coastal_relief_m: f32,
    coastal_relief_scale: f32,
    area_ptr: *const f64,
    neighbors_ptr: *const i32,
    elevation_ptr: *const f32,
    relief_ptr: *const f32,
    ocean_mask_out_ptr: *mut f32,
    ocean_fraction_out_ptr: *mut f32,
    surface_elevation_out_ptr: *mut f32,
    ocean_depth_out_ptr: *mut f32,
    shelf_fraction_out_ptr: *mut f32,
    coastal_mask_out_ptr: *mut f32,
    inland_below_sea_level_out_ptr: *mut f32,
    stats_out: *mut SeaLevelStats,
) -> i32 {
    if cell_count <= 0
        || area_ptr.is_null()
        || neighbors_ptr.is_null()
        || elevation_ptr.is_null()
        || relief_ptr.is_null()
        || ocean_mask_out_ptr.is_null()
        || ocean_fraction_out_ptr.is_null()
        || surface_elevation_out_ptr.is_null()
        || ocean_depth_out_ptr.is_null()
        || shelf_fraction_out_ptr.is_null()
        || coastal_mask_out_ptr.is_null()
        || inland_below_sea_level_out_ptr.is_null()
    {
        return 1;
    }
    if !target_ocean_area_fraction.is_finite()
        || !(0.05..=0.95).contains(&target_ocean_area_fraction)
        || !shelf_depth_m.is_finite()
        || shelf_depth_m <= 0.0
        || shelf_depth_m > 2_000.0
        || !minimum_coastal_relief_m.is_finite()
        || minimum_coastal_relief_m <= 0.0
        || minimum_coastal_relief_m > 2_000.0
        || !coastal_relief_scale.is_finite()
        || !(0.0..=4.0).contains(&coastal_relief_scale)
    {
        return 2;
    }

    let total = cell_count as usize;
    let edge_len = match total.checked_mul(D4_NEIGHBORS) {
        Some(value) => value,
        None => return 1,
    };
    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let neighbors = unsafe { slice::from_raw_parts(neighbors_ptr, edge_len) };
    let elevation = unsafe { slice::from_raw_parts(elevation_ptr, total) };
    let relief = unsafe { slice::from_raw_parts(relief_ptr, total) };
    if areas.iter().any(|area| !area.is_finite() || *area <= 0.0)
        || !finite_slice(elevation)
        || !finite_slice(relief)
    {
        return 3;
    }
    for cell in 0..total {
        let mut unique = [0i32; D4_NEIGHBORS];
        unique.copy_from_slice(&neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS]);
        unique.sort_unstable();
        if unique.windows(2).any(|pair| pair[0] == pair[1])
            || unique
                .iter()
                .any(|neighbor| *neighbor < 0 || *neighbor as usize >= total)
        {
            return 4;
        }
    }

    let ocean_mask_out = unsafe { slice::from_raw_parts_mut(ocean_mask_out_ptr, total) };
    let ocean_fraction_out = unsafe { slice::from_raw_parts_mut(ocean_fraction_out_ptr, total) };
    let surface_elevation_out =
        unsafe { slice::from_raw_parts_mut(surface_elevation_out_ptr, total) };
    let ocean_depth_out = unsafe { slice::from_raw_parts_mut(ocean_depth_out_ptr, total) };
    let shelf_fraction_out = unsafe { slice::from_raw_parts_mut(shelf_fraction_out_ptr, total) };
    let coastal_mask_out = unsafe { slice::from_raw_parts_mut(coastal_mask_out_ptr, total) };
    let inland_below_sea_level_out =
        unsafe { slice::from_raw_parts_mut(inland_below_sea_level_out_ptr, total) };

    let stats = run_sea_level(
        target_ocean_area_fraction,
        shelf_depth_m,
        minimum_coastal_relief_m,
        coastal_relief_scale,
        areas,
        neighbors,
        elevation,
        relief,
        ocean_mask_out,
        ocean_fraction_out,
        surface_elevation_out,
        ocean_depth_out,
        shelf_fraction_out,
        coastal_mask_out,
        inland_below_sea_level_out,
    );
    if !stats_out.is_null() {
        unsafe { stats_out.write(stats) };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_topology(count: usize) -> (Vec<f64>, Vec<i32>) {
        let areas = vec![1.0; count];
        let mut neighbors = vec![0i32; count * D4_NEIGHBORS];
        for cell in 0..count {
            neighbors[cell * 4] = ((cell + count - 1) % count) as i32;
            neighbors[cell * 4 + 1] = ((cell + 1) % count) as i32;
            neighbors[cell * 4 + 2] = ((cell + count - 2) % count) as i32;
            neighbors[cell * 4 + 3] = ((cell + 2) % count) as i32;
        }
        (areas, neighbors)
    }

    #[test]
    fn connected_ocean_excludes_isolated_low_basin() {
        let (areas, neighbors) = ring_topology(20);
        let mut elevation: Vec<f32> = (0..20).map(|cell| cell as f32 * 10.0).collect();
        elevation[12] = -500.0;
        let solution = solve_connected_ocean(&elevation, &areas, &neighbors, 10.0);
        assert!(!solution.ocean[12]);
        assert!(solution.largest_inland_area > 0.0);
        assert!(solution.ocean.iter().filter(|value| **value).count() >= 10);
    }

    #[test]
    fn fractional_coast_hits_target_area() {
        let (areas, neighbors) = ring_topology(40);
        let elevation: Vec<f32> = (0..40).map(|cell| cell as f32 - 25.0).collect();
        let relief = vec![100.0; 40];
        let solution = solve_connected_ocean(&elevation, &areas, &neighbors, 26.0);
        let (fractions, _) = fractional_ocean_coverage(
            &solution, &elevation, &relief, &areas, &neighbors, 26.0, 40.0, 0.5,
        );
        let area: f64 = fractions.iter().map(|value| *value as f64).sum();
        assert!((area - 26.0).abs() < 1e-5);
    }

    #[test]
    fn kernel_outputs_are_bounded_and_relative_to_datum() {
        let (areas, neighbors) = ring_topology(40);
        let elevation: Vec<f32> = (0..40).map(|cell| cell as f32 * 50.0 - 1_200.0).collect();
        let relief = vec![150.0; 40];
        let mut ocean_mask = vec![0.0f32; 40];
        let mut ocean_fraction = vec![0.0f32; 40];
        let mut surface_elevation = vec![0.0f32; 40];
        let mut ocean_depth = vec![0.0f32; 40];
        let mut shelf_fraction = vec![0.0f32; 40];
        let mut coastal_mask = vec![0.0f32; 40];
        let mut inland_below_sea_level = vec![0.0f32; 40];
        let stats = run_sea_level(
            0.65,
            200.0,
            40.0,
            0.5,
            &areas,
            &neighbors,
            &elevation,
            &relief,
            &mut ocean_mask,
            &mut ocean_fraction,
            &mut surface_elevation,
            &mut ocean_depth,
            &mut shelf_fraction,
            &mut coastal_mask,
            &mut inland_below_sea_level,
        );
        assert!((stats.ocean_fractional_area_fraction - 0.65).abs() < 1e-6);
        assert!(ocean_fraction
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
        for cell in 0..40 {
            assert_eq!(surface_elevation[cell], elevation[cell] - stats.sea_level_m);
        }
    }
}
