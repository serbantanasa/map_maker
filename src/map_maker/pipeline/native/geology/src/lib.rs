use std::collections::VecDeque;
use std::mem;
use std::slice;

const PLATE_COMPONENTS: usize = 7;
const D4_NEIGHBORS: usize = 4;

const PROVINCE_SHIELD: u8 = 1;
const PROVINCE_PLATFORM: u8 = 2;
const PROVINCE_SEDIMENTARY_BASIN: u8 = 3;
const PROVINCE_OROGEN: u8 = 4;
const PROVINCE_CONTINENTAL_RIFT: u8 = 5;
const PROVINCE_CONTINENTAL_ARC: u8 = 6;
const PROVINCE_SHELF: u8 = 7;
const PROVINCE_ABYSSAL_BASIN: u8 = 8;
const PROVINCE_OCEANIC_RIDGE: u8 = 9;
const PROVINCE_INTRA_OCEANIC_ARC: u8 = 10;
const PROVINCE_VOLCANIC: u8 = 11;

const REGIME_INACTIVE: u8 = 1;
const REGIME_CONTINENTAL_COLLISION: u8 = 2;
const REGIME_SUBDUCTION_MARGIN: u8 = 3;
const REGIME_INTRA_OCEANIC_SUBDUCTION: u8 = 4;
const REGIME_CONTINENTAL_RIFT: u8 = 5;
const REGIME_SPREADING_RIDGE: u8 = 6;
const REGIME_TRANSFORM: u8 = 7;

#[no_mangle]
pub extern "C" fn geology_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_geology_abi_version() -> u32 {
    1
}

#[repr(C)]
pub struct ProvinceRecord {
    pub province_id: i32,
    pub class_code: i32,
    pub parent_plate_id: i32,
    pub cell_count: i32,
    pub area_steradians: f64,
    pub mean_crust_age_ga: f32,
    pub mean_rock_strength: f32,
    pub mean_accommodation: f32,
    pub mean_confidence: f32,
}

#[repr(C)]
pub struct ProvinceRecordArray {
    pub data: *mut ProvinceRecord,
    pub len: usize,
}

#[repr(C)]
pub struct BoundarySegmentRecord {
    pub segment_id: i32,
    pub regime_code: i32,
    pub plate_a: i32,
    pub plate_b: i32,
    pub edge_count: i32,
    pub angular_length: f64,
    pub mean_compression: f32,
    pub mean_extension: f32,
    pub mean_shear: f32,
    pub mean_subduction: f32,
    pub mean_confidence: f32,
}

struct BoundaryEdge {
    source: usize,
    target: usize,
    source_slot: usize,
    target_slot: usize,
    plate_a: i32,
    plate_b: i32,
    regime: u8,
    confidence: f32,
    compression: f32,
    extension: f32,
    shear: f32,
    subduction: f32,
}

fn adjacent_boundary_edges(
    edge_index: usize,
    edges: &[BoundaryEdge],
    edge_by_slot: &[i32],
    neighbors: &[i32],
) -> Vec<usize> {
    let edge = &edges[edge_index];
    let mut adjacent_edges = Vec::new();
    for cell in [edge.source, edge.target] {
        let mut nearby_cells = [0usize; 5];
        nearby_cells[0] = cell;
        for (offset, neighbor) in neighbors[cell * 4..cell * 4 + 4].iter().enumerate() {
            nearby_cells[offset + 1] = *neighbor as usize;
        }
        for nearby in nearby_cells {
            for adjacent_index in &edge_by_slot[nearby * 4..nearby * 4 + 4] {
                if *adjacent_index < 0 {
                    continue;
                }
                let adjacent_index = *adjacent_index as usize;
                if adjacent_index != edge_index && !adjacent_edges.contains(&adjacent_index) {
                    adjacent_edges.push(adjacent_index);
                }
            }
        }
    }
    adjacent_edges
}

fn smooth_boundary_regimes(edges: &mut [BoundaryEdge], edge_by_slot: &[i32], neighbors: &[i32]) {
    for _ in 0..4 {
        let mut updates = Vec::with_capacity(edges.len());
        for (edge_index, edge) in edges.iter().enumerate() {
            let mut scores = [0.0f32; 8];
            scores[edge.regime as usize] = edge.confidence * 1.25;
            for adjacent_index in
                adjacent_boundary_edges(edge_index, edges, edge_by_slot, neighbors)
            {
                let adjacent = &edges[adjacent_index];
                if adjacent.plate_a == edge.plate_a && adjacent.plate_b == edge.plate_b {
                    scores[adjacent.regime as usize] += adjacent.confidence;
                }
            }
            let mut selected = edge.regime;
            let mut selected_score = scores[selected as usize];
            for (regime, score) in scores.iter().enumerate().skip(1) {
                if *score > selected_score {
                    selected = regime as u8;
                    selected_score = *score;
                }
            }
            updates.push((selected, selected_score / 6.0));
        }
        for (edge, (regime, neighborhood_confidence)) in edges.iter_mut().zip(updates) {
            if edge.regime != regime {
                edge.regime = regime;
                edge.confidence = edge
                    .confidence
                    .min(neighborhood_confidence.clamp(0.5, 0.75));
            }
        }
    }
}

fn merge_short_boundary_segments(
    edges: &mut [BoundaryEdge],
    edge_by_slot: &[i32],
    neighbors: &[i32],
    areas: &[f64],
) {
    const MIN_SEGMENT_ANGULAR_LENGTH: f64 = 0.035;
    let mut queue = VecDeque::new();
    for _ in 0..4 {
        let mut component_id = vec![-1i32; edges.len()];
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut lengths = Vec::new();
        for start in 0..edges.len() {
            if component_id[start] >= 0 {
                continue;
            }
            let id = components.len() as i32;
            let regime = edges[start].regime;
            let plate_a = edges[start].plate_a;
            let plate_b = edges[start].plate_b;
            let mut members = Vec::new();
            let mut length = 0.0f64;
            component_id[start] = id;
            queue.push_back(start);
            while let Some(edge_index) = queue.pop_front() {
                let edge = &edges[edge_index];
                members.push(edge_index);
                length += 0.5 * (areas[edge.source].sqrt() + areas[edge.target].sqrt());
                for adjacent_index in
                    adjacent_boundary_edges(edge_index, edges, edge_by_slot, neighbors)
                {
                    let adjacent = &edges[adjacent_index];
                    if component_id[adjacent_index] < 0
                        && adjacent.regime == regime
                        && adjacent.plate_a == plate_a
                        && adjacent.plate_b == plate_b
                    {
                        component_id[adjacent_index] = id;
                        queue.push_back(adjacent_index);
                    }
                }
            }
            components.push(members);
            lengths.push(length);
        }

        let mut updates = Vec::new();
        for (index, members) in components.iter().enumerate() {
            if lengths[index] >= MIN_SEGMENT_ANGULAR_LENGTH {
                continue;
            }
            let plate_a = edges[members[0]].plate_a;
            let plate_b = edges[members[0]].plate_b;
            let mut contacts = vec![0usize; components.len()];
            for &edge_index in members {
                for adjacent_index in
                    adjacent_boundary_edges(edge_index, edges, edge_by_slot, neighbors)
                {
                    let adjacent_component = component_id[adjacent_index] as usize;
                    let adjacent = &edges[adjacent_index];
                    if adjacent_component != index
                        && adjacent.plate_a == plate_a
                        && adjacent.plate_b == plate_b
                        && lengths[adjacent_component] > lengths[index]
                    {
                        contacts[adjacent_component] += 1;
                    }
                }
            }
            let target = contacts
                .iter()
                .enumerate()
                .max_by_key(|(component, count)| (**count, lengths[*component].to_bits()))
                .filter(|(_, count)| **count > 0)
                .map(|(component, _)| component);
            if let Some(target_component) = target {
                let target_regime = edges[components[target_component][0]].regime;
                updates.push((members.clone(), target_regime));
            }
        }
        if updates.is_empty() {
            break;
        }
        for (members, target_regime) in updates {
            for edge_index in members {
                edges[edge_index].regime = target_regime;
                edges[edge_index].confidence = edges[edge_index].confidence.min(0.55);
            }
        }
    }
}

#[repr(C)]
pub struct BoundarySegmentRecordArray {
    pub data: *mut BoundarySegmentRecord,
    pub len: usize,
}

#[repr(C)]
pub struct GeologyStats {
    pub province_count: i32,
    pub boundary_segment_count: i32,
    pub active_boundary_segment_count: i32,
    pub mixed_plate_province_count: i32,
    pub continental_area: f64,
    pub oceanic_area: f64,
    pub mean_crust_age_ga: f32,
    pub mean_confidence: f32,
}

#[no_mangle]
/// Release province records allocated by [`geology_run_cubed_sphere`].
///
/// # Safety
///
/// `array` must be returned by one successful kernel call and released once.
pub unsafe extern "C" fn geology_free_provinces(array: ProvinceRecordArray) {
    if !array.data.is_null() && array.len > 0 {
        let records = std::ptr::slice_from_raw_parts_mut(array.data, array.len);
        drop(Box::from_raw(records));
    }
}

#[no_mangle]
/// Release boundary records allocated by [`geology_run_cubed_sphere`].
///
/// # Safety
///
/// `array` must be returned by one successful kernel call and released once.
pub unsafe extern "C" fn geology_free_boundary_segments(array: BoundarySegmentRecordArray) {
    if !array.data.is_null() && array.len > 0 {
        let records = std::ptr::slice_from_raw_parts_mut(array.data, array.len);
        drop(Box::from_raw(records));
    }
}

fn valid_ffi_len<T>(len: usize) -> bool {
    len.checked_mul(mem::size_of::<T>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

#[allow(clippy::too_many_arguments)]
fn province_class(
    oceanic: bool,
    compression: f32,
    extension: f32,
    subduction: f32,
    uplift: f32,
    subsidence: f32,
    isostasy: f32,
    margin: f32,
    stiffness: f32,
    basin_isostasy_threshold: f32,
) -> (u8, f32) {
    let activity = compression.max(extension).max(subduction);
    if oceanic {
        if extension > 0.22 && extension >= compression.max(subduction) {
            return (PROVINCE_OCEANIC_RIDGE, (0.58 + 0.35 * extension).min(0.98));
        }
        if subduction > 0.22 && compression > 0.12 {
            return (
                PROVINCE_INTRA_OCEANIC_ARC,
                (0.56 + 0.24 * subduction + 0.12 * compression).min(0.98),
            );
        }
        return (PROVINCE_ABYSSAL_BASIN, (0.62 + 0.2 * stiffness).min(0.9));
    }

    if subduction > 0.24 && compression > 0.14 {
        return (
            PROVINCE_CONTINENTAL_ARC,
            (0.56 + 0.22 * subduction + 0.12 * compression).min(0.98),
        );
    }
    if compression > 0.26 && uplift > 0.16 {
        return (
            PROVINCE_OROGEN,
            (0.56 + 0.24 * compression + 0.12 * uplift).min(0.98),
        );
    }
    if extension > 0.22 && extension > compression {
        return (
            PROVINCE_CONTINENTAL_RIFT,
            (0.56 + 0.34 * extension).min(0.98),
        );
    }
    if margin > 0.58 && activity < 0.28 {
        return (
            PROVINCE_SHELF,
            (0.55 + 0.32 * margin - 0.15 * activity).clamp(0.5, 0.92),
        );
    }
    if activity < 0.18 && isostasy < basin_isostasy_threshold && subsidence >= uplift * 0.3 {
        let signal = (basin_isostasy_threshold - isostasy).max(subsidence - uplift * 0.3);
        return (
            PROVINCE_SEDIMENTARY_BASIN,
            (0.56 + 0.35 * signal).clamp(0.52, 0.92),
        );
    }
    if stiffness > 0.915 {
        return (PROVINCE_SHIELD, (0.58 + 0.38 * stiffness).min(0.97));
    }
    (PROVINCE_PLATFORM, (0.55 + 0.3 * stiffness).clamp(0.5, 0.9))
}

fn boundary_regime(
    continental_a: bool,
    continental_b: bool,
    compression: f32,
    extension: f32,
    shear: f32,
    subduction: f32,
) -> (u8, f32) {
    let compressive = compression.max(subduction * 0.9);
    if extension > 0.1 && extension >= compressive.max(shear) {
        let regime = if continental_a && continental_b {
            REGIME_CONTINENTAL_RIFT
        } else {
            REGIME_SPREADING_RIDGE
        };
        return (regime, (0.52 + 0.4 * extension).min(0.98));
    }
    if compressive > 0.1 && compressive >= shear {
        let regime = match (continental_a, continental_b) {
            (true, true) => REGIME_CONTINENTAL_COLLISION,
            (false, false) => REGIME_INTRA_OCEANIC_SUBDUCTION,
            _ => REGIME_SUBDUCTION_MARGIN,
        };
        return (
            regime,
            (0.52 + 0.25 * compressive + 0.18 * subduction).min(0.98),
        );
    }
    if shear > 0.08 {
        return (REGIME_TRANSFORM, (0.52 + 0.42 * shear).min(0.96));
    }
    (REGIME_INACTIVE, 0.5)
}

fn plate_id(plate_field: &[f32], cell: usize, components: usize) -> i32 {
    plate_field[cell * components].round() as i32
}

fn is_continental(plate_field: &[f32], cell: usize, components: usize) -> bool {
    plate_field[cell * components + 1] >= 0.5
}

fn minimum_province_area(class_code: u8, total_area: f64) -> f64 {
    match class_code {
        PROVINCE_SHIELD | PROVINCE_PLATFORM | PROVINCE_SEDIMENTARY_BASIN | PROVINCE_VOLCANIC => {
            total_area / 8192.0
        }
        _ => total_area / 16384.0,
    }
}

fn merge_small_provinces(
    classes: &mut [u8],
    confidence: &mut [f32],
    oceanic: &[bool],
    neighbors: &[i32],
    areas: &[f64],
    total_area: f64,
) {
    let mut queue = VecDeque::new();
    for _ in 0..12 {
        let mut component_id = vec![-1i32; classes.len()];
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut component_areas = Vec::new();
        for start in 0..classes.len() {
            if component_id[start] >= 0 {
                continue;
            }
            let id = components.len() as i32;
            let class_code = classes[start];
            let mut cells = Vec::new();
            let mut area = 0.0f64;
            component_id[start] = id;
            queue.push_back(start);
            while let Some(cell) = queue.pop_front() {
                cells.push(cell);
                area += areas[cell];
                for &neighbor in &neighbors[cell * 4..cell * 4 + 4] {
                    let adjacent = neighbor as usize;
                    if component_id[adjacent] < 0 && classes[adjacent] == class_code {
                        component_id[adjacent] = id;
                        queue.push_back(adjacent);
                    }
                }
            }
            components.push(cells);
            component_areas.push(area);
        }

        let mut updates = Vec::new();
        for (index, cells) in components.iter().enumerate() {
            let source_class = classes[cells[0]];
            if component_areas[index] >= minimum_province_area(source_class, total_area) {
                continue;
            }
            let source_oceanic = oceanic[cells[0]];
            let mut shared_edge_length = [0.0f64; 12];
            for &cell in cells {
                for &neighbor in &neighbors[cell * 4..cell * 4 + 4] {
                    let adjacent = neighbor as usize;
                    let target_class = classes[adjacent];
                    let target_component = component_id[adjacent] as usize;
                    let target_is_larger = component_areas[target_component]
                        > component_areas[index]
                        || (component_areas[target_component] == component_areas[index]
                            && target_component < index);
                    if target_class != source_class
                        && oceanic[adjacent] == source_oceanic
                        && target_is_larger
                        && (target_class as usize) < shared_edge_length.len()
                    {
                        shared_edge_length[target_class as usize] +=
                            0.5 * (areas[cell].sqrt() + areas[adjacent].sqrt());
                    }
                }
            }
            let target = shared_edge_length
                .iter()
                .enumerate()
                .skip(1)
                .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
                .filter(|(_, weight)| **weight > 0.0)
                .map(|(class_code, _)| class_code as u8);
            if let Some(target_class) = target {
                updates.push((cells.clone(), target_class));
            }
        }
        if updates.is_empty() {
            for (index, cells) in components.iter().enumerate() {
                let class_code = classes[cells[0]];
                if component_areas[index] < minimum_province_area(class_code, total_area) {
                    for &cell in cells {
                        confidence[cell] = confidence[cell].min(0.5);
                    }
                }
            }
            break;
        }
        for (cells, target_class) in updates {
            for cell in cells {
                classes[cell] = target_class;
                confidence[cell] = confidence[cell].min(0.55);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
/// Initialize connected geological process provinces and plate-boundary segments.
///
/// This classifies current structural evidence. It does not claim a simulated
/// stratigraphic or terrane history.
///
/// # Safety
///
/// Every input and output must be aligned, correctly sized, and non-overlapping.
/// Neighbor IDs must reference four unique valid global cells. Catalog and
/// stats pointers may be null; null catalog pointers discard generated records.
pub unsafe extern "C" fn geology_run_cubed_sphere(
    cell_count: i32,
    world_age_ga: f32,
    plate_components: i32,
    area_ptr: *const f64,
    neighbors_ptr: *const i32,
    plate_field_ptr: *const f32,
    subduction_ptr: *const f32,
    isostasy_ptr: *const f32,
    uplift_ptr: *const f32,
    subsidence_ptr: *const f32,
    compression_ptr: *const f32,
    extension_ptr: *const f32,
    shear_ptr: *const f32,
    margin_ptr: *const f32,
    stiffness_ptr: *const f32,
    proto_ocean_ptr: *const f32,
    province_id_out_ptr: *mut i32,
    province_class_out_ptr: *mut u8,
    crust_age_out_ptr: *mut f32,
    rock_strength_out_ptr: *mut f32,
    accommodation_out_ptr: *mut f32,
    province_confidence_out_ptr: *mut f32,
    boundary_segment_id_out_ptr: *mut i32,
    boundary_regime_out_ptr: *mut u8,
    boundary_confidence_out_ptr: *mut f32,
    provinces_out: *mut ProvinceRecordArray,
    segments_out: *mut BoundarySegmentRecordArray,
    stats_out: *mut GeologyStats,
) -> i32 {
    if cell_count <= 0
        || !world_age_ga.is_finite()
        || world_age_ga <= 0.0
        || plate_components < PLATE_COMPONENTS as i32
        || area_ptr.is_null()
        || neighbors_ptr.is_null()
        || plate_field_ptr.is_null()
        || subduction_ptr.is_null()
        || isostasy_ptr.is_null()
        || uplift_ptr.is_null()
        || subsidence_ptr.is_null()
        || compression_ptr.is_null()
        || extension_ptr.is_null()
        || shear_ptr.is_null()
        || margin_ptr.is_null()
        || stiffness_ptr.is_null()
        || proto_ocean_ptr.is_null()
        || province_id_out_ptr.is_null()
        || province_class_out_ptr.is_null()
        || crust_age_out_ptr.is_null()
        || rock_strength_out_ptr.is_null()
        || accommodation_out_ptr.is_null()
        || province_confidence_out_ptr.is_null()
        || boundary_segment_id_out_ptr.is_null()
        || boundary_regime_out_ptr.is_null()
        || boundary_confidence_out_ptr.is_null()
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
        || !valid_ffi_len::<i32>(total)
        || !valid_ffi_len::<u8>(total)
        || !valid_ffi_len::<f32>(neighbor_len)
    {
        return 2;
    }

    let areas = slice::from_raw_parts(area_ptr, total);
    let neighbors = slice::from_raw_parts(neighbors_ptr, neighbor_len);
    let plate_field = slice::from_raw_parts(plate_field_ptr, plate_len);
    let subduction = slice::from_raw_parts(subduction_ptr, total);
    let isostasy = slice::from_raw_parts(isostasy_ptr, total);
    let uplift = slice::from_raw_parts(uplift_ptr, total);
    let subsidence = slice::from_raw_parts(subsidence_ptr, total);
    let compression = slice::from_raw_parts(compression_ptr, total);
    let extension = slice::from_raw_parts(extension_ptr, total);
    let shear = slice::from_raw_parts(shear_ptr, total);
    let margin = slice::from_raw_parts(margin_ptr, total);
    let stiffness = slice::from_raw_parts(stiffness_ptr, total);
    let proto_ocean = slice::from_raw_parts(proto_ocean_ptr, total);
    let province_ids = slice::from_raw_parts_mut(province_id_out_ptr, total);
    let province_classes = slice::from_raw_parts_mut(province_class_out_ptr, total);
    let crust_age = slice::from_raw_parts_mut(crust_age_out_ptr, total);
    let rock_strength = slice::from_raw_parts_mut(rock_strength_out_ptr, total);
    let accommodation = slice::from_raw_parts_mut(accommodation_out_ptr, total);
    let province_confidence = slice::from_raw_parts_mut(province_confidence_out_ptr, total);
    let segment_ids = slice::from_raw_parts_mut(boundary_segment_id_out_ptr, neighbor_len);
    let regimes = slice::from_raw_parts_mut(boundary_regime_out_ptr, neighbor_len);
    let boundary_confidence = slice::from_raw_parts_mut(boundary_confidence_out_ptr, neighbor_len);

    let scalar_inputs = [
        subduction,
        isostasy,
        uplift,
        subsidence,
        compression,
        extension,
        shear,
        margin,
        stiffness,
        proto_ocean,
    ];
    let mut total_area = 0.0f64;
    for cell in 0..total {
        if !areas[cell].is_finite()
            || areas[cell] <= 0.0
            || plate_field[cell * components..cell * components + PLATE_COMPONENTS]
                .iter()
                .any(|value| !value.is_finite())
            || scalar_inputs.iter().any(|values| !values[cell].is_finite())
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
        total_area += areas[cell];
    }

    let mut continental_area = 0.0f64;
    let mut continental_isostasy_sum = 0.0f64;
    let mut continental_isostasy_sq_sum = 0.0f64;
    for cell in 0..total {
        if proto_ocean[cell] < 0.5 {
            let area = areas[cell];
            continental_area += area;
            continental_isostasy_sum += isostasy[cell] as f64 * area;
            continental_isostasy_sq_sum += isostasy[cell] as f64 * isostasy[cell] as f64 * area;
        }
    }
    let basin_isostasy_threshold = if continental_area > 0.0 {
        let continental_isostasy_mean = continental_isostasy_sum / continental_area;
        let continental_isostasy_std = (continental_isostasy_sq_sum / continental_area
            - continental_isostasy_mean * continental_isostasy_mean)
            .max(0.0)
            .sqrt();
        (continental_isostasy_mean - 0.45 * continental_isostasy_std) as f32
    } else {
        f32::NEG_INFINITY
    };

    let oceanic: Vec<bool> = proto_ocean.iter().map(|value| *value >= 0.5).collect();
    for cell in 0..total {
        let (class_code, confidence) = province_class(
            oceanic[cell],
            compression[cell].clamp(0.0, 1.0),
            extension[cell].clamp(0.0, 1.0),
            subduction[cell].clamp(0.0, 1.0),
            uplift[cell].max(0.0),
            subsidence[cell].max(0.0),
            isostasy[cell],
            margin[cell].clamp(0.0, 1.0),
            stiffness[cell].clamp(0.0, 1.0),
            basin_isostasy_threshold,
        );
        province_classes[cell] = class_code;
        province_confidence[cell] = confidence;
        rock_strength[cell] = (0.28 + 0.58 * stiffness[cell].clamp(0.0, 1.0)
            - 0.18 * extension[cell].clamp(0.0, 1.0)
            - 0.1 * subduction[cell].clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        accommodation[cell] = (0.48 * subsidence[cell].max(0.0)
            + 0.28 * extension[cell].clamp(0.0, 1.0)
            + 0.18 * (1.0 - stiffness[cell].clamp(0.0, 1.0))
            + 0.12 * (-isostasy[cell]).max(0.0))
        .clamp(0.0, 1.0);
    }

    merge_small_provinces(
        province_classes,
        province_confidence,
        &oceanic,
        neighbors,
        areas,
        total_area,
    );

    let mut age_area_sum = 0.0f64;
    let mut confidence_area_sum = 0.0f64;
    for cell in 0..total {
        let class_code = province_classes[cell];
        crust_age[cell] = if oceanic[cell] {
            let oceanic_max_age = world_age_ga.min(0.25);
            (oceanic_max_age
                * (0.12 + 0.72 * stiffness[cell].clamp(0.0, 1.0)
                    - 0.5 * extension[cell].clamp(0.0, 1.0)))
            .clamp(0.005, oceanic_max_age)
        } else {
            let age_fraction = match class_code {
                PROVINCE_SHIELD => 0.78 + 0.2 * stiffness[cell],
                PROVINCE_PLATFORM => 0.38 + 0.34 * stiffness[cell],
                PROVINCE_SEDIMENTARY_BASIN => 0.3 + 0.3 * stiffness[cell],
                PROVINCE_OROGEN | PROVINCE_CONTINENTAL_ARC => 0.42 + 0.32 * stiffness[cell],
                PROVINCE_CONTINENTAL_RIFT => 0.3 + 0.3 * stiffness[cell],
                PROVINCE_SHELF => 0.36 + 0.3 * stiffness[cell],
                PROVINCE_VOLCANIC => 0.12 + 0.24 * stiffness[cell],
                _ => 0.35 + 0.3 * stiffness[cell],
            };
            (world_age_ga
                * (age_fraction
                    - 0.14 * extension[cell].clamp(0.0, 1.0)
                    - 0.08 * subduction[cell].clamp(0.0, 1.0)))
            .clamp(0.05, world_age_ga)
        };
        let area = areas[cell];
        age_area_sum += crust_age[cell] as f64 * area;
        confidence_area_sum += province_confidence[cell] as f64 * area;
    }

    province_ids.fill(-1);
    let mut province_records = Vec::new();
    let mut queue = VecDeque::new();
    for start in 0..total {
        if province_ids[start] >= 0 {
            continue;
        }
        let province_id = province_records.len() as i32;
        let class_code = province_classes[start];
        let initial_plate = plate_id(plate_field, start, components);
        let mut parent_plate = initial_plate;
        let mut cells = 0i32;
        let mut area_sum = 0.0f64;
        let mut age_sum = 0.0f64;
        let mut strength_sum = 0.0f64;
        let mut accommodation_sum = 0.0f64;
        let mut confidence_sum = 0.0f64;
        province_ids[start] = province_id;
        queue.push_back(start);
        while let Some(cell) = queue.pop_front() {
            cells += 1;
            let area = areas[cell];
            area_sum += area;
            age_sum += crust_age[cell] as f64 * area;
            strength_sum += rock_strength[cell] as f64 * area;
            accommodation_sum += accommodation[cell] as f64 * area;
            confidence_sum += province_confidence[cell] as f64 * area;
            if plate_id(plate_field, cell, components) != initial_plate {
                parent_plate = -1;
            }
            for &neighbor in &neighbors[cell * 4..cell * 4 + 4] {
                let adjacent = neighbor as usize;
                if province_ids[adjacent] < 0 && province_classes[adjacent] == class_code {
                    province_ids[adjacent] = province_id;
                    queue.push_back(adjacent);
                }
            }
        }
        province_records.push(ProvinceRecord {
            province_id,
            class_code: class_code as i32,
            parent_plate_id: parent_plate,
            cell_count: cells,
            area_steradians: area_sum,
            mean_crust_age_ga: (age_sum / area_sum) as f32,
            mean_rock_strength: (strength_sum / area_sum) as f32,
            mean_accommodation: (accommodation_sum / area_sum) as f32,
            mean_confidence: (confidence_sum / area_sum) as f32,
        });
    }

    segment_ids.fill(-1);
    regimes.fill(0);
    boundary_confidence.fill(0.0);
    let mut edge_by_slot = vec![-1i32; neighbor_len];
    let mut edges = Vec::new();
    for source in 0..total {
        let source_plate = plate_id(plate_field, source, components);
        for source_slot in 0..D4_NEIGHBORS {
            let target = neighbors[source * 4 + source_slot] as usize;
            if source >= target {
                continue;
            }
            let target_plate = plate_id(plate_field, target, components);
            if source_plate == target_plate {
                continue;
            }
            let Some(target_slot) = neighbors[target * 4..target * 4 + 4]
                .iter()
                .position(|neighbor| *neighbor as usize == source)
            else {
                return 4;
            };
            let mean_compression = 0.5 * (compression[source] + compression[target]);
            let mean_extension = 0.5 * (extension[source] + extension[target]);
            let mean_shear = 0.5 * (shear[source] + shear[target]);
            let mean_subduction = 0.5 * (subduction[source] + subduction[target]);
            let (regime, confidence) = boundary_regime(
                is_continental(plate_field, source, components),
                is_continental(plate_field, target, components),
                mean_compression.clamp(0.0, 1.0),
                mean_extension.clamp(0.0, 1.0),
                mean_shear.clamp(0.0, 1.0),
                mean_subduction.clamp(0.0, 1.0),
            );
            let edge_index = edges.len() as i32;
            edge_by_slot[source * 4 + source_slot] = edge_index;
            edge_by_slot[target * 4 + target_slot] = edge_index;
            edges.push(BoundaryEdge {
                source,
                target,
                source_slot,
                target_slot,
                plate_a: source_plate.min(target_plate),
                plate_b: source_plate.max(target_plate),
                regime,
                confidence,
                compression: mean_compression,
                extension: mean_extension,
                shear: mean_shear,
                subduction: mean_subduction,
            });
        }
    }
    smooth_boundary_regimes(&mut edges, &edge_by_slot, neighbors);
    merge_short_boundary_segments(&mut edges, &edge_by_slot, neighbors, areas);

    let mut segment_records = Vec::new();
    let mut edge_segments = vec![-1i32; edges.len()];
    for start in 0..edges.len() {
        if edge_segments[start] >= 0 {
            continue;
        }
        let segment_id = segment_records.len() as i32;
        let regime = edges[start].regime;
        let plate_a = edges[start].plate_a;
        let plate_b = edges[start].plate_b;
        let mut edge_count = 0i32;
        let mut angular_length = 0.0f64;
        let mut compression_sum = 0.0f64;
        let mut extension_sum = 0.0f64;
        let mut shear_sum = 0.0f64;
        let mut subduction_sum = 0.0f64;
        let mut confidence_sum = 0.0f64;
        edge_segments[start] = segment_id;
        queue.push_back(start);
        while let Some(edge_index) = queue.pop_front() {
            let edge = &edges[edge_index];
            edge_count += 1;
            angular_length += 0.5 * (areas[edge.source].sqrt() + areas[edge.target].sqrt());
            compression_sum += edge.compression as f64;
            extension_sum += edge.extension as f64;
            shear_sum += edge.shear as f64;
            subduction_sum += edge.subduction as f64;
            confidence_sum += edge.confidence as f64;
            for adjacent_index in
                adjacent_boundary_edges(edge_index, &edges, &edge_by_slot, neighbors)
            {
                let adjacent = &edges[adjacent_index];
                if edge_segments[adjacent_index] < 0
                    && adjacent.regime == regime
                    && adjacent.plate_a == plate_a
                    && adjacent.plate_b == plate_b
                {
                    edge_segments[adjacent_index] = segment_id;
                    queue.push_back(adjacent_index);
                }
            }
        }
        let count = edge_count as f64;
        segment_records.push(BoundarySegmentRecord {
            segment_id,
            regime_code: regime as i32,
            plate_a,
            plate_b,
            edge_count,
            angular_length,
            mean_compression: (compression_sum / count) as f32,
            mean_extension: (extension_sum / count) as f32,
            mean_shear: (shear_sum / count) as f32,
            mean_subduction: (subduction_sum / count) as f32,
            mean_confidence: (confidence_sum / count) as f32,
        });
    }

    for (edge_index, edge) in edges.iter().enumerate() {
        let segment_id = edge_segments[edge_index];
        let source_slot = edge.source * 4 + edge.source_slot;
        let target_slot = edge.target * 4 + edge.target_slot;
        for slot in [source_slot, target_slot] {
            segment_ids[slot] = segment_id;
            regimes[slot] = edge.regime;
            boundary_confidence[slot] = edge.confidence;
        }
    }

    let mixed_plate_provinces = province_records
        .iter()
        .filter(|record| record.parent_plate_id < 0)
        .count();
    let active_segments = segment_records
        .iter()
        .filter(|record| record.regime_code != REGIME_INACTIVE as i32)
        .count();
    let province_count = province_records.len();
    let segment_count = segment_records.len();

    if !provinces_out.is_null() {
        (*provinces_out).data = std::ptr::null_mut();
        (*provinces_out).len = 0;
        if province_count > 0 {
            let records = province_records.into_boxed_slice();
            (*provinces_out).data = Box::into_raw(records) as *mut ProvinceRecord;
            (*provinces_out).len = province_count;
        }
    }
    if !segments_out.is_null() {
        (*segments_out).data = std::ptr::null_mut();
        (*segments_out).len = 0;
        if segment_count > 0 {
            let records = segment_records.into_boxed_slice();
            (*segments_out).data = Box::into_raw(records) as *mut BoundarySegmentRecord;
            (*segments_out).len = segment_count;
        }
    }
    if !stats_out.is_null() {
        (*stats_out) = GeologyStats {
            province_count: province_count.min(i32::MAX as usize) as i32,
            boundary_segment_count: segment_count.min(i32::MAX as usize) as i32,
            active_boundary_segment_count: active_segments.min(i32::MAX as usize) as i32,
            mixed_plate_province_count: mixed_plate_provinces.min(i32::MAX as usize) as i32,
            continental_area,
            oceanic_area: total_area - continental_area,
            mean_crust_age_ga: (age_area_sum / total_area) as f32,
            mean_confidence: (confidence_area_sum / total_area) as f32,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_major_structural_regimes() {
        assert_eq!(
            province_class(false, 0.7, 0.0, 0.5, 0.6, 0.0, 0.2, 0.0, 0.7, 0.0).0,
            PROVINCE_CONTINENTAL_ARC
        );
        assert_eq!(
            province_class(false, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.4, 0.0).0,
            PROVINCE_CONTINENTAL_RIFT
        );
        assert_eq!(
            province_class(true, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0).0,
            PROVINCE_OCEANIC_RIDGE
        );
    }

    #[test]
    fn distinguishes_collision_subduction_rift_and_transform() {
        assert_eq!(boundary_regime(true, true, 0.8, 0.0, 0.1, 0.2).0, 2);
        assert_eq!(boundary_regime(true, false, 0.6, 0.0, 0.1, 0.7).0, 3);
        assert_eq!(boundary_regime(true, true, 0.0, 0.7, 0.1, 0.0).0, 5);
        assert_eq!(boundary_regime(false, false, 0.0, 0.0, 0.7, 0.0).0, 7);
    }
}
