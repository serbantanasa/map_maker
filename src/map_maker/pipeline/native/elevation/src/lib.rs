use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::slice;

const D4_NEIGHBORS: usize = 4;
const EARTH_RADIUS_KM: f32 = 6371.0;

const REGIME_CONTINENTAL_COLLISION: u8 = 2;
const REGIME_SUBDUCTION_MARGIN: u8 = 3;
const REGIME_INTRA_OCEANIC_SUBDUCTION: u8 = 4;
const REGIME_CONTINENTAL_RIFT: u8 = 5;
const REGIME_SPREADING_RIDGE: u8 = 6;
const REGIME_TRANSFORM: u8 = 7;

#[repr(C)]
pub struct ElevationStats {
    pub elevation_min_m: f32,
    pub elevation_mean_m: f32,
    pub elevation_max_m: f32,
    pub continental_mean_m: f32,
    pub oceanic_mean_m: f32,
    pub maximum_orogenic_m: f32,
    pub maximum_basin_m: f32,
    pub high_mountain_area_fraction: f64,
    pub deep_ocean_area_fraction: f64,
    pub active_relief_area_fraction: f64,
}

#[derive(Clone, Copy)]
struct QueueEntry {
    distance: f64,
    cell: usize,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

#[no_mangle]
pub extern "C" fn elevation_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_elevation_abi_version() -> u32 {
    1
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn random_signed(seed: u64, cell: usize) -> f32 {
    let bits = splitmix64(seed ^ (cell as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let unit = (bits >> 40) as f32 / ((1u32 << 24) - 1) as f32;
    unit * 2.0 - 1.0
}

fn smooth_graph(values: &mut [f32], neighbors: &[i32], passes: usize) {
    let mut next = vec![0.0f32; values.len()];
    for _ in 0..passes {
        for cell in 0..values.len() {
            let adjacent_sum: f32 = neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS]
                .iter()
                .map(|neighbor| values[*neighbor as usize])
                .sum();
            next[cell] = values[cell] * 0.5 + adjacent_sum * 0.125;
        }
        values.copy_from_slice(&next);
    }
}

fn smooth_within_plates(values: &mut [f32], plate_ids: &[i32], neighbors: &[i32], passes: usize) {
    let mut next = values.to_vec();
    for _ in 0..passes {
        for cell in 0..values.len() {
            if !values[cell].is_finite() {
                next[cell] = values[cell];
                continue;
            }
            let mut sum = values[cell] * 4.0;
            let mut weight = 4.0f32;
            for neighbor in &neighbors[cell * D4_NEIGHBORS..(cell + 1) * D4_NEIGHBORS] {
                let adjacent = *neighbor as usize;
                if plate_ids[adjacent] == plate_ids[cell] && values[adjacent].is_finite() {
                    sum += values[adjacent];
                    weight += 1.0;
                }
            }
            next[cell] = sum / weight;
        }
        values.copy_from_slice(&next);
    }
}

fn normalized_noise(seed: u64, areas: &[f64], neighbors: &[i32]) -> Vec<f32> {
    let mut broad: Vec<f32> = (0..areas.len())
        .map(|cell| random_signed(seed ^ 0xA076_1D64_78BD_642F, cell))
        .collect();
    let mut medium: Vec<f32> = (0..areas.len())
        .map(|cell| random_signed(seed ^ 0xE703_7ED1_A0B4_28DB, cell))
        .collect();
    smooth_graph(&mut broad, neighbors, 36);
    smooth_graph(&mut medium, neighbors, 9);
    let mut result: Vec<f32> = broad
        .iter()
        .zip(medium.iter())
        .map(|(low, mid)| 0.72 * low + 0.28 * mid)
        .collect();
    let area_total: f64 = areas.iter().sum();
    let mean = result
        .iter()
        .zip(areas.iter())
        .map(|(value, area)| *value as f64 * *area)
        .sum::<f64>()
        / area_total;
    let variance = result
        .iter()
        .zip(areas.iter())
        .map(|(value, area)| {
            let centered = *value as f64 - mean;
            centered * centered * *area
        })
        .sum::<f64>()
        / area_total;
    let scale = variance.sqrt().max(1e-8) as f32;
    for value in &mut result {
        *value = (((*value - mean as f32) / scale) * 0.72).tanh();
    }
    result
}

fn nearest_source(
    source_strength: &[f32],
    plate_ids: &[i32],
    neighbors: &[i32],
    areas: &[f64],
) -> (Vec<f32>, Vec<f32>) {
    let mut distances = vec![f64::INFINITY; source_strength.len()];
    let mut amplitudes = vec![0.0f32; source_strength.len()];
    let mut queue = BinaryHeap::new();
    for (cell, strength) in source_strength.iter().enumerate() {
        if *strength > 0.0 {
            distances[cell] = 0.0;
            amplitudes[cell] = *strength;
            queue.push(QueueEntry {
                distance: 0.0,
                cell,
            });
        }
    }
    while let Some(entry) = queue.pop() {
        if entry.distance > distances[entry.cell] + 1e-12 {
            continue;
        }
        for neighbor in &neighbors[entry.cell * D4_NEIGHBORS..(entry.cell + 1) * D4_NEIGHBORS] {
            let target = *neighbor as usize;
            if plate_ids[target] != plate_ids[entry.cell] {
                continue;
            }
            let step = 0.5 * (areas[entry.cell].sqrt() + areas[target].sqrt());
            let candidate = entry.distance + step;
            let improved = candidate + 1e-12 < distances[target];
            let tied_but_stronger = (candidate - distances[target]).abs() <= 1e-12
                && amplitudes[entry.cell] > amplitudes[target];
            if improved || tied_but_stronger {
                distances[target] = candidate;
                amplitudes[target] = amplitudes[entry.cell];
                queue.push(QueueEntry {
                    distance: candidate,
                    cell: target,
                });
            }
        }
    }
    (
        distances
            .into_iter()
            .map(|value| {
                if value.is_finite() {
                    value.min(f32::MAX as f64) as f32
                } else {
                    f32::INFINITY
                }
            })
            .collect(),
        amplitudes,
    )
}

fn gaussian(distance: f32, center: f32, sigma: f32) -> f32 {
    if !distance.is_finite() {
        return 0.0;
    }
    let z = (distance - center) / sigma.max(1e-6);
    (-0.5 * z * z).exp()
}

fn age_factor_for_noise(crust_age_ga: f32) -> f32 {
    (crust_age_ga.max(0.0) / 0.22).clamp(0.0, 1.0).powf(0.85)
}

fn corridor_activity(value: f32) -> f32 {
    let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
    let smooth = normalized * normalized * (3.0 - 2.0 * normalized);
    0.18 + 0.82 * smooth
}

fn km_to_radians(km: f32) -> f32 {
    km / EARTH_RADIUS_KM
}

fn update_source(source: &mut [f32], cell: usize, value: f32) {
    source[cell] = source[cell].max(value.clamp(0.0, 1.0));
}

fn finite_slice(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_elevation(
    total: usize,
    seed: u64,
    collision_height_m: f32,
    arc_height_m: f32,
    ridge_height_m: f32,
    trench_depth_m: f32,
    rift_depth_m: f32,
    areas: &[f64],
    neighbors: &[i32],
    plate_field: &[f32],
    plate_components: usize,
    crust_thickness: &[f32],
    isostasy: &[f32],
    uplift: &[f32],
    subsidence: &[f32],
    compression: &[f32],
    extension: &[f32],
    shear: &[f32],
    stiffness: &[f32],
    proto_ocean: &[f32],
    hotspot: &[f32],
    crust_age: &[f32],
    rock_strength: &[f32],
    accommodation: &[f32],
    province_confidence: &[f32],
    boundary_regime: &[u8],
    boundary_confidence: &[f32],
    crustal_out: &mut [f32],
    orogenic_out: &mut [f32],
    basin_out: &mut [f32],
    bedrock_out: &mut [f32],
    relief_out: &mut [f32],
    confidence_out: &mut [f32],
) -> ElevationStats {
    let plate_ids: Vec<i32> = plate_field
        .chunks_exact(plate_components)
        .map(|plate| plate[0].round() as i32)
        .collect();
    let continental: Vec<bool> = proto_ocean.iter().map(|value| *value < 0.5).collect();
    let mut land_area = 0.0f64;
    let mut ocean_area = 0.0f64;
    let mut land_isostasy = 0.0f64;
    let mut ocean_isostasy = 0.0f64;
    let mut land_thickness = 0.0f64;
    let mut ocean_thickness = 0.0f64;
    for cell in 0..total {
        if continental[cell] {
            land_area += areas[cell];
            land_isostasy += isostasy[cell] as f64 * areas[cell];
            land_thickness += crust_thickness[cell] as f64 * areas[cell];
        } else {
            ocean_area += areas[cell];
            ocean_isostasy += isostasy[cell] as f64 * areas[cell];
            ocean_thickness += crust_thickness[cell] as f64 * areas[cell];
        }
    }
    let mean_land_iso = (land_isostasy / land_area.max(f64::EPSILON)) as f32;
    let mean_ocean_iso = (ocean_isostasy / ocean_area.max(f64::EPSILON)) as f32;
    let mean_land_thickness = (land_thickness / land_area.max(f64::EPSILON)) as f32;
    let mean_ocean_thickness = (ocean_thickness / ocean_area.max(f64::EPSILON)) as f32;

    let mut collision = vec![0.0f32; total];
    let mut descending = vec![0.0f32; total];
    let mut overriding = vec![0.0f32; total];
    let mut ridge = vec![0.0f32; total];
    let mut rift = vec![0.0f32; total];
    let mut transform = vec![0.0f32; total];
    let noise = normalized_noise(seed, areas, neighbors);
    let corridor_noise = normalized_noise(seed ^ 0x8EBC_6AF0_9C88_C6E3, areas, neighbors);

    for source in 0..total {
        for slot in 0..D4_NEIGHBORS {
            let target = neighbors[source * D4_NEIGHBORS + slot] as usize;
            if source >= target {
                continue;
            }
            let edge = source * D4_NEIGHBORS + slot;
            let regime = boundary_regime[edge];
            let confidence = boundary_confidence[edge].clamp(0.0, 1.0);
            if confidence <= 0.0 {
                continue;
            }
            let activity =
                corridor_activity(0.5 * (corridor_noise[source] + corridor_noise[target]));
            let compression_signal = (0.45
                + 0.35 * (compression[source] + compression[target])
                + 0.10 * (uplift[source] + uplift[target]))
                .clamp(0.0, 1.0);
            let extension_signal =
                (0.45 + 0.40 * (extension[source] + extension[target])).clamp(0.0, 1.0);
            let shear_signal = (0.35 + 0.45 * (shear[source] + shear[target])).clamp(0.0, 1.0);
            match regime {
                REGIME_CONTINENTAL_COLLISION => {
                    let strength = confidence * compression_signal * activity;
                    update_source(&mut collision, source, strength);
                    update_source(&mut collision, target, strength);
                }
                REGIME_SUBDUCTION_MARGIN | REGIME_INTRA_OCEANIC_SUBDUCTION => {
                    let source_ocean = !continental[source];
                    let target_ocean = !continental[target];
                    let source_density = plate_field[source * plate_components + 3];
                    let target_density = plate_field[target * plate_components + 3];
                    let source_descends = if source_ocean != target_ocean {
                        source_ocean
                    } else if (crust_age[source] - crust_age[target]).abs() > 1e-6 {
                        crust_age[source] > crust_age[target]
                    } else {
                        source_density >= target_density
                    };
                    let (down, over) = if source_descends {
                        (source, target)
                    } else {
                        (target, source)
                    };
                    let strength = confidence * compression_signal * activity;
                    update_source(&mut descending, down, strength);
                    update_source(&mut overriding, over, strength);
                }
                REGIME_SPREADING_RIDGE => {
                    let strength = confidence * extension_signal * activity;
                    update_source(&mut ridge, source, strength);
                    update_source(&mut ridge, target, strength);
                }
                REGIME_CONTINENTAL_RIFT => {
                    let strength = confidence * extension_signal * activity;
                    update_source(&mut rift, source, strength);
                    update_source(&mut rift, target, strength);
                }
                REGIME_TRANSFORM => {
                    let strength = confidence * shear_signal * activity;
                    update_source(&mut transform, source, strength);
                    update_source(&mut transform, target, strength);
                }
                _ => {}
            }
        }
    }

    let (mut collision_distance, mut collision_strength) =
        nearest_source(&collision, &plate_ids, neighbors, areas);
    let (mut descending_distance, mut descending_strength) =
        nearest_source(&descending, &plate_ids, neighbors, areas);
    let (mut overriding_distance, mut overriding_strength) =
        nearest_source(&overriding, &plate_ids, neighbors, areas);
    let (mut ridge_distance, mut ridge_strength) =
        nearest_source(&ridge, &plate_ids, neighbors, areas);
    let (mut rift_distance, mut rift_strength) =
        nearest_source(&rift, &plate_ids, neighbors, areas);
    let (mut transform_distance, mut transform_strength) =
        nearest_source(&transform, &plate_ids, neighbors, areas);
    let (mut hotspot_distance, mut hotspot_strength) =
        nearest_source(hotspot, &plate_ids, neighbors, areas);
    for field in [
        &mut collision_distance,
        &mut descending_distance,
        &mut overriding_distance,
        &mut ridge_distance,
        &mut rift_distance,
        &mut transform_distance,
    ] {
        smooth_within_plates(field, &plate_ids, neighbors, 2);
    }
    for (distance, sources) in [
        (&mut collision_distance, &collision),
        (&mut descending_distance, &descending),
        (&mut overriding_distance, &overriding),
        (&mut ridge_distance, &ridge),
        (&mut rift_distance, &rift),
        (&mut transform_distance, &transform),
    ] {
        for (value, source) in distance.iter_mut().zip(sources.iter()) {
            if *source > 0.0 {
                *value = 0.0;
            }
        }
    }
    for field in [
        &mut collision_strength,
        &mut descending_strength,
        &mut overriding_strength,
        &mut ridge_strength,
        &mut rift_strength,
        &mut transform_strength,
    ] {
        smooth_within_plates(field, &plate_ids, neighbors, 3);
    }
    smooth_within_plates(&mut hotspot_distance, &plate_ids, neighbors, 1);
    smooth_within_plates(&mut hotspot_strength, &plate_ids, neighbors, 2);
    for (distance, source) in hotspot_distance.iter_mut().zip(hotspot.iter()) {
        if *source > 0.0 {
            *distance = 0.0;
        }
    }
    let mut regional_compression = compression.to_vec();
    smooth_graph(&mut regional_compression, neighbors, 30);

    let mut elevation_min = f32::INFINITY;
    let mut elevation_max = f32::NEG_INFINITY;
    let mut elevation_sum = 0.0f64;
    let mut continental_sum = 0.0f64;
    let mut oceanic_sum = 0.0f64;
    let mut high_area = 0.0f64;
    let mut deep_area = 0.0f64;
    let mut active_area = 0.0f64;
    let mut maximum_orogenic = 0.0f32;
    let mut maximum_basin = 0.0f32;
    let total_area = land_area + ocean_area;

    for cell in 0..total {
        let structural_variation = noise[cell];
        let corridor_variation = corridor_activity(corridor_noise[cell]);
        let crustal = if continental[cell] {
            // Continental freeboard: modest plateaus and platforms before orogeny.
            380.0
                + (isostasy[cell] - mean_land_iso) * 780.0
                + (crust_thickness[cell] - mean_land_thickness) * 68.0
        } else {
            // Age-deepened abyssal plain with more vertical range than the prior
            // near-constant mid-ocean floor.
            let age_factor = (crust_age[cell].max(0.0) / 0.22).clamp(0.0, 1.0).powf(0.85);
            -2_150.0 - age_factor * 3_350.0
                + (isostasy[cell] - mean_ocean_iso) * 520.0
                + (crust_thickness[cell] - mean_ocean_thickness) * 95.0
        };

        // Config heights are peak-of-corridor *scales*. Cell-mean orogeny uses a
        // reduced factor so ~5e3 km² tiles stay in range/plateau-mean territory;
        // TerrainReliefM uses the full scales for subgrid peaks.
        const MEAN_OROG_SCALE: f32 = 0.62;
        const OROGENIC_MEAN_CAP_M: f32 = 3_000.0;
        const CONTINENTAL_MEAN_CAP_M: f32 = 3_500.0;

        let collision_width = km_to_radians(240.0 + 190.0 * stiffness[cell]);
        let collision_offset = km_to_radians(55.0 + 70.0 * (corridor_noise[cell] + 1.0));
        let collision_kernel = collision_strength[cell]
            * gaussian(collision_distance[cell], collision_offset, collision_width)
            * (0.82 + 0.38 * structural_variation);
        let collision_relief = collision_height_m * MEAN_OROG_SCALE * collision_kernel;
        let arc_center =
            if continental[cell] { 170.0 } else { 135.0 } + 75.0 * corridor_noise[cell];
        let arc_width = if continental[cell] { 130.0 } else { 100.0 };
        let arc_kernel = overriding_strength[cell]
            * gaussian(
                overriding_distance[cell],
                km_to_radians(arc_center),
                km_to_radians(arc_width),
            )
            * (0.80 + 0.45 * structural_variation);
        let arc_relief = arc_height_m * MEAN_OROG_SCALE * arc_kernel;
        let ridge_relief = if continental[cell] {
            0.0
        } else {
            ridge_height_m
                * MEAN_OROG_SCALE
                * ridge_strength[cell]
                * gaussian(
                    ridge_distance[cell],
                    km_to_radians((80.0 + 80.0 * corridor_noise[cell]).max(0.0)),
                    km_to_radians(250.0),
                )
                * corridor_variation
        };
        let rift_shoulder = rift_depth_m
            * 0.55
            * MEAN_OROG_SCALE
            * rift_strength[cell]
            * gaussian(
                rift_distance[cell],
                km_to_radians(200.0),
                km_to_radians(105.0),
            );
        let transform_relief = 680.0
            * MEAN_OROG_SCALE
            * transform_strength[cell]
            * gaussian(transform_distance[cell], 0.0, km_to_radians(75.0))
            * (0.55 + 0.45 * structural_variation.abs());
        let volcanic_relief = hotspot_strength[cell].clamp(0.0, 1.5)
            * if continental[cell] { 1_400.0 } else { 2_200.0 }
            * gaussian(
                hotspot_distance[cell],
                0.0,
                km_to_radians(if continental[cell] { 200.0 } else { 150.0 }),
            )
            * (0.72 + 0.34 * structural_variation);
        let distributed_compression = if continental[cell] {
            let compression_factor = (regional_compression[cell].clamp(0.0, 1.0) / 0.11)
                .clamp(0.0, 1.0)
                .powf(0.95);
            2_800.0
                * MEAN_OROG_SCALE
                * compression_factor
                * (0.58 + 0.42 * uplift[cell].clamp(0.0, 1.0))
                * (0.74 + 0.36 * structural_variation)
                * (0.42 + 0.58 * corridor_variation)
        } else {
            0.0
        };
        // Max-of-processes for cell-mean massifs (no stacking to sky plateaus).
        let orogenic_raw = (collision_relief
            + arc_relief
            + ridge_relief
            + rift_shoulder
            + transform_relief
            + volcanic_relief)
            .max(distributed_compression)
            .max(0.0);
        let orogenic = orogenic_raw.min(OROGENIC_MEAN_CAP_M);

        let accommodation_depth = accommodation[cell].clamp(0.0, 1.0)
            * if continental[cell] { 780.0 } else { 640.0 }
            * (0.45 + 0.55 * subsidence[cell].clamp(0.0, 1.0));
        let extension_depth = if continental[cell] {
            extension[cell].clamp(0.0, 1.0) * 420.0
        } else {
            subsidence[cell].clamp(0.0, 1.0) * 380.0
        };
        // Full trench/forearc only on oceanic lithosphere. Continental cells may
        // feel a shallow margin trench shadow, not multi-kilometre dry holes.
        let trench_weight = if continental[cell] { 0.12 } else { 1.0 };
        let trench = trench_depth_m
            * trench_weight
            * descending_strength[cell]
            * gaussian(
                descending_distance[cell],
                km_to_radians((100.0 + 80.0 * corridor_noise[cell]).max(20.0)),
                km_to_radians(if continental[cell] { 90.0 } else { 140.0 }),
            )
            * corridor_variation;
        let forearc = trench_depth_m
            * if continental[cell] { 0.04 } else { 0.16 }
            * overriding_strength[cell]
            * gaussian(
                overriding_distance[cell],
                km_to_radians(50.0),
                km_to_radians(42.0),
            );
        let rift_axis = rift_depth_m
            * rift_strength[cell]
            * gaussian(
                rift_distance[cell],
                km_to_radians((90.0 + 90.0 * corridor_noise[cell]).max(0.0)),
                km_to_radians(190.0),
            )
            * corridor_variation;
        let mut basin =
            (accommodation_depth + extension_depth + trench + forearc + rift_axis).max(0.0);
        if continental[cell] {
            // Keep emerged continental basins as sedimentary lowlands, not
            // unfilled oceanic chasms without ice or lake cover.
            basin = basin.min(1_350.0);
        }

        let background_amplitude = if continental[cell] {
            280.0 + 620.0 * (1.0 - rock_strength[cell].clamp(0.0, 1.0))
        } else {
            // Abyssal roughness and fracture-zone scale variation.
            180.0 + 260.0 * shear[cell].clamp(0.0, 1.0) + 140.0 * age_factor_for_noise(crust_age[cell])
        };
        let background =
            structural_variation * (background_amplitude + 0.08 * orogenic + 0.04 * basin);
        let mut bedrock = crustal + orogenic - basin + background;
        if continental[cell] {
            // Bound dry continental lows; true hydrologic lakes are later stages.
            bedrock = bedrock.max(-650.0);
            // Hard stop: no ~5e3 km² sky plateaus (user contract).
            bedrock = bedrock.min(CONTINENTAL_MEAN_CAP_M);
        }
        // Subgrid amplitude: peaks over a moderate cell-mean range.
        // Uses full config height scales + uncapped orogenic potential.
        let relief_orogen = orogenic_raw.max(orogenic);
        let relief = (if continental[cell] { 220.0 } else { 110.0 }
            + 1.05 * relief_orogen
            + 0.22 * basin
            + 620.0 * shear[cell].clamp(0.0, 1.0)
            + 480.0 * (1.0 - rock_strength[cell].clamp(0.0, 1.0))
            + 0.55 * collision_kernel * collision_height_m
            + 0.35 * arc_kernel * arc_height_m)
        .clamp(100.0, 5_500.0);
        let boundary_evidence = (collision_strength[cell]
            * gaussian(collision_distance[cell], 0.0, km_to_radians(700.0)))
        .max(
            descending_strength[cell]
                * gaussian(descending_distance[cell], 0.0, km_to_radians(420.0)),
        )
        .max(
            overriding_strength[cell]
                * gaussian(overriding_distance[cell], 0.0, km_to_radians(520.0)),
        )
        .max(ridge_strength[cell] * gaussian(ridge_distance[cell], 0.0, km_to_radians(520.0)))
        .max(rift_strength[cell] * gaussian(rift_distance[cell], 0.0, km_to_radians(420.0)))
        .max(
            transform_strength[cell]
                * gaussian(transform_distance[cell], 0.0, km_to_radians(260.0)),
        );
        let confidence =
            (0.28 + 0.57 * province_confidence[cell].clamp(0.0, 1.0) + 0.15 * boundary_evidence)
                .clamp(0.0, 1.0);

        crustal_out[cell] = crustal;
        orogenic_out[cell] = orogenic;
        basin_out[cell] = basin;
        bedrock_out[cell] = bedrock;
        relief_out[cell] = relief;
        confidence_out[cell] = confidence;

        elevation_min = elevation_min.min(bedrock);
        elevation_max = elevation_max.max(bedrock);
        elevation_sum += bedrock as f64 * areas[cell];
        maximum_orogenic = maximum_orogenic.max(orogenic);
        maximum_basin = maximum_basin.max(basin);
        if continental[cell] {
            continental_sum += bedrock as f64 * areas[cell];
        } else {
            oceanic_sum += bedrock as f64 * areas[cell];
        }
        if bedrock > 3000.0 {
            high_area += areas[cell];
        }
        if bedrock < -5000.0 {
            deep_area += areas[cell];
        }
        if orogenic > 250.0 || basin > 400.0 {
            active_area += areas[cell];
        }
    }

    ElevationStats {
        elevation_min_m: elevation_min,
        elevation_mean_m: (elevation_sum / total_area) as f32,
        elevation_max_m: elevation_max,
        continental_mean_m: (continental_sum / land_area.max(f64::EPSILON)) as f32,
        oceanic_mean_m: (oceanic_sum / ocean_area.max(f64::EPSILON)) as f32,
        maximum_orogenic_m: maximum_orogenic,
        maximum_basin_m: maximum_basin,
        high_mountain_area_fraction: high_area / total_area,
        deep_ocean_area_fraction: deep_area / total_area,
        active_relief_area_fraction: active_area / total_area,
    }
}

/// Generate causal pre-erosion elevation on canonical cubed-sphere cells.
///
/// Returns 0 on success, 1 for invalid dimensions or pointers, 2 for invalid
/// parameters, 3 for invalid numeric inputs, and 4 for invalid topology.
///
/// # Safety
///
/// Every input pointer must reference a readable buffer of the length implied
/// by `cell_count` and `plate_components`. Every output pointer must reference
/// a distinct writable `cell_count`-element buffer and must not alias any input
/// or other output. `stats_out` may be null or must reference one writable
/// `ElevationStats` value. Neighbor IDs must be valid global cell indices.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn elevation_run_cubed_sphere(
    cell_count: i32,
    seed: u64,
    collision_height_m: f32,
    arc_height_m: f32,
    ridge_height_m: f32,
    trench_depth_m: f32,
    rift_depth_m: f32,
    plate_components: i32,
    area_ptr: *const f64,
    neighbors_ptr: *const i32,
    plate_ptr: *const f32,
    crust_thickness_ptr: *const f32,
    isostasy_ptr: *const f32,
    uplift_ptr: *const f32,
    subsidence_ptr: *const f32,
    compression_ptr: *const f32,
    extension_ptr: *const f32,
    shear_ptr: *const f32,
    stiffness_ptr: *const f32,
    proto_ocean_ptr: *const f32,
    hotspot_ptr: *const f32,
    crust_age_ptr: *const f32,
    rock_strength_ptr: *const f32,
    accommodation_ptr: *const f32,
    province_confidence_ptr: *const f32,
    boundary_regime_ptr: *const u8,
    boundary_confidence_ptr: *const f32,
    crustal_out_ptr: *mut f32,
    orogenic_out_ptr: *mut f32,
    basin_out_ptr: *mut f32,
    bedrock_out_ptr: *mut f32,
    relief_out_ptr: *mut f32,
    confidence_out_ptr: *mut f32,
    stats_out: *mut ElevationStats,
) -> i32 {
    if cell_count <= 0
        || plate_components < 4
        || area_ptr.is_null()
        || neighbors_ptr.is_null()
        || plate_ptr.is_null()
        || crust_thickness_ptr.is_null()
        || isostasy_ptr.is_null()
        || uplift_ptr.is_null()
        || subsidence_ptr.is_null()
        || compression_ptr.is_null()
        || extension_ptr.is_null()
        || shear_ptr.is_null()
        || stiffness_ptr.is_null()
        || proto_ocean_ptr.is_null()
        || hotspot_ptr.is_null()
        || crust_age_ptr.is_null()
        || rock_strength_ptr.is_null()
        || accommodation_ptr.is_null()
        || province_confidence_ptr.is_null()
        || boundary_regime_ptr.is_null()
        || boundary_confidence_ptr.is_null()
        || crustal_out_ptr.is_null()
        || orogenic_out_ptr.is_null()
        || basin_out_ptr.is_null()
        || bedrock_out_ptr.is_null()
        || relief_out_ptr.is_null()
        || confidence_out_ptr.is_null()
    {
        return 1;
    }
    let parameters = [
        collision_height_m,
        arc_height_m,
        ridge_height_m,
        trench_depth_m,
        rift_depth_m,
    ];
    if parameters
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 20_000.0)
    {
        return 2;
    }

    let total = cell_count as usize;
    let components = plate_components as usize;
    let edge_len = match total.checked_mul(D4_NEIGHBORS) {
        Some(value) => value,
        None => return 1,
    };
    let plate_len = match total.checked_mul(components) {
        Some(value) => value,
        None => return 1,
    };
    let areas = unsafe { slice::from_raw_parts(area_ptr, total) };
    let neighbors = unsafe { slice::from_raw_parts(neighbors_ptr, edge_len) };
    let plate_field = unsafe { slice::from_raw_parts(plate_ptr, plate_len) };
    let crust_thickness = unsafe { slice::from_raw_parts(crust_thickness_ptr, total) };
    let isostasy = unsafe { slice::from_raw_parts(isostasy_ptr, total) };
    let uplift = unsafe { slice::from_raw_parts(uplift_ptr, total) };
    let subsidence = unsafe { slice::from_raw_parts(subsidence_ptr, total) };
    let compression = unsafe { slice::from_raw_parts(compression_ptr, total) };
    let extension = unsafe { slice::from_raw_parts(extension_ptr, total) };
    let shear = unsafe { slice::from_raw_parts(shear_ptr, total) };
    let stiffness = unsafe { slice::from_raw_parts(stiffness_ptr, total) };
    let proto_ocean = unsafe { slice::from_raw_parts(proto_ocean_ptr, total) };
    let hotspot = unsafe { slice::from_raw_parts(hotspot_ptr, total) };
    let crust_age = unsafe { slice::from_raw_parts(crust_age_ptr, total) };
    let rock_strength = unsafe { slice::from_raw_parts(rock_strength_ptr, total) };
    let accommodation = unsafe { slice::from_raw_parts(accommodation_ptr, total) };
    let province_confidence = unsafe { slice::from_raw_parts(province_confidence_ptr, total) };
    let boundary_regime = unsafe { slice::from_raw_parts(boundary_regime_ptr, edge_len) };
    let boundary_confidence = unsafe { slice::from_raw_parts(boundary_confidence_ptr, edge_len) };

    if areas.iter().any(|area| !area.is_finite() || *area <= 0.0)
        || plate_field.iter().any(|value| !value.is_finite())
        || [
            crust_thickness,
            isostasy,
            uplift,
            subsidence,
            compression,
            extension,
            shear,
            stiffness,
            proto_ocean,
            hotspot,
            crust_age,
            rock_strength,
            accommodation,
            province_confidence,
            boundary_confidence,
        ]
        .iter()
        .any(|values| !finite_slice(values))
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

    let crustal_out = unsafe { slice::from_raw_parts_mut(crustal_out_ptr, total) };
    let orogenic_out = unsafe { slice::from_raw_parts_mut(orogenic_out_ptr, total) };
    let basin_out = unsafe { slice::from_raw_parts_mut(basin_out_ptr, total) };
    let bedrock_out = unsafe { slice::from_raw_parts_mut(bedrock_out_ptr, total) };
    let relief_out = unsafe { slice::from_raw_parts_mut(relief_out_ptr, total) };
    let confidence_out = unsafe { slice::from_raw_parts_mut(confidence_out_ptr, total) };

    let stats = unsafe {
        run_elevation(
            total,
            seed,
            collision_height_m,
            arc_height_m,
            ridge_height_m,
            trench_depth_m,
            rift_depth_m,
            areas,
            neighbors,
            plate_field,
            components,
            crust_thickness,
            isostasy,
            uplift,
            subsidence,
            compression,
            extension,
            shear,
            stiffness,
            proto_ocean,
            hotspot,
            crust_age,
            rock_strength,
            accommodation,
            province_confidence,
            boundary_regime,
            boundary_confidence,
            crustal_out,
            orogenic_out,
            basin_out,
            bedrock_out,
            relief_out,
            confidence_out,
        )
    };
    if !stats_out.is_null() {
        unsafe { *stats_out = stats };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_peaks_at_center() {
        assert!((gaussian(0.2, 0.2, 0.1) - 1.0).abs() < 1e-6);
        assert!(gaussian(0.0, 0.2, 0.1) < gaussian(0.1, 0.2, 0.1));
    }

    #[test]
    fn random_field_is_deterministic() {
        assert_eq!(random_signed(42, 9), random_signed(42, 9));
        assert_ne!(random_signed(42, 9), random_signed(43, 9));
    }
}
