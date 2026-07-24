use std::slice;

const SYSTEM_COUNT: usize = 10;
const COMMODITY_COUNT: usize = 15;

#[cfg(test)]
const ARC_MAGMATIC: usize = 0;
#[cfg(test)]
const OROGENIC_SHEAR: usize = 1;
#[cfg(test)]
const MAFIC_ULTRAMAFIC: usize = 2;
#[cfg(test)]
const VOLCANOGENIC_SEAFLOOR: usize = 3;
#[cfg(test)]
const SEDIMENT_HOSTED: usize = 4;
#[cfg(test)]
const ANCIENT_IRON: usize = 5;
#[cfg(test)]
const WEATHERING_SUPERGENE: usize = 6;
#[cfg(test)]
const PLACER: usize = 7;
#[cfg(test)]
const EVAPORITE: usize = 8;
#[cfg(test)]
const COAL: usize = 9;

#[derive(Clone, Copy)]
struct CellInputs {
    ocean: f64,
    shelf: f64,
    relief_m: f64,
    elevation_m: f64,
    terrain_slope: f64,
    province_class: u8,
    crust_age_ga: f64,
    rock_strength: f64,
    accommodation: f64,
    province_confidence: f64,
    elevation_confidence: f64,
    convergence: f64,
    divergence: f64,
    shear: f64,
    subduction: f64,
    hotspot: f64,
    uplift: f64,
    subsidence: f64,
    compression: f64,
    extension: f64,
    stiffness: f64,
    temperature_c: f64,
    precipitation_mm: f64,
    aridity: f64,
    contributing_area_km2: f64,
    stream_power_w: f64,
    river: f64,
    floodplain: f64,
    lake: f64,
    wetland: f64,
    bedrock: f64,
    residual_regolith: f64,
    alluvium: f64,
    lacustrine: f64,
    volcaniclastic: f64,
    soil_depth_m: f64,
    salinity: f64,
    drainage: f64,
    hydric_soil: f64,
    soil_confidence: f64,
    annual_npp: f64,
    standing_biomass: f64,
    vegetation_cover: f64,
    biosphere_confidence: f64,
}

#[derive(Clone, Copy)]
struct Supports {
    source: f64,
    process: f64,
    transport: f64,
    trap: f64,
    timing: f64,
    preservation: f64,
    confidence: f64,
}

impl Supports {
    fn masked(self, eligibility: f64) -> Self {
        let mask = clamp01(eligibility);
        Self {
            source: clamp01(self.source) * mask,
            process: clamp01(self.process) * mask,
            transport: clamp01(self.transport) * mask,
            trap: clamp01(self.trap) * mask,
            timing: clamp01(self.timing) * mask,
            preservation: clamp01(self.preservation) * mask,
            confidence: clamp01(self.confidence),
        }
    }

    fn values(self) -> [f64; 6] {
        [
            self.source,
            self.process,
            self.transport,
            self.trap,
            self.timing,
            self.preservation,
        ]
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn smoothstep01(value: f64) -> f64 {
    let bounded = clamp01(value);
    bounded * bounded * (3.0 - 2.0 * bounded)
}

fn ramp(value: f64, lower: f64, upper: f64) -> f64 {
    smoothstep01((value - lower) / (upper - lower))
}

fn window(value: f64, lower: f64, plateau_lower: f64, plateau_upper: f64, upper: f64) -> f64 {
    ramp(value, lower, plateau_lower) * (1.0 - ramp(value, plateau_upper, upper))
}

fn class_is(class_code: u8, classes: &[u8]) -> f64 {
    if classes.contains(&class_code) {
        1.0
    } else {
        0.0
    }
}

fn max_many(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}

fn combined_confidence(geological: f64, process: f64, observation: f64) -> f64 {
    let values = [
        0.05 + 0.95 * clamp01(geological),
        0.05 + 0.95 * clamp01(process),
        0.05 + 0.95 * clamp01(observation),
    ];
    let geometric = values.iter().product::<f64>().powf(1.0 / 3.0);
    let bottleneck = values.iter().copied().fold(1.0, f64::min);
    clamp01(0.72 * geometric + 0.28 * bottleneck)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn signed_hash(value: u64) -> f64 {
    let mixed = splitmix64(value);
    2.0 * ((mixed >> 11) as f64 / ((1u64 << 53) as f64)) - 1.0
}

fn coherent_wave(
    seed: u64,
    system: usize,
    band: usize,
    lane: usize,
    xyz: [f64; 3],
    face_resolution: usize,
) -> f64 {
    let wave_index = 2 * band + lane;
    let base = seed
        ^ (system as u64 + 1).wrapping_mul(0xa0761d6478bd642f)
        ^ (wave_index as u64 + 1).wrapping_mul(0xe7037ed1a0b428db);
    let mut axis = [
        signed_hash(base),
        signed_hash(base ^ 0x8ebc6af09c88c6e3),
        signed_hash(base ^ 0x589965cc75374cc3),
    ];
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(1e-12);
    for component in &mut axis {
        *component /= length;
    }
    let frequency = face_resolution as f64
        * [1.5, 3.0, 6.0][band]
        * (0.85 + 0.30 * clamp01(0.5 + 0.5 * signed_hash(base ^ 0x1d8e4e27c47d124f)));
    let phase = std::f64::consts::PI * signed_hash(base ^ 0xeb44accab455d165);
    let projected = xyz[0] * axis[0] + xyz[1] * axis[1] + xyz[2] * axis[2];
    (frequency * projected + phase).sin()
}

fn unresolved_support(seed: u64, system: usize, xyz: [f64; 3], face_resolution: usize) -> f64 {
    let mut value = 0.5;
    for (band, amplitude) in [0.15, 0.09, 0.05].iter().copied().enumerate() {
        let paired = (coherent_wave(seed, system, band, 0, xyz, face_resolution)
            + coherent_wave(seed, system, band, 1, xyz, face_resolution))
            * std::f64::consts::FRAC_1_SQRT_2;
        value += amplitude * paired;
    }
    clamp01(value)
}

fn support_fields(cell: CellInputs) -> [Supports; SYSTEM_COUNT] {
    let ocean = clamp01(cell.ocean);
    let land = 1.0 - ocean;
    let shelf = clamp01(cell.shelf);
    let low_slope = 1.0 - ramp(cell.terrain_slope, 0.002, 0.022);
    let steep = ramp(cell.terrain_slope, 0.003, 0.030);
    let high_relief = ramp(cell.relief_m, 250.0, 1_800.0);
    let high_elevation = ramp(cell.elevation_m, 300.0, 2_800.0);
    let exposed = clamp01(0.62 * cell.bedrock + 0.38 * (1.0 - ramp(cell.soil_depth_m, 0.2, 2.5)));
    let covered = clamp01(
        0.30 * cell.residual_regolith
            + 0.25 * cell.alluvium
            + 0.30 * cell.lacustrine
            + 0.15 * ramp(cell.soil_depth_m, 0.2, 2.2),
    );
    let ancient = ramp(cell.crust_age_ga, 1.4, 3.2);
    let young = 1.0 - ramp(cell.crust_age_ga, 0.15, 1.4);
    let intermediate_age = window(cell.crust_age_ga, 0.1, 0.6, 2.4, 3.8);
    let active_tectonics = max_many(&[
        cell.convergence,
        cell.divergence,
        cell.shear,
        cell.subduction,
        cell.hotspot,
        cell.uplift,
        cell.compression,
        cell.extension,
    ]);
    let stable = clamp01(0.55 * cell.stiffness + 0.45 * (1.0 - active_tectonics));
    let basin = max_many(&[
        class_is(cell.province_class, &[3, 7, 8]),
        cell.accommodation,
        cell.subsidence,
        0.55 * cell.extension,
    ]);
    let arc = max_many(&[
        class_is(cell.province_class, &[6, 10]),
        ramp(cell.subduction, 0.01, 0.20),
        ramp(cell.convergence * cell.hotspot, 0.025, 0.30),
    ]);
    let orogen = max_many(&[
        class_is(cell.province_class, &[4]),
        cell.convergence,
        cell.shear,
        cell.compression,
    ]);
    let mafic = max_many(&[
        class_is(cell.province_class, &[9, 10, 11]),
        ramp(cell.hotspot, 0.08, 0.35),
        ramp(cell.divergence, 0.02, 0.55),
        ramp(cell.extension, 0.05, 0.70),
        0.55 * ramp(cell.subduction, 0.01, 0.20),
    ]);
    let craton = max_many(&[
        class_is(cell.province_class, &[1]),
        class_is(cell.province_class, &[2]) * (0.35 + 0.65 * ancient) * cell.stiffness,
        ancient * cell.stiffness,
    ]);
    let river_scale = max_many(&[
        clamp01(cell.river),
        ramp(cell.contributing_area_km2.max(0.0).ln_1p(), 6.0, 12.0),
        ramp(cell.stream_power_w.max(0.0).ln_1p(), 8.0, 19.0),
    ]);
    let stream_energy = max_many(&[
        ramp(cell.stream_power_w.max(0.0).ln_1p(), 9.0, 20.0),
        0.55 * steep,
        0.45 * river_scale,
    ]);
    let depositional_water = max_many(&[
        cell.floodplain,
        cell.lake,
        cell.wetland,
        cell.hydric_soil,
        cell.alluvium,
        cell.lacustrine,
    ]);
    let wet_climate =
        clamp01(ramp(cell.precipitation_mm, 250.0, 1_800.0) * (1.0 - 0.72 * clamp01(cell.aridity)));
    let warm = window(cell.temperature_c, -2.0, 12.0, 30.0, 48.0);
    let dry = clamp01(
        0.72 * ramp(cell.aridity, 0.42, 0.92)
            + 0.28 * (1.0 - ramp(cell.precipitation_mm, 120.0, 700.0)),
    );
    let productive = clamp01(
        0.45 * ramp(cell.annual_npp, 0.03, 0.75)
            + 0.30 * ramp(cell.standing_biomass, 0.5, 12.0)
            + 0.25 * cell.vegetation_cover,
    );
    let geo_conf = clamp01(cell.province_confidence);
    let process_conf = clamp01(0.55 * geo_conf + 0.45 * cell.elevation_confidence);
    let surface_conf =
        clamp01(0.45 * cell.soil_confidence + 0.30 * cell.elevation_confidence + 0.25 * geo_conf);
    let bio_conf = clamp01(cell.biosphere_confidence);
    let weathering_setting = max_many(&[
        ramp(warm * wet_climate, 0.10, 0.58),
        0.65 * ramp(cell.residual_regolith, 0.15, 0.75),
    ]);

    let arc_system = Supports {
        source: 0.12 + 0.46 * young + 0.42 * max_many(&[mafic, cell.volcaniclastic]),
        process: 0.08 + 0.72 * arc + 0.20 * cell.compression,
        transport: 0.10
            + 0.42 * cell.subduction
            + 0.20 * cell.uplift
            + 0.14 * cell.hotspot
            + 0.14 * high_elevation,
        trap: 0.12 + 0.38 * cell.shear + 0.30 * arc + 0.20 * cell.rock_strength,
        timing: 0.10 + 0.58 * active_tectonics + 0.32 * young,
        preservation: 0.12 + 0.32 * cell.rock_strength + 0.28 * exposed + 0.28 * low_slope,
        confidence: combined_confidence(geo_conf, process_conf, cell.elevation_confidence),
    }
    .masked(land * arc);

    let orogenic_system = Supports {
        source: 0.15 + 0.45 * intermediate_age + 0.25 * ancient + 0.15 * cell.rock_strength,
        process: 0.06 + 0.36 * cell.convergence + 0.30 * cell.shear + 0.28 * cell.compression,
        transport: 0.08
            + 0.38 * cell.shear
            + 0.24 * cell.uplift
            + 0.15 * high_relief
            + 0.15 * high_elevation,
        trap: 0.08 + 0.52 * orogen + 0.25 * cell.shear + 0.15 * cell.rock_strength,
        timing: 0.12 + 0.48 * active_tectonics + 0.40 * intermediate_age,
        preservation: 0.14 + 0.40 * cell.rock_strength + 0.24 * exposed + 0.22 * stable,
        confidence: combined_confidence(geo_conf, process_conf, cell.elevation_confidence),
    }
    .masked(land * max_many(&[orogen, 0.65 * class_is(cell.province_class, &[1, 2])]));

    let mafic_system = Supports {
        source: 0.08 + 0.72 * mafic + 0.20 * young,
        process: 0.06
            + 0.37 * cell.hotspot
            + 0.25 * cell.divergence
            + 0.20 * cell.extension
            + 0.12 * cell.subduction,
        transport: 0.12 + 0.38 * active_tectonics + 0.28 * cell.uplift + 0.22 * cell.volcaniclastic,
        trap: 0.15 + 0.42 * cell.rock_strength + 0.28 * mafic + 0.15 * cell.shear,
        timing: 0.12 + 0.55 * young + 0.33 * active_tectonics,
        preservation: 0.12 + 0.42 * cell.rock_strength + 0.26 * exposed + 0.20 * stable,
        confidence: combined_confidence(geo_conf, process_conf, cell.elevation_confidence),
    }
    .masked(land * mafic);

    let seafloor_system = Supports {
        source: 0.10 + 0.50 * mafic + 0.25 * young + 0.15 * cell.volcaniclastic,
        process: 0.05
            + 0.35 * cell.divergence
            + 0.25 * cell.extension
            + 0.24 * cell.subduction
            + 0.11 * cell.hotspot,
        transport: 0.08 + 0.52 * active_tectonics + 0.24 * cell.divergence + 0.16 * cell.subduction,
        trap: 0.12 + 0.38 * basin + 0.30 * shelf + 0.20 * cell.volcaniclastic,
        timing: 0.08 + 0.62 * young + 0.30 * active_tectonics,
        preservation: 0.12 + 0.42 * basin + 0.28 * (1.0 - cell.uplift) + 0.18 * stable,
        confidence: combined_confidence(geo_conf, process_conf, cell.elevation_confidence),
    }
    .masked(ocean * mafic);

    let sediment_system = Supports {
        source: 0.18 + 0.28 * ancient + 0.24 * orogen + 0.18 * exposed + 0.12 * productive,
        process: 0.10 + 0.42 * basin + 0.28 * cell.subsidence + 0.20 * wet_climate,
        transport: 0.10 + 0.48 * river_scale + 0.22 * wet_climate + 0.20 * cell.drainage,
        trap: 0.08 + 0.52 * basin + 0.25 * cell.accommodation + 0.15 * covered,
        timing: 0.15 + 0.45 * stable + 0.25 * cell.subsidence + 0.15 * intermediate_age,
        preservation: 0.10 + 0.42 * covered + 0.30 * cell.accommodation + 0.18 * stable,
        confidence: combined_confidence(geo_conf, process_conf, surface_conf),
    }
    .masked(land * basin);

    let iron_system = Supports {
        source: 0.08 + 0.72 * ancient + 0.20 * craton,
        process: 0.10 + 0.50 * ancient + 0.25 * stable + 0.15 * basin,
        transport: 0.16 + 0.34 * basin + 0.28 * shelf + 0.22 * wet_climate,
        trap: 0.12 + 0.50 * craton + 0.22 * basin + 0.16 * cell.rock_strength,
        timing: 0.06 + 0.76 * ancient + 0.18 * stable,
        preservation: 0.10 + 0.48 * stable + 0.25 * cell.rock_strength + 0.17 * exposed,
        confidence: combined_confidence(geo_conf, process_conf, cell.elevation_confidence),
    }
    .masked(land * max_many(&[craton, 0.45 * class_is(cell.province_class, &[3, 7])]));

    let weathering_system = Supports {
        source: 0.12
            + 0.34 * max_many(&[mafic, craton, ancient])
            + 0.30 * exposed
            + 0.24 * cell.rock_strength,
        process: 0.06 + 0.39 * warm + 0.35 * wet_climate + 0.20 * cell.residual_regolith,
        transport: 0.14 + 0.34 * wet_climate + 0.28 * cell.drainage + 0.24 * low_slope,
        trap: 0.10 + 0.42 * low_slope + 0.28 * cell.residual_regolith + 0.20 * stable,
        timing: 0.12 + 0.44 * stable + 0.28 * warm + 0.16 * ancient,
        preservation: 0.08 + 0.48 * cell.residual_regolith + 0.24 * covered + 0.20 * low_slope,
        confidence: combined_confidence(geo_conf, surface_conf, cell.soil_confidence),
    }
    .masked(land * weathering_setting);

    let placer_system = Supports {
        source: 0.12
            + 0.30 * max_many(&[arc, orogen, mafic, craton])
            + 0.30 * exposed
            + 0.28 * high_relief,
        process: 0.05 + 0.55 * stream_energy + 0.25 * river_scale + 0.15 * steep,
        transport: 0.04 + 0.72 * river_scale + 0.24 * stream_energy,
        trap: 0.06 + 0.38 * cell.floodplain + 0.34 * cell.alluvium + 0.22 * low_slope,
        timing: 0.16 + 0.42 * river_scale + 0.24 * active_tectonics + 0.18 * wet_climate,
        preservation: 0.08 + 0.44 * cell.alluvium + 0.28 * cell.floodplain + 0.20 * low_slope,
        confidence: combined_confidence(geo_conf, surface_conf, cell.soil_confidence),
    }
    .masked(land * max_many(&[river_scale, cell.alluvium, cell.floodplain]));

    let evaporite_system = Supports {
        source: 0.18 + 0.32 * basin + 0.28 * cell.salinity + 0.22 * shelf,
        process: 0.05 + 0.58 * dry + 0.22 * warm + 0.15 * cell.salinity,
        transport: 0.10
            + 0.36 * max_many(&[cell.lake, shelf])
            + 0.30 * basin
            + 0.24 * cell.drainage,
        trap: 0.06 + 0.42 * cell.lacustrine + 0.34 * cell.accommodation + 0.18 * basin,
        timing: 0.12 + 0.44 * stable + 0.30 * dry + 0.14 * cell.subsidence,
        preservation: 0.08 + 0.42 * covered + 0.32 * cell.salinity + 0.18 * stable,
        confidence: combined_confidence(geo_conf, surface_conf, cell.soil_confidence),
    }
    .masked(land * max_many(&[basin, cell.lake, cell.salinity]));

    let coal_system = Supports {
        source: 0.04 + 0.74 * productive + 0.22 * cell.vegetation_cover,
        process: 0.04 + 0.40 * cell.wetland + 0.34 * cell.hydric_soil + 0.22 * depositional_water,
        transport: 0.10
            + 0.36 * cell.floodplain
            + 0.30 * (1.0 - cell.drainage)
            + 0.24 * river_scale,
        trap: 0.06 + 0.42 * cell.accommodation + 0.30 * cell.subsidence + 0.22 * cell.lacustrine,
        timing: 0.12 + 0.38 * stable + 0.28 * productive + 0.22 * cell.subsidence,
        preservation: 0.06
            + 0.38 * cell.hydric_soil
            + 0.34 * covered
            + 0.22 * (1.0 - cell.drainage),
        confidence: combined_confidence(geo_conf, surface_conf, bio_conf),
    }
    .masked(
        land * max_many(&[
            cell.wetland,
            cell.hydric_soil,
            productive * cell.accommodation,
        ]),
    );

    [
        arc_system,
        orogenic_system,
        mafic_system,
        seafloor_system,
        sediment_system,
        iron_system,
        weathering_system,
        placer_system,
        evaporite_system,
        coal_system,
    ]
}

fn system_potential(supports: Supports, unresolved: f64) -> f64 {
    let values = supports.values();
    if values.iter().any(|value| *value <= 0.0) {
        return 0.0;
    }
    let geometric = values.iter().product::<f64>().powf(1.0 / 6.0);
    let bottleneck = values.iter().copied().fold(1.0, f64::min);
    clamp01((0.68 * geometric + 0.32 * bottleneck) * (0.99 + 0.02 * unresolved))
}

const COMMODITY_WEIGHTS: [[f64; SYSTEM_COUNT]; COMMODITY_COUNT] = [
    [0.60, 0.00, 0.00, 0.20, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.42, 0.34, 0.00, 0.08, 0.00, 0.00, 0.00, 0.16, 0.00, 0.00],
    [0.42, 0.08, 0.00, 0.25, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.12, 0.00, 0.00, 0.33, 0.55, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.66, 0.00, 0.12, 0.00, 0.22, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.86, 0.00, 0.00, 0.00, 0.00, 0.14, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.76, 0.14, 0.10, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.48, 0.30, 0.22, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.66, 0.12, 0.00, 0.00, 0.22, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.80, 0.00, 0.00, 0.00],
    [0.62, 0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.18, 0.82, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0.00, 0.92, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.85, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
];

#[no_mangle]
pub extern "C" fn mineral_systems_native_abi_version() -> u32 {
    2
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
/// Compute coarse causal mineral-system and commodity prospectivity.
///
/// # Safety
/// Every pointer must reference the declared number of contiguous elements.
/// Input and output buffers must not overlap.
pub unsafe extern "C" fn mineral_systems_run(
    cell_count: i32,
    face_resolution: i32,
    seed: u64,
    minimum_dominant_potential: f64,
    xyz_ptr: *const f32,
    ocean_ptr: *const f32,
    shelf_ptr: *const f32,
    relief_ptr: *const f32,
    elevation_ptr: *const f32,
    terrain_slope_ptr: *const f32,
    province_class_ptr: *const u8,
    crust_age_ptr: *const f32,
    rock_strength_ptr: *const f32,
    accommodation_ptr: *const f32,
    province_confidence_ptr: *const f32,
    elevation_confidence_ptr: *const f32,
    convergence_ptr: *const f32,
    divergence_ptr: *const f32,
    shear_ptr: *const f32,
    subduction_ptr: *const f32,
    hotspot_ptr: *const f32,
    uplift_ptr: *const f32,
    subsidence_ptr: *const f32,
    compression_ptr: *const f32,
    extension_ptr: *const f32,
    stiffness_ptr: *const f32,
    temperature_ptr: *const f32,
    precipitation_ptr: *const f32,
    aridity_ptr: *const f32,
    contributing_area_ptr: *const f64,
    stream_power_ptr: *const f32,
    river_ptr: *const f32,
    floodplain_ptr: *const f32,
    lake_ptr: *const f32,
    wetland_ptr: *const f32,
    bedrock_ptr: *const f32,
    residual_regolith_ptr: *const f32,
    alluvium_ptr: *const f32,
    lacustrine_ptr: *const f32,
    volcaniclastic_ptr: *const f32,
    soil_depth_ptr: *const f32,
    salinity_ptr: *const f32,
    drainage_ptr: *const f32,
    hydric_soil_ptr: *const f32,
    soil_confidence_ptr: *const f32,
    annual_npp_ptr: *const f32,
    standing_biomass_ptr: *const f32,
    vegetation_cover_ptr: *const f32,
    biosphere_confidence_ptr: *const f32,
    source_out_ptr: *mut f32,
    process_out_ptr: *mut f32,
    transport_out_ptr: *mut f32,
    trap_out_ptr: *mut f32,
    timing_out_ptr: *mut f32,
    preservation_out_ptr: *mut f32,
    unresolved_out_ptr: *mut f32,
    potential_out_ptr: *mut f32,
    confidence_out_ptr: *mut f32,
    commodity_out_ptr: *mut f32,
    dominant_system_out_ptr: *mut u8,
) -> i32 {
    if cell_count <= 0 || face_resolution <= 0 {
        return 1;
    }
    let input_ptrs = [
        xyz_ptr.cast::<()>(),
        ocean_ptr.cast::<()>(),
        shelf_ptr.cast::<()>(),
        relief_ptr.cast::<()>(),
        elevation_ptr.cast::<()>(),
        terrain_slope_ptr.cast::<()>(),
        province_class_ptr.cast::<()>(),
        crust_age_ptr.cast::<()>(),
        rock_strength_ptr.cast::<()>(),
        accommodation_ptr.cast::<()>(),
        province_confidence_ptr.cast::<()>(),
        elevation_confidence_ptr.cast::<()>(),
        convergence_ptr.cast::<()>(),
        divergence_ptr.cast::<()>(),
        shear_ptr.cast::<()>(),
        subduction_ptr.cast::<()>(),
        hotspot_ptr.cast::<()>(),
        uplift_ptr.cast::<()>(),
        subsidence_ptr.cast::<()>(),
        compression_ptr.cast::<()>(),
        extension_ptr.cast::<()>(),
        stiffness_ptr.cast::<()>(),
        temperature_ptr.cast::<()>(),
        precipitation_ptr.cast::<()>(),
        aridity_ptr.cast::<()>(),
        contributing_area_ptr.cast::<()>(),
        stream_power_ptr.cast::<()>(),
        river_ptr.cast::<()>(),
        floodplain_ptr.cast::<()>(),
        lake_ptr.cast::<()>(),
        wetland_ptr.cast::<()>(),
        bedrock_ptr.cast::<()>(),
        residual_regolith_ptr.cast::<()>(),
        alluvium_ptr.cast::<()>(),
        lacustrine_ptr.cast::<()>(),
        volcaniclastic_ptr.cast::<()>(),
        soil_depth_ptr.cast::<()>(),
        salinity_ptr.cast::<()>(),
        drainage_ptr.cast::<()>(),
        hydric_soil_ptr.cast::<()>(),
        soil_confidence_ptr.cast::<()>(),
        annual_npp_ptr.cast::<()>(),
        standing_biomass_ptr.cast::<()>(),
        vegetation_cover_ptr.cast::<()>(),
        biosphere_confidence_ptr.cast::<()>(),
    ];
    let output_ptrs = [
        source_out_ptr.cast::<()>(),
        process_out_ptr.cast::<()>(),
        transport_out_ptr.cast::<()>(),
        trap_out_ptr.cast::<()>(),
        timing_out_ptr.cast::<()>(),
        preservation_out_ptr.cast::<()>(),
        unresolved_out_ptr.cast::<()>(),
        potential_out_ptr.cast::<()>(),
        confidence_out_ptr.cast::<()>(),
        commodity_out_ptr.cast::<()>(),
        dominant_system_out_ptr.cast::<()>(),
    ];
    if input_ptrs.iter().any(|ptr| ptr.is_null()) || output_ptrs.iter().any(|ptr| ptr.is_null()) {
        return 1;
    }
    if !minimum_dominant_potential.is_finite() || !(0.0..=1.0).contains(&minimum_dominant_potential)
    {
        return 2;
    }

    let total = cell_count as usize;
    let xyz = unsafe { slice::from_raw_parts(xyz_ptr, 3 * total) };
    macro_rules! input_f32 {
        ($ptr:ident) => {
            unsafe { slice::from_raw_parts($ptr, total) }
        };
    }
    let ocean = input_f32!(ocean_ptr);
    let shelf = input_f32!(shelf_ptr);
    let relief = input_f32!(relief_ptr);
    let elevation = input_f32!(elevation_ptr);
    let terrain_slope = input_f32!(terrain_slope_ptr);
    let province_class = unsafe { slice::from_raw_parts(province_class_ptr, total) };
    let crust_age = input_f32!(crust_age_ptr);
    let rock_strength = input_f32!(rock_strength_ptr);
    let accommodation = input_f32!(accommodation_ptr);
    let province_confidence = input_f32!(province_confidence_ptr);
    let elevation_confidence = input_f32!(elevation_confidence_ptr);
    let convergence = input_f32!(convergence_ptr);
    let divergence = input_f32!(divergence_ptr);
    let shear = input_f32!(shear_ptr);
    let subduction = input_f32!(subduction_ptr);
    let hotspot = input_f32!(hotspot_ptr);
    let uplift = input_f32!(uplift_ptr);
    let subsidence = input_f32!(subsidence_ptr);
    let compression = input_f32!(compression_ptr);
    let extension = input_f32!(extension_ptr);
    let stiffness = input_f32!(stiffness_ptr);
    let temperature = input_f32!(temperature_ptr);
    let precipitation = input_f32!(precipitation_ptr);
    let aridity = input_f32!(aridity_ptr);
    let contributing_area = unsafe { slice::from_raw_parts(contributing_area_ptr, total) };
    let stream_power = input_f32!(stream_power_ptr);
    let river = input_f32!(river_ptr);
    let floodplain = input_f32!(floodplain_ptr);
    let lake = input_f32!(lake_ptr);
    let wetland = input_f32!(wetland_ptr);
    let bedrock = input_f32!(bedrock_ptr);
    let residual_regolith = input_f32!(residual_regolith_ptr);
    let alluvium = input_f32!(alluvium_ptr);
    let lacustrine = input_f32!(lacustrine_ptr);
    let volcaniclastic = input_f32!(volcaniclastic_ptr);
    let soil_depth = input_f32!(soil_depth_ptr);
    let salinity = input_f32!(salinity_ptr);
    let drainage = input_f32!(drainage_ptr);
    let hydric_soil = input_f32!(hydric_soil_ptr);
    let soil_confidence = input_f32!(soil_confidence_ptr);
    let annual_npp = input_f32!(annual_npp_ptr);
    let standing_biomass = input_f32!(standing_biomass_ptr);
    let vegetation_cover = input_f32!(vegetation_cover_ptr);
    let biosphere_confidence = input_f32!(biosphere_confidence_ptr);

    let source_out = unsafe { slice::from_raw_parts_mut(source_out_ptr, SYSTEM_COUNT * total) };
    let process_out = unsafe { slice::from_raw_parts_mut(process_out_ptr, SYSTEM_COUNT * total) };
    let transport_out =
        unsafe { slice::from_raw_parts_mut(transport_out_ptr, SYSTEM_COUNT * total) };
    let trap_out = unsafe { slice::from_raw_parts_mut(trap_out_ptr, SYSTEM_COUNT * total) };
    let timing_out = unsafe { slice::from_raw_parts_mut(timing_out_ptr, SYSTEM_COUNT * total) };
    let preservation_out =
        unsafe { slice::from_raw_parts_mut(preservation_out_ptr, SYSTEM_COUNT * total) };
    let unresolved_out =
        unsafe { slice::from_raw_parts_mut(unresolved_out_ptr, SYSTEM_COUNT * total) };
    let potential_out =
        unsafe { slice::from_raw_parts_mut(potential_out_ptr, SYSTEM_COUNT * total) };
    let confidence_out =
        unsafe { slice::from_raw_parts_mut(confidence_out_ptr, SYSTEM_COUNT * total) };
    let commodity_out =
        unsafe { slice::from_raw_parts_mut(commodity_out_ptr, COMMODITY_COUNT * total) };
    let dominant_system_out = unsafe { slice::from_raw_parts_mut(dominant_system_out_ptr, total) };

    for cell in 0..total {
        let cell_xyz = [
            f64::from(xyz[3 * cell]),
            f64::from(xyz[3 * cell + 1]),
            f64::from(xyz[3 * cell + 2]),
        ];
        if cell_xyz.iter().any(|value| !value.is_finite()) {
            return 3;
        }
        let raw = [
            f64::from(ocean[cell]),
            f64::from(shelf[cell]),
            f64::from(relief[cell]),
            f64::from(elevation[cell]),
            f64::from(terrain_slope[cell]),
            f64::from(crust_age[cell]),
            f64::from(rock_strength[cell]),
            f64::from(accommodation[cell]),
            f64::from(province_confidence[cell]),
            f64::from(elevation_confidence[cell]),
            f64::from(convergence[cell]),
            f64::from(divergence[cell]),
            f64::from(shear[cell]),
            f64::from(subduction[cell]),
            f64::from(hotspot[cell]),
            f64::from(uplift[cell]),
            f64::from(subsidence[cell]),
            f64::from(compression[cell]),
            f64::from(extension[cell]),
            f64::from(stiffness[cell]),
            f64::from(temperature[cell]),
            f64::from(precipitation[cell]),
            f64::from(aridity[cell]),
            contributing_area[cell],
            f64::from(stream_power[cell]),
            f64::from(river[cell]),
            f64::from(floodplain[cell]),
            f64::from(lake[cell]),
            f64::from(wetland[cell]),
            f64::from(bedrock[cell]),
            f64::from(residual_regolith[cell]),
            f64::from(alluvium[cell]),
            f64::from(lacustrine[cell]),
            f64::from(volcaniclastic[cell]),
            f64::from(soil_depth[cell]),
            f64::from(salinity[cell]),
            f64::from(drainage[cell]),
            f64::from(hydric_soil[cell]),
            f64::from(soil_confidence[cell]),
            f64::from(annual_npp[cell]),
            f64::from(standing_biomass[cell]),
            f64::from(vegetation_cover[cell]),
            f64::from(biosphere_confidence[cell]),
        ];
        if raw.iter().any(|value| !value.is_finite())
            || !(0.0..=1.0).contains(&raw[0])
            || raw[4] < 0.0
            || raw[5] < 0.0
            || raw[23] < 0.0
            || raw[24] < 0.0
        {
            return 3;
        }
        let inputs = CellInputs {
            ocean: raw[0],
            shelf: raw[1],
            relief_m: raw[2],
            elevation_m: raw[3],
            terrain_slope: raw[4],
            province_class: province_class[cell],
            crust_age_ga: raw[5],
            rock_strength: raw[6],
            accommodation: raw[7],
            province_confidence: raw[8],
            elevation_confidence: raw[9],
            convergence: raw[10],
            divergence: raw[11],
            shear: raw[12],
            subduction: raw[13],
            hotspot: raw[14],
            uplift: raw[15],
            subsidence: raw[16],
            compression: raw[17],
            extension: raw[18],
            stiffness: raw[19],
            temperature_c: raw[20],
            precipitation_mm: raw[21],
            aridity: raw[22],
            contributing_area_km2: raw[23],
            stream_power_w: raw[24],
            river: raw[25],
            floodplain: raw[26],
            lake: raw[27],
            wetland: raw[28],
            bedrock: raw[29],
            residual_regolith: raw[30],
            alluvium: raw[31],
            lacustrine: raw[32],
            volcaniclastic: raw[33],
            soil_depth_m: raw[34],
            salinity: raw[35],
            drainage: raw[36],
            hydric_soil: raw[37],
            soil_confidence: raw[38],
            annual_npp: raw[39],
            standing_biomass: raw[40],
            vegetation_cover: raw[41],
            biosphere_confidence: raw[42],
        };
        let supports = support_fields(inputs);
        let mut best_system = 0usize;
        let mut best_potential = 0.0f64;
        for (system, support) in supports.iter().copied().enumerate() {
            let offset = system * total + cell;
            let unresolved = unresolved_support(seed, system, cell_xyz, face_resolution as usize);
            let potential = system_potential(support, unresolved);
            source_out[offset] = support.source as f32;
            process_out[offset] = support.process as f32;
            transport_out[offset] = support.transport as f32;
            trap_out[offset] = support.trap as f32;
            timing_out[offset] = support.timing as f32;
            preservation_out[offset] = support.preservation as f32;
            unresolved_out[offset] = unresolved as f32;
            potential_out[offset] = potential as f32;
            confidence_out[offset] = support.confidence as f32;
            if potential > best_potential {
                best_potential = potential;
                best_system = system;
            }
        }
        dominant_system_out[cell] = if best_potential >= minimum_dominant_potential {
            (best_system + 1) as u8
        } else {
            0
        };
        for (commodity, weights) in COMMODITY_WEIGHTS.iter().enumerate() {
            let mut prospectivity = 0.0;
            for (system, weight) in weights.iter().copied().enumerate() {
                let offset = system * total + cell;
                prospectivity += weight
                    * f64::from(potential_out[offset])
                    * (0.60 + 0.40 * f64::from(confidence_out[offset]));
            }
            commodity_out[commodity * total + cell] = clamp01(prospectivity) as f32;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cell() -> CellInputs {
        CellInputs {
            ocean: 0.0,
            shelf: 0.0,
            relief_m: 400.0,
            elevation_m: 500.0,
            terrain_slope: 0.004,
            province_class: 2,
            crust_age_ga: 2.2,
            rock_strength: 0.7,
            accommodation: 0.2,
            province_confidence: 0.9,
            elevation_confidence: 0.85,
            convergence: 0.1,
            divergence: 0.1,
            shear: 0.1,
            subduction: 0.0,
            hotspot: 0.0,
            uplift: 0.1,
            subsidence: 0.1,
            compression: 0.1,
            extension: 0.1,
            stiffness: 0.8,
            temperature_c: 18.0,
            precipitation_mm: 900.0,
            aridity: 0.35,
            contributing_area_km2: 100.0,
            stream_power_w: 1_000.0,
            river: 0.05,
            floodplain: 0.05,
            lake: 0.0,
            wetland: 0.05,
            bedrock: 0.25,
            residual_regolith: 0.55,
            alluvium: 0.08,
            lacustrine: 0.02,
            volcaniclastic: 0.02,
            soil_depth_m: 1.0,
            salinity: 0.05,
            drainage: 0.65,
            hydric_soil: 0.08,
            soil_confidence: 0.85,
            annual_npp: 0.35,
            standing_biomass: 4.0,
            vegetation_cover: 0.6,
            biosphere_confidence: 0.85,
        }
    }

    #[test]
    fn arc_support_responds_to_subduction_and_arc_setting() {
        let background = support_fields(base_cell())[ARC_MAGMATIC];
        let mut arc_cell = base_cell();
        arc_cell.province_class = 6;
        arc_cell.subduction = 0.95;
        arc_cell.convergence = 0.8;
        arc_cell.uplift = 0.7;
        arc_cell.volcaniclastic = 0.7;
        let enriched = support_fields(arc_cell)[ARC_MAGMATIC];
        assert!(system_potential(enriched, 0.5) > system_potential(background, 0.5));
    }

    #[test]
    fn coal_support_requires_productive_waterlogged_accommodation() {
        let background = support_fields(base_cell())[COAL];
        let mut coal_cell = base_cell();
        coal_cell.province_class = 3;
        coal_cell.accommodation = 0.9;
        coal_cell.subsidence = 0.7;
        coal_cell.wetland = 0.8;
        coal_cell.hydric_soil = 0.9;
        coal_cell.annual_npp = 0.9;
        coal_cell.standing_biomass = 15.0;
        coal_cell.vegetation_cover = 0.95;
        coal_cell.drainage = 0.1;
        let enriched = support_fields(coal_cell)[COAL];
        assert!(system_potential(enriched, 0.5) > system_potential(background, 0.5));
    }

    #[test]
    fn accumulated_strain_elevation_and_confidence_are_active_inputs() {
        let mut low_orogen = base_cell();
        low_orogen.province_class = 4;
        low_orogen.compression = 0.0;
        low_orogen.elevation_m = 100.0;
        let mut high_orogen = low_orogen;
        high_orogen.compression = 1.0;
        high_orogen.elevation_m = 2_800.0;
        assert!(
            system_potential(support_fields(high_orogen)[OROGENIC_SHEAR], 0.5)
                > system_potential(support_fields(low_orogen)[OROGENIC_SHEAR], 0.5)
        );

        let mut low_mafic = base_cell();
        low_mafic.province_class = 9;
        low_mafic.extension = 0.0;
        let mut high_mafic = low_mafic;
        high_mafic.extension = 1.0;
        assert!(
            system_potential(support_fields(high_mafic)[MAFIC_ULTRAMAFIC], 0.5)
                > system_potential(support_fields(low_mafic)[MAFIC_ULTRAMAFIC], 0.5)
        );

        let mut low_confidence = high_orogen;
        low_confidence.elevation_confidence = 0.1;
        let mut high_confidence = high_orogen;
        high_confidence.elevation_confidence = 0.95;
        let low_support = support_fields(low_confidence)[OROGENIC_SHEAR];
        let high_support = support_fields(high_confidence)[OROGENIC_SHEAR];
        assert_eq!(
            system_potential(low_support, 0.5),
            system_potential(high_support, 0.5)
        );
        assert!(high_support.confidence > low_support.confidence);
    }

    #[test]
    fn unresolved_support_is_exact_and_bounded() {
        let xyz = [0.2, -0.4, 0.8944271909999159];
        let first = unresolved_support(42, PLACER, xyz, 128);
        assert_eq!(first, unresolved_support(42, PLACER, xyz, 128));
        assert!((0.0..1.0).contains(&first));
        assert_ne!(first, unresolved_support(43, PLACER, xyz, 128));
    }

    #[test]
    fn unresolved_support_is_spatially_coherent() {
        let first = unresolved_support(42, PLACER, [1.0, 0.0, 0.0], 128);
        let nearby = unresolved_support(42, PLACER, [0.99999999995, 0.00001, 0.0], 128);
        assert!((first - nearby).abs() < 0.02);
    }

    #[test]
    fn unresolved_wavelength_scales_with_parent_resolution() {
        fn mean_increment(face_resolution: usize) -> f64 {
            let delta = 0.001;
            let mut sum = 0.0;
            let mut count = 0usize;
            for system in 0..SYSTEM_COUNT {
                for sample in 0..24 {
                    let angle = 0.17 * f64::from(sample);
                    let first = [angle.cos(), angle.sin(), 0.0];
                    let nearby = [(angle + delta).cos(), (angle + delta).sin(), 0.0];
                    sum += (unresolved_support(42, system, first, face_resolution)
                        - unresolved_support(42, system, nearby, face_resolution))
                    .abs();
                    count += 1;
                }
            }
            sum / count as f64
        }

        let coarse = mean_increment(64);
        let fine = mean_increment(256);
        assert!(fine > 1.8 * coarse);
    }

    #[test]
    fn every_causal_axis_is_a_real_bottleneck() {
        let support = Supports {
            source: 0.8,
            process: 0.8,
            transport: 0.8,
            trap: 0.8,
            timing: 0.8,
            preservation: 0.8,
            confidence: 0.9,
        };
        let baseline = system_potential(support, 0.5);
        for axis in 0..6 {
            let mut values = support.values();
            values[axis] = 0.0;
            let missing = Supports {
                source: values[0],
                process: values[1],
                transport: values[2],
                trap: values[3],
                timing: values[4],
                preservation: values[5],
                confidence: support.confidence,
            };
            assert_eq!(system_potential(missing, 0.5), 0.0);
            assert!(system_potential(missing, 0.5) < baseline);
        }
    }

    #[test]
    fn constants_cover_every_supported_family() {
        assert_eq!(
            [
                ARC_MAGMATIC,
                OROGENIC_SHEAR,
                MAFIC_ULTRAMAFIC,
                VOLCANOGENIC_SEAFLOOR,
                SEDIMENT_HOSTED,
                ANCIENT_IRON,
                WEATHERING_SUPERGENE,
                PLACER,
                EVAPORITE,
                COAL,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
    }
}
