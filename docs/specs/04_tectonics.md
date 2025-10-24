# Stage 3 – Tectonics & Plate Modeling

## Objectives
- Generate plate geometry with controlled size distribution and continent/ocean assignment.
- Simulate motion to derive convergence/divergence/shear metrics.
- Output fields consumed by uplift, volcanism, and later erosion stages.
- Implemented as Rust modules (SIMD + Rayon); Python orchestrates configuration/tests only.

## Inputs
- `Topology`, `ResolutionSet`, `BaseOceanMask`, `ProtoBasinField`.
- Config keys:
  - `num_plates`, `continental_fraction`, `lloyd_iterations`.
  - `velocity_scale`, `drift_bias`, `wrap_x`, `wrap_y`.
  - `hotspot_density`, `subduction_bias`.

## Steps
1. **Plate Seeding (Rust)**
   - SIMD Poisson disk sampling respecting `continental_fraction`.
   - Lloyd relaxation (vectorized) to control plate size variance.
2. **Plate Typing**
   - Assign continental/oceanic classification with configurable ratios; store crust thickness/density in SoA buffers.
3. **Velocity Assignment**
   - Sample velocity vectors with biases (equatorial drift, hotspot attraction) using deterministic RNG.
   - Enforce net-zero momentum.
4. **Motion Simulation**
   - Time-stepped integration (N steps, dt) computing relative velocities along neighbor tables.
   - Rayon parallel loops with cache-friendly tiling.
5. **Boundary Classification**
   - Identify boundaries, compute convergence/divergence/shear; estimate subduction probability using plate type + velocity.
6. **Outputs**
   - `PlateField` (id, type, thickness, velocity vector).
   - `BoundaryField` (convergence rate, divergence rate, shear).
   - `HotspotMap` (probabilistic field for volcanic activity).
   - Metadata with seeds, parameter snapshot, plate polygons.

## Performance Requirements
- 8 k grid, 12–40 plates processed in ≤3 s using Rust (Rayon + SIMD).
- Memory footprint <1 GB per stage.
- Python pays only for configuration marshalling; no data copies.

## Logging
- Plate count, continental/oceanic area ratio.
- Velocity statistics (mean, std).
- Boundary length by type.
- Hotspot count and locations.

## Testing
- Determinism with fixed seed.
- Plate area distribution within tolerance of target ratio.
- Convergence/divergence fields antisymmetric along shared boundaries.
- Serialization/deserialization of plate metadata.
