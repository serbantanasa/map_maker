# Stage 7 – Atmospheric Circulation & Hydrology

## Objectives
- Model global temperature, precipitation, and moisture transport.
- Compute full drainage network with hierarchical catchments and river properties.

## Inputs
- `Elevation`, `Slope`, `Topology`, `SeaSurfaceTemperature` (optional).
- Config:
  - Climate: `insolation_profile`, `hadley_extent`, `orographic_coeff`, `evaporation_rate`, `gpu_backend` (`metal`, `cuda`, or `cpu`).
  - Hydrology: `flow_algorithm`, `river_thresholds`, `lake_evap_factor`, `reservoir_capacity`, `tile_size`.

## Climate Pipeline
1. **Radiative Balance**
   - Solve simplified energy balance (zonal or spectral) using GPU tensor ops (Metal or CUDA) with fallback Rust CPU kernels.
   - Output temperature baseline per latitude, adjust by elevation lapse rate in SIMD.
2. **Moisture Transport**
   - Advection-diffusion solver for moisture flux using semi-Lagrangian scheme (GPU) or vectorized Rust for CPU.
   - Orographic precipitation: `P = base + orographic_factor * max(0, upslope_wind)` computed tile-wise using cached slope/aspect.
   - Handle rain shadows with leeward decay using exponential attenuation.
3. **Outputs**
   - `TemperatureMean`, `TemperatureSeasonal` (12-month tensor or harmonic amplitudes), `TemperatureVariance`.
   - `PrecipitationMean`, `PrecipitationSeasonal`, `PrecipitationVariance`.
   - `MoistureFlux`, `WindVectors` (SoA buffers).

## Hydrology Pipeline
1. **Flow Routing**
   - Use D-infinity (or hex grid) implemented in Rust SIMD with optional GPU path; processed in tiles.
   - Multi-resolution progressive refinement (coarse to fine) to reduce iterations.
2. **Catchment Extraction**
   - Build drainage tree: each cell links to downstream neighbor until ocean.
   - Determine basins, sub-basins; compute Strahler/Shreve orders (graph stored in Arrow/CSR for zero-copy access).
3. **River Geometry & Delta Detection**
   - Extract polylines with smoothing; store multi-scale routes with jitter/curvature handling (Rust polyline fitter).
   - Compute discharge = precipitation runoff * contributing area.
   - Detect delta candidates at river mouths (river meets ocean, low slope); compute sediment load and stagnation metrics to feed `DeltaCatalog`.
4. **Lake & Wetland Modeling**
   - Fill depressions with priority-flood, consider infiltration/evaporation; produce water balance metrics.
5. **Outputs**
   - `HydroGraph` (Arrow/Parquet), `RiverMask` (per threshold), `CatchmentMetadata`, `LakeCatalog`, `DeltaCatalog` (river mouth metadata with discharge, sediment load, typology cues).

## Performance
- Climate GPU solver ~1 s (8 k grid) on MPS/CUDA; CPU fallback (Rust) aims for <2 s using multi-threading.
- Hydrology flows using Rust + Rayon target <2 s; GPU optional for further acceleration.
- Asynchronous pipeline overlaps climate GPU kernels with CPU hydrology preprocessing.
- Tile-based workload ensures cache locality and allows streaming for large grids.

- Temperature/precip summary statistics.
- Number of basins, river segments per order.
- Total discharge vs precipitation check.
- Delta detection summary (count, discharge totals, fertile vs swamp indicators).
- Temperature/precip summary statistics.
- Number of basins, river segments per order.
- Total discharge vs precipitation check.

## Testing
- Energy balance test (no elevation) matches analytic solution.
- Seasonal precipitation/temperature integrate to annual means within tolerance.
- Flow routing determinism with seeded random noise; CPU/GPU parity within tolerance.
- Catchment tree acyclic and drains to ocean or terminal lakes.
- Discharge integrates to precipitation minus evaporation within tolerance.
