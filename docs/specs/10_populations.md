# Stage 9 – Proto-Societies & Emergent Civilization Candidates

## Objectives
- Derive population density distributions for pre-civilized bands based on environmental affordances.
- Identify regions with favorable conditions for emergent civilizations without simulating full trade networks.

## Inputs
- `BiomeMap`, `SuitabilityMap`, `StressIndex`, `HydroGraph`, `CatchmentMetadata`, `Elevation`, `Topology`.
- Config: `band_density_scale`, `civilization_threshold`, `mobility_cost_weights`, `tile_size`.

## Workflow (Rust SIMD with tile-based execution)
1. **Band Density Estimation**
   - Compute baseline hunter-gatherer carrying capacity from `SuitabilityMap`, adjusted by `StressIndex` and proximity to rivers/coasts.
   - Apply diffusion to simulate seasonal mobility; output `PopulationDensity` (float32 raster).
2. **Mobility Graph**
   - Build local movement cost surface (terrain slope, river crossing difficulty).
   - Derive `MobilityGraph` (lightweight adjacency per macro-cell) for later use.
3. **Civilization Candidate Scoring**
   - Identify clusters where population density, fertile soils, and hydropower exceed `civilization_threshold`.
   - Output `CandidateCatalog` with location, carrying capacity, environmental richness.

## Performance Targets
- Density and mobility computations in Rust SIMD; <1 s for 8 k grid.
- Candidate clustering uses Rust (k-means or DBSCAN) with deterministic seeds.

## Logging
- Mean/min/max population density, total estimated population.
- List candidate regions with scores, area, supporting metrics.
- Tile throughput metrics for profiling.

## Testing
- Density field non-negative, integrates to expected population within tolerance.
- Mobility cost surface respects impassable terrain (e.g., ocean).
- Candidate detection deterministic with fixed seed and matches synthetic scenarios.
