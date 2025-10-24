# Stage 8 – Soil Formation & Biome Classification

## Objectives
- Translate geology + climate into soil properties.
- Classify land cover/biomes and agricultural suitability.

## Inputs
- `Elevation`, `TemperatureMean`, `TemperatureSeasonal`, `PrecipitationMean`, `PrecipitationSeasonal`, `CatchmentMetadata`, `SedimentDepth`.
- Config:
  - Soil: `weathering_rate`, `organic_input`, `soil_texture_map`, `tile_size`.
  - Biome: `classification_table`, `snowline`, `aridity_thresholds`, `gpu_backend` (optional for large grids).

## Soil Model (Rust SIMD with optional GPU kernels)
1. **Parent Material**
   - Derive lithology from plate type and sediment depth (SoA lookups generated in previous stages).
2. **Weathering**
   - Compute chemical + physical weathering using degree-day approximations leveraging `TemperatureSeasonal`.
   - Accumulate organic matter from vegetation potential; tile-based SIMD loops minimize cache misses.
3. **Floodplain & Delta Modeling**
   - Use `CatchmentMetadata`, `HydroGraph`, and `DeltaCatalog` to flag floodplain cells (e.g., discharge above threshold, low slope) and ingest precomputed delta metadata.
   - Classify deltas into fertile vs swamp typologies using hydrology-provided sediment load, tidal range, and stagnation metrics; fertile deltas get high fertility boosts, swamp deltas get high organic content but low suitability.
   - Apply alluvial deposition model (exponential decay from channel centerline) for fertile deltas; organic accrual + malaria risk scoring for swamp deltas.
   - Update buffers in-place while conserving total sediment mass and record delta metadata.
4. **Soil Horizon Outputs**
   - Produce SOC, texture fractions, fertility index (post enrichment/delta adjustment); store in arena-backed SoA buffers.

## Biome Classification
1. **Köppen-like Decision Tree**
   - Use temperature + precipitation seasonality grid (mean + amplitude).
   - Adjust for elevation (snowline) and soil moisture using SIMD decision tree or GPU kernel for large runs.
2. **Agricultural Suitability**
   - Score per biome/soil combination; identify pasture vs cropland; output additional `StressIndex` using precipitation variability.
3. **Outputs**
   - `SoilField` (horizons, fertility), `BiomeMap`, `SuitabilityMap`, `StressIndex`.

## Performance
- Rust SIMD tile-based loops target <1 s for 8 k grid; optional GPU backend for biome classification if `gpu_backend` enabled.
- Python orchestrator touches only metadata handles.

## Logging
- Histogram of fertility, biome area percentages, suitability coverage, stress index distribution.
- Alluvial enrichment stats (total enriched area, fertility delta) and delta classification summary (counts, discharge stats, swamp vs fertile ratios).
- Tile throughput metrics to diagnose hotspots.

## Testing
- Soil depth non-negative, fertility bounded [0,1].
- Floodplain/delta enrichment increases fertility near major rivers while conserving total sediment; swamp deltas reduce suitability as expected.
- Biome classification matches reference cases (e.g., desert climate).
- Suitability correlates with high fertility and moderate climate.
- CPU/GPU parity tests when GPU backend enabled.
