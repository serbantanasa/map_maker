# Stage 6 – Elevation Consolidation

## Purpose
- Convert erosion outputs into finalized multi-resolution elevation products and derived terrain metrics.

## Inputs
- `ElevationRaw`, `SedimentDepth`, `BaseOceanMask`, `Topology`.
- Config: `smoothing_kernel`, `shelf_depth`, `min_island_size`, `tile_size`, `enable_gpu`.

## Steps (Rust SIMD w/ optional GPU kernels)
1. **Finalize Sea Level**
   - Adjust baseline via area-weighted binary search to ensure target ocean fraction (respecting sediment adjustments).
   - Remove tiny islands below `min_island_size` using connected-component analysis on bit-packed masks.
2. **Sediment & Bedrock Merge**
   - Combine raw elevation with sediment depth; clamp bathymetry to non-negative values in place, using fused operations for cache efficiency.
3. **Derived Metrics**
   - Compute slope/aspect via gradient kernels (Sobel/Prewitt) implemented in Rust SIMD; GPU path uses Metal compute shader when enabled.
   - Curvature and relief computed through windowed convolution with sliding tiles to keep cache hot.
   - Hypsometric statistics derived via parallel histogram over elevation.
4. **Multi-resolution Outputs**
   - Downsample using area-weighted averaging (summed-area tables) to produce pyramid levels without re-reading from Python.
   - Generate hillshade textures (vectorized lighting) and store as uint8 buffers.
5. **Outputs**
   - `Elevation` (arena-backed native grid + pyramid handles), `Slope`, `Aspect`, `Relief`, `Hillshade`.

## Performance
- Rust SIMD + Rayon path target <1 s for 8 k grid; GPU hillshade optional for faster previews.
- All computations tile-based to avoid thrashing caches; Python never touches raw buffers.

## Logging
- Elevation min/max, quartiles.
- Percentage land vs sea after cleanup.
- Slope distribution summary, relief histogram, tile throughput (cells/sec).

## Testing
- Sea-level adjustment achieves configured fraction.
- Derived metrics match analytic results on test surfaces (plane, cone).
- Multi-resolution downsampling conserves mean elevation.
- CPU/GPU paths match within tolerance when both enabled.
