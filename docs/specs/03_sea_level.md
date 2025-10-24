# Stage 2 – Ocean Fraction & Base Sea-Level *(Deprecated)*

> **Status:** Deprecated. As of 2025-10-24 00:33 UTC we removed the sea-level stage from the next-generation pipeline. Land/ocean classification now emerges from tectonic plate outputs (Stage 3), where continental/oceanic plates are assigned directly and downstream stages derive elevation and coastlines from tectonics-driven data.

## Responsibilities
- Establish initial land/ocean mask hitting target sea fraction.
- Provide coarse proto-basins to guide subsequent tectonic uplift.
- Expose tunable biases (latitudinal, noise amplitude).
- Execute primarily in Rust (SIMD) with optional GPU backend deferred for later.

## Inputs
- `TopologyMetadata`, `ResolutionSet`.
- Config: `target_ocean_fraction` (0–1), `lat_bias_curve`, `smoothing_kernel`, `proto_basin_seed`.

## Algorithm
1. **Low-frequency Noise Generation (Rust)**
   - SIMD-friendly fBm noise generated directly at target resolution using `wide` or `packed_simd`; GPU path optional later.
   - Parameters `noise_octaves`, `noise_scale`, `noise_gain`.
2. **Latitudinal Bias**
   - Compute weighting function `w(lat)` from precomputed topology trig tables.
   - Combine via fused multiply-add for cache efficiency: `score = noise + bias`.
3. **Binary Search Threshold**
   - Rust routine performing area-weighted histogram to find `t` achieving target fraction within ±0.3%.
4. **Morphological Smoothing**
   - SIMD Gaussian blur (separable convolution) plus optional diffusion iterations.
   - Re-threshold to maintain fraction.
5. **Outputs**
   - `BaseOceanMask`: boolean grid (bit-packed to reduce memory).
   - `SeaLevelThreshold`: float.
   - `ProtoBasinField`: float map for tectonics seeding.

## Performance
- Rust SIMD implementation targets <0.3 s on 8 k grid.
- Optional GPU backend (Metal) may be added later but not required initially.
- Python receives only handles; no large array copies.

## Logging
- Report target vs actual fraction, number of smoothing iterations, min/max noise values, threshold.
- Save preview (PNG) if debug flag.

## Testing
- Unit tests verifying:
  - Binary search converges to within tolerance for sample configs.
  - Lat bias extremes (all ocean at poles) behave as expected.
  - Morphological smoothing maintains ocean connectivity.
  - Rust/Python results match reference within epsilon.

### Deprecation Notes
- The ocean mask and proto-basin outputs described here are no longer generated. Instead:
  - `PlateField[..., 1]` from Stage 3 stores continental vs oceanic classification.
  - Boundary and hotspot rasters supply the cues previously derived from `ProtoBasinField`.
- Any references to `BaseOceanMask`, `SeaLevelThreshold`, or `ProtoBasinField` should be updated to consume the tectonics outputs.
