# Stage 5 – Erosion & Sedimentation

## Objective
- Evolve the elevation field through uplift, fluvial erosion, coastal abrasion, and sediment deposition.

## Inputs
- `CrustThickness`, `UpliftRate`, `SubsidenceRate`, `HotspotEvents`, `PlateField`, `BaseOceanMask`.
- Config: `erosion_steps`, `dt`, `stream_power_k`, `sediment_capacity`, `coastal_wave_energy`, `tile_size`, `gpu_backend` (optional).

## Algorithm Outline (Rust core with optional GPU kernels)
1. **Initialize Elevation Prototype**
   - Combine isostatic offsets and uplift/subsidence to produce `InitialDEM` (arena-backed float32 grid).
2. **Iterative Solver (T timesteps)** – executed in Rust with tile-based parallelism (configurable `tile_size`, halo width):
   - **Uplift Update:** fused multiply-add on elevation buffer using `UpliftRate` and `SubsidenceRate`.
   - **Fluvial Erosion:**
     - Compute flow direction/accumulation via D-infinity implemented with SIMD (fallback) or GPU kernel when `gpu_backend == metal`.
     - Apply stream power law `dz = k * (flow^m) * (slope^n) * dt` on tiles; maintain mass balance accumulator per tile.
     - Enforce sediment capacity, advect excess downstream via queue-based solver.
   - **Mass Wasting:** slope-threshold cellular automata using bitmasks for active cells to avoid branching.
   - **Coastal Erosion:** process shoreline tiles only, applying precomputed wave energy coefficients.
   - **Sediment Deposition:** deposit into basins/continental shelves; maintain `SedimentDepth` buffer (float32) and per-tile residual mass.
   - Multi-resolution loop: coarse level update (1/4 res) first, then inject correction into native grid to reduce iteration count.
3. **Outputs**
   - `ElevationRaw`, `SedimentDepth`, `RiverIncision`, intermediate diagnostics (erosion flux, deposition flux) stored in Arrow tables.

## Performance Targets
- Rust SIMD + Rayon path: <4 s for 8 k grid with default 20 timesteps.
- GPU backend (Metal compute shader) optional to cut flow solver to <1 s; asynchronous overlap with CPU tasks where possible.
- Tile scheduler avoids loading entire grid into cache; memory residency limited to a few tiles + global buffers.

## Logging
- Per iteration: mean elevation, mass removed/added, max incision, tile-level throughput (cells/sec).
- Final summary: erosion vs uplift balance, global sediment budget, GPU/CPU split timings.

## Testing
- Mass conservation within tolerance (<0.5% drift) verified per tile and globally.
- River incision increases with higher flow power.
- Coastal erosion reduces protruding landforms over iterations.
- GPU and CPU backends produce matching results within tolerance.
