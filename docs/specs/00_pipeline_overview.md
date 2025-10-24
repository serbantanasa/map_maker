# Next-Generation World Pipeline Overview

## Goals
- Support multiple topologies (sphere, cylinder, torus) with consistent neighbor queries and area weights.
- Run each domain stage (geometry → trade) on 8 k × 8 k grids within single-digit seconds by using compiled kernels where necessary.
- Produce deterministic, cacheable outputs with clear metadata for downstream tooling.
- Leave the legacy generator untouched under `map_maker.legacy`.

### Performance Tenets
- Rasters live in contiguous, aligned buffers managed by a shared arena; stages pass handles, not copies.
- Heavy kernels execute in Rust/C++/GPU; Python is orchestration only.
- Pipeline supports chunked/streaming execution to avoid loading entire worlds when not required.
- Stage outputs are immutable; downstream stages allocate new buffers when mutation is needed.

## Execution Model
1. **Pipeline Runner** loads configuration, initializes topology and RNG, and executes stages registered in a dependency DAG.
2. Each **Stage** declares explicit inputs/outputs, performs its computation, and returns a structured payload.
3. Outputs are persisted to a dataset directory (rasters, graphs, metadata) and a run log with timing/memory stats is written.
4. A thin compatibility layer continues to expose the legacy generator for existing consumers.

## Stage Ordering
1. Geometry Topology
2. Ocean Fraction / Sea-Level Initialization
3. Tectonics (plates, velocities, boundaries)
4. World Age & Thermal Adjustments
5. Erosion & Sedimentation
6. Current Elevation Map
7. Atmospheric & Hydrology
8. Soil Types & Biomes
9. Societies, Settlements & Trade

Legacy pipeline remains callable through `map_maker.legacy.*` but does not participate in this DAG.

Each section below has its own detailed specification in this folder.
