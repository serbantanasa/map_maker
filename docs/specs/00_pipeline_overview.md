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

## Canonical Stage Ordering
1. Geometry and cubed-sphere topology.
2. Kinematic tectonic skeleton.
3. Age-conditioned crust state.
4. Connected geological provinces and boundary segments.
5. Initial elevation and tectonic morphology.
6. Planetary boundary conditions and monthly orbital forcing.
7. Climate pass 1.
8. Hydrology pass 1.
   - A sparse selected-basin refinement gate proves inherited river topology,
     subgrid physical width, lateral corridor capacity, and parent/child
     conservation before erosion. It reports complete source-to-sink readiness
     through physical channels and zero-width hydrologic connectors.
9. Erosion and sedimentation.
10. Hydrology pass 2.
11. Soils and biomes.
12. Mineral and energy systems.
13. Selected-region refinement and map export.

The current canonical cubed-sphere implementation reaches Hydrology Pass 1 and
passes the selected-basin refinement readiness gate with a
causal, pre-erosion bedrock surface and separate crustal, orogenic, basin, and
relief-prior artifacts, persisted monthly orbital forcing, and a first seasonal
climate/orography pass. The older
rectangular compatibility path runs directly from world age into provisional
erosion and final rendering; it is reference behavior, not the canonical stage
order.

Legacy pipeline remains callable through `map_maker.legacy.*` but does not participate in this DAG.

Each section below has its own detailed specification in this folder.
