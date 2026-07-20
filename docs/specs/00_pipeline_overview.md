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
6. Connected sea level and fractional surface geography.
   - Continental crust and emerged land are independent states.
   - A Rust union-find solve creates one connected global ocean, preserves
     isolated low basins, and publishes fractional coarse coasts and shelves.
7. Planetary boundary conditions and monthly orbital forcing.
8. Atmospheric composition, pressure, and greenhouse forcing.
   - Earth is the default validation profile, not a hard operating boundary.
   - Composition and pressure remain explicit conditioning artifacts rather
     than being folded irreversibly into climate labels.
9. Climate pass 1.
10. Seasonal snow, firn, and glacier mass balance.
   - A Rust kernel separates seasonal snow and glacier ice, transfers ice
     conservatively downslope, and publishes melt-aware runoff potential.
11. Hydrology pass 1.
   - A sparse selected-basin refinement gate proves inherited river topology,
     subgrid physical width, lateral corridor capacity, and parent/child
     conservation before erosion. It reports complete source-to-sink readiness
     through physical channels and zero-width hydrologic connectors.
12. Erosion and sedimentation.
   - The sparse selected-basin pass now solves junction-consistent physical bed
     profiles, applies volume-based subgrid incision, routes newly eroded
     sediment through connectors, and deposits only on allocated floodplain or
     terminal support.
13. Hydrology pass 2.
    - The sparse selected-basin pass uses volume-adjusted terrain means and
      subgrid channel beds, preserves inherited trunk/connector identities,
      applies one bounded local reroute, and publishes depression candidates.
14. Refined seasonal surface-water balance.
    - Local candidates receive monthly catchment inflow, fractional inundation,
      fill/spill propagation, and provisional lake or hydrologic-wetland classes.
15. Bounded subgrid outlet incision and final surface-water balance.
    - Narrow outlet beds modify terrain by physical eroded volume, retain
      ordinary-cell semantics, rerun conservative routing in Rust, and iterate
      monthly balance until outlet feedback is resolved or the hard round bound
      is reached.
16. Final lake-to-reach hydrograph coupling and hydrology validation.
    - Solved terminal lake-network overflow replaces its inherited runoff
      component in reach-entry and reach-exit monthly hydrographs. Negative
      adjustments stay branch-local, are bounded by represented discharge, and
      persist any unresolved scale mismatch as pre-channel interception.
17. Surface materials and initial soils.
    - Mutually exclusive L2 component fractions preserve exposed bedrock,
      residual regolith, colluvium, alluvium, lacustrine sediment, glacial
      deposit, and volcaniclastic material without pretending each coarse cell
      is homogeneous.
    - A Rust kernel derives mineral-soil properties and a conservative monthly
      soil-water partition. Hydric-soil evidence is not yet an ecological
      wetland label.
18. Environmental and biosphere resource envelope.
   - Monthly light, liquid-water opportunity, thermal opportunity, atmospheric
     substrates, oxygen support, and land-surface support remain separate raw
     fields plus diagnostics.
19. Trait-first potential biosphere.
   - A Rust kernel converts the bounded resource envelope into potential NPP,
     cover, biomass, growing season, adaptation pressures, and continuous
     producer-community traits without assigning species or biome labels.
20. Earth biosphere validation.
   - Versioned global carbon totals, upstream-climate distribution strata, and
     fixed-seed ensemble tolerances gate calibration before functional types.
21. Functional vegetation mixtures.
   - Eight continuous producer-community strategies plus five nonvegetated
     classes close the physical land-cell partition exactly.
   - Fire, grazing, forest-resource, pasture, and crop fields describe bounded
     physical suitability, not disturbance history or actual land use.
   - `earth_functional_vegetation_v1` gates broad global cover, upstream-climate
     response, resource-potential shape, and six-seed stability.
22. Derived biome mixtures and Earth profile.
   - Thirteen familiar biome fractions are derived from causal climate,
     functional vegetation, terrain, wet support, and physical surface state.
   - `earth_biomes_v1` gates global shape, causal response, and six-seed
     stability while retaining the full mixture for ecotones and refinement.
23. Bounded vegetation feedback.
24. Mineral and energy systems.
25. Selected-region refinement and map export.

## Current Implementation Focus

Late simulation development is frozen at stage 22 until the global map-export
milestone is accepted. Stages 23 and 24, additional hydrology or biosphere
features, and regional L3 refinement receive bug fixes and contract corrections
only. Current product work is limited to canonical surface geography,
tectonically legible elevation and orogeny, shelf and passive-margin morphology,
versioned physical atlas rendering, and the fixed multi-seed acceptance gallery.

Map export is a presentation layer over existing canonical artifacts and may be
implemented before later physical stages without changing their causal order.
The freeze ends only through a new recorded decision after the export acceptance
bar is met.

The current canonical cubed-sphere implementation reaches converged bounded
outlet incision, final lake-coupled river hydrographs, and a bounded V1
cryosphere after erosion, sedimentation, and Hydrology Pass 2. It now reaches
fractional L2 surface materials, property-first initial soils, explicit
atmospheric composition and pressure, and a Rust-backed environmental resource
envelope plus continuous potential producer-community traits. It includes a
passing Earth-profile gate and the first Rust-backed functional-vegetation
mixture, exact cover partition, derived dominant-cover query, and resource-
suitability layers plus passing functional-vegetation and derived-biome Earth
profiles. The bounded vegetation feedback remains deferred. It includes a causal,
pre-erosion bedrock surface; separate crustal,
orogenic, basin, and relief-prior artifacts; persisted monthly orbital forcing;
an explicit connected sea-level datum with fractional coasts and shelves; and a
first seasonal climate/orography pass. Bed-profile and sediment budgets
are conservative but remain uncalibrated. Pass 2 audits their local routing
consequences without replacing the accepted coarse trunk graph; the
surface-water stage solves periodic monthly storage, and the outlet stage
consumes its explicit erosion feedback through bounded persistent corrections.
The older rectangular compatibility path runs directly
from world age into provisional erosion and final rendering; it is reference
behavior, not the canonical stage order.

Legacy pipeline remains callable through `map_maker.legacy.*` but does not participate in this DAG.

Each section below has its own detailed specification in this folder.
