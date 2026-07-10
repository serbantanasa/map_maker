# Planet Engine Decision Log

This file records design decisions for the next-generation world generator.
Entries are intentionally concise. A decision marked "approved" is accepted
unless later superseded by a new numbered decision. A decision marked
"accepted for prototype, tentative" is a working hypothesis that must pass an
early executable experiment before promotion to approved status.

## Decision 001: Canonical Planet Grid

Status: approved, provisional

Decision:
Use a cubed sphere as the canonical simulation and storage topology.

Rationale:
- Avoids equirectangular polar distortion as a core simulation artifact.
- Supports true spherical wraparound through topology-aware neighbor links.
- Keeps storage raster-like and chunkable by face and tile.
- Supports multi-resolution refinement.
- Allows beautiful projected exports without making the projection the
  simulation grid.

Implications:
- Equirectangular is export/debug only.
- All simulation stages must use centralized topology APIs for neighbors,
  distances, bearings, and area weights.
- Hydrology, erosion, climate, and movement across face edges must treat face
  edges as normal connected neighbors.
- Map rendering samples from the cubed sphere into cartographic projections.

Alternatives considered:
- Equirectangular lat/lon raster: simpler, but polar distortion and adjacency
  artifacts are unacceptable for canonical simulation.
- Icosahedral/geodesic grid: physically elegant, but harder for chunked raster
  storage, image-like layers, and tooling.
- HEALPix/equal-area grid: useful for analysis, awkward for terrain morphology
  and game-oriented map layers.

## Decision 002: Planetary Boundary Conditions

Status: approved, provisional

Decision:
The first engine version targets Earth-like dense rocky water worlds around
stable main-sequence stars. Parameter variation is allowed only inside ranges
that plausibly preserve long-term habitability and avoid rapid tidal locking.

Default baseline:

```text
planet_radius = 1 Earth
surface_gravity = 1 g
mean_density = Earth-like

star_luminosity = 1 Lsun
semi_major_axis = 1 AU
eccentricity = 0.0167
obliquity = 23.44 deg
rotation_period = 24 h

moon_mass = 1 Luna
moon_distance = 384400 km
```

Derived outputs feed downstream stages:
- Annual mean insolation by cell.
- Seasonal insolation amplitude.
- Eccentricity-driven seasonal asymmetry.
- Equator-to-pole energy gradient.
- Polar day/night tendency.
- Tide strength index.
- Obliquity stability index.

Non-goal:
V1 does not simulate arbitrary exotic planets.

## Decision 003: Resolution Hierarchy

Status: approved, provisional

Decision:
The engine always produces a global coarse/mid world, while high-resolution
detail is generated only for selected regions.

Initial hierarchy:

| Tier | Scope | Approx Resolution | Purpose |
| --- | --- | ---: | --- |
| L0 Planet | global | 50-100 km | plates, crust age, ocean basins, macro climate |
| L1 Planet | global | 10-25 km | major elevation, mountain belts, shelves, climate zones |
| L2 World | global or large regions | 2-5 km | major hydrology, biomes, large basins, resource provinces |
| L3 Region | selected regions | 500 m-1 km | county-scale hydrology, erosion, soils, terrain constraints |
| L4 Local | selected local maps | 50-100 m | game tiles, settlements, tactical detail; likely separate workstream |

Key rules:
- Finer levels refine coarser levels. They do not independently regenerate
  unrelated geography.
- L3 inherits coastlines, drainage basins, climate, uplift/erosion context,
  biome tendencies, lithology, and resource potential from L2.

First vertical slice target:
Small global L0-L1 plus one L3 region.

## Decision 004: Depression-Aware Hydrology

Status: approved, provisional

Decision:
Hydrology is basin- and depression-aware. The engine models basins, lakes,
spill points, overflow, and breach erosion before producing final river
networks.

Rationale:
Closed depressions are not always errors. They may become permanent lakes,
seasonal lakes, wetlands, salt flats/playas, inland seas, overflow basins,
breached paleolakes, or canyon/gorge systems at outlet points.

Minimum invariants:
- Every land cell drains to ocean, a stable lake/wetland/basin sink, a glacier
  sink in later versions, or an explicitly modeled endorheic terminal basin.
- Drainage graphs are acyclic except for explicitly modeled water bodies.
- Flow accumulation increases downstream.
- Rivers follow slope unless passing through lake overflow or breached outlets.
- Basins and watersheds are explicit outputs.
- Deltas appear only where significant discharge meets low-gradient coast or
  shelf.
- Endorheic basins are allowed, especially in arid interiors.
- Discharge should relate to precipitation, catchment area, evaporation, and
  infiltration.

Required process:
1. Compute initial flow routing on smoothed terrain.
2. Detect depressions and closed basins.
3. Estimate basin water balance: inflow, precipitation/runoff, evaporation,
   seepage/infiltration, seasonality, and sedimentation tendency.
4. Classify each basin as stable lake, endorheic salt basin, wetland, overflow
   lake, or breached/drained basin.
5. Apply fill-spill-merge behavior through lowest spill points.
6. Model sustained overflow as outlet carving at the basin-graph level:
   lowered outlet elevation, incision field, gorge/canyon metadata, and
   downstream sediment pulse.
7. Recompute or stabilize hydrology after outlet carving and lake
   classification.

Implementation guidance:
- At 2-5 km resolution, do not try to simulate rapid erosional events cell by
  cell. Model overflow carving at basin-graph scale, then refine in L3.
- Do not blindly fill all depressions. Preserve legitimate closed basins.
- Major drainage is established at L1/L2. L3 adds tributaries and local detail
  without inventing a disconnected river system.
- Rivers are rendered from smoothed discharge-aware polylines, not raw pixel
  masks.

Game-relevant outputs:
- Fertile lake basins.
- Salt flats.
- Marshes and wetlands.
- Strategic river outlets.
- Canyon chokepoints.
- Floodplains.
- Drained paleolake plains.
- Inland seas.
- Deltas.

## Decision 005: Rivers As Graph/Vector Features

Status: approved, provisional

Decision:
Rivers are canonical graph/vector features with raster support layers, not
raster-only terrain classes. Coarse cells may contain river corridors without
implying the river is cell-sized.

Rationale:
Many rivers are much narrower than 2-5 km coarse cells, especially county-level
tributaries. Representing rivers only as raster cells makes small rivers too
wide, hides channel character, and produces poor downstream game data.

Hydrology outputs:
- A drainage graph.
- River reach tables.
- Smoothed spherical/cubed-sphere polylines.
- Raster support fields for erosion, rendering, and regional refinement.

Required river reach fields:
- `reach_id`
- `from_node`
- `to_node`
- `upstream_reach_ids`
- `downstream_reach_id`
- `basin_id`
- `polyline_on_cubed_sphere`
- `flow_direction_vector`
- `slope`
- `strahler_order`
- `discharge_mean`
- `discharge_seasonal`
- `velocity_mean`
- `velocity_seasonal`
- `stream_power`
- `channel_width_m`
- `channel_depth_m`
- `valley_width_m`
- `floodplain_width_m`
- `meander_index`
- `braiding_index`
- `incision_m`
- `sediment_load`
- `bed_material`
- `morphology_class`

Required raster support layers:
- `basin_id`
- `flow_dir_x`
- `flow_dir_y`
- `flow_accumulation`
- `flow_velocity_index`
- `stream_power`
- `river_corridor`
- `floodplain_potential`

Initial morphology classes:
- Mountain torrent.
- Upland river.
- Lowland meander.
- Braided river.
- Delta distributary.
- Endorheic inflow.

Implementation guidance:
- At L2, a raster cell indicates a river corridor or valley influence, not a
  river occupying the whole cell.
- L3 refines river corridors and tributaries.
- L4 or separate local-map generation places actual channels, banks, meanders,
  oxbows, marshes, levees, fords, and human-scale river features.
- Velocity may initially be approximate and classified from slope, discharge,
  roughness, and channel geometry. Full hydraulic simulation is not required in
  v1.

## Decision 006: Storage And Layer Versioning

Status: approved, provisional

Decision:
Canonical simulation data is stored as immutable, versioned, chunked artifacts.
Dense gridded layers use Zarr. Tables, graphs, catalogs, and sparse records use
Parquet/Arrow. Rendered maps are exports only.

Rationale:
The engine needs resumable runs, inspectable intermediate state, regional
refinement, feedback-loop history, and future training data. Single-file dense
array dumps are insufficient for huge maps and selected high-resolution
regions.

Canonical storage split:
- Dense layers: Zarr.
- Tables, graphs, catalogs, sparse records: Parquet/Arrow.
- Config, manifests, and layer metadata: JSON/YAML.
- Pretty maps and previews: image exports only.

Key rule:
Stages never mutate previous canonical layers in place. They write new layer
versions.

Examples:
- `elevation_initial`
- `elevation_eroded_pass_1`
- `hydrology_initial`
- `hydrology_after_breach`
- `elevation_after_hydro_erosion`

Suggested run layout:

```text
runs/<run_id>/
  manifest.json
  config.yaml
  topology/
    cubed_sphere.json
  layers/
    L0/
    L1/
    L2/
  regions/
    <region_id>/
      manifest.json
      layers/
  tables/
    river_reaches.parquet
    basin_graph.parquet
    lake_catalog.parquet
    deposit_catalog.parquet
  previews/
  logs/
```

Implications:
- Existing `.npy` dense-grid persistence should evolve into a Zarr backend.
- Chunk sizes must be benchmarked.
- Layer metadata and versioning become required for canonical outputs.
- Scratch outputs are still allowed during development, but downstream stages
  should consume registered canonical artifacts.

## Decision 007: Stage Artifact Contract

Status: approved, provisional

Decision:
Any artifact consumed by downstream stages must be registered, typed, versioned,
validated, and semantically documented. Scratch artifacts are allowed, but they
cannot become dependencies until promoted.

Rationale:
The pipeline must not become a pile of mysterious arrays. Registered artifacts
make runs reproducible, inspectable, resumable, trainable, and easier to
validate.

Canonical artifact metadata:
- `artifact_id`
- `name`
- `kind`: `dense_layer`, `table`, `graph`, `metadata`, `preview`, or `scratch`
- `level`: `L0`, `L1`, `L2`, `L3`, or `L4`
- `topology`
- `shape`
- `dtype`
- `units`
- `nodata`
- `valid_range`
- `semantic_type`
- `stage_name`
- `stage_version`
- `config_hash`
- `seed`
- `dependencies`
- `checksum`
- `created_at`

Validation examples:
- Elevation has no NaNs.
- Biome codes exist in the biome taxonomy.
- Basin graph is acyclic except for explicitly modeled water bodies.
- River discharge is nonnegative.
- Fertility is in `[0, 1]`.
- Plate IDs match the plate catalog.
- Dense layers match their declared topology and resolution level.

Implementation guidance:
- Prototype freely with scratch artifacts.
- Promote only stable, semantically meaningful outputs to registered artifacts.
- No downstream stage may depend on scratch artifacts.

## Decision 008: V1 World Stack

Status: approved, provisional

Decision:
V1 targets a concrete world stack that produces a beautiful, causally plausible
global world plus one refined region. It does not include human/civilization
simulation or local tactical maps.

V1 stage stack:
1. Planet Setup: star/orbit/moon defaults, cubed sphere, insolation fields.
2. Tectonic Skeleton: plates, plate boundaries, velocities, crust type, crust
   age.
3. Geological Provinces: shields, basins, arcs, rifts, orogens, shelves,
   volcanic provinces.
4. Initial Elevation: ocean basins, continents, mountain belts, rifts, shelves,
   roughness.
5. Climate Pass 1: temperature, precipitation, wind/moisture transport,
   aridity.
6. Hydrology Pass 1: basins, lakes, drainage graph, trunk rivers, endorheic
   basins.
7. Erosion/Sedimentation Pass: valley incision, floodplains, sediment basins,
   deltas, adjusted elevation.
8. Hydrology Pass 2: stabilized rivers after fill-spill-carve and erosion.
9. Soils/Biomes: fertility, soil moisture, biome class, vegetation potential.
10. Mineral/Energy Systems: ore potential, coal basins, evaporites, placers,
    deposit catalog.
11. Region Refinement: one selected L3 region with inherited rivers, soils,
    terrain, and resources.
12. Map Export: beautiful projected world map and region overview.

First-success acceptance bar:
- Recognizable non-blob continents.
- Plausible ocean basins and shelves.
- Mountain belts that follow tectonic causes.
- No absurd continent-spanning rectangular mountains.
- Climate zones that make visual sense.
- Rivers drain basins, form lakes/deltas where appropriate, and do not look
  decorative.
- At least one refined region has county-scale valleys and tributaries.
- Soils/biomes are consistent with climate and drainage.
- Mineral/coal potential follows geological history.
- One beautiful projected world map plus one beautiful regional map are
  generated from canonical layers.

## Decision 009: Geological History Through Process-Constrained Synthesis

Status: approved, provisional

Decision:
Use a hybrid causal generator. A coarse, event-driven geological history
establishes crustal state, geological provinces, process constraints, and
provenance. Specialized morphology stages then generate detailed terrain
inside those constraints. V1 does not attempt comprehensive numerical
geophysics.

Required distinctions:
- Kinematic plates are not continents.
- Plates may carry oceanic crust, continental crust, and multiple terranes.
- Terranes, cratons, sedimentary basins, arcs, rifts, sutures, and orogens
  retain identities and histories independent of the current plate mosaic.
- Elevation is a state derived from crustal structure, tectonic events,
  isostasy, volcanism, erosion, and sedimentation. Plate type alone must not
  determine continental elevation.

Geological history requirements:
- Plate boundaries are segmented along strike and retain persistent state.
- Boundary segments progress through regimes such as oceanic subduction,
  terminal ocean closure, terrane collision, continental collision, mature
  suture, and post-collisional collapse.
- Outcomes depend on remaining oceanic lithosphere, slab behavior, accumulated
  shortening, margin geometry, crustal strength, microplate mobility,
  extension, and erosion.
- Microplates arise in plausible settings such as arcs, rifts, triple
  junctions, and fragmented collision zones. They may rotate, accrete, split,
  or be consumed.
- High plateaus require an explicit supporting history. They are not the
  default expression of continental crust.

Basin and stratigraphy requirements:
- Basins are persistent geological objects governed by subsidence, flexure,
  sea level, sediment supply, compaction, uplift, and erosion.
- Seas may deepen, fill with sediment, become terrestrial basins, or later be
  inverted into folded and uplifted mountain belts.
- Coarse stratigraphic packages preserve age, facies, thickness, provenance,
  organic content, burial, and thermal history for later soil and resource
  generation.

Realism and validation:
- Hard causal invariants must be enforced per stage.
- Statistical morphology is evaluated across generated worlds.
- Deterministic benchmark scenarios cover subduction arcs, mature continental
  collision, segmented closure and rollback, continental rifting, old eroded
  orogens, basin inversion, and hotspot tracks.
- Failed realism checks are repaired by changing causal events or parameters
  and replaying affected stages, not by cosmetically repainting final terrain.
- Rare outcomes are allowed when their event history supports them.

Implementation implication:
The current model that assigns one crust type and nearly constant thickness to
an entire plate is reference code only. Preserve the pipeline architecture,
deterministic execution, caching, artifact persistence, and Rust-kernel option,
but replace the canonical tectonic/elevation model incrementally.

## Decision 010: Explicit History Window And Adaptive Nested Sweeps

Status: accepted for prototype, tentative

Working decision:
Represent ancient planetary history with an inherited deep-time substrate,
then simulate approximately the final 600-800 million years explicitly. The
explicit history proceeds through repeated coarse global predictor sweeps and
event-triggered higher-resolution regional periods.

Proposed cycle:
1. Run a coarse global prediction for a geological interval.
2. Detect events or uncertainty requiring refinement, including rifting, ocean
   closure, collision, rollback, rapid basin evolution, and major sediment
   transfer.
3. Restore the interval checkpoint and rerun active corridors at higher
   resolution with coarse boundary conditions.
4. Conservatively restrict fine results into global state.
5. Expand and replay a patch if boundary reconciliation exceeds tolerance.
6. Persist the committed checkpoint and continue.

Required safeguards:
- Refinement follows geological systems rather than fixed geographic tiles.
- Mass, crust, water, and sediment budgets remain conserved across level
  transitions.
- Vector geometry, provenance, stratigraphy, event history, and subgrid
  distributions survive restriction.
- Broad outcomes must be stable under reasonable changes in refinement level.

Reason for tentative status:
Repeated refinement and restriction may introduce resolution-dependent
history, expensive reconciliation, or unstable boundary behavior. A small
deterministic prototype must test those risks before this becomes canonical.

## Decision 011: Persistent Subgrid Geological Features

Status: accepted for prototype, tentative

Working decision:
The coarse process raster is not the authoritative geometry of continents,
islands, narrow seas, straits, arcs, terranes, basin margins, or continental
margins. Important geological features retain persistent vector, object, or
parametric identities below the current raster resolution.

Hard working invariant:
A feature may disappear through a recorded physical process or sea-level
change, but not merely because simulation resolution decreased.

Representation rules:
- Coarse coastal cells store area fractions and elevation distributions rather
  than a single majority land/ocean category.
- Plate boundaries, terranes, rifts, sutures, basin margins, arcs, and passive
  margins retain adaptive subcell geometry.
- Island systems preserve origin, geometry or parametric backbone, bathymetric
  platform, provenance, emergence state, and stable child-generation seeds.
- Coarsening may simplify geometry but must preserve topology-critical
  features and stable IDs.
- Fine artifacts remain archived even when a coarse state is produced for the
  next history sweep.
- Map rendering may use a cartographic minimum display size without altering
  physical feature area.

Reason for tentative status:
Maintaining vector/object features alongside changing raster resolutions adds
substantial topology and reconciliation complexity. The prototype must prove
that islands, conjugate margins, narrow seas, and microplates survive repeated
coarse/fine cycles without excessive cost or artificial persistence.

## Decision 012: Seasonal Intermediate-Complexity Climate

Status: approved, provisional

Decision:
V1 uses a causal seasonal climate model of intermediate complexity. It is more
physical than painted latitude bands and less ambitious than a general
circulation model. The modern climate pass produces twelve monthly
climatologies.

Required modern outputs:
- Surface temperature.
- Precipitation.
- Horizontal wind vector.
- Humidity.
- Snow accumulation and melt potential.
- Evaporation.
- Runoff potential.

Required drivers and processes:
- Orbital insolation from latitude, season, obliquity, and eccentricity.
- Different land and ocean heat capacities.
- Elevation lapse rate.
- Broad pressure-gradient and planetary-circulation tendencies.
- Coriolis deflection through topology-aware spherical vectors.
- Moisture transport from oceans and major lakes.
- Orographic precipitation and rain shadows.
- Continental drying.
- Snow persistence and seasonal melt.
- Simplified sea-surface temperature and coastal moderation.
- Iteration of temperature, wind, and moisture to a stable climatology.

Paleoclimate rule:
Geological history sweeps use a cheaper climate proxy based on insolation,
latitude, continentality, broad elevation, prevailing wind, orographic
precipitation, and a configurable global warm/cold state. The full monthly
model is reserved for the present world and selected checkpoints.

Feedback rule:
V1 may perform one bounded biome/soil feedback pass through albedo,
evapotranspiration, infiltration, and erosion resistance. It does not run an
open-ended climate/vegetation convergence loop.

V1 non-goals:
- Daily weather.
- Individual storms.
- ENSO-like internal variability.
- Full atmospheric or oceanic general circulation.

## Decision 013: Historical Source-To-Sink Coupling

Status: approved, provisional

Decision:
Geological history sweeps use a bounded, basin-scale source-to-sink model for
erosion, sediment transport, deposition, compaction, crustal loading, and
isostatic response. Detailed channel networks are reserved for the present
world.

Historical process sequence:
1. Apply tectonic uplift and subsidence.
2. Estimate erosivity from coarse paleoclimate, relief, lithology, and broad
   vegetation state.
3. Build temporary basin-scale drainage and transport graphs.
4. Erode bedrock and move sediment conservatively from source to sink.
5. Deposit sediment in terrestrial basins, lakes, foreland basins, deltas,
   shelves, and marine basins according to transport capacity and
   accommodation.
6. Apply compaction, sediment loading, flexural subsidence, and isostatic
   rebound.

Historical sediment classes:
- Coarse clastic.
- Fine clastic.
- Carbonate.
- Organic-rich.
- Volcaniclastic.
- Chemical and evaporitic.

Persistent stratigraphic properties:
- Source terranes and provenance.
- Deposition interval.
- Environment and facies.
- Thickness and grain class.
- Organic content.
- Burial and thermal history.

Persistence rule:
Ordinary temporary paleorivers are not canonical features. Major drainage
captures, lake breaches, paleovalleys, placer-forming systems, and other events
with lasting geological consequences are preserved in the event and
stratigraphy ledgers.

Feedback boundary:
Historical erosion and deposition may change topography, crustal load,
flexure, isostatic elevation, and basin evolution. In V1 they do not alter
plate velocities, mantle dynamics, or the plate-boundary kinematic solution.

Modern-world boundary:
The final world receives the detailed depression-aware hydrology, permanent
river graph, discharge, channel morphology, floodplain, delta, incision, and
regional refinement passes described in Decisions 004 and 005.

## Decision 014: Modern Hydrology Realism Gates

Status: approved, provisional

Decision:
Modern hydrology and erosion must pass independent topology/conservation,
process-consistency, statistical, deterministic-scenario, and visual gates.
A plausible-looking rendered river network alone is not sufficient.

Hard topology and conservation gates:
- Every land cell drains to ocean, a registered water-body or glacier sink, or
  an explicit endorheic basin.
- Drainage graphs contain no accidental cycles or dangling reaches.
- Lakes have coherent water levels and explicit spill points.
- Contributing area increases downstream.
- Monthly water balances account for precipitation, snowmelt, evaporation,
  infiltration, storage, upstream inflow, and downstream discharge.
- Eroded sediment is deposited, retained in transport, or exported.
- Rivers and watersheds cross cubed-sphere face boundaries continuously.

Process-consistency gates:
- Discharge follows catchment runoff and seasonal water balance.
- Width, depth, velocity, slope, stream power, sediment load, and morphology
  class agree with one another.
- Endorheic systems, losing reaches, intermittent rivers, braided reaches,
  floodplains, and deltas occur only under supporting conditions.

Statistical gates:
Generated regions are compared with broad Earth-derived distributions for
analogous climate, relief, and geology. Metrics include basin area and shape,
main-channel length, stream order, drainage density, slope-area relations,
river width and discharge, lake size, endorheic frequency, and delta
frequency. Targets are conditional distributions, not a single global Earth
average.

Required deterministic scenarios:
- Humid mountain-to-ocean drainage.
- Arid enclosed basin and terminal salt lake.
- Low-gradient meandering continental river.
- Fill-spill-merge lake chain.
- Overflow lake with outlet incision and drainage.
- Sediment-rich river building a delta on a shallow shelf.
- River and watershed crossing cubed-sphere faces.
- L3 refinement inheriting the correct L2 trunk network.

Bounded iteration:
V1 runs two standard modern hydrology passes separated by erosion,
sedimentation, and outlet carving. One exceptional correction pass is allowed
when topology remains unstable. Continued instability rejects the output or
replays the causal terrain stage rather than iterating indefinitely.

## Decision 015: Property-First Soils And Functional Biomes

Status: approved, provisional

Decision:
Canonical soils are physical and chemical property fields derived from causal
soil-forming factors. Canonical vegetation is a mixture of plant functional
types and productivity traits. Familiar soil and biome classes are derived
labels rather than primary painted categories.

Soil drivers:
- Parent material, lithology, and depositional origin.
- Climate and weathering regime.
- Terrain position and slope.
- Drainage, groundwater, flooding, and seasonal saturation.
- Surface exposure age and reset events.
- Erosion, deposition, and sediment provenance.
- Vegetation and organic inputs.

Required soil properties:
- Depth to bedrock and regolith/soil depth.
- Sand, silt, clay, and coarse-fragment fractions.
- Bulk density.
- Organic carbon.
- pH, carbonate content, and salinity.
- Drainage and seasonal saturation.
- Water-holding capacity.
- Nutrient and fertility potential.
- Erodibility.
- Parent material and surface-reset age.

Scale rule:
L2 soil map units store component mixtures and fractions. L3 refinement expands
those components into deterministic terrain catenas, including ridge, slope,
footslope, floodplain, wetland, and closed-basin positions.

Vegetation outputs:
- Plant functional type fractions.
- Potential biomass and productivity.
- Canopy cover and rooting depth.
- Growing-season length and seasonality.
- Fire tendency.
- Grazing and forest-resource potential.
- Bare, saline, desert, wetland, ice, and other non-vegetated fractions.

Bounded feedback:
V1 generates initial mineral soil, estimates vegetation potential, adjusts
organic matter and soil moisture, finalizes functional vegetation and biome
labels, then permits one bounded hydrology and erosion-resistance correction.

V1 non-goals:
- Individual plant species.
- Full ecological succession.
- Detailed wildfire, pest, or disturbance history.
- Human land use.

## Decision 016: System-First Mineral And Energy Resources

Status: approved, provisional

Decision:
Generate resources from persistent geological mineral and energy systems.
Canonical outputs include continuous prospectivity fields and an explicit
ground-truth catalog of major deposits. Minor occurrences are generated
deterministically during regional refinement.

Required causal chain:
- Geological and tectonic setting.
- Source material.
- Energy or concentrating process.
- Transport medium and pathway.
- Structural, physical, or chemical trap.
- Correct timing.
- Preservation, exposure, or later reworking.

Initial system families:
- Arc magmatic-hydrothermal.
- Orogenic and shear-zone.
- Mafic and ultramafic magmatic.
- Volcanogenic seafloor.
- Sediment-hosted basin.
- Ancient iron and inherited cratonic.
- Weathering, residual, and supergene enrichment.
- Placer and heavy-mineral.
- Evaporite and chemical sediment.
- Coal-forming basin.
- Petroleum basin.

Major deposit records:
- Stable deposit and mineral-system IDs.
- Commodities and byproducts.
- Formation age, host unit, source terrane, and structural setting.
- Geometry, depth, exposure, size class, and grade distribution.
- Alteration halo, weathering state, and preservation confidence.

Scale rule:
L2 stores regional prospectivity, mineral systems, and major deposits. L3
refines orebody geometry, exposed veins, coal seams, placer reaches, and minor
occurrences from stable parent records and seeds.

Coal rule:
Coal requires supported peat productivity, persistent waterlogging,
accommodation, preservation, burial, and thermal history. Rank, seam potential,
depth, impurities, deformation, and exposure derive from that history.

Petroleum rule:
Petroleum potential requires source, maturity, reservoir, seal, trap,
migration, timing, and preservation. V1 models basin-scale systems rather than
multiphase pore-scale flow.

Game-utility rule:
Configurable abundance envelopes may change eligible system productivity and
replay deposit formation. They may not sprinkle deposits without geological
causes. Ground-truth deposits are separate from any future civilization's
knowledge or discovery state.

## Decision 017: Probabilistic Constrained Regional Refinement

Status: approved, provisional

Decision:
Regional refinement is a coherent conditional realization of inherited parent
state, not an independent regeneration. L0-L2 cells store subgrid
distributions, component mixtures, spatial structure, feature references, and
refinement priors rather than only mean values.

Required distinction:
- Actual subgrid composition records unresolved area fractions and material
  mixtures.
- Refinement freedom describes allowed geometry and morphology within parent
  constraints.
- Uncertainty records confidence in unresolved state.
These concepts must not be collapsed into one undifferentiated probability.

Hard inherited constraints:
- Coastline, island, strait, lake, basin, and ocean topology.
- Stable feature and parent-child IDs.
- Trunk river connectivity, direction, mouth, and inherited fluxes.
- Geological structures, terranes, lithology, and stratigraphy.
- Monthly climate boundary conditions.
- Major deposit and mineral-system identity and mass.

Subgrid terrain priors may include:
- Elevation distribution or hypsometric curve.
- Land, water, shelf, and terrain-component fractions.
- Relief and slope distributions.
- Dominant aspect, anisotropy, and geological orientation.
- Roughness by spatial scale.
- Lithology, surface-material, soil, and vegetation mixtures.
- Boundary corridors and topology-critical feature references.

Refinement sequence:
1. Extract the requested area plus process-specific hydrological, geological,
   terrain, and climate halos.
2. Conservatively prolong parent fields and install inherited feature
   constraints.
3. Generate correlated, geology- and process-conditioned higher-frequency
   detail across tiles and halos.
4. Route and stabilize local drainage, erosion, soils, vegetation, and
   resources.
5. Reaggregate the child result and compare it with all parent levels.
6. Expand, replay, or reject refinement when constraints fail.

Tolerance rule:
- Topology and stable identities have zero tolerance.
- Conserved mass and flux quantities are preserved to numerical tolerance.
- Land/water fractions, elevation quantiles, relief, basin area, discharge,
  and other non-conserved statistical properties use property-specific
  absolute or relative tolerances, generally in the 1-5 percent range.
- Boundary positions are constrained by explicit corridors rather than
  percentage error.

Determinism and adjacency:
- Hierarchical seeds are stable across generation order.
- Neighboring refinements use overlapping halos and shared boundary artifacts.
- Separately refined adjacent regions must meet without seams.
- Committed child realizations remain stable; alternate realizations require
  explicit branch/version IDs.

## Decision 018: Truth And Atlas Rendering

Status: approved, provisional

Decision:
Every successful world produces unsmoothed diagnostic truth renders and
separately styled physical atlas renders. Cartographic styling and
generalization may improve legibility but may not modify canonical geography.

Required products:
- Rotatable globe view for topology, wraparound, and polar inspection.
- Equal-area physical world map.
- Physical regional map for each refined region.
- Diagnostic geology, climate, hydrology, soil, biome, resource, topology, and
  refinement-boundary maps.
- Projected data layers suitable for later game tooling.

Projection rules:
- Equal Earth is the default static world projection.
- The central meridian is selected to avoid cutting important continents,
  archipelagos, or basins where practical.
- Appropriate local projections are selected for regional maps.
- Equirectangular output is limited to texture and diagnostic interchange.

Physical atlas composition:
- Natural land color derived from vegetation, aridity, rock exposure, and
  snow, with restrained elevation tinting.
- Bathymetric relief and visible continental shelves.
- Scale-aware multidirectional hillshade.
- Vector-derived coasts, islands, lakes, wetlands, glaciers, salt flats, and
  discharge-aware rivers.
- No political borders or human labels in V1.

Truth-render rule:
Truth renders expose unsmoothed elevation, exact water masks and vectors,
cubed-sphere face boundaries, refinement boundaries, and canonical feature
dimensions without decorative texture.

Atlas-render rule:
Atlas renders may apply topology-preserving generalization, minimum visible
sizes for small islands and rivers, tonal adjustment, and relief enhancement.
These choices alter appearance only.

Beauty and legibility gates:
- No face seams, projection holes, or incoherent refinement boundaries.
- Coastlines do not visibly stair-step at the target display scale.
- Islands, straits, river hierarchy, lake connections, shelves, and terrain
  remain legible.
- Relief shading does not overwhelm or invert terrain information.
- Major surface classes remain distinguishable, including under common
  color-vision deficiencies.
- A fixed multi-seed gallery receives side-by-side subjective review in
  addition to automated checks.

Versioning rule:
Simulation, projection configuration, and atlas style carry independent
versions so cartography can improve without regenerating the world.

## Decision 019: Gate Then Score Realism Evaluation

Status: approved, provisional

Decision:
World acceptance uses non-negotiable validity gates followed by separate
subsystem quality dimensions. A single composite realism score may be reported
for convenience but may never conceal a hard failure or a subsystem below its
minimum threshold.

Subsystem quality dimensions:
- Causal coherence.
- Statistical realism.
- Morphological and visual quality.
- Cross-scale stability.
- Game utility.
- Confidence and validation coverage.

Required benchmark suites:
- Deterministic process scenarios for isolated mechanisms and edge cases.
- Earth-analog regions selected by comparable climate, relief, geology, and
  process regime.
- A fixed random-world seed gallery for integration, variety, and subjective
  review.

Required cross-layer tests:
- Terrain aligns with tectonic and geological history.
- Climate responds coherently to latitude, elevation, land/ocean distribution,
  and orography.
- Hydrology agrees with climate and terrain.
- Sediment sinks agree with erosional sources and transport paths.
- Soils agree with parent material, climate, drainage, terrain, and age.
- Biomes agree with climate, soil moisture, and soil properties.
- Resources agree with geological systems, timing, and preservation.
- Refined regions preserve parent topology, distributions, and fluxes.

Threshold rule:
Numeric thresholds are calibrated from reference-data variation, dataset
disagreement, benchmark scenarios, and clearly failed synthetic cases before
being frozen. Different subsystems use appropriate distribution, topology,
graph, spatial, and correlation metrics rather than one generic distance.

Realism and utility rule:
Game utility is evaluated separately from scientific plausibility. Utility
envelopes may reject or causally replay a world but may not repaint resources,
terrain, climate, or drainage after generation.

Atlas-grade acceptance requires:
- All hard invariants pass.
- Every subsystem exceeds its own minimum dimensions.
- Cross-layer and cross-resolution tests pass.
- Performance and storage budgets pass.
- Truth renders pass automated inspection.
- The fixed physical-map gallery passes human review.

## Decision 020: Rust Computational Core With Python Orchestration

Status: approved, provisional

Decision:
Preserve the repository's hybrid architecture but make its ownership boundary
explicit. Rust owns computational state and inner simulation loops. A small
Python package owns configuration, stage orchestration, persisted datasets,
experiments, diagnostics, and cartography.

Rust ownership:
- Cubed-sphere indexing, topology, geometry, and native neighbor operations.
- Geological state and chronological integration.
- Plate, terrane, boundary-event, basin, erosion, sediment, and hydrology
  kernels.
- Refinement/restriction kernels and conservation checks.
- Performance-critical realism validators.

Python ownership:
- User configuration and experiment definitions.
- Stage DAG, cache keys, checkpoints, and run manifests.
- Zarr and Parquet/Arrow dataset orchestration.
- Benchmark-suite control and statistical analysis.
- Diagnostic visualization, projected exports, and atlas rendering.

Boundary rule:
Python invokes coarse native operations such as a complete geological interval,
surface-process pass, or refinement operation. It must not cross the language
boundary per cell or per numerical timestep.

Build and packaging rule:
- Create one root Cargo workspace for native crates.
- Add a small `pyproject.toml` for the existing Python package.
- Replace Cargo build-on-import behavior with an explicit reproducible native
  build and development workflow.
- Keep native-library and Python-package versions compatible and recorded in
  run manifests.
- Select the final binding mechanism after benchmarking; PyO3/maturin and a
  narrow C ABI remain candidates.

Migration rule:
Reuse the current Python stage registry, execution, caching, artifact, and
visualization concepts while incrementally replacing the current tectonic and
elevation kernels. Do not rewrite the full pipeline solely to make the project
Rust-only.
