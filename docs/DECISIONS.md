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

Implementation contract:
- A year is represented by twelve equal-time climatological bins, not Gregorian
  calendar months.
- Insolation is daily-mean top-of-atmosphere shortwave forcing computed at each
  bin midpoint from a Keplerian orbit.
- Orbital period, perihelion day, and northern vernal equinox day are explicit
  controls so the seasonal phase is reproducible.
- Moon properties produce tide-strength and obliquity-stability indices in V1;
  detailed tidal and long-term obliquity evolution remain future work.

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

Implementation contract:
- Climate consumes persisted monthly orbital forcing rather than recreating a
  latitude-only insolation approximation.
- Effective atmospheric orography is a persisted, smoothed land-surface field;
  ocean bathymetry is not atmospheric elevation.
- Horizontal wind is stored as global tangent XYZ vectors so cube-face edges do
  not define local component discontinuities.
- Moisture transport combines directed advection with unresolved synoptic mixing;
  rate-limited orographic rainout must allow continental penetration.
- The first pass writes pre-soil runoff potential only. It does not route rivers
  or silently resolve depressions.

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

## Decision 021: Causal Pre-Erosion Elevation Components

Status: approved, provisional

Decision:
Canonical elevation is generated before surface-process erosion as separate,
inspectable crustal, orogenic, and basin components on the cubed sphere. The
stage produces an initial bedrock surface and unresolved relief priors; it does
not finalize sea level or claim eroded landform realism.

Causality rule:
Broad elevation derives from crust thickness, density, age, buoyancy, and
subsidence. Localized tectonic relief derives from boundary regime, rate,
polarity, strength, and angular distance from connected boundary segments.
Geological province classes may condition those calculations but may not map
directly to fixed elevations.

Morphology rule:
Collision belts, volcanic arcs, trenches, ridges, rifts, and transforms use
distinct cross-strike profiles. Profiles cross cube-face seams through the
canonical neighbor graph. Long structures receive coherent along-strike
variation so they do not become uniform walls.

Corridor realization rule:
The discrete plate edge is the causal skeleton of a finite-width process
corridor, not a mandatory terrain centerline. Multi-scale spherical curvature,
coherent along-strike activity, regime-specific lateral offsets, and
plate-constrained distance smoothing may realize morphology inside that
corridor. These operations may not alter plate connectivity or move evidence
onto unrelated plates.

Ownership rule:
Rust owns graph propagation and component synthesis. Python owns configuration,
persistence, diagnostics, and cartographic previews.

Failure rule:
Continental-sized flat plateaus, uniform boundary walls, cube-face seams,
arbitrary class-to-height painting, and loss of inherited islands are hard
failures.

## Decision 022: Fractional Coarse Surface Coverage

Status: approved, provisional

Decision:
Coarse cells are mixed-cover containers rather than atomic terrain labels.
Surface classes whose physical features are commonly smaller than a cell must
store area fractions. Hydrologic feature identity and drainage topology remain
explicit and discrete.

Hydrology rule:
- Permanent lake and wetland coverage are separate fractions in `[0, 1]`.
- Lakes retain waterbody identity, level, area, volume, inflow, outflow, spill
  point, and downstream graph relationships.
- Rivers remain vector reaches with physical width and discharge rather than
  converting every crossed cell into a river cell.
- A sparse cell-to-waterbody table records membership and covered physical area.

Subgrid rule:
Unresolved terrain relief defines a provisional cell hypsometry. Lake level and
connected-basin occupancy determine fractional inundation and volume. This is a
computational approximation, not a substitute for refined bathymetry.

Refinement rule:
Regional refinement spatially realizes parent fractions while preserving parent
waterbody identity, drainage connectivity, aggregate area, storage, and flux.
Refined geometry may redistribute coverage inside the parent. It may instantiate
multiple minor waterbody objects from explicitly unresolved parent coverage, but
may not change aggregate water area, storage, or major connectivity merely to
improve appearance.

Classification rule:
Permanent lakes, present-day wetlands, seasonal floodplains, and paleowetlands
are distinct products. Paleowetlands may contribute to geological resource
history, including coal formation, but are not counted as present-day lakes.

## Decision 023: Hierarchical Rivers And Subgrid Fluvial Incision

Status: approved, provisional

Decision:
River channels, valleys, and floodplains remain explicit subgrid features at
every level where their physical width is smaller than a cell. A river may cross
a coarse cell without converting that cell into an atomic river or lowering its
entire bedrock surface. Literal raster incision is permitted only where the
incised landform is genuinely resolved.

Scale interpretation:
- Face-128 is a global structural level whose average Earth-size cell is about
  5,200 km2, or 72 km across. It may support drainage topology, basin-scale
  denudation, and source-to-sink budgets, but not resolved channel or valley
  morphology.
- Approximately 2-5 km cells may resolve major valley corridors and
  floodplains, while most channels remain vector or fractional features.
- Approximately 500 m-1 km regional cells may resolve many valleys and broad
  floodplains, but ordinary rivers and streams still retain explicit reach
  geometry and physical width.
- Approximately 100 m cells may resolve large rivers; minor channels remain
  subgrid. Human-scale channel realization belongs to later local refinement.

Canonical reach rule:
Each river reach preserves centerline geometry, upstream and downstream
relationships, entry and exit anchors, bed elevation, discharge, velocity,
physical width and depth, valley and floodplain width, incision depth, eroded
volume, sediment load, and provenance where available. Rendering width is not
simulation width and may not feed physical calculations.

Coarse-cell rule:
A sparse cell-to-reach relation stores reach length inside the cell and derived
channel, valley, and floodplain area fractions. Channel coverage is based on
physical reach width times in-cell length divided by physical cell area.
Incision lowers the reach bed and changes the unresolved low-elevation portion
of the cell hypsometry. It does not lower the whole cell by the reach incision
depth. Any change to cell-mean elevation derives only from a conserved physical
volume divided by cell area.

Refinement contract:
- Child drainage inherits the parent reach entry and exit cells as spatial
  constraints, downstream identity, monthly discharge, sediment flux, and any
  accepted lake or ocean sink. A coarse junction anchor is not an arbitrary
  fine-cell center that every connected reach must hit exactly.
- Fine reaches are realized downstream-first. An upstream reach may merge at
  any cell of its inherited downstream path inside the shared coarse junction
  cell. The actual fine join cell is stored explicitly. The union must remain a
  convergent directed acyclic graph with no reverse-used edge.
- Fine routing may add tributaries, meanders, local wetlands, and distributaries
  or redistribute unresolved coverage, but may not silently redirect a major
  trunk or change an inherited water and sediment budget.
- Fine incised and deposited volumes restrict to their parent totals within the
  configured conservation tolerance.
- Refined terrain must reproduce parent area-weighted elevation and unresolved
  relief constraints before local process adjustments are applied.
- Aggregation publishes conserved budgets and distribution updates; it does not
  replace accepted coarse topology solely because a finer realization looks
  preferable.

Implementation sequence:
1. Establish registered multiresolution reach and sparse membership artifacts.
2. Select one complete coarse drainage-basin footprint and refine it while
   preserving its registered reach graph, trunk identity, discharge, and known
   sediment boundary conditions. "Complete" describes spatial basin coverage;
   it does not conceal reach gaps inherited from a coarse extraction threshold.
3. Realize constrained local routing and subgrid valley properties in that
   basin.
4. Run fluvial incision and conservative sediment routing at the finest active
   level, then restrict budgets and terrain-distribution changes upward.
5. Generalize the proven regional path to additional basins and production
   resolution levels.

Failure conditions:
- Cell-wide trenches produced by subcell rivers.
- Simulation width inferred from cartographic stroke width.
- Refined rivers that do not connect to inherited parent reaches or sinks.
- Applied erosion while any inherited terminal lacks a registered downstream
  reach, lake/wetland/endorheic sink, or ocean boundary.
- Water, sediment, or incision volumes that change under restriction.
- Global fine grids used where selected regional refinement provides the same
  physical result within the accepted hierarchy.

## Decision 024: Hydrologic Connectors And Conservative Corridor Support

Status: implemented, provisional

Decision:
The canonical reach graph distinguishes physical river channels from
zero-width hydrologic connectors. A connector preserves source-to-sink topology
where routed discharge crosses preserved coarse depression or waterbody support
in which Hydrology Pass 1 cannot justify open-channel geometry. Ordinary
below-threshold land gaps may not become connectors. A connector is not a
hidden river and cannot contribute channel, valley, floodplain, or incision
area.

Connector contract:
- Connectors retain reach identity, ordered cell paths, upstream/downstream
  relationships, discharge, sediment flux, and sink topology needed by later
  stages.
- They publish `reach_kind = connector`, zero physical width and depth, zero
  valley and floodplain width, zero local velocity and stream power, zero
  incision, `hydrologic_connector` morphology, and a not-applicable bed
  material.
- A terminal reach is valid only at an ocean boundary or registered
  lake/wetland/endorheic sink. Any other terminal blocks refinement and erosion.
- Regional refinement may replace a connector with resolved inlet, waterbody,
  outlet, or local channel geometry, but may not infer erosion from the coarse
  connector itself.
- Topological path length and physical channel length remain separate. Shared
  connector endpoints preserve graph continuity but do not add physical channel
  length or support.

Corridor support contract:
- Requested channel, floodplain, and valley area derives from physical reach
  width times in-cell reach length. Rendering width never enters the budget.
- Physical channel support is reserved first. Valley support is distributed
  from the centerline into nearby sparse fine cells, and floodplain support is
  constrained to the allocated valley footprint.
- Per membership and in aggregate per fine cell, channel support must be no
  greater than floodplain support, which must be no greater than valley
  support. The summed support of each type cannot exceed the cell's physical
  area even where multiple reaches share it.
- Lateral support records carry zero reach length and zero incision volume.
  They describe occupied area, not additional channel centerline.
- Preserved depression parents are process-excluded: topological paths may cross
  them, but channel, valley, floodplain, and incision memberships may not enter
  them until resolved local geometry replaces the connector.
- Requested and represented areas must agree within the configured conservation
  tolerance. Allocation failure is fatal rather than silently clipping area.
- Competing valley demands are ordered by spatial scarcity with deterministic
  fallback orders. Floodplain area is allocated per reach from that reach's
  already-conserved valley footprint, guaranteeing nested support when the
  input widths are nested.

Current approximation:
Lateral allocation is deterministic and prefers nearby lower terrain, but it
does not yet solve cross-valley flow direction, flood recurrence, or
process-driven valley morphology. Those refinements may redistribute support
while preserving the registered reach graph and physical area budgets.

Failure conditions:
- A connector with nonzero physical channel dimensions or incision.
- A connector created across ordinary non-waterbody land merely to satisfy the
  readiness gate.
- An unresolved land terminal in the canonical reach graph.
- Physical process support in a connector-owned or preserved-depression parent.
- Corridor support clipped because it does not fit in a centerline cell.
- Per-cell support greater than physical cell area or non-nested channel,
  floodplain, and valley footprints.
- Lateral support counted as extra reach length or erodible channel area.

## Decision 025: Junction Beds And Conservative First-Pass Fluvial Budgets

Status: implemented, provisional

Decision:
The first refined fluvial pass solves the least-incision downstream-graded bed
surface compatible with the inherited physical channel graph. It then converts
that cut to solid volume and routes every removed volume unit to allocated
floodplain support, a registered inland sink, or an ocean terminal. It does not
force the provisional coarse potential-incision field into the fine terrain.

Bed-profile contract:
- Consecutive physical centerline memberships form a directed node DAG.
- A fine cell shared at a confluence has exactly one bed elevation.
- Bed elevation does not rise downstream and meets the configured minimum
  grade within numerical tolerance.
- Bed/profile coordinates and grade diagnostics persist as float64, and the
  grade gate is recomputed from emitted records after native serialization.
- The terrain prior is lowered only where required by the downstream envelope.
- A path-order gap across process-excluded support breaks the physical bed
  component. A connector does not invent a bed through an unresolved lake or
  depression.

Incision contract:
- Physical volume is channel width times represented in-cell length times solved
  incision depth.
- Incision remains a subgrid channel property. Full-cell mean elevation changes
  only by net process volume divided by full physical cell area.
- Child erosion and deposition volumes restrict exactly to their coarse parent.
- The inherited potential-incision volume is reported as a comparison. It is
  neither a target nor a cap until calibrated process history exists.

Sediment contract:
- Only newly eroded solid volume enters this historical routing budget.
- Physical reaches may retain a bounded, support-area-capped fraction on their
  allocated floodplains. Connectors retain none and transfer all incoming
  sediment downstream.
- Registered inland terminals retain their remainder in a terminal inventory.
  Ocean terminals export their remainder to the future delta/shelf model.
- Eroded volume must equal floodplain deposition plus terminal deposition plus
  ocean export. Reach-level incoming, local, deposited, and transferred budgets
  are persisted for inspection and surrogate training.
- Python independently reconciles profile, reach, fine-cell, parent, and native
  totals, including every reach's input/output balance and downstream transfer.
- Instantaneous inherited sediment load remains a separate flux diagnostic and
  may not be added dimensionally to historical solid volume.

Current approximation:
The minimum-grade envelope is terrain conditioning, not a calibrated
geological-time stream-power model. Floodplain retention uses inherited slope
and allocated floodplain-to-valley support with bounded depth. Its parameters
remain provisional until multi-seed morphology and Earth-derived sediment
benchmarks exist.

Failure conditions:
- Different bed elevations for reaches sharing a physical junction cell.
- A downstream physical bed rise above the configured grade tolerance.
- Connector bed, incision, floodplain deposition, or local sediment production.
- Process volume in a preserved-depression parent.
- Cell-mean terrain change based on incision depth rather than conserved volume.
- Eroded sediment disappearing, appearing, or changing under parent
  restriction.
- Native statistics agreeing internally while emitted catalogs disagree.
- Treating the canonical seed's potential-incision value as calibration.

## Decision 026: Bounded Sparse Hydrology Pass 2

Status: implemented, provisional

Decision:
Hydrology Pass 2 is a sparse selected-basin stabilization pass over the refined
terrain and accepted fluvial graph. It compares local drainage before and after
erosion, publishes every accepted receiver or depression change, and rejects
basin-scale reorganization. It is not a second unconstrained global hydrology
generation and may not silently replace Pass 1 identities.

Routing-surface contract:
- Ordinary child cells use their volume-adjusted full-cell mean terrain.
- A physical centerline cell uses its solved float64 channel bed as its subgrid
  routing surface; incision depth is not applied to the whole cell.
- Preserved coarse depressions and zero-width connectors remain nonphysical
  handoffs until dedicated bathymetric refinement resolves them.
- Outside-basin boundary support remains an inherited terminal and may not open
  a new lateral leak through the selected watershed boundary.

Topology contract:
- Physical trunk edges, junction cells, downstream reach references, and
  connector handoffs are fixed inputs.
- A deterministic sparse cubed-sphere D4 priority flood routes ordinary child
  cells to the accepted trunk, preserved waterbody support, or inherited ocean
  terminal.
- Every routed child has one receiver, the stabilized graph is acyclic, and
  contributing area is accumulated exactly once through the graph.
- Local receiver changes are allowed only within configured area and count
  bounds. Exceeding either bound rejects the pass instead of iterating.

Depression contract:
- Priority-fill depth is evaluated on both the pre-erosion and stabilized
  routing surfaces.
- Connected cells deeper than the configured minimum become local depression
  candidates with deterministic IDs, area, potential storage, spill cell, and
  before/after status.
- On a flat priority-fill component, the persisted spill is the component's
  final downstream exit in the accepted receiver DAG. This prevents arbitrary
  equal-level tree tie-breaking from creating self-edges or candidate cycles.
- A candidate is not automatically a permanent lake or wetland. Water balance,
  hydroperiod, soils, and vegetation determine those later.
- Inherited preserved lakes and depressions are audited as excluded handoffs,
  not regenerated from incomplete local bathymetry.

Bounded-correction rule:
Pass 2 itself applies at most one local receiver correction from the pre-erosion
to post-erosion surface. It reports whether any correction occurred and whether
additional erosion correction would be required. V1 rejects a result requiring
another unbounded terrain/hydrology loop.

Failure conditions:
- A changed physical trunk edge, bed, junction, reach identity, or connector
  process exclusion.
- A cycle, uncovered active child, dangling nonterminal receiver, or
  contributing-area conservation failure.
- Receiver-change area or count above the configured stability bounds.
- Process routing through a preserved-depression interior.
- Treating every priority-fill candidate as standing surface water.

## Decision 027: Refined Seasonal Surface-Water Balance

Status: implemented, provisional

Decision:
New local depression candidates from Hydrology Pass 2 receive a deterministic
monthly storage balance before any lake or wetland label is accepted. The
balance runs only on resolved ordinary child cells. Inherited coarse lakes and
preserved depressions remain excluded handoffs until their bathymetry is
refined explicitly.

Catchment contract:
- Parent monthly runoff and evaporation depths are copied to refined children;
  physical child areas therefore conserve each represented parent's water
  volume without inventing unsupported fine-scale climate.
- Every active source child belongs to its first downstream local candidate, if
  one exists. Direct candidate catchments are disjoint.
- Candidate spill receivers define an acyclic upstream-to-downstream candidate
  graph. Monthly overflow is transferred through that graph once; it is not
  counted as new runoff at the downstream candidate.
- Candidate-network direct inflow equals evaporation, seepage, terminal
  overflow, and storage change within floating-point tolerance.

Fractional-water contract:
- A refined child is still a container, not an atomic water label.
- Parent relief is scaled to unresolved child relief. A uniform subcell
  elevation distribution provides deterministic area and volume curves between
  the local bottom and the Pass-2 spill elevation.
- Because these candidates were unresolved at the parent scale, a configured
  connected-basin fraction caps how much low subcell terrain may belong to one
  local waterbody. Inherited resolved lakes are outside this pass.
- The monthly balance publishes storage, water area, overflow, evaporation,
  seepage, and fractional inundation for every participating candidate and
  child.

Classification contract:
- `dry_depression`: no material surface-water month.
- `transient_storage`: short-hydroperiod inundation, or a filled local candidate
  whose sustained overflow requires outlet-incision feedback before standing
  water can be accepted.
- `seasonal_lake`: deeper material inundation that is not present year-round.
- `permanent_lake`: material water area in every climatological month and mean
  depth above the wetland threshold.
- `hydrologic_wetland`: recurrent material inundation whose annual wetted mean
  depth remains below the wetland threshold.
- A hydrologic wetland is a surface-water and hydroperiod result, not a final
  ecological biome. Soil saturation, vegetation, and groundwater may revise it
  later.
- Sustained overflow, available head, weak rock, and low sediment accommodation
  produce an explicit outlet-erosion score. The pre-incision monthly state is
  retained for conservation auditing, while the accepted class is transient
  until a later bounded incision/reroute pass resolves the outlet.

Failure conditions:
- A cyclic candidate graph, overlapping direct catchments, unknown receiver or
  candidate identity, or process-excluded candidate support.
- Fractional inundation outside `[0, 1]`, storage above candidate capacity, or a
  non-finite monthly state.
- Candidate-network water imbalance above the configured tolerance.
- Treating provisional land evaporation as calibrated open-water potential
  evaporation.
