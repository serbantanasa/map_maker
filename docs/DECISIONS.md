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

## Decision 028: Bounded Subgrid Outlet Incision

Status: implemented, provisional

Decision:
Surface-water candidates that request outlet erosion receive one bounded
subgrid incision and local-routing correction before soils or biomes may run.
The correction follows the accepted Pass-2 receiver graph from each candidate's
final spill cell to the first downstream candidate, lower ordinary cell, fixed
trunk, preserved handoff, or terminal. It does not search globally for a more
convenient outlet.

Cascade contract:
- Candidate outlets are planned downstream-to-upstream. An upstream candidate
  may connect to the corrected level of a downstream candidate in the same
  bounded pass, which permits linked lake chains to drain coherently.
- The requested start level is the old spill level minus the published
  recommended incision. It is raised when necessary to retain a positive
  downstream bed slope into fixed or uncorrected support.
- A path that cannot reach valid downstream support within the configured cell
  bound is reported as blocked. It is not carved speculatively.
- A candidate whose requested bed cannot descend into its downstream water
  level is grade-limited, not an unresolved erosion failure. After that bounded
  feasibility proof it may retain its lake class without blocking soils.
- Shared path cells receive the lowest compatible requested bed and the widest
  contributing outlet, while their physical erosion volume is counted once.

Subgrid and terrain contract:
- Outlet beds are subgrid channel elevations. A 10-100 metre outlet does not
  lower an entire approximately five-kilometre child by its incision depth.
- Channel width is a bounded function of pre-incision mean overflow discharge.
  Eroded volume is incision depth times channel width times the child's
  representative linear length (`sqrt(area)`); exact channel geometry remains
  a regional-refinement responsibility.
- Cell-mean terrain feedback is eroded volume divided by child area. Parent
  aggregates must reconstruct the exact emitted child volume.
- Existing physical trunk beds and receivers remain fixed. Outlet paths may
  join them but may not rewrite them.
- Applied outlet-path cells remain ordinary terrain cells and persist their
  accepted downstream receiver as a separate subgrid constraint. They are not
  promoted to full-cell physical channels, but later priority-flood passes
  exclude the constrained bed support from depression membership.
- If local priority flooding would route an unconstrained neighbor back into a
  fixed outlet and form a cycle, only the cyclic ordinary support is restored
  to its previously accepted receiver. Repair rounds and total constrained
  support are bounded explicitly.

Post-correction contract:
- The Rust Hydrology Pass-2 kernel reruns over the corrected routing surface and
  must publish an acyclic, conservative receiver graph with valid process
  exclusions and unchanged physical trunks.
- Monthly surface-water balance then reruns over the corrected candidates,
  followed by further bounded incision/balance rounds up to the configured
  limit.
- Soils become ready only when that post-incision solve requests no further
  outlet correction. Grade-limited paths and paths whose persisted bed already
  satisfies the requested cut are retained as explicit accepted standing-water
  reasons across rounds. Other residual feedback remains a hard blocker.

Failure conditions:
- Unknown or cyclic source routing, a mismatched downstream candidate, or a
  correction through excluded/outside support.
- A path longer than the configured bound, excessive corrected area or receiver
  change, or a non-descending emitted outlet bed.
- Any mismatch among cell, parent, and total eroded volume.
- A changed physical trunk receiver or bed, invalid terminal, uncovered active
  cell, or contributing-area conservation failure after rerouting.

## Decision 029: Scale-Aware Hydrology KPIs And Cryosphere Boundary

Status: implemented, provisional

Decision:
Hydrology validation publishes a persistent KPI catalog after final outlet
incision. Every KPI identifies its spatial scope, unit, reference envelope,
comparison status, and whether it is a hard invariant, an Earth diagnostic, or
an implementation-capability marker. A global Earth statistic may inform a
single-basin result but may not hard-fail it without matching climate, relief,
geology, scale, and inventory threshold.

Hard invariants:
- Candidate and river graphs are acyclic, closed over known identifiers, and
  reach only allowed terminals.
- Every major river has a resolved source-to-terminal path.
- Mean and monthly reach discharge do not decrease downstream unless the loss
  is represented by an explicit water-storage or loss node.
- Candidate-network water balance and outlet-correction convergence remain
  hard gates.

Earth diagnostics:
- Global lake area, closed-drainage area, and runoff depth are compared with a
  versioned Earth reference profile.
- Selected-basin standing water, runoff, lake throughflow, and seasonality are
  reported without a global pass/fail interpretation.
- Wetlands and floodplains are definition-sensitive. Hydrologic wetness,
  ecological wetland, geomorphic floodplain, and monthly inundation remain
  separate products.

Cryosphere boundary:
- Seasonal snowfall, snow-water equivalent, melt, and melt-fed runoff are
  existing monthly climate products and receive explicit KPIs.
- Seasonal snow cover means the annual maximum snowpack exceeds a threshold;
  perennial snow means the annual minimum exceeds a threshold. These terms may
  not be interchanged.
- Persistent snow is not a glacier. Glacier ice requires a separate firn/ice
  reservoir, multi-year mass balance, downslope ice transport, and snow-versus-
  ice meltwater provenance.
- Terrain above the local climatic snowline with sustained positive
  accumulation must be able to retain a multi-year firn/ice reservoir. The
  reservoir gains mass primarily during its cold season and can release melt
  through the spring, summer, and fall without resetting to zero each year.
  Elevation alone is not sufficient: temperature, snowfall, sublimation, and
  exposure determine whether high terrain actually carries persistent ice.
- Seasonal snowmelt and glacier-ice melt remain separate hydrology inputs so
  long-lived ice can buffer river flow across dry or warm seasons.
- The V1 subsystem now supplies an age-tracked snow reservoir, firn conversion,
  separate ice storage and melt, and conservative parameterized downslope ice
  transfer. `glacier_mass_balance_implemented` is therefore one, while dynamic
  ice-stress flow and calibrated glacier runoff remain explicitly incomplete.

Known capability gates:
- Final terminal lake-network overflow and consumptive losses are coupled into
  monthly reach-entry and reach-exit hydrographs with a persistent adjustment
  audit catalog.
- Floodplain widths exist, but monthly inundation depth and duration do not.
- Hydrologic wetland candidates await groundwater, soil, and vegetation
  confirmation.

## Decision 030: Bounded V1 Lake Hydrographs And Cryosphere

Status: implemented, provisional calibration

Decision:
The V1 hydrology stack closes two causal gaps without attempting research-grade
limnology or glaciology.

Lake hydrographs:
- Final surface-water candidates form terminal lake networks. Their summed
  direct runoff is replaced downstream by the network's solved monthly terminal
  overflow, rather than added to inherited discharge and counted twice.
- Coupling preserves both pre-lake hydrographs and final reach-entry and
  reach-exit hydrographs. Every terminal-network/month adjustment records its
  nominal and effective reach.
- If coarse inherited discharge has not yet accumulated a fine tributary's
  water, a negative adjustment moves to the first downstream reach where it can
  be applied without negative flow. This scale-remapping event remains visible
  in metadata and may not silently clamp or discard water.
- Lake-network balance, nonnegative discharge, and accounted downstream losses
  remain hard validation gates.

Cryosphere:
- The atmosphere stage remains responsible for temperature, precipitation, and
  evaporation. A separate Rust cryosphere stage owns canonical seasonal snow,
  firn, glacier ice, glacier melt, and melt-inclusive runoff potential.
- Snow age is tracked as a reservoir age moment. Long-lived snow converts to
  firn/ice gradually; seasonal snowmelt and glacier-ice melt remain separate
  outputs and training targets.
- Coarse-cell unresolved relief defines an upper-terrain climate sample using a
  configurable relief multiplier and bounded highland fraction. It is not a
  fixed altitude or latitude glacier mask.
- Ice above a storage threshold transfers conservatively toward the lowest
  neighboring bedrock cell; transfer into ocean is calving. This is a
  parameterized V1 spreading rule, not stress-driven glacier dynamics.
- Ocean cells carry a separate thermodynamic sea-ice thickness reservoir.
  Configurable seawater freezing and melt thresholds drive seasonal growth and
  retreat; fractional concentration is derived from thickness. Sea ice does not
  enter land runoff or create atmospheric precipitation.
- Glaciers affect runoff immediately. Dynamic ice flow, ice sheets, glacial
  erosion, moraines, sea-ice drift and ridging, and calibrated equilibrium-line
  statistics remain later milestones.

Reason:
Lake regulation is necessary for correct river continuity. A bounded glacier
reservoir is necessary for persistent high-mountain ice and seasonal meltwater,
but a full ice-dynamics solver would exceed V1's structural-realism target.

## Decision 031: Fractional Surface Materials Before Functional Biomes

Status: implemented, provisional calibration

Decision:
The first post-hydrology land-surface milestone is a property-first L2 surface-
material and initial-soil model. Functional vegetation, familiar biome labels,
and the optional bounded soil-to-hydrology feedback remain later work.

Scale contract:
- Each L2 land cell stores mutually exclusive area fractions of exposed
  bedrock, residual regolith, colluvium, alluvium, lacustrine sediment, glacial
  deposit, and volcaniclastic material. Fractions sum to one on land and zero
  on ocean.
- A coarse cell is a map unit containing component mixtures, not one atomic
  soil polygon. L3 refinement will place those components deterministically
  along ridge, slope, footslope, floodplain, wetland, and closed-basin catenas.
- Fine selected-basin erosion, deposition, and final surface water restrict to
  L2 parent cells by physical area or volume. Unrefined parents retain the
  accepted global hydrology priors.

Property contract:
- Sand, silt, and clay are normalized fine-earth fractions. Coarse fragments
  remain a separate fraction.
- Regolith depth and pedogenic soil depth are separate. Soil depth may not
  exceed regolith depth.
- Initial properties include bulk density, potential organic carbon, pH,
  carbonate, salinity, drainage, available water capacity, nutrient and
  fertility potential, erodibility, reset age, and confidence.
- Province class is evidence for parent-material priors, not a direct soil-class
  lookup. Climate, relief, drainage, erosion, deposition, water persistence,
  and exposure age all modify the result.

Water contract:
- A Rust kernel partitions monthly rain and snow/glacier melt over non-open
  land into soil storage, actual evapotranspiration, quick runoff, and deep
  drainage after a bounded periodic spinup.
- Monthly fluxes and storage are L2-cell-equivalent water depths. A persisted
  soil-bearing fraction separates open water, active ice, and substantially
  exposed bedrock from soil-supporting area.
- Input equals evapotranspiration plus runoff plus deep drainage plus storage
  change within the configured tolerance.
- Hydric-soil fraction is physical saturation evidence. It is not yet an
  ecological wetland, swamp biome, or vegetation label.

V1 boundaries:
- No canonical soil taxonomy or biome is painted by this stage.
- No routed groundwater aquifer, lateral soil-water flow, permafrost physics,
  soil horizons, or explicit chemical reaction network is claimed.
- Potential organic carbon and fertility are pre-vegetation estimates and must
  remain distinct from their later vegetation-adjusted canonical values.
- Soil-derived runoff does not modify the accepted hydrology in this milestone.
  A later stage may apply at most the one bounded feedback allowed by Decision
  015, with a new validation pass.

Reason:
Surface materials close the causal path from geology, erosion, climate, and
hydrology into land properties needed by vegetation and the simulation game.
Fractional map units preserve subgrid floodplains and catenas without inventing
false L2 precision.

## Decision 032: Earth Calibration Profile And Open Environmental Envelope

Status: implemented, provisional calibration

Decision:
Earth is the V1 default calibration profile, not the permanent validity
boundary of the simulator. The artifact contracts must support later dense
rocky water-world experiments including snowball climates, continentless or
island-dominated worlds, ice-free hothouses, and high-CO2/high-productivity
scenarios without requiring a new architecture.

Validation contract:
- Hard failures enforce physical, numerical, topology, and conservation
  invariants.
- Earth-relative ranges and morphology statistics are versioned diagnostics
  attached to the `earthlike` profile.
- Named non-Earth profiles select diagnostic expectations and scenario tests;
  they do not alter conservation laws or silently clamp state into Earth
  ranges.
- A scenario outside the calibrated envelope must identify its missing physics
  and reduced confidence rather than being called realistic by configuration
  alone.

State contract:
- Atmospheric pressure and composition, light, temperature, liquid-water
  opportunity, oxygen support, carbon substrate, nutrient/surface support, and
  stress remain inspectable fields.
- Canonical biosphere state is continuous and trait-first. Earth plant
  functional types and biome names are derived products.
- High CO2 alone is not a proxy for biological productivity or organism size.
  Energy, oxygen, pressure, gravity, temperature, water, nutrients, and
  organism physiology remain separate constraints.
- Cross-scenario neural surrogates must receive the planetary/environmental
  conditioning vector. Training only on Earth defaults cannot produce a
  general planetary surrogate.

Implementation boundary:
- Milestone 15b0 adds a pre-climate atmosphere contract and a post-soil raw
  biosphere-resource envelope.
- Current Earthlike planet bounds, climate-temperature bounds, atomic coastal
  cells, fixed albedo, incomplete sea/land ice, and absent non-Earth calibration
  remain explicit capability gaps.
- Generalizing those systems is staged work; this decision prevents new stages
  from making the current limitations permanent.

Reason:
Earth calibration gives V1 measurable targets, while raw causal state keeps the
world stack useful for future climate and biosphere regimes. Separating hard
physics gates from profile diagnostics prevents both false generality and an
unnecessary Earth-only rewrite later.

## Decision 033: Trait-First Potential Biosphere Before Functional Types

Status: implemented, provisional calibration

Decision:
Milestone 15b1 predicts continuous terrestrial producer-community potentials
before assigning plant functional types or biome labels.

Interpretation contract:
- Outputs describe equilibrium potential under an explicit carbon-based,
  photosynthetic-colonization assumption. They are not actual vegetation or an
  evolutionary-history simulation.
- Environmental adaptation pressures remain separate from predicted community
  traits. Similar pressure does not guarantee convergent organisms.
- Potential NPP is auditably bounded by the 15b0 chemical-energy artifact through
  a configured energy-per-carbon conversion.
- Cover, biomass, woody allocation, resource-conservation strategy, rooting,
  canopy, leaf area, and fuel continuity remain continuous fields.
- Species, PFT fractions, biome labels, consumers, disturbance events, and
  vegetation feedback are not emitted by 15b1.

Validation contract:
- Hard gates enforce energy bounds, monthly/annual aggregation, normalized
  traits and pressures, regolith-bounded roots, configured morphology bounds,
  finite state, and zero terrestrial production over ocean.
- Earth trait databases and land-model relationships feed the separate
  `earth_biosphere_v1` calibration profile; they are not categorical lookup
  tables or generation-time physical gates.
- Non-Earth validation profiles reduce confidence where calibration is absent;
  they do not clamp physical inputs or force Earth vegetation.

Reason:
The game needs productivity, biomass, cover, and terrain-scale ecological
constraints before it needs named biomes. Continuous, inspectable potentials
also preserve useful training targets and leave room for later Earth and
non-Earth functional models without rerunning the physical world stack.

## Decision 034: Earth Biosphere Profile Before Functional Types

Status: implemented and passing at both screening scales

Decision:
Establish the versioned `earth_biosphere_v1` profile before tuning or
implementing 15b2 functional vegetation mixtures and biome labels.

Validation contract:
- Per-world hard gates cover finite state, exclusive climate-stratum area, and
  reconstruction of global NPP and biomass from area-weighted strata.
- Earth comparison uses physical global totals. The initial potential-natural
  targets are `50-75 Pg C/year` NPP and `771-1,107 Pg C` vegetation biomass.
- Cover, canopy, LAI, rooting, and latent trait state are distributional
  outputs until an accepted reference transformation gives them comparable
  semantics; they may not receive arbitrary one-number targets.
- Exclusive polar, cold, cool-dry, cool-moist, warm-dry, warm-seasonal, and
  warm-humid strata are derived only from upstream climate. They are validation
  bins, not canonical biomes.
- Fixed-seed evaluation reports hard execution, Earth agreement, and ensemble
  dispersion separately. Failed worlds remain failed observations and may not
  be removed from the seed set to improve the score.

Scale contract:
- Face-64 fixed seeds are the coarse screening ensemble.
- The canonical face-128 seed confirms high-resolution behavior.
- A model change is not calibrated from one scale or one seed alone.

Scenario contract:
- Earth ranges are calibration diagnostics for the `earthlike` profile, not
  physical clamps and not validity rules for snowball, hothouse, archipelago,
  or future non-Earth biospheres.
- Ordinary generation can complete outside Earth ranges while preserving a
  visible `outside_reference` status. The dedicated Earth validation command
  passes only when hard gates, ensemble tolerances, and Earth diagnostics pass.

Reason:
15b2 labels could make an underproductive or seed-fragile biosphere look
finished while preserving the wrong underlying carbon cycle. A source-backed,
multi-seed profile makes calibration failures explicit before categorical
vegetation is added and creates stable training-data acceptance metadata for
future surrogate models.

## Decision 035: Convergent Fine Routing And Branch-Local Lake Losses

Status: implemented

Decision:
Refined reaches may share an already occupied directed fine edge when they
continue in the same downstream direction. They may not reuse the reverse edge,
cross an occupied edge, or leave a shared node through different downstream
targets. Shared directed edges represent unresolved co-routing or parallel
subgrid channels inside an approximately 10 km fine cell; they are not evidence
that the modeled rivers have cell-scale width.

Hydrology contract:
- Refinement persists directed-edge multiplicity and separates all shared edges
  from shared physical-channel edges. Fluvial erosion deduplicates the shared
  bed edge while retaining each reach's subgrid erosion width and discharge.
- Hydrology Pass 2 assigns one spill-surface level to each depression ID. Cells
  at different filled levels cannot be merged merely because they are adjacent.
- A negative terminal-lake hydrograph adjustment remains on its own downstream
  branch and is bounded by discharge physically represented there. Any
  unprojectable part is persisted as `pre_channel_interception_km3`; it may not
  be silently discarded or consume water from a sibling tributary.
- Monthly storage-aware downstream regressions remain hard-gated. Raw
  annual-mean regressions are diagnostic because legitimate lake storage can
  reduce mean flow.

This decision supersedes Decision 030's provisional rule that moved a negative
lake adjustment to the first downstream reach with sufficient inherited flow.
That remapping could incorrectly debit a sibling branch after confluence.

Validation:
All six fixed face-64 seeds and the canonical face-128 seed now reach final
hydrology with zero unaccounted monthly reach losses. Directed reuse is audited,
reverse-edge conflicts remain forbidden, lake-network balance closes, and
projected discharge remains nonnegative.

Reason:
At coarse resolution, distinct tributaries can be narrower than one fine cell
and cannot always be embedded as disjoint raster paths. Directed convergence
preserves the drainage DAG without inventing crossings. Branch-local bounded
lake losses preserve causal ownership when inherited coarse discharge does not
resolve every fine tributary.

## Decision 036: Calibrate Carbon Amplitude Without Hiding Climate Errors

Status: implemented across the coarse ensemble and canonical resolution

Decision:
The Earthlike 15b1 amplitude calibration retains `39.9 MJ/kg C` as the chemical
energy-per-carbon conversion. Nutrient support uses a normalized saturating
response

`response = (1 + K) * support / (support + K)`

with `K = 0.5`, so zero and full support remain fixed at zero and one while
moderate nutrient supply is not treated as a linearly proportional carbon
ceiling. The upstream peak PAR-to-chemical efficiency is `0.0421`, and cover
uses a `0.30 kg C/m2/year` NPP half-saturation response. Biomass uses
a bounded residence-time response rather than a single multiplier:

- residence bounds are `0.5-45 years`;
- the normalized response has a `0.10` baseline;
- woody and resource-conservative structure weights are `0.60` and `0.40`;
- low productivity has weight `2.50`, representing slower turnover under
  resource-limited conditions;
- standing biomass remains capped at `40 kg C/m2`.

The six-seed face-64 screen completes every world. NPP is `62.56-75.98 Pg
C/year`, biomass is `977.48-1,089.41 Pg C`, NPP coefficient of variation is
`0.071`, and biomass coefficient of variation is `0.040`. Five of six global
NPP diagnostics and all global biomass diagnostics pass, satisfying the
predeclared `80%` requirement. Every land-mean diagnostic and ensemble-
dispersion gate passes. The configured `35.0%` landmass is inside the approved
`18-36%` generated-Earthlike land band.

The canonical face-128 world produces `57.26 Pg C/year` NPP, `0.321 kg
C/m2/year` land-mean NPP, `932.65 Pg C` biomass, and `5.22 kg C/m2` land-mean
biomass. It therefore passes all four carbon-amplitude diagnostics. Functional
types and biome labels may build on these continuous outputs without treating
their Earth calibration as a universal biological law.

Reason:
Calibration must preserve causal information. A biosphere scalar that forces a
dry generated climate to reproduce Earth totals would make the final map look
plausible while corrupting both simulation state and future surrogate-training
targets.

## Decision 037: Resolution-Aware Climate Transport Controls

Status: implemented

Decision:
Climate transport controls are defined relative to face resolution 128 and
converted to numerical work at the active cubed-sphere resolution:

- physical transport steps scale linearly with face resolution;
- moisture-diffusion substeps scale linearly, while advection per numerical
  substep is divided by the substep count;
- supersaturation removal and maximum condensation are converted to equivalent
  per-step fractions that preserve the same monthly relaxation; and
- synoptic smoothing passes scale with the square of face resolution so their
  physical mixing length remains approximately stable.

These conversions are persisted in climate metadata. They are numerical-
resolution controls, not Earth-profile productivity multipliers.

Validation:
For seed 42, face-64 and face-128 land precipitation are `505.39` and `497.92
mm/year`, respectively, after the correction. Face 64 uses eight transport
steps, one diffusion substep, `0.38` advection, and four synoptic passes. Face
128 uses 16 transport steps, two diffusion substeps, 32 numerical moisture
steps, `0.19` advection, and 16 synoptic passes. The resulting carbon state
passes Decision 036 at both resolutions.

Reason:
Applying per-cell and per-timestep coefficients unchanged at multiple grid
resolutions changes physical transport distance and monthly condensation. That
numerical artifact made the canonical world appear substantially drier and
could not be repaired honestly in the biosphere stage.

## Decision 038: Earthlike Emerged Land Band Is 18 To 36 Percent

Status: approved and implemented

Decision:
`earth_biosphere_v1` and provisional morphology validation accept generated
Earthlike emerged-land fractions from a **minimum of `18%` to a maximum of
`36%`**. Earth's observed approximately `29%` emerged-land fraction remains the
reference point inside that band. A configured ocean-area target (for example
the current canonical `65%` ocean → `35%` land) is only one legal setpoint; it
is not a required exclusive land fraction.

This decision is a profile and validation band. It does not force every world
to the same land share. Non-Earth scenarios may leave the band when they use a
different validation profile.

Validation:
All six face-64 worlds and the canonical face-128 world currently land at
`35.0%` under the canonical ocean target, which remains inside `18-36%`.

Reason:
Useful Earthlike game worlds may be somewhat wetter or drier than modern Earth,
but not arbitrarily land-dominated or nearly ocean-only. The user-specified
acceptance band is `18%` minimum and `36%` maximum emerged land.

## Decision 039: Functional Vegetation Is A Conservative Cover Mixture

Status: implemented; calibrated by Decision 041

Decision:
Milestone 15b2a stores eight continuous producer-community strategies rather
than a painted biome class: cold-adapted woody, warm evergreen woody, seasonal
woody, xeric shrub, cool-season herbaceous, warm-season herbaceous,
hydrophytic, and low-stature resource-conservative vegetation.

Each strategy value is a fraction of total coarse-cell area. The strategy sum
equals functional vegetation cover. Bare ground, saline barren, persistent
ice, inland open water, and unsupported surface close the remaining land-cell
partition exactly. Ocean remains outside the terrestrial partition.

Fire, grazing, forest-resource, pasture, and crop outputs are bounded physical
potentials. Pasture and crop potential do not place land use or imply
domestication. A dominant functional-cover code is a derived query and
rendering artifact; familiar biome names remain deferred until mixture
distributions have independent validation.

Reason:
The game needs compositional land cover and resource suitability, but a single
biome code would discard ecotones, subgrid mixtures, and causal trait state.
An exact partition is inspectable, conserves physical area, supports later L3
refinement, and produces better surrogate-training targets than categorical
painting.

Validation:
The canonical face-128 seed-42 world closes every land-cell partition within
`4.00e-8`, emits no terrestrial state over ocean, reproduces its dominant-cover
codes independently, and passes deterministic native-kernel tests. Its land is
`47.52%` functionally vegetated. All five resource potentials are finite,
bounded, spatially nondegenerate, and covered by the Decision 041 structural
calibration contract.

The dominant code compares aggregate vegetation against each individual
nonvegetated class before selecting the leading functional strategy. Directly
comparing all thirteen leaf classes biased mixed vegetated cells toward bare
ground because vegetation is intentionally split into eight strategies.

## Decision 040: Refined Surface State Uses Physical Child Area

Status: implemented

Decision:
Fine selected-basin lake, wetland, erosion, and deposition state is restricted
to L2 parents using the summed `area_km2` of persisted child cells. Cubed-sphere
`CellArea` is a solid angle in steradians and must never be divided directly
into a physical area or volume.

The surface-material stage persists `EffectiveLakeFraction`,
`EffectiveWetlandFraction`, `EffectiveSurfaceWaterHydroperiod`, and
`RefinedSurfaceWaterMask`. Soils and all downstream biosphere stages consume
the same coarse/refined water selection. Parent lake/wetland area and
erosion/deposition volume must reconstruct within `1e-6` relative error.

Validation:
The corrected canonical world has `82.58%` soil-bearing land, `2.859%`
effective inland open water, and `14.56%` unsupported or effectively soil-free
surface. The previously reported `17.60%` unsupported fraction was invalid:
roughly 3.5 percentage points were refined water/area projection error rather
than physical barren terrain. The corrected canonical carbon profile and the
six-seed ensemble both pass without widening any Earth reference range.

Reason:
The prior projection divided fine water area in square kilometres by parent
solid angle in steradians. Clipping hid the dimensional error by turning many
selected-basin parents into apparent 100% water cells; soils then removed their
support and functional vegetation mislabeled the missing area as unsupported.
Persisting one conservative effective surface state prevents this class of
cross-stage mismatch.

## Decision 041: Calibrate Functional Cover Before Naming Biomes

Status: implemented and passing

Decision:
Milestone 15b2b defines `earth_functional_vegetation_v1` as the acceptance
profile for functional cover. Functional scoring first preserves the upstream
woody-allocation budget, reserves hydrophytic cover from wet nonwoody support,
and only then uses climate response to divide those budgets among strategies.
This prevents normalization across eight independent scores from erasing the
causal trait state.

The profile gates broad global potential-natural cover ranges, absolute and
directional climate-stratum behavior, resource-potential dynamic range, and a
fixed six-seed ensemble. Modern cropland, pasture, deforestation, ignition,
and fire suppression are explicitly excluded as calibration targets. Resource
potentials describe physical suitability rather than realized human use.

Validation:
The canonical face-128 seed-42 world has `47.52%` functional vegetation,
including `15.71%` woody, `17.86%` herbaceous, `12.70%` xeric plus low-stature,
and `1.25%` hydrophytic cover over land. Cool-moist woody cover is `23.0%`,
where the pre-calibration scorer produced only `7.4%`; warm-humid hydrophytic
cover is `10.3%`, down from an implausibly broad `28.1%`. Every canonical hard
and Earth-profile gate passes.

All six face-64 worlds pass. Global functional vegetation spans
`50.01-54.54%`, woody cover `17.29-21.58%`, herbaceous cover `17.96-19.04%`,
xeric plus low-stature cover `12.12-12.78%`, and hydrophytic cover
`1.45-2.08%`. Their coefficients of variation are `0.030`, `0.079`, `0.019`,
`0.018`, and `0.118`. Every climate relationship and resource-amplitude gate
passes in all six seeds.

Reason:
Named biomes are easy to make visually persuasive even when the underlying
ecology is wrong. Calibrating conserved mixtures and causal climate responses
first gives later biome labels an auditable basis and preserves better state
for regional refinement and surrogate training.

## Decision 042: Continental Crust Is Not The Shoreline

Status: approved and implemented

Decision:
Replace the six-focus continental-crust score and the downstream use of
`BaseOceanMask` as literal water. Continental crust is assembled from multiple
curved terrane chains, accreted branches, rift corridors, interior basins,
microcontinents, and oceanic separators. Compact-support fields prevent weak
Gaussian tails from joining unrelated assemblies. This crust state remains
geological substrate, not a finished map.

After elevation, a dedicated Rust sea-level stage solves the largest connected
low-elevation component to the configured ocean target. Other low components
remain inland depressions. The stage publishes a discrete connected-ocean mask
for topology, an area-conservative fractional coast for coarse physical state,
signed elevation relative to the solved datum, ocean depth, shelf fraction,
coastal cells, and inland-below-sea-level evidence.

For the canonical Earthlike profile, continental-crust candidate area is
`42%` and the default ocean-area target is `65%` (emerged land `35%`). Crust
share and land share are independent controls. Emerged land must stay inside
the Decision 038 band of `18-36%`; the `35%` setpoint is one legal value in
that band, not a fixed product requirement. `BaseOceanMask` retains only its
documented oceanic-crust-candidate semantics; surface stages must consume the
new sea-level artifacts.

Validation:
The face-128 seed-42 surface has `35.0%` emerged land, ten significant
landmasses, a largest-landmass share of `44.6%`, `4,948` directed land-to-ocean
coast edges, `3.96%` shelf area, and `0.95%` inland below-sea-level area. The
fixed six-seed face-64 screen closes fractional ocean area at `65.0%` in every
world. Largest-landmass share spans `39.4-67.9%`, significant landmasses span
`8-16`, and coastline edges span `2,064-2,960`.

The truth-render screen includes split continents, rifted supercontinents,
archipelago-heavy worlds, deep gulfs, long narrow seas, peninsulas, islands,
and enclosed basins. Occasional supercontinent-like outcomes remain valid, but
six similarly sized round blobs and one accidental almost-global land weld are
hard failures.

The largest-landmass coastline-complexity metric compares its approximate
coast length with the minimum perimeter of a same-area spherical cap. The
canonical face-128 value is `4.43`; the six face-64 Earth-profile seeds span
`3.12-6.27`. The regression floor is `2.0`, well above the approximately
round, grid-sampled baseline.

Reason:
The prior model used six orthogonal crust foci and selected the top area-ranked
cells. It therefore generated six compact blobs by construction. Giving
continental and oceanic crust a multi-kilometre elevation gap, then targeting
the ocean fraction equal to oceanic-crust area, forced sea level into that gap
and made every coastline identical to the crust boundary. No downstream
climate, erosion, hydrology, or biome work could repair that topology.

## Decision 043: Recalibrate After Surface-Geography Repair

Status: approved and implemented

Decision:
Treat the Decision 042 geography correction as a changed physical model, not
as a render-only adjustment. Rebuild native libraries explicitly, invalidate
all downstream stage caches, replay the fixed ensemble, and recalibrate only
named process parameters whose old values depended on the invalid shoreline.
Decision 041's figures remain the historical pre-repair baseline and are
superseded by this decision for current runs.

The Earthlike peak PAR-to-chemical conversion efficiency is `0.0295`. The
subgrid connected-basin fraction is `0.85`, preserving open overflow while
allowing unresolved depressions to retain realistic fractional water area.
Climate-stratum area absolute range is allowed up to `0.22`; inland open water
has a dedicated coefficient-of-variation ceiling of `0.75` because sparse
closed basins are intrinsically more seed-sensitive than vegetation fractions.
No generated field is clamped to an Earth observation range.

`SurfaceOceanFraction` is physically active rather than merely persisted:
coastal albedo, thermal response, initial atmospheric moisture, evaporation,
and land/ocean condensation fluxes are area-weighted. Binary
`SurfaceOceanMask` remains authoritative only where connected topology or one
explicit surface class is required.

Validation:
The canonical face-128 seed passes every hard and Earth diagnostic with
`50.28 Pg C/year` potential NPP, `844.54 Pg C` potential biomass, `44.35%`
functional vegetation, and `3.17%` inland open water over land. All six
face-64 worlds pass both Earth profiles and ensemble tolerances. Their NPP
spans `54.78-73.02 Pg C/year`, biomass `888.33-1,061.11 Pg C`, functional
vegetation `47.27-56.45%`, and inland open water `0.93-3.32%`.

Reason:
Replacing compact crust blobs with separated terranes, rifts, shelves, and a
connected ocean increased coastal exposure and changed precipitation,
hydrology, and productivity. Retaining the old calibration would have hidden
that causal change and made cache provenance misleading.

## Decision 044: Familiar Biomes Are Derived Conserved Mixtures

Status: approved and implemented

Decision:
Derive familiar game-facing biomes only after climate, soils, potential
biosphere, and functional vegetation. Persist 13 nonnegative full-cell biome
fractions rather than painting one categorical label per coarse cell. Inland
open water and persistent ice remain separate physical landscape classes;
ocean remains outside the terrestrial partition. Primary, secondary, and
dominant-landscape codes are reproducible query and rendering views, not new
canonical physical state.

This supersedes only Decision 039's temporary deferral of familiar biome names;
its functional-cover ownership and conservation rules remain in force.

The V1 taxonomy is tropical rainforest, tropical seasonal forest, savanna, hot
desert, xeric shrubland, temperate forest, temperate grassland, steppe, boreal
forest, tundra, cold desert, alpine, and wetland. It is deliberately broad.
Finer ecoregions may later derive from these mixtures and physical context
without replacing upstream state.

`earth_biomes_v1` validates partition closure, code integrity, broad global
abundance, and directional climate, highland, and wet-support relationships.
The fixed six-seed profile retains the established 80% per-diagnostic rule and
adds global and climate-zone dispersion gates. A generated world is never
clamped into an Earth diagnostic range. This stage has no vegetation feedback;
feedback belongs to the next bounded process pass.

Validation:
The canonical face-128 seed passes all 36 checks. Its land mixture is `26.63%`
forest, `5.41%` warm open, `32.72%` temperate open, `21.49%` core dryland,
`4.67%` tundra, `2.24%` alpine, and `3.67%` wetland, plus `3.17%` inland open
water. Every one of the 13 biome classes is a nontrivial dominant class.

The fixed six-world face-64 ensemble passes the profile and stability gates;
every diagnostic passes in at least five seeds and every biome remains
nontrivially represented in every world. Four seeds pass every individual
diagnostic; seeds 101 and 404 each miss one soft Earth diagnostic. Automated
passage does not replace visual acceptance: the ensemble command writes
`biome_gallery.png`, and human gallery review remains a required gate against
plausible statistics over visibly implausible maps.

Reason:
A single coarse categorical biome would erase ecotones and misrepresent cells
that span tens of kilometres. Conserved mixtures preserve subgrid diversity,
support later 100 m refinement, and provide richer surrogate-training targets,
while familiar labels give the game and map renderer an ergonomic vocabulary.

## Decision 045: Freeze Late Simulation Work Until Global Map Export

Status: approved, provisional

Decision:
Freeze new simulation features after the implemented derived-biome stage until
the first global map-export milestone is accepted. Bounded vegetation feedback,
mineral and energy systems, regional L3 refinement, additional hydrology work,
and further biome calibration are bugfix-only during this interval.

Allowed work is deliberately narrow:
- canonical surface-geography morphology and multi-seed acceptance;
- tectonically legible elevation and orogeny;
- shelf, passive-margin, and bathymetric morphology;
- truth rendering, projection, physical atlas composition, and export tooling;
- regressions, broken invariants, artifact-contract corrections, and performance
  defects that block those products.

The existing stage order remains the causal order. A renderer may consume the
current immutable artifacts before later physical stages are implemented; this
does not move map export into the simulation DAG.

Thaw gate:
1. The canonical six-seed `surface_geography_global` gallery is generated by a
   reproducible command and receives an explicit human accept/reject review.
2. One canonical projected physical world map meets the applicable Decision 008
   and Decision 018 visual bars.
3. Mountain belts, continental shelves, passive margins, islands, straits, and
   inland seas remain legible at the chosen world-map scale.
4. Truth and atlas outputs remain separate and their style/projection versions
   do not invalidate canonical simulation state.

Implementation:
`validate-biosphere` now writes `surface_geography_gallery.png` from the exact
hypsometric inputs used by each seed's `surface_geography_global` diagnostic,
alongside the biome gallery and numerical report. The older `map-maker validate`
gallery remains a framework/legacy-path test and is not evidence about canonical
cubed-sphere geography.

Reason:
The pipeline already contains enough late-stage breadth to produce a useful
world, but it does not yet produce the beautiful physical map required by V1.
Freezing feature growth makes visual product quality the controlling milestone
and prevents more ecological or hydrologic detail from hiding unresolved
geography, relief, margin, or cartographic failures.

## Decision 046: Continental Aggregation Is Scenario-Conditioned

Status: approved, provisional

Decision:
Allow supercontinent-like outcomes in the Earthlike scenario as a rare tail,
but make them substantially more likely under an explicit Pangean scenario.
Do not impose one universal largest-landmass cap. Named scenario profiles own
distributions over landmass concentration, significant landmass count, rift and
inland-sea expression, archipelago share, and related morphology diagnostics.

The default Earthlike profile should favor several major irregular landmasses
while retaining occasional supercontinent, archipelago-heavy, and highly rifted
worlds. A Pangean profile should deliberately bias crust assembly and surface
connectivity toward one dominant mass rather than accepting accidental seeds
after generation. Scenario choice must affect causal generation controls, not
only post-generation filtering.

Continental aggregation and total emerged-land area are separate profile
dimensions. Earthlike scenarios accept `18-36%` emerged land but do not use one
fixed setpoint. Each scenario may configure or sample its own ocean-area target;
non-Earthlike profiles may generate outside the Earthlike band. The generator
must not clamp all worlds to `18-36%` or to the current default `35%` land.

Surface connectivity is evaluated on the spherical topology at the scenario's
sea level. Glacial lowstands may expose shelf bridges and merge some continental
regions without implying that every shelf-separated continent joins one global
landmass. Persistent deep-water straits remain separators.

Current disposition:
Seed 202, with `67.85%` of emerged land in its largest connected component, is
accepted as a physically valid rare Earthlike tail and retained as a
supercontinent regression fixture. Its appearance in one six-seed gallery is
not a calibrated probability. If a larger Earthlike ensemble produces similar
concentration this often, the Earthlike prior must be adjusted; the Pangean
profile should target it positively.

Reason:
Supercontinents are geologically plausible phases, so a hard global ban would
discard valid worlds. Treating them as equally likely in every scenario would
also make an Earthlike request unreliable. Profile-conditioned generation gives
the user control while preserving natural variation and useful edge cases.

## Decision 047: Breach Erosion Does Not Imply Complete Lake Drainage

Status: implemented, provisional

Decision:
Treat outlet breaching and basin drainage as separate outcomes. An accepted
breach erodes its bounded outlet path, after which a second priority flood
defines the post-breach hydraulic control surface. If that surface still stands
at least one configured minimum-depression depth above the basin sink and
retains connected hypsometric area, the depression becomes a smaller open lake
or inland sea. Its breach remains in `BreachCatalog`, while its residual water
is registered as ordinary process-excluded open-water support. Only a basin
without resolved residual head remains a fully drained breached depression.

The Earthlike basin-erosion profile also rejects a prospective trunk profile
deeper than `2,200 m`. This is a diagnostic stop for a broken upstream waterbody
or graph boundary, not permission to apply that incision to coarse raster
terrain and not a universal planetary constant.

Reason:
The earlier model equated an accepted breach with removal of the entire water
body. In seed 303 it therefore treated a roughly Mediterranean-sized,
`4.46 km`-deep inland basin as dry after only `800 m` of bounded outlet erosion.
The fluvial grade solver inherited the basin floor as a river node and propagated
a roughly `-5.45 km` bed through ordinary downstream terrain, producing
`5.75 km` of incision and about `91,000 km3` of bank erosion. Re-flooding after
the breach preserves the physically unavoidable inland sea and keeps later
channel processes out of its abyssal floor.

## Decision 048: Terrain Resolution Owns Physical River Morphology

Status: approved; global and sparse safeguards implemented, local refinement pending

Decision:
Treat the global and sparse-basin hydrology products as multiscale constraints,
not as a raster landscape-evolution solve. At canonical face-128 resolution an
Earth-sized global cell averages about `5,190 km2` and is roughly `72 km`
across. Factor-16 sparse refinement reduces this to about `20.3 km2` and
`4.5 km` across. That is sufficient to preserve an inherited major-trunk graph,
but not to discover or physically excavate county-scale rivers and valleys.

This decision supersedes Decision 025's sparse cell-mean terrain feedback and
Decision 026's use of candidate channel beds as routing-raster elevations. It
narrows Decision 023's provisional permission for 2-5 km valley realization:
that scale may carry fractional support and major-trunk constraints, but the V1
pipeline does not yet have the process history or lateral terrain resolution to
claim evolved valley morphology there. It also supersedes Decision 028's
indefinite soil blocker once the bounded outlet correction exhausts its coarse
resolution budget; an explicit regional-refinement handoff is then required.

The River Thames is the concrete regression example. Its roughly `350 km`
length spans only about five face-128 cell widths, while its roughly
`13,000 km2` basin occupies only about two and a half average global cells. It
also falls far below the canonical `200,000 km2` global river-area threshold.
The global product therefore must not pretend to resolve the Thames, its
tributaries, floodplain, banks, or incision history.

Resolution ownership is:

- global L2 hydrology owns major drainage basins, lake and ocean terminals,
  monthly runoff and discharge, and major trunk topology;
- sparse factor-16 refinement realizes inherited trunk vectors and routing
  constraints, but its candidate bed and sediment volumes remain prospective;
- regional refinement discovers smaller tributaries and owns physical valley,
  bank, floodplain, and terrain incision, initially at roughly `100-250 m`;
- channels narrower than the active terrain cell remain vectors with physical
  width, depth, velocity, discharge, and cross-section attributes.

The representation contract applies at every resolution:

- the river vector spine and reach graph remain canonical, including for large
  rivers whose water surface is visible in a raster;
- a typical roughly `30 m` channel remains subcell on a `100-250 m` terrain
  grid, so raster layers store fractional channel-water, floodplain, wetland,
  and valley effects rather than categorical river cells or whole-cell channel
  excavation;
- physical width is a process attribute. Cartographic stroke width is a
  separate view- and scale-dependent style attribute and never feeds back into
  physical state;
- deterministic finer local refinement is created only where banks, meanders,
  crossings, or other lateral morphology must be resolved. Its cell size is
  chosen from physical channel width so the channel spans multiple cells, with
  stable refinement bounds, seeding, and tie-breaking.

The current global and sparse stages implement the persistent vector graph,
fractional support, and no-whole-cell-excavation safeguards. Deterministic local
bank and meander realization remains a regional-refinement requirement. A finer
raster may sample or derive support from the vector spine; it never replaces
the spine or reach graph as canonical river identity.

Consequently, the sparse fluvial stage must publish separate prospective and
applied process fields. Candidate channel excavation and floodplain deposition
budgets may be routed and conserved, but applied erosion, applied deposition,
cell-mean terrain change, and parent-restricted terrain change are exactly zero.
Bank carving is disabled. Hydrology Pass 2 preserves accepted trunk receivers
over the unchanged terrain prior instead of substituting the candidate vector
bed as raster elevation. Surface materials consume only explicitly applied
process volumes, never prospective budgets.

The existing bounded lake-outlet spill correction is a separate hydraulic
topology repair. It may apply its independently capped, volume-accounted outlet
lowering, but it must not become a back door for coarse trunk or valley erosion.
When its configured correction rounds end with a moving spill edge, remaining
candidates are persisted as `regional_refinement_deferred` standing water with
their exact spill IDs and area. The coarse pipeline must stop carving rather
than increasing its iteration cap until a narrow channel happens to migrate
across kilometre-scale cells.

Reason:
A conserved volume can still be attached to the wrong spatial support. Dividing
a 10-100 m channel prism across a `20 km2` child cell creates a numerically neat
but physically false terrain mean, and feeding that mean into hydrology and
soils compounds the error. Persisting vector constraints and prospective
budgets preserves useful causal information for later regional generation and
surrogate training without claiming morphology the current grid cannot carry.

## Decision 049: Fractional Water Does Not Break River Identity

Status: implemented, provisional atlas acceptance

Decision:
Keep fractional standing-water occupancy, physical channel support, hydraulic
connectivity, and cartographic continuity as separate fields and decisions.

- Any positive lake or wetland fraction remains persisted as subgrid water.
- A coarse parent is process-excluded only when standing water occupies at
  least half of it. Smaller fractions do not erase river identity, physical
  vector support, or parent terrain.
- A registered lake/spill control crossing, the routed edge exiting a
  contiguous unresolved-fill region, or an edge requiring an unresolved uphill
  hydraulic step of at least one minimum depression depth is a zero-width
  hydraulic connector. It carries reach identity, discharge, and
  source-to-terminal topology but no physical bed, incision, or sediment
  process at this scale. The rest of the depression path is not categorically
  erased.
- Sparse refined terrain retains its parent-mean elevation. A vector crossing a
  priority-flood depression gets a separate `channel_surface_prior_m` from its
  filled hydraulic control surface; a registered lake surface overrides that
  prior where available. On a physical reach entering that control, the same
  surface extends upstream only across child samples that fall below the
  receiving surface, stopping at the first emerged sample. This backwater rule
  prevents refinement interpolation from inserting a false kilometre-deep
  shoreline ramp. It includes locally dry coarse members where hydraulically
  required, prevents buried basin floors from leaking into the bed profile,
  and never changes raster terrain.
- The atlas begins with major physical channels, completes their upstream and
  downstream graph paths, and draws intervening connector geometry. Connector
  stroke width is cartographic and never becomes physical channel width.
- Fractional lakes contribute area-proportional color mass in the atlas. The
  visual must not paint every supporting coarse cell as a full lake.

Validation reports hydrologic and physical source-to-terminal lengths
separately, their ratio, and basin counts above `3000`, `4000`, and `5000 km`.
The Earthlike longest-path diagnostic is `4000-8000 km`; it is a profile check,
not a universal constraint on archipelago, water, or other scenario classes.

Reason:
Treating every fractional lake cell as wholly non-river made major channels
appear to die at the first depression. Treating every such cell as a physical
channel created the opposite error: priority-flood routes inherited kilometre-
deep basin floors and demanded multi-kilometre prospective incision through
ordinary downstream terrain. The split representation preserves long rivers
without pretending that coarse terrain resolves their banks or lake passages.

## Decision 050: L2 Regional Handoff Is A Conditional Package

Status: implemented

Decision:
Build the first literal L2 product as a selected regional handoff package, not
as a globally dense replay and not as an independent world generation.

The canonical face-128 globe is the roughly `72 km` L0 parent state. Existing
stage names and older prose that call those cells L2 describe semantic map-unit
fields, not literal spatial resolution. A factor-16 child realization has a
face-2048-equivalent cell area of roughly `20 km2` and width of roughly `4.5
km`; that is the first literal L2 terrain support.

The package contract is:

- select one complete drainage basin by stable `BasinID` and include a
  configurable number of L0 neighbor rings as boundary context;
- realize every selected parent into deterministic, parent-mean-conserving L2
  child terrain using the established refinement kernel;
- place ocean, lake, and wetland occupancy among child cells by terrain rank,
  conserving each source parent's fractional areas and never exceeding unit
  surface occupancy;
- persist all other grid-shaped tectonic, geological, climate, hydrologic,
  material, soil, biosphere, and biome values as parent priors joined through
  stable parent IDs; do not relabel repeated parent values as downscaled L2
  physics;
- preserve inherited river identity as graph and vector tables. L2 terrain
  does not discover tributaries or apply channel incision;
- write chunked Zarr arrays, Parquet graph/catalog tables, a preview, a
  validation report, and a checksummed manifest;
- identify basin core, required reach-path support, halo, source artifacts,
  software/config versions, selection, topology, resolution, units, and
  approximation semantics explicitly.

The package is accepted only when child area and parent-mean terrain conserve
their parents, surface fractions conserve parent ocean/lake/wetland area,
surface occupancy closes, river paths reference packaged children, source
checksums are complete, and a repeated export is deterministic.

Reason:
L3 cannot safely refine directly from a `72 km` global cell, while a globally
dense 2-5 km planet would force the current whole-array pipeline beyond its
memory and storage architecture. A selected, chunked L2 bridge gives L3 stable
terrain, boundary context, causal priors, and vector identities without
inventing precision or committing to a global high-resolution solve.

## Decision 051: First L3 Slice Is A Bounded Complete Catchment

Status: approved; target selected

Decision:
Use the seed-42 `temperate-highland-catchment` ending at parent cell `80324` as
the first L3 vertical slice. Select its complete upstream L0 closure, include
two context rings from the L2 handoff, and retain one explicit outlet. The
target is roughly `102,000 km2`, cool and wet, carries strong runoff, and spans
upland to lower terrain without requiring an entire continental basin.

Use a `200 m` base grid with deterministic adaptive `25-50 m` river corridors.
The base grid is capped at three million cells. Terrain arrays are chunked;
graphs and vectors remain canonical and live in Parquet. L0 physical and
ecological values remain priors until an L3 process recomputes them.

L3 V0 owns seamless conditioned terrain, conservative runoff forcing,
depression-aware routing, tributary discovery, vector channel geometry, and
physically supported fluvial incision and deposition. Narrow channels remain
vectors with fractional support until adaptive refinement resolves their banks.
The acceptance contract is `docs/specs/13_l3_vertical_slice.md`.

This decision supersedes Decision 045 only for this bounded regional slice.
New global hydrology, biosphere, and resource stages remain frozen.

Reason:
Refining all roughly eight million square kilometres of basin 395 would require
about `128-800 million` cells at `250-100 m` before process layers. The selected
catchment is large enough to exercise a real hierarchy, floodplains, lakes, and
seasonality while keeping the first implementation and validation cycle within
the 32 GB workstation budget.
