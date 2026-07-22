# Sparse Selected-Basin Refinement

## Status

Implemented prototype following Hydrology Pass 1. This stage proves the
cross-resolution river and terrain contract consumed by the sparse trunk-profile
pass. It does not generate new tributaries, physically evolve valleys, or
replace Hydrology Pass 2.

## Objective

Refine one complete coarse drainage-basin footprint without allocating a global
fine raster and without converting subcell rivers into atomic raster trenches.
Preserve the accepted coarse basin, registered reach graph, junctions, physical
dimensions, and flux attributes while realizing fine child terrain and reach
paths. Zero-width hydrologic connectors preserve routed source-to-sink topology
across coarse depression support where open-channel geometry is unresolved.

## Selection

The default selects the largest ocean-draining basin that contains registered
river reaches. An explicit `basin_id` may be configured. Every coarse basin cell
is refined, together with any adjacent ocean or terminal cell required by an
inherited reach path. Boundary cells are identified separately and are not
counted as basin land area.

The canonical factor is 16. Face-128 parents therefore receive globally unique
face-2048 child IDs, whose average Earth-size cell is about 20 km2 or 4.5 km
across. Only selected child records are allocated and persisted.

## Terrain Realization

- Every parent produces `factor * factor` same-face children.
- Fine cell centers, areas, and D4 neighbors use the canonical equiangular
  cubed-sphere geometry.
- Child terrain blends toward available neighboring parent means and adds a
  deterministic, locally smoothed relief realization.
- The child field is area-weight recentered so its restricted mean reproduces
  the adjusted parent bedrock elevation.
- This is a terrain prior. It does not claim final valley, hillslope, or channel
  morphology.

## Reach Realization

- Every inherited coarse reach retains its reach ID, basin, downstream reach,
  discharge, velocity, width, depth, valley and floodplain widths, incision
  potential, sediment load, morphology, and bed material.
- Physical channel reaches and zero-width hydrologic connectors retain distinct
  `reach_kind` semantics. Connectors preserve topology and flux but contribute
  no channel dimensions, corridor area, or incision volume.
- Topological path length includes the shared graph anchors required to cross
  connectors. Physical channel length includes only fine segments whose parent
  Hydrology Pass 1 cell contains channel support.
- Reaches are processed downstream-first. Terminal reaches retain fine anchors;
  each upstream reach merges onto its registered downstream path inside the
  inherited coarse junction cell, and records the actual fine join cell.
- Each coarse-cell transition receives a valid D4 child transition, including
  cube-face and land-to-ocean boundaries.
- A deterministic downstream-rooted least-cost path connects each entry to its
  next boundary portal, terminal anchor, or downstream merge using the terrain
  prior. Collapsing adjacent fine parents must exactly reproduce the inherited
  coarse cell path.
- The union of fine paths must be a directed acyclic drainage graph. A path may
  neither reuse an edge in reverse nor cross a downstream path and depart from
  it again.
- Path geometry remains a vector property. Cartographic stroke width does not
  enter physical calculations.

## Fractional Support

Each centerline membership contributes physical corridor demand:

```text
channel_fraction = channel_width_m * reach_length_m / cell_area_m2
valley_fraction = valley_width_m * reach_length_m / cell_area_m2
floodplain_fraction = floodplain_width_m * reach_length_m / cell_area_m2
```

Fractions are bounded by `[0, 1]`. Broad valley and floodplain rectangles can
exceed their centerline fine cell. The allocator reserves physical channel
support, spreads valley area into nearby sparse cells, and constrains floodplain
area to the allocated valley footprint. Per-cell capacity is shared across
reaches, so aggregate support cannot exceed physical cell area. Lateral records
carry zero reach length and do not duplicate channel length or incision.
Potential incision volume is physical channel width times in-cell reach length
times inherited incision depth. It remains a diagnostic volume and does not
lower the full fine cell.

Preserved-depression parents are process-excluded. Reach topology may cross
them, but no physical channel, valley, floodplain, or incision membership may
enter them until local waterbody and outlet geometry is resolved. Connector
reaches therefore have no `RefinedReachCellCatalog` records.

## Outputs

- `RefinedBasinCellCatalog`: sparse child IDs, parent IDs, cubed-sphere
  coordinates, areas, terrain prior, offset, inherited relief, and process
  exclusion.
- `RefinedBasinParentCatalog`: parent/child area and elevation restriction
  audit, including boundary membership and preserved-depression process
  exclusion.
- `RefinedRiverReachCatalog`: inherited attributes, fine paths, downstream join
  cells, terminal classification/readiness, separate topological and physical
  channel lengths, and spherical polylines.
- `RefinedReachCellCatalog`: sparse reach length, centerline/lateral support
  role, channel/valley/floodplain fractions, and potential incision volume per
  fine cell.
- `BasinRefinementMetadata`: selection, dimensions, conservation errors,
  channel/connector counts, topology gates, physical totals, and semantic
  versioning.

## Hard Gates

- Child areas restrict to every parent area.
- Area-weighted child elevation restricts to every adjusted parent elevation.
- Every fine reach path is D4-contiguous.
- Fine paths collapse exactly to inherited parent paths.
- Every connected upstream reach ends on its inherited downstream fine path
  inside the shared coarse junction cell.
- The combined directed fine network is acyclic and contains no reverse-used
  edge.
- Every terminal reach ends at an ocean boundary or registered hydrologic sink.
- Hydrologic connectors have zero physical width, local velocity, stream power,
  and incision.
- Physical memberships do not enter preserved-depression parents, including
  shared connector endpoint cells.
- Inherited discharge is unchanged.
- Channel, valley, and floodplain fractions remain finite, nested, and bounded.
- Summed support of each corridor type does not exceed any fine cell's physical
  capacity, and represented corridor area conserves requested physical area.
- Identical seed, configuration, inputs, and native build produce identical
  artifacts.

## Current Limits

- Fine routing realizes inherited trunks only. It does not discover tributaries
  below the coarse hydrology threshold.
- Terrain interpolation reduces but cannot recover information absent from the
  face-128 parent surface. Final geomorphology requires process-driven local
  terrain and erosion.
- Lake and wetland fractions are not yet spatially realized inside refined
  parents.
- Potential incision remains a diagnostic comparison rather than a mandatory
  fine-scale cut. `basin_erosion` solves a prospective least-incision vector
  envelope and routes its prospective sediment budget, but applies no bank
  carve or raster terrain change at this resolution.
- At the canonical factor 16, a child is still about `20.3 km2` or `4.5 km`
  across. Regional refinement, initially around `100-250 m`, owns tributary
  discovery and physical valley, floodplain, bank, and incision morphology.
- The prototype refines one basin per stage run and stores Arrow catalogs rather
  than chunked regional rasters.
- Coarse connectors are topological placeholders. Refined inlet, lake-crossing,
  outlet, and channel geometry remain unresolved in those cells, and connectors
  cannot erode terrain.
- Lateral support currently uses deterministic proximity and lower terrain. It
  does not yet solve cross-valley direction, flood recurrence, or physically
  evolved valley morphology.
- Scarcity ordering and deterministic retries avoid known greedy allocation
  failures, but the valley allocator is not a general global optimizer.

The implemented downstream pass is specified in `08b_basin_erosion.md`.
