# Sparse Selected-Basin Refinement

## Status

Implemented prototype following Hydrology Pass 1. This stage proves the
cross-resolution river and terrain contract required before fluvial erosion. It
does not yet apply erosion, generate new tributaries, or replace Hydrology Pass
2.

## Objective

Refine one complete coarse drainage-basin footprint without allocating a global fine raster and
without converting subcell rivers into atomic raster trenches. Preserve the
accepted coarse basin, registered reach graph, junctions, physical dimensions,
and flux attributes while realizing fine child terrain and reach paths. A
complete footprint does not imply that thresholded Hydrology Pass 1 reaches
already form a complete source-to-sink network.

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

For every fine reach-cell membership:

```text
channel_fraction = channel_width_m * reach_length_m / cell_area_m2
valley_fraction = valley_width_m * reach_length_m / cell_area_m2
floodplain_fraction = floodplain_width_m * reach_length_m / cell_area_m2
```

Fractions are bounded by `[0, 1]`. Broad valley and floodplain rectangles can
exceed their centerline fine cell; current records cap that cell's represented
support and report both represented and requested unclipped areas. Later lateral
corridor realization must place the retained area in neighboring cells rather
than discard it. Potential incision volume is physical channel width times
in-cell reach length times inherited incision depth. It remains a diagnostic
volume and does not lower the full fine cell.

## Outputs

- `RefinedBasinCellCatalog`: sparse child IDs, parent IDs, cubed-sphere
  coordinates, areas, terrain prior, offset, and inherited relief.
- `RefinedBasinParentCatalog`: parent/child area and elevation restriction
  audit, including boundary membership.
- `RefinedRiverReachCatalog`: inherited attributes, fine paths, downstream join
  cells, terminal classification/readiness, path length, and spherical
  polylines.
- `RefinedReachCellCatalog`: sparse reach length, channel/valley/floodplain
  fractions, and potential incision volume per fine cell.
- `BasinRefinementMetadata`: selection, dimensions, conservation errors,
  topology gates, physical totals, and semantic versioning.

## Hard Gates

- Child areas restrict to every parent area.
- Area-weighted child elevation restricts to every adjusted parent elevation.
- Every fine reach path is D4-contiguous.
- Fine paths collapse exactly to inherited parent paths.
- Every connected upstream reach ends on its inherited downstream fine path
  inside the shared coarse junction cell.
- The combined directed fine network is acyclic and contains no reverse-used
  edge.
- Inherited discharge is unchanged.
- Channel, valley, and floodplain fractions remain finite and bounded.
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
- Potential incision volume is not applied, and sediment is not yet routed or
  deposited.
- Junction-consistent channel-bed profiles are not yet solved. They belong to
  the conservative erosion pass and may not be inferred from rendering strokes.
- The prototype refines one basin per stage run and stores Arrow catalogs rather
  than chunked regional rasters.
- Reach gaps already present in Hydrology Pass 1 remain visible; refinement does
  not invent connectivity that the accepted parent graph does not contain.
- `source_to_sink_ready` remains false while those gaps exist. Conservative
  erosion and sediment routing must not run on such a basin.
- Capped valley and floodplain support is not yet spread laterally into adjacent
  fine cells. Requested and represented areas therefore differ.

The next pass closes inherited source-to-sink threshold gaps and realizes broad
corridors laterally. Conservative fluvial incision and sediment routing can
follow only after those readiness gates pass, then aggregate physical budgets
upward before Hydrology Pass 2.
