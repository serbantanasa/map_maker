# L2 Regional Handoff

Status: implemented

## Role

The L2 handoff is an immutable conditional realization between the global L0
parent state and later 100-250 m L3 terrain. It packages one complete drainage
basin plus a coarse-cell halo at approximately 4.5 km child support. It is an
export product over canonical artifacts, not a new independent geography.

The default command is:

```bash
uv run map-maker regional-handoff
```

Defaults live in `configs/l2_regional_handoff.yaml`.

## Selection And Geometry

Selection uses a stable hydrology `BasinID`. `auto` chooses the same largest
ocean-draining basin with registered reaches used by basin refinement. The core
contains every parent in that basin. Any inherited reach cell beyond the
BasinID is explicit terminal-path support. The halo is an exact breadth-first
set around the union of core and terminal-path support; it may cross face edges
or include ocean. These three roles are stored separately.

Every packaged parent is refined by the existing factor-16 terrain kernel. L2
children retain global face-2048-equivalent IDs, cubed-sphere face/row/column,
XYZ center, physical area, parent ID, parent row, core/halo role, terrain
elevation, terrain offset, and hydraulic surface priors.

The residual terrain field is evaluated from global spherical coordinates as a
domain-warped multiscale field with a bounded ridged component. Its current
wavelengths span roughly `5-72 km`, with the strongest mesoscale form at
`18-72 km`. This is where coherent regional hills, uplands, valleys, and
drainage divides are realized; L3 is not allowed to recreate that band later.

Overlapping spherical radial basis functions condition the result against L0
means across three parent rings. The condition is deliberately soft: each
parent mean must lie within `15 m` and `5%` of inherited relief. Exact
parent-local correction shapes are forbidden because they emboss the L0 grid.
Validation compares p95 boundary residuals with ordinary interior edge jumps
and independently rejects repeated parent motifs. Decision 055 owns this
contract.

## Surface Realization

Children are ordered by terrain elevation with stable child-ID tie-breaking.
Ocean and effective wetland area are assigned independently within each parent
against remaining child capacity. Effective lake area is grouped by stable
hydrology `LakeID`, conserved over the represented hydraulic basin, and placed
by continuous child terrain relative to the inherited hydraulic surface. This
avoids stamping a separate rectangular lake quota into every L0 parent. At most
one boundary child per allocation group is fractional, and combined occupancy
may not exceed one.

The L0 priority-flood surface is a spill and connectivity constraint, not
literal 72 km bathymetry. Raw source bedrock is retained as a prior. Where an
inland hydraulically controlled parent lies farther below that surface than the
scenario permits, L2 terrain uses a separate bounded conditioning target. The
current Earthlike limit is the larger of `1,200 m` and four times inherited
relief. Validation also rejects any single parent's L2 offset span above
`2,500 m` or eight times inherited relief. Decision 056 owns these controls.

Child surface elevation is the refined terrain plus the source parent's
sea-level-relative offset. Its parent mean follows the same bounded soft
conditioning contract as terrain. Surface occupancy and parent area remain
exactly conservative. This does not claim L2 coastal erosion, lake bathymetry,
or climate feedback.

## Priors And Vectors

Every grid-shaped artifact available on the causal path through derived biomes
and validated mineral systems is restricted to packaged parent IDs and stored
parent-first under its source stage and artifact name. These are inherited
priors. Monthly, class-mixture, mineral-system, commodity, and vector
components remain trailing dimensions. The manifest carries the complete
versioned mineral-system, commodity, and causal axes plus the upstream mineral
validation checksum; publication is forbidden when that hard gate is red.

The package also includes the selected and halo drainage graph, represented
basin and waterbody catalogs, inherited source reaches, refined reaches, and
fractional reach support. River vectors remain canonical. Tributary discovery,
bank placement, meanders, and applied erosion remain L3 work.

## Layout

```text
<output>/
  manifest.json
  validation.json
  preview.png
  region.zarr/
    parent/
    parent_priors/<stage>/<artifact>/
    l2/geometry/
    l2/surface/
  tables/
    basins.parquet
    drainage_graph.parquet
    waterbody_cells.parquet
    depression_catalog.parquet
    river_reaches.parquet
    refined_river_reaches.parquet
    refined_reach_cells.parquet
    mineral_systems.parquet
    major_deposit_candidates.parquet
```

Zarr arrays are row-chunked and Parquet owns variable-length graph/vector
records. The manifest records checksums for the Zarr tree, every table, the
preview, source artifacts, and both configuration files.

## Non-Goals

- globally dense 2-5 km generation;
- recomputed L2 climate or soil chemistry;
- physical river incision or sediment deposition;
- newly discovered tributaries;
- realized mineral-deposit geometry, reserves, economics, or energy systems;
- L3 banks, floodplains, settlements, or game tiles.

## Canonical Evidence

The seed-42 face-128 export selects basin `395`, packages `1,587` core parents,
`832` halo parents, and one additional source-context parent, and realizes
`619,520` factor-16 children. The package contains `190` inherited parent-prior
fields, including validated mineral-system and commodity axes. Its validation
report passes area, bounded terrain-mean, surface-mean,
occupancy, child-ID, parent-membership, hydraulic-depth, local-relief, seam,
motif, and refined river-graph gates. Maximum terrain-mean error is `10.19 m`,
its parent-boundary residual p95 ratio is `1.04`, and motif correlation is
`0.257` at p50 and `0.657` at p95. The largest local parent offset span is
`1,830 m`, or `5.90` times inherited relief. Forty-two unresolved hydraulic
parents use bounded terrain targets; the largest adjustment from retained raw
source bedrock is about `3,218 m`, with zero post-condition depth-bound excess.
Re-running the same source and handoff configurations must produce an identical
manifest and output checksums.
