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

## Surface Realization

Within each parent, children are ordered by terrain elevation with stable child
ID tie-breaking. Ocean, effective lake, and effective wetland area are assigned
in that order against remaining child capacity. At most one boundary child per
class is fractional. Child area times occupancy must reproduce the source
parent fraction times parent area within numerical tolerance, and combined
occupancy may not exceed one.

Child surface elevation is the refined terrain plus the source parent's
sea-level-relative offset. This preserves the source parent mean. It does not
claim L2 coastal erosion, lake bathymetry, or climate feedback.

## Priors And Vectors

Every grid-shaped artifact available on the causal path through derived biomes
is restricted to packaged parent IDs and stored parent-first under its source
stage and artifact name. These are inherited priors. Monthly, class-mixture,
and vector components remain trailing dimensions.

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
```

Zarr arrays are row-chunked and Parquet owns variable-length graph/vector
records. The manifest records checksums for the Zarr tree, every table, the
preview, source artifacts, and both configuration files.

## Non-Goals

- globally dense 2-5 km generation;
- recomputed L2 climate or soil chemistry;
- physical river incision or sediment deposition;
- newly discovered tributaries;
- mineral or energy deposits not yet present in the parent world;
- L3 banks, floodplains, settlements, or game tiles.

## Canonical Evidence

The seed-42 face-128 export selects basin `395`, packages `1,587` core parents
and `336` context parents, and realizes `492,288` factor-16 children. The
package contains `178` inherited parent-prior fields and occupies approximately
`20 MB` with the current compressors. Its validation report passes area,
terrain-mean, surface-mean, occupancy, child-ID, parent-membership, and refined
river-graph gates. Re-running the same source and handoff configurations must
produce an identical manifest and output checksums.
