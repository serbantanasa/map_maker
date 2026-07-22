# Physical Atlas Export

Status: implemented, provisional visual acceptance

## Role

The physical atlas is a versioned presentation layer over immutable canonical
artifacts. It is not a simulation stage and does not write inputs back into the
pipeline DAG. Cartographic changes therefore neither regenerate nor invalidate
the physical world state.

The canonical command is intentionally small:

```bash
uv run map-maker atlas
```

The matching hydrologic diagnostic is:

```bash
uv run map-maker basin-atlas
```

Its defaults live in `configs/physical_atlas.yaml`.

## Inputs

The V1 world render consumes:

- surface elevation, fractional ocean, ocean depth, and continental shelf area;
- local terrain relief;
- monthly snow water equivalent and glacier fraction;
- fractional bedrock exposure, soil salinity, wetlands, and inland open water;
- the complete 13-component derived-biome mixture;
- coarse vector river reaches and mean discharge.

Every input artifact checksum is recorded in the output metadata. Missing inputs
fail the export rather than silently falling back to unrelated data.

## Projection And Seam

Equal Earth is the canonical static projection. Automatic seam placement scores
every equirectangular longitude by area-weighted emerged land and centers the map
on the opposite meridian. An explicit central meridian remains available for a
particular campaign or comparison render.

The PNG has a neutral opaque exterior. The GeoTIFF stores RGB plus an alpha band,
an Equal Earth CRS carrying the selected longitude of natural origin, and the
projected affine transform. Equirectangular sampling is only an internal
interchange step from the cubed sphere to the final projection.

## Composition

Land color is an area-weighted mixture of biome colors, then bounded by physical
highland, local-relief, bedrock, salinity, wetland, snow, and glacier signals.
Ocean color follows logarithmic bathymetry with visible shelf tinting. Fractional
ocean and inland-water area are retained at coasts and coarse subgrid lakes.
Multidirectional hillshade is computed from elevation and applied more strongly
on land than under water.

The canonical world style seeds display from channel reaches with mean
discharge of at least `3000 m3/s`, then completes a discharge-dominant upstream
path and every downstream continuation. Intervening zero-width hydraulic
connectors are drawn so a major river does not disappear at unresolved standing
water. This is a legibility rule, not a claim that smaller rivers do not exist
or that a connector has physical channel width. Two passes of
endpoint-preserving corner cutting round the coarse cell-center path for display
only; reach identity, connectivity, and canonical geometry are unchanged.

Fractional inland water uses area-proportional color weight. Supporting cells
are not painted as categorical lakes, because their combined footprint can be
several times larger than represented water area.

## Outputs

The default export directory contains:

- `physical_world_map.png`: presentation image;
- `physical_world_map.tif`: projected RGB-alpha raster for downstream tooling;
- `physical_world_map.json`: style, projection, source checksums, output
  checksums, and rendering semantics.

The independent `physical_atlas_v5` style version includes physical-width-aware
river hierarchy, connected hydraulic handoffs, area-proportional fractional
lakes, and seasonal/perennial sea-ice composition. Later cartographic changes
remain independent of simulation-stage versions.

The `drainage_basin_atlas_v1` export adds:

- `drainage_basins.png`: deterministic adjacency-colored basins, all basin
  boundaries, emphasized basins at least `250,000 km2`, gold inland-terminal
  boundaries, fractional lakes, and the connected major-river overlay;
- `drainage_basin_ids.tif`: an integer Equal Earth raster where `-2` is outside
  the globe, `-1` is ocean, and nonnegative values are `BasinCatalog` IDs;
- `drainage_basins.json`: projection, checksums, palette, display semantics,
  and the complete basin-to-color/sink/area catalog.

The categorical export uses nearest-neighbor reprojection for basin identity.
Its visible one-cell coastal basins are therefore model evidence, not blended
or generalized cartography.

## Acceptance And Limits

Automated checks require deterministic cubed-sphere sampling, correct seam
rotation, a nonblank Equal Earth footprint with empty exterior corners, valid
GeoTIFF bands and CRS, and a working discharge-aware river overlay. The
canonical face-128 image is also rendered and inspected directly.

The current map is intentionally honest about its source. Face-128 truth is
sampled through a `1024 x 512` equirectangular interchange grid, roughly `0.35`
degrees or `39 km` per equatorial sample. A `4096`-pixel export improves
projection and antialiasing but cannot create county-scale shoreline detail.
The remaining visible coastal stepping and angular river paths must be resolved
by later topology-preserving vector generalization or finer regional
realization, not decorative random noise.

Still open under Decision 018:

- explicit vector coast, island, lake, wetland, and glacier geometry;
- regional physical maps from refined state;
- rotatable globe inspection;
- multi-seed atlas gallery review and color-vision checks;
- explicit user acceptance of the canonical world-map visual bar.
