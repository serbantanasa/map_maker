# Connected Sea Level And Surface Geography

## Status

Canonical cubed-sphere V1 implemented. All canonical surface consumers now use
the solved sea-level artifacts; only geological and explicitly legacy stages
retain the crust-class compatibility mask.

## Purpose

Convert continuous post-tectonic elevation into present-day emerged land,
connected ocean, continental shelf, coarse fractional coastline, and a solved
vertical datum. Continental versus oceanic crust is geological evidence; it is
not a shoreline.

The Earthlike profile currently uses approximately `42%` continental-crust
candidate area and independently targets ocean area so emerged land stays in
the Decision 038 band of `18-36%`. The default config targets `65%` ocean
(`35%` land), one legal point in that band, while allowing continental margins
and rifts to flood. Other world profiles may choose different setpoints inside
or outside the Earthlike band.

## Inputs

- Cubed-sphere cell areas and reciprocal D4 neighbors.
- `BedrockElevationM`: continuous provisional-datum elevation.
- `TerrainReliefM`: unresolved within-cell relief used as coastal hypsometry.
- Target ocean area fraction and shelf/coastal controls.

`BaseOceanMask` is deliberately not an input. It remains a compatibility name
for oceanic-crust candidates in `world_age` and must not be interpreted as
surface water.

## Connected-Ocean Solve

1. Sort cells by bedrock elevation, activate equal-elevation groups from low to
   high, and maintain connected component area with union-find.
2. Select the first level whose largest connected low component reaches the
   target ocean area. That component is the global ocean.
3. Low components not connected to the global ocean remain inland
   below-sea-level terrain. Hydrology may later make them dry basins, lakes,
   wetlands, or connected seas; sea level does not silently flood them.
4. Subtract the solved datum from bedrock to publish signed surface elevation.

This is an area-constrained V1 approximation. A later water-volume and
isostatic solver may replace the prescribed target without changing the output
contract.

## Fractional Coast And Shelf

Coarse shoreline cells are mixed-area containers. Interior ocean cells have
ocean fraction one and interior land cells zero. Cells touching the discrete
shore use a logistic hypsometric response derived from relative elevation and
unresolved relief. A deterministic logit shift closes the area-weighted global
fraction exactly while preserving spatial rank.

`SurfaceOceanMask` remains discrete for drainage topology and connectivity.
`SurfaceOceanFraction` is authoritative for physical coarse area, climate
mixing, storage, rendering, and regional refinement. Neither representation
deletes persistent subgrid straits, islands, or basin objects required by
Decision 011.

Continental shelf fraction is the ocean-covered fraction whose unresolved
surface plausibly lies within the configured shelf depth. It is not a crust
class and is allowed over either crust type.

## Outputs

- `SurfaceOceanMask`: center-class connected global-ocean mask.
- `SurfaceOceanFraction`: fractional ocean area in `[0, 1]`.
- `SurfaceElevationM`: bedrock elevation relative to solved sea level.
- `OceanDepthM`: center depth for connected-ocean cells.
- `ContinentalShelfFraction`: fractional shallow connected ocean.
- `CoastalCellMask`: cells crossing the discrete shoreline.
- `InlandBelowSeaLevelMask`: low cells excluded from the connected ocean.
- `SeaLevelMetadata`: datum, global fractions, component morphology, shelves,
  inland basins, extremes, coastline edge count, and largest-landmass spherical
  coastline complexity.

## Required Invariants

- Area-weighted fractional ocean coverage meets its target within configured
  tolerance.
- The center-class ocean is one connected component across cube-face seams.
- Isolated low basins are not mislabeled as ocean.
- Surface elevation differs from bedrock by one global datum only.
- Fractions are finite and bounded; depths and shelf fractions are nonnegative.
- Identical seed, configuration, topology, and native version reproduce
  identical outputs.
- Crust class and surface-ocean class differ in an Earthlike run.
- Earthlike morphology retains multiple significant landmasses and may not
  collapse into six compact, near-circular components.
- The largest landmass must exceed the configured anti-blob coastline-
  complexity floor. This is measured against the minimum perimeter of a
  same-area spherical cap; the fixed Earthlike regression floor is `2.0`.

## Validation

- Analytic connected and isolated basins.
- Exact area reconstruction from fractional coverage.
- Cross-face global-ocean connectivity.
- Determinism and FFI buffer validation.
- Fixed six-seed morphology screen using component share, significant-landmass
  count, coastline complexity, shelf area, and inland-basin area.
- Canonical cube-net and equirectangular truth renders reviewed for round blobs,
  accidental isthmuses, straight corridors, polar projection artifacts, lost
  islands, and implausible fragmentation.

The canonical face-128 seed has ten significant landmasses, a `44.6%`
largest-landmass share, `4,948` land-to-ocean edges, and a `4.43` largest-
landmass coastline-complexity ratio. The six face-64 Earth-profile seeds span
`3.12-6.27`, above the `2.0` anti-blob gate.

## Deferred Work

- Water-volume, thermal-expansion, glacial, and sediment-displacement sea-level
  feedback.
- Tides and moon-dependent intertidal area.
- Dynamic shelf sedimentation, coastal erosion, deltas, and barrier systems.
- Persistent vector coastlines, straits, and island identities through all
  resolution levels.
- Cartographic projection and generalized coastline products for the atlas.
