# Planet Engine Notes

Working notes, unresolved questions, and speculative ideas. Promote stable
items to `PLANET_ENGINE_SPEC.md` or `DECISIONS.md` after discussion.

## Project Identity

The codebase is currently named `map_maker`, but the ambition is broader:
planetary history and world-stack generation. Do not rename the repository yet.
Consider a rename only after the next-generation pipeline runs end to end.

Possible future names:
- planet-engine
- geogenesis
- tellus
- orbis
- worldforge

## Current Concerns

- Hydrology is high risk. Previous river attempts failed because plausible
  rivers require hydrologically conditioned terrain, basin/lake handling,
  outlet carving, discharge, and coarse-to-fine inheritance.
- Beautiful maps require a projection/rendering system. The canonical cubed
  sphere should not be shown raw except for debugging.
- 100 m global resolution is not a normal workstation target. Use global
  coarse/mid plus selected high-resolution regions.
- Neural surrogate training is interesting, but the simulator must not depend
  on neural models to be useful.

## Current Executable Prototype

The canonical command now runs the implemented stack from tectonics through
erosion, persists every stage, writes a run manifest, and renders a physical
preview. This is a working integration baseline, not the accepted geological
model.

The tectonics V4 kernel separates continental crustal provinces from mechanical
plate ownership using a deterministic multi-scale correlated field. Plate cells
are spatially warped to avoid straight Voronoi boundaries, while boundary forcing
is segmented along strike and diffused into deformation belts. This removes the
earlier one-plate/one-crust-type failure and the most obvious linear uplift
artifacts while exercising the approved artifact contracts. It does not yet
create terranes, cratons, or continental crust through the explicit event history
required by Decision 009, so it must be replaced or made subordinate to that
history model.

The final preview intentionally does not draw `RiverIncision` as rivers. That
field is an erosion diagnostic, not a routed hydrological network. Ocean depth
in the preview is generalized from a longitude-wrapped Gaussian coastal shelf
field until credible basin and bathymetric history exists; raw simulation fields
remain unchanged in the datasets and diagnostic renders.

The fixed-seed validation harness implements the first portion of Decision 019.
It gates deterministic cold runs, cache replay, seed uniqueness, finite fields,
prototype morphology, longitude seams, and plate imprint, then writes a gallery
for mandatory human review. The thresholds remain provisional until geological
scenarios and reference distributions can calibrate them.

The first cubed-sphere milestone now provides native equiangular geometry,
steradian cell areas, and reciprocal cross-face D4 neighbors. Geological kernels
remain on the provisional grid until they can consume topology-owned neighbor
tables or face-aware halos; a flattened compatibility mode is explicitly
rejected.

Native ABI version and binary SHA-256 fingerprints are now verified before FFI
use, included in simulation cache keys, and recorded in run manifests. Debug and
release native profiles both execute the physical pipeline. Cached artifact
content is checksum-verified during hydration and corrupt stage entries are
recomputed.
