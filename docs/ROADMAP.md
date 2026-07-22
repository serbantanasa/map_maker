# Current Roadmap

Status: current sequencing as of 2026-07-21. Numbered decisions remain the
authority when this summary and the decision log differ.

## Controlling Milestone

[Decision 045](DECISIONS.md#decision-045-freeze-late-simulation-work-until-global-map-export)
continues to freeze new global hydrology, biosphere, and resource-system stages.
[Decision 051](DECISIONS.md#decision-051-first-l3-slice-is-a-bounded-complete-catchment)
explicitly opens only the bounded first L3 vertical slice. It does not authorize
another global-stage expansion.

## Current Work

1. Implement seamless `200 m` conditioned terrain for the selected
   `temperate-highland-catchment`, using chunked storage and the `24 GB` memory
   ceiling in the L3 contract.
2. Add conservative regional runoff forcing and depression-aware routing over
   that terrain, preserving the sole target outlet and inherited trunk identity.
3. In parallel, record explicit acceptance or a bounded blocker list for the
   current six-seed physical-atlas gallery and face-128 release candidate.

The July 2026 six-seed surface-geography gallery passed its broad morphology
review. A connected-river `physical_atlas_v5` release candidate and six-seed
gallery now exist, but the overall milestone remains open until the projected
physical map receives explicit user acceptance. The selected-basin L2 handoff
now passes a quantitative terrain-seam gate. The first L3 catchment, resolution,
budget, and acceptance contract are selected in Decision 051. Detailed earlier evidence is retained in the
[archived validation baseline](archive/validation/2026-07-canonical-validation-baseline.md).

## Thaw Gate

The freeze ends only when all of the following hold:

- The canonical six-seed surface-geography gallery is reproducible and has an
  explicit dated human disposition.
- A canonical projected physical world map meets the product and rendering
  bars in Decisions 008 and 018.
- Mountain belts, shelves, passive margins, islands, straits, and inland seas
  remain legible at the selected world-map scale.
- Truth artifacts remain separate from atlas style and projection versions.

## After The Milestone

The deferred backlog includes deterministic regional refinement, bounded
vegetation feedback, additional hydrology and calibration, mineral and energy
systems, and later population/civilization work. Their ordering requires a new
roadmap decision after the thaw gate.

Regional refinement must implement [Decision 048](DECISIONS.md#decision-048-terrain-resolution-owns-physical-river-morphology):
retain canonical river vectors and reach graphs, refine only where morphology
is needed, and choose local cells fine enough to span physical channels with
multiple cells before resolving banks and meanders. The archived population
stage plan is historical input, not an approved next-stage contract.
