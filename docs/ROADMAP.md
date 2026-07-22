# Current Roadmap

Status: current sequencing as of 2026-07-21. Numbered decisions remain the
authority when this summary and the decision log differ.

## Controlling Milestone

[Decision 045](DECISIONS.md#decision-045-freeze-late-simulation-work-until-global-map-export)
freezes new simulation features after derived biomes until the first global
map-export milestone is accepted. Hydrology, biosphere, regional refinement,
and resource-system work is limited to regressions, broken invariants, artifact
contract corrections, and blockers for the map milestone.

## Current Work

1. Review the implemented seed-42 L2 regional handoff package, its preview,
   manifest, and validation report.
2. Choose the first L3 target and define the 100-250 m vertical-slice acceptance
   contract without treating inherited L0 priors as downscaled physics.
3. In parallel, record explicit acceptance or a bounded blocker list for the
   current six-seed physical-atlas gallery and face-128 release candidate.

The July 2026 six-seed surface-geography gallery passed its broad morphology
review. A connected-river `physical_atlas_v5` release candidate and six-seed
gallery now exist, but the overall milestone remains open until the projected
physical map receives explicit user acceptance. The first selected-basin L2
handoff package is implemented as a bounded export and contract milestone; it
does not thaw new regional process simulation. Detailed earlier evidence is retained in the
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
