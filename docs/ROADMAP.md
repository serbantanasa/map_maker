# Current Roadmap

Status: current sequencing as of 2026-07-22. Numbered decisions remain the
authority when this summary and the decision log differ.

## Controlling Milestone

[Decision 045](DECISIONS.md#decision-045-freeze-late-simulation-work-until-global-map-export)
continues to freeze new global hydrology, biosphere, and resource-system stages.
[Decision 051](DECISIONS.md#decision-051-first-l3-slice-is-a-bounded-complete-catchment)
explicitly opens only the bounded first L3 vertical slice. It does not authorize
another global-stage expansion.

## Current Work

1. **Complete:** regenerate the continuous L3 terrain window with explicit
   core, process halo, outside mask, legend, and scale bar.
2. **Complete:** add conservative regional forcing, depression-aware routing,
   and a fine routed-core refinement around the regional outlet handoff.
3. **Complete:** discover the first L3 tributary graph, preserve inherited
   trunk identity, publish explicit lake connectors, and leave terrain intact.
4. **Complete:** repair the L0-to-L3 morphology gap by making L2 own coherent
   `5-72 km` hills, uplands, valleys, and drainage divides under bounded soft
   L0 conditioning; bound unresolved hydraulic pits and realize lake area over
   stable basin identities; then regenerate terrain and hydrology.
5. **Complete:** route the entire stored L3 rectangle behind a four-L2-cell
   hidden halo, crop terrain and hydrology to one fully processed display, and
   replace the artificial inland outlet terminal with dominant natural-basin
   refinement.
6. **Next:** turn selected raw D8 reach paths into smooth physical centerline
   geometry and adaptively refine representative `25-50 m` channel corridors
   before applying any fluvial incision or resolving banks.
7. In parallel, record explicit acceptance or a bounded blocker list for the
   current six-seed physical-atlas gallery and face-128 release candidate.

The July 2026 six-seed surface-geography gallery passed its broad morphology
review. A connected-river `physical_atlas_v5` release candidate and six-seed
gallery now exist, but the overall milestone remains open until the projected
physical map receives explicit user acceptance. The selected-basin L2 handoff
now passes a quantitative terrain-seam gate. The first L3 catchment, resolution,
budget, and acceptance contract are selected in Decision 051. Detailed earlier evidence is retained in the
[archived validation baseline](archive/validation/2026-07-canonical-validation-baseline.md).

The original sparse-core L3 terrain artifact passed its numerical gates but
failed visual/domain review because its catchment-shaped storage left large
internal no-data regions in the rectangular diagnostic and provided an unsafe
hydrology boundary. Decision 053 replaces that extent with an approximately
`6.04 million`-cell continuous working window. Decision 054 now records the
passing L3 hydrology V0: a fine routed core, continuous vector network, explicit
lake connectors, and no unexplained downstream discharge losses. Decision 055
now assigns coherent regional morphology to L2 rather than stretching L3 noise
or stamping one correction shape into every L0 parent. Decision 056 separates
raw coarse hydraulic pits from literal L2 terrain and replaces parent-stamped
lake quotas with basin-coherent area conservation. Raw D8
centerline geometry remains unsuitable as final bank or meander geometry.
Decision 057 now routes all `6.04 million` stored cells behind an approximately
`17.3 km` hidden boundary halo and exposes a common `5.20 million`-cell terrain
and hydrology display. No visible region is terrain-only. The inherited inland
outlet remains an alignment prior; the accepted `89,852 km2` natural basin is
selected by dominant overlap with the coarse target and exits the display at
about `1,079 m3/s`.

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
