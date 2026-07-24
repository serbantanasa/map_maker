# Current Roadmap

Status: current sequencing as of 2026-07-24. Numbered decisions remain the
authority when this summary and the decision log differ.

## Controlling Milestone

[Decision 045](DECISIONS.md#decision-045-freeze-late-simulation-work-until-global-map-export)
continues to freeze unbounded global hydrology and biosphere expansion.
[Decision 051](DECISIONS.md#decision-051-first-l3-slice-is-a-bounded-complete-catchment)
explicitly opens only the bounded first L3 vertical slice. It does not authorize
another global-stage expansion.
[Decision 061](DECISIONS.md#decision-061-mineral-systems-v0-is-causal-coarse-and-explicitly-incomplete)
now opens one bounded global resource-stage exception: the causal mineral
inventory required before L3 deposit geometry.

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
6. **Complete:** turn selected raw D8 reach paths into smooth physical
   centerline geometry and publish channel, riparian, floodplain, and valley
   support suitable for L3 soils and biomes. The dated execution contract is
   [L3 Ecology Readiness: 2026-07-23](plans/2026-07-23-l3-ecology-readiness.md).
7. **Complete:** realize L3 surface materials and initial soils from inherited
   priors plus accepted L3 terrain, water, and channel geometry. The canonical
   artifact covers the complete stored window in durable chunks and keeps
   active channel width separate from inherited alluvial history.
8. **Complete:** replay the causal ecology stack from L3 soil water through
   resource envelope, potential biosphere, functional vegetation, and
   fractional biome mixtures. Broad climate is interpolated continuously;
   accepted fine soil water controls productivity, and coarse biome colors are
   comparison priors only.
9. **Complete:** implement Causal Mineral Systems V0 under Decision 061:
   first correct the global geomorphic-slope driver, then persist causal
   supports, system and commodity prospectivity, stable catalogs, diagnostics,
   and multi-seed validation. The canonical face-128 artifact and fixed
   six-seed face-64 ensemble pass; petroleum and L3 deposit geometry remain
   explicitly deferred.
10. **Next decision after V0:** select the first L3 resource-realization
   vertical slice or adaptive `25-50 m` river-corridor refinement. Do not
   resolve deposits or banks before their respective upstream contracts exist.
11. In parallel, record explicit acceptance or a bounded blocker list for the
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
about `1,079 m3/s`. Decisions 058-060 now cover passing regional soils and
ecology, including the rule that hydraulic receiver slope cannot stand in for
physical hillslope gradient at L3. Decision 061 extends that correction to
global materials and records the passing causal coarse mineral inventory used
as the future L3 resource prior.

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
