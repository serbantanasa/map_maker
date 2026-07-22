# Bounded Subgrid Outlet Incision

## Status

Implemented, provisional milestone following the first refined surface-water
balance.

## Objective

Consume every explicit outlet-erosion feedback record, cut a physically narrow
and topologically bounded drainage path, rerun local hydrology, and then rerun
monthly surface-water balance. This pass resolves false standing water caused
by erodible local sills without lowering whole refined cells by channel depth.
It is bounded by the active terrain resolution: residual moving spill edges are
handed to regional refinement rather than carved indefinitely.

## Inputs

- `surface_water`: requested candidates, discharge, erosion score, recommended
  incision, and candidate DAG.
- `hydrology_pass2`: accepted receiver graph, subgrid routing surface, physical
  trunk anchors, child area, terrain means, and topology coordinates.
- `planet`: radius for path length and slope calculations.

## Outlet Paths

Each requested candidate begins at its persisted final spill cell and follows
the accepted receiver graph. The path stops at the first downstream candidate,
ordinary cell already below the requested bed, fixed physical channel,
preserved handoff, or terminal. Candidate cascades are planned
downstream-to-upstream so an upstream outlet may connect to a downstream
candidate's corrected level.

No unconstrained least-cost search is performed. The accepted graph supplies
the causal overflow route, and the configured path-cell bound limits the blast
radius of one correction pass.

## Bed And Volume Model

The requested spill bed is the old spill elevation minus recommended incision.
Where downstream support is higher, the applied incision is reduced enough to
retain the configured positive bed slope. Path-cell bed requests are linearly
graded downstream and combined by minimum elevation on shared support.

Outlet width is a bounded power-law function of mean overflow discharge.
Physical eroded volume is integrated from bed lowering, width, and the child's
representative linear length (`sqrt(area)`). Dividing volume by cell area
produces a small cell-mean terrain change while preserving the much deeper
subgrid outlet bed. Parent aggregates must exactly reproduce child volume. The
representative length is a structural approximation until regional refinement
resolves exact channel geometry.

## Local Reroute

The Hydrology Pass-2 native kernel runs again with corrected ordinary routing
support and the original fixed channel anchors. Applied outlet cells remain
ordinary terrain cells but persist an `outlet_fixed_receiver_id`; they are
excluded from depression membership without claiming that a narrow outlet fills
the whole child. The kernel recomputes all remaining receivers, priority-fill
candidates, contributing area, slopes, and direction vectors.

If a newly free neighbor routes back into a constrained outlet, a bounded repair
loop identifies the cyclic component and restores only its unconstrained
ordinary cells to their previously accepted receivers. These repair constraints
persist across later rounds but count as outlet termination support only when
the cell also carries measurable incision. Independent audits repeat all graph,
exclusion, trunk, fixed-receiver, and area-conservation checks.

## Post-Incision Balance

The monthly surface-water solver runs over the corrected cells and candidate
catalog with the same climate, geology, and fractional-water controls as the
first pass. Incision and monthly balance repeat up to a configured bound. A
candidate whose requested outlet cannot descend into fixed or downstream water
support is grade-limited; a candidate whose existing persisted bed already
satisfies the requested profile is already-satisfied. These accepted spill
identities accumulate monotonically across the loop so alternating candidate
sets cannot oscillate by forgetting the previous feasibility result. Other
residual records remain explicit blockers until the configured round bound. At
that bound, each remaining spill is persisted as
`regional_refinement_deferred`, retained as standing water, and suppressed from
further coarse outlet carving. This is an explicit scale handoff, not simulated
convergence.

## Persistent Outputs

- `OutletIncisionCandidateCatalog`: one record per requested candidate with
  requested/applied depth, path termination, width, and status.
- `OutletIncisionCellCatalog`: unique corrected support with old/new bed,
  channel width, contributing-candidate count, and eroded volume.
- `OutletIncisionParentCatalog`: parent-level eroded-volume reconstruction.
- `OutletCorrectedBasinCellCatalog`: corrected mean terrain, subgrid routing
  surface, cumulative outlet state, fixed receiver constraints, and rerun
  hydrology fields.
- `PostIncisionDepressionCandidateCatalog`: candidates discovered after local
  rerouting.
- `OutletIncisionMetadata`: bounds, topology, volume, and trunk diagnostics.
- `OutletIncisionIterationCatalog`: correction and residual counts, volume, and
  standing-water area for each bounded round.
- Final post-incision candidate, cell, monthly, correction, and metadata
  surface-water catalogs.

## Hard Gates

- Every requested feedback record is applied or explicitly blocked.
- Corrected paths are finite, bounded, and downstream-descending.
- Corrected support excludes outside, preserved, and fixed-channel cells except
  as unmodified termination targets.
- Child, parent, and total erosion volumes agree within floating-point
  tolerance.
- Corrected area and rerouted area remain inside configured bounds.
- Cycle-repair rounds and total constrained ordinary support remain inside
  configured bounds.
- Physical trunk identities and beds are unchanged.
- The corrected receiver graph is acyclic and conserves active area.
- Post-incision monthly water balance and immediate-edge transfer audits pass.
- Soil readiness requires a zero residual feedback count, not merely exhaustion
  of the configured round count. A round-limited candidate reaches zero coarse
  feedback only through an explicit regional-refinement deferral record.
- True outlet convergence and resolution deferral are reported separately.

## Canonical Provisional Result

The current fixed-seed face-128 world genuinely converges in 12 rounds with no
regional deferral. Across all rounds it accounts for about `1.45e10 m3` of
bounded outlet-channel erosion. Final accepted standing-water mean area is
about `167,211 km2`, or `4.72%` of the selected basin's active area. The
corrected graph conserves contributing area and preserves all 966 physical
trunk cells.

In the fixed face-64 six-seed screen, four worlds genuinely converge in 6-17
rounds. Seeds 42 and 101 reach the 20-round cap with two and one moving spill
candidates respectively; `6,970 km2` and `5,725 km2` of standing-water area are
explicitly handed to regional refinement. All six worlds pass hydrology hard
gates and the biosphere, functional-vegetation, and derived-biome ensemble
profiles.

This is a structural milestone, not lake calibration. Candidate count, size
distribution, hydroperiod, bathymetry, and the canonical world's globally high
lake fraction still require multi-seed and multi-resolution Earth comparison.

The depression-catalog aggregation is grouped by candidate rather than scanning
the complete sparse basin once per candidate. On the canonical catalog this
preserves byte-identical Arrow output while reducing that operation from about
`4.77 s` to `0.17 s` on the development machine. Outlet path planning remains
in Python; routing, fill, contributing area, and monthly balance remain native
Rust kernels.
