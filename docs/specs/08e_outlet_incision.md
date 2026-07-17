# Bounded Subgrid Outlet Incision

## Status

Implemented, provisional milestone following the first refined surface-water
balance.

## Objective

Consume every explicit outlet-erosion feedback record, cut a physically narrow
and topologically bounded drainage path, rerun local hydrology, and then rerun
monthly surface-water balance. This pass resolves false standing water caused
by erodible local sills without lowering whole refined cells by channel depth.

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
residual records remain an explicit blocker for soils and biomes.

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
  of the configured round count.

## Canonical Provisional Result

The fixed-seed face-128 world converges in seven rounds with residual feedback
counts `2181, 671, 109, 22, 3, 1, 0`. Across all rounds it removes about
`1.65e10 m3` from narrow outlet channels. Final accepted standing-water mean
area is about `238,049 km2`, or `3.74%` of the selected basin's active area.
The corrected graph conserves contributing area and preserves all 2,434
physical trunk cells.

This is a structural milestone, not lake calibration. The canonical result has
10,062 permanent candidates, seven hydrologic-wetland candidates, and no
seasonal candidates. Candidate count, size distribution, hydroperiod, and
bathymetry still require multi-seed and multi-resolution Earth comparison.

The depression-catalog aggregation is grouped by candidate rather than scanning
the complete sparse basin once per candidate. On the canonical catalog this
preserves byte-identical Arrow output while reducing that operation from about
`4.77 s` to `0.17 s` on the development machine. Outlet path planning remains
in Python; routing, fill, contributing area, and monthly balance remain native
Rust kernels.
