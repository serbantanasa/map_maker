# Sparse Hydrology Pass 2

## Status

Implemented provisional stabilization pass following `basin_erosion`.

## Objective

Consume the selected basin's volume-adjusted terrain and solved subgrid channel
beds, recompute local child drainage without changing the accepted trunk graph,
and measure whether erosion materially changed receivers or depression
candidates. The pass is bounded: accepted corrections are persisted once;
large-scale instability rejects the result.

## Dual Routing Surface

The full-cell terrain after erosion is the routing surface for ordinary cells.
Physical centerline cells instead expose the solved channel bed to routing while
retaining their separately conserved full-cell mean. This preserves narrow
rivers without turning a 10-100 metre channel into a multi-kilometre trench.

Preserved-depression children and outside-basin boundary children are terminal
anchors. Connector reaches remain graph handoffs with no physical bed or local
process support.

## Native Stabilization

The Rust `hydrology_pass2_native` kernel builds sparse cubed-sphere D4 adjacency
without allocating a global refined raster. It runs deterministic multi-source
priority floods on the pre-erosion and post-erosion surfaces. Physical channel
cells, preserved waterbody support, and inherited ocean support are anchors.

Ordinary cells receive the priority-flood parent as their local receiver.
Physical channel receivers are then restored from the accepted bed-profile DAG.
The complete stabilized receiver graph is topologically audited and used to
accumulate contributing area once from source to terminal.

## Depression Candidates

Connected ordinary cells whose priority-fill depth exceeds the configured
minimum receive a deterministic candidate ID. The stage aggregates candidate
area, potential fill volume, maximum depth, spill cell, and overlap with the
pre-erosion state. Candidate status is `stable`, `changed`, or `new`.

These are topographic storage candidates, not lake labels. Later water balance,
groundwater, soils, and vegetation stages decide whether they hold permanent,
seasonal, wetland, or no surface water.

## Persistent Outputs

- `StabilizedBasinCellCatalog`: dual surfaces, pre/post receivers, hydrologic
  elevations, fill depths, flow slope and direction, contributing area,
  depression IDs, and change flags.
- `StabilizedRiverReachCatalog`: accepted reaches plus Pass-2 preservation
  status.
- `LocalDepressionCandidateCatalog`: aggregated post-erosion candidates and
  before/after status.
- `HydrologyCorrectionCatalog`: the sparse set of accepted receiver or
  depression-membership changes.
- `HydrologyPass2Metadata`: topology, conservation, stability, exclusion, and
  deterministic-process diagnostics.

## Hard Gates

- Every active child routes to the fixed trunk or an inherited terminal.
- The stabilized receiver graph is acyclic.
- Every physical trunk receiver exactly matches the bed-profile DAG.
- Preserved-depression interiors remain process excluded.
- Contributing area is nondecreasing downstream and terminal accumulation equals
  active selected-basin area within floating-point tolerance.
- Receiver-change count and area remain below configured bounds.
- Identical inputs produce identical catalogs and cache keys.

## Current Limits

- D4 routing is retained to match the accepted topology contract.
- Local depression candidates do not yet run monthly lake water balance.
- Preserved coarse waterbodies still require bathymetric refinement before their
  inlet, lake-crossing, and outlet geometry becomes physical.
- The pass stabilizes one selected basin, not the entire planet at refined
  resolution.
- Earth-derived receiver-change and local-depression distributions are not yet
  calibrated.
