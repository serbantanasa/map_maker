# Sparse Hydrology Pass 2

## Status

Implemented provisional stabilization pass following `basin_erosion`.

## Objective

Consume the selected basin's unchanged terrain prior plus accepted vector trunk
graph, recompute local child drainage without changing that graph, and publish
local receiver and depression candidates. The pass is bounded: accepted
topological constraints are persisted once; large-scale instability rejects
the result.

## Routing Surface

Every child uses the unchanged sparse terrain prior as its raster routing
surface. Physical centerline cells retain fixed downstream receivers from the
accepted vector profile, but the candidate channel-bed elevation is not
substituted for raster terrain. This preserves trunk topology without turning a
10-100 metre channel into a multi-kilometre trench or implying completed
erosion.

Preserved-depression children and outside-basin boundary children are terminal
anchors. Connector reaches remain graph handoffs with no physical bed or local
process support.

## Native Stabilization

The Rust `hydrology_pass2_native` kernel builds sparse cubed-sphere D4 adjacency
without allocating a global refined raster. It runs deterministic multi-source
priority floods on the same unchanged terrain surface for baseline and
stabilized audits. Physical channel cells, preserved waterbody support, and
inherited ocean support are anchors.

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

- `StabilizedBasinCellCatalog`: unchanged routing surface,
  baseline/stabilized receivers, hydrologic elevations, fill depths, flow slope
  and direction, contributing area, depression IDs, and change flags.
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
- The routing raster exactly equals the sparse terrain prior.
- The stabilized receiver graph is acyclic.
- Every physical trunk receiver exactly matches the bed-profile DAG.
- Preserved-depression interiors remain process excluded.
- Contributing area is nondecreasing downstream and terminal accumulation equals
  active selected-basin area within floating-point tolerance.
- Receiver-change count and area remain below configured bounds.
- Identical inputs produce identical catalogs and cache keys.

## Current Limits

- D4 routing is retained to match the accepted topology contract.
- The downstream surface-water stage runs monthly balance for local depression
  candidates. Its separately bounded outlet-spill correction is a hydraulic
  topology repair, not physical major-trunk or valley erosion.
- Preserved coarse waterbodies still require bathymetric refinement before their
  inlet, lake-crossing, and outlet geometry becomes physical.
- The pass stabilizes one selected basin, not the entire planet at refined
  resolution.
- Earth-derived receiver-change and local-depression distributions are not yet
  calibrated.
