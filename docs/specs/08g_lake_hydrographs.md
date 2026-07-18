# Final Lake To Reach Hydrographs

## Status

Implemented, provisional calibration. The `lake_hydrographs` stage runs after
`surface_water_final` and publishes `LakeCoupledRiverReachCatalog`,
`LakeHydrographAdjustmentCatalog`, and `LakeHydrographMetadata`.

## Contract

The inherited reach hydrograph already contains the runoff that later feeds
refined lake candidates. Final lake overflow therefore replaces that source
component downstream; it is never added on top of inherited discharge.

For each terminal candidate network, the stage:

1. Sums monthly direct runoff from every upstream candidate catchment.
2. Subtracts that source hydrograph from solved terminal overflow.
3. Routes the resulting monthly delta from the final spill receiver to a fine
   channel, a preserved coarse handoff, or an allowed outside terminal.
4. Applies the delta through the downstream reach DAG.
5. Preserves separate pre-lake, coupled-entry, and coupled-exit hydrographs.

If a negative fine-scale adjustment reaches a coarse reach before the inherited
hydrograph contains that tributary flow, it moves downstream to the first
nonnegative application point. Every remap is persisted; no adjustment is
silently clamped.

## Hard Gates

- Candidate and reach graphs are acyclic and closed over known identifiers.
- Final entry and exit discharge is finite and nonnegative in every month.
- Candidate-network direct runoff equals terminal overflow plus evaporation,
  seepage, and periodic storage change.
- Every downstream discharge decrease is attributable to registered coarse
  storage or a final refined surface-water network.

## Current Boundary

The stage couples monthly volume and loss at reach scale. It does not yet solve
travel time within a reach, stratified lake temperature, groundwater exchange,
or event-scale flood waves. Those processes require finer temporal and spatial
models.
