# Sparse Trunk Profiles And Prospective Sediment

## Status

Implemented provisional constraint pass following `basin_refinement`. Despite
the historical stage name `basin_erosion`, this is not a physical terrain
erosion pass. Decision 048 defines the resolution boundary.

## Objective

Turn an accepted sparse reach graph into junction-consistent major-trunk vector
profiles and a conservative prospective sediment budget. Preserve topology,
physical channel dimensions, source-to-sink accounting, and regional-generator
constraints without widening a river to cell size or claiming that unresolved
terrain has already been excavated.

At canonical face 128, global cells average about `5,190 km2` and `72 km`
across. Factor-16 sparse children still average about `20.3 km2` and `4.5 km`
across. They can carry an inherited trunk vector but cannot represent ordinary
river banks, tributaries, cross-valley profiles, or evolved floodplains as
raster terrain.

## Native Solver

The Rust `fluvial_native` kernel receives compact structure-of-arrays inputs for
selected child cells, reaches, and sparse memberships. It builds:

- a physical node DAG from consecutive inherited channel memberships;
- the complete reach DAG, including zero-width hydrologic connectors.

The physical DAG supplies a candidate downstream profile. The reach DAG routes
the sediment volume that would result if regional generation realized that
profile. Python performs orchestration, Arrow conversion, independent audits,
persistence, and visualization.

## Candidate Profile

Fine terrain is an unresolved prior, not an accepted channel bed. The solver
visits physical nodes in deterministic topological order and constructs the
least-incision profile satisfying the configured minimum downstream grade.
Multiple incoming reaches share one node and therefore one candidate elevation
at a confluence.

Terrain, candidate bed, prospective incision depth, reach-end elevation, and
grade persist as float64. Path-order gaps mark discontinuities across
process-excluded support. No profile edge crosses a gap. Connectors retain
topology and flux but never enter the physical profile.

The profile is a vector constraint for regional refinement. It is not written
into `terrain_elevation_after_m` or the Hydrology Pass 2 routing raster.

## Prospective Budgets

Every centerline membership computes a candidate channel prism:

```text
prospective_excavation = channel_width * represented_length * candidate_depth
```

The prospective solid volume is routed through the reach graph. Bounded
floodplain retention, inland-terminal storage, and ocean export remain useful
capacity and provenance constraints. Their conservation identity is:

```text
prospective_excavation
  = prospective_floodplain_deposition
  + prospective_terminal_deposition
  + prospective_ocean_export
```

These volumes are not geological history and are not passed to soils as recent
erosion or deposition. Bank carving is fixed to zero in this stage. The native
bank-carve capability remains tested as a primitive for a future regional
process stage, where terrain resolution and lateral support are adequate.

## Published Terrain Contract

`ErodedBasinCellCatalog` retains its historical artifact name for pipeline
compatibility, but publishes an explicit split:

- `prospective_channel_excavation_volume_m3`;
- `prospective_floodplain_deposition_volume_m3`;
- `prospective_maximum_channel_incision_m`;
- `applied_terrain_erosion_volume_m3`, exactly zero;
- `applied_terrain_deposition_volume_m3`, exactly zero;
- `terrain_mean_delta_m`, exactly zero;
- `terrain_elevation_after_m`, exactly equal to the terrain prior.

The parent catalog makes the same prospective/applied distinction. Hydrology
Pass 2 preserves inherited vector receiver constraints over unchanged terrain.
Surface materials consume only applied fields.

The separately bounded lake-outlet spill correction may later alter its local
routing support under its own volume and scope gates. It is a hydraulic
topology repair, not fluvial trunk or valley erosion.

## Determinism

- Cell and reach identifiers define all map and queue tie breaks.
- Physical and reach DAG queues use ascending stable identifiers.
- Sparse outputs are sorted by reach/path/cell or fine-cell identity.
- The native binary fingerprint participates in the stage cache key.

## Hard Gates

- The inherited reach graph is source-to-sink ready and acyclic.
- Shared physical junctions have one candidate profile elevation.
- Every candidate physical edge meets the configured downstream grade.
- Candidate bed elevation never exceeds the terrain prior.
- Connectors have no profile, local sediment production, or deposition.
- No prospective process volume enters a preserved-depression parent.
- Candidate prism volumes equal width times length times depth.
- Prospective sediment is conserved across reaches, cells, and parents.
- Bank erosion is exactly zero.
- Applied child and parent process volumes are exactly zero.
- Cell and parent terrain mean changes are exactly zero.
- Hydrology Pass 2 does not substitute candidate bed elevations into raster
  terrain.

## Validation

Native tests retain coverage of profile construction, confluences, connectors,
sediment transfer, and bank-carve mechanics as a reusable lower-level
capability. Pipeline tests independently recompute grades, candidate channel
volume, junction equality, reach balances, cell and parent prospective sums,
connector emptiness, vector slopes, deterministic output, cache reuse, and the
zero-applied-terrain contract.

The Earthlike profile rejects a prospective trunk depth above `2,000 m`. This
is a fail-fast diagnostic for invalid upstream lake or graph boundaries, not a
calibration target and not permission to apply that depth to coarse terrain.

## Current Limits

- Only inherited major trunks are realized. Tributaries below the global river
  threshold are absent.
- The candidate profile can expose large conditioning depths where an
  unresolved terrain prior rises along an inherited route. Such depths are
  constraints to investigate during regional realization, not accepted cuts.
- Floodplain retention is a bounded capacity approximation, not a calibrated
  settling, competence, grain-size, avulsion, or delta model.
- Regional refinement must generate a finer DEM, discover tributaries, evolve
  valleys, and decide how much candidate incision and deposition is physically
  realized from lithology, regolith, discharge history, uplift, and time.
