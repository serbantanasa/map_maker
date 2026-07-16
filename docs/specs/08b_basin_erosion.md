# Sparse Selected-Basin Erosion

## Status

Implemented provisional process pass following `basin_refinement`.

## Objective

Turn the accepted sparse reach and corridor contract into a physically
accounted first erosion state without widening subgrid rivers to cell size.
Solve one bed at confluences, cut only physical centerline area, route every
removed cubic metre to a floodplain, registered inland sink, or ocean terminal,
and restrict the resulting volume changes to coarse parents.

## Native Solver

The Rust `fluvial_native` kernel receives compact structure-of-arrays inputs for
selected child cells, reaches, and sparse memberships. It builds two DAGs:

- a physical node DAG from consecutive channel centerline memberships;
- the complete reach DAG, including zero-width connectors.

The physical DAG supplies bed elevations and incision volume. The complete DAG
routes sediment source to sink. Python performs orchestration, Arrow conversion,
independent conservation audits, persistence, and visualization.

## Profile Construction

Fine terrain is an unresolved prior, not an accepted channel bed. The solver
visits physical nodes in deterministic topological order and lowers a downstream
node only when needed to satisfy the configured minimum grade. Multiple incoming
reaches reference the same fine-cell node, so their confluence elevation is
identical by construction.

Terrain, bed elevation, incision depth, reach-end elevation, and realized grade
cross the native ABI and persist as float64. The emitted profile, rather than an
internal pre-serialization statistic, must still satisfy the minimum grade.

Membership path-order gaps mark physical discontinuities across process-excluded
support. No bed edge crosses such a gap. Connectors remain in the reach graph for
sediment transfer but never enter the physical DAG.

## Volume Budgets

Incision is calculated at every centerline membership from physical channel
width, represented in-cell length, and solved depth. Eroded volume accumulates
to the reach and fine cell. Floodplain deposition is allocated in proportion to
the already-conserved per-reach floodplain support area, including lateral
memberships, and is capped by support area times maximum deposition depth.

Sparse cell budgets are expanded onto the full selected child catalog only when
artifacts are published. Coarse-parent budgets are exact sums of child erosion
and deposition. Neither routing nor restriction changes total material.

## Determinism

- Cell and reach identifiers define all map and queue tie breaks.
- Physical and reach DAG queues use ascending stable identifiers.
- Sparse outputs are sorted by reach/path/cell or fine-cell identity.
- The native binary fingerprint participates in the stage cache key.

## Validation

Synthetic native tests cover a downstream terrain rise, a shared confluence,
and channel-to-connector-to-channel transport. CFFI tests also cover a grade
smaller than float32 precision at the profile elevation, connector transfer,
physical-component separation at a path gap, native record layout, and rejected
connector membership. Pipeline tests independently recompute emitted grade,
physical volume, junction equality, reach flow balances, profile-to-reach and
profile-to-cell allocations, cell-mean feedback, parent sums, connector
emptiness, source-to-sink conservation, independent-run determinism, and cache
reuse. A fixture with real connectors and process-excluded parents prevents
those gates from passing vacuously.

## Downstream Stabilization

The sparse Hydrology Pass 2 now consumes the volume-derived terrain and solved
channel beds, applies one bounded local reroute, and reports depression
stability. Calibration of profile depths and floodplain retention still requires
multi-seed distributions and Earth-derived benchmarks rather than tuning the
canonical seed alone.
