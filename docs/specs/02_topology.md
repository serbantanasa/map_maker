# Stage 1: Geometry And Topology

## Responsibilities

- Provide the canonical cubed-sphere simulation and storage grid.
- Supply global cell IDs, face-local coordinates, neighbors, geometry, and area.
- Make face boundaries ordinary topology links for every downstream process.
- Support conservative multi-resolution projection and regional refinement.
- Project canonical data into atlas and diagnostic map projections.

Equirectangular, cylindrical, and toroidal grids remain temporary diagnostics or
legacy test fixtures. They are not valid canonical planetary state.

## Canonical Layout

The first native implementation uses six equiangular cube faces in this order:

```text
0  +X
1  -X
2  +Y
3  -Y
4  +Z
5  -Z
```

Every face is an `n x n` row-major raster. The global cell ID is:

```text
face * n * n + row * n + col
```

Cell centers and corners are sampled uniformly in face angle over
`[-pi/4, +pi/4]`, projected onto a cube face, then normalized onto the unit
sphere. This equiangular mapping limits the center-to-corner spacing variation
while retaining raster-like storage.

## Native Interface

The Rust `topology_native` crate owns canonical geometry generation. Its coarse
C ABI fills preallocated buffers for:

- `xyz`: float32, shape `(6, n, n, 3)`.
- `longitude`: float64, shape `(6, n, n)`.
- `latitude`: float64, shape `(6, n, n)`.
- `cell_area`: float64 steradians, shape `(6, n, n)`.
- `neighbors_d4`: int32 global IDs, shape `(6, n, n, 4)`.

Python exposes these immutable buffers through `CubedSphereGrid`. Python must
not call native code per cell.

## Neighbor Semantics

D4 order is north, south, west, east in face-local raster coordinates. Crossing
a face edge returns the global ID of the adjoining face cell. Required
invariants:

- Every cell has four distinct valid neighbors.
- Every D4 edge is reciprocal.
- Face-edge links are indistinguishable from interior links to consumers.
- The total number of directed cross-face links is `24 * n`.
- Neighbor angular spacing remains inside the accepted distortion envelope.

D8 and vector-basis transport are deferred until a consumer requires them.
They must define cube-corner behavior and rotate tangent vectors between face
bases explicitly rather than inferring orientation from array layout.

## Cell Geometry

Cell area is the sum of two spherical-triangle areas from the four normalized
cell corners. On the unit sphere:

- Every cell area is positive.
- Each face sums to `4*pi/6` steradians.
- All six faces sum to `4*pi` steradians within floating-point tolerance.

Physical area is canonical steradian area multiplied by planet radius squared.

## Multi-Resolution Requirements

Later levels use face resolutions related by powers of two. Required operations:

- Parent/child global-ID mapping.
- Area-conserving restriction for extensive quantities.
- Area-weighted restriction for intensive quantities.
- Constrained interpolation for refinement priors.
- Cross-face halo exchange with no duplicated authoritative cells.

These operations remain pending; the current milestone establishes only the
single-resolution geometry and D4 graph they depend on.

## Diagnostics

```bash
uv run map-maker topology --face-resolution 96 --output-dir out/topology
```

The command writes:

- `cube_net.png`: global XYZ colors on an unrotated net, exposing orientation
  discontinuities at touching edges.
- `topology.json`: area totals, area distortion, and neighbor-angle statistics.

The cube net is a topology diagnostic, not an atlas projection.

## Acceptance

- Rust and Python area totals agree with `4*pi` within `1e-12` at tested sizes.
- XYZ cell centers are unit length within float32 tolerance.
- D4 neighbors are valid, unique, reciprocal, and cross every face boundary.
- Maximum/minimum cell area ratio is below `1.5` for the equiangular grid.
- Maximum/minimum D4 angular-spacing ratio is below `1.5`.
- The fixed cube-net diagnostic has no discontinuity on touching face edges.
- Native generation remains a coarse call and does not compile during import.

## Migration Boundary

The current tectonics, world-age, and erosion kernels still consume a
provisional two-dimensional grid. They must not be switched to cubed sphere by
flattening the six faces, because that would create false adjacency between face
rows. Migration requires those kernels to consume global IDs and topology-owned
neighbor tables or face-aware halo buffers.
