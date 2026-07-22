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

## Multi-Resolution Operations

Levels use integer refinement factors, with powers of two as the normal
production hierarchy. Refinement never changes a cell's face. Children are
ordered row-major within each parent, and global IDs remain local to a specific
resolution.

The native topology library provides coarse-buffer operations for:

- Fine-cell to parent and parent to child global-ID mappings.
- Restriction of extensive quantities by summing children.
- Restriction of intensive quantities by spherical-area-weighted mean.
- Constant prolongation that copies a parent prior to all children.
- Width-one float32/float64 D4 face halos sourced through canonical cross-face
  neighbors without dtype narrowing.

Restriction rejects non-finite values and non-positive or non-finite areas.
It also rejects accumulation overflow instead of publishing non-finite parent
fields.
Constant prolongation is a prior transfer, not the final constrained
interpolation needed for detailed terrain refinement. Halo interiors and edges
are populated, while the four cube-corner diagonals are `NaN` until D8 and
vector-basis transport define their semantics. Authoritative state remains the
unhaloed six-face field.

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
- Parent and child maps are inverse under every tested refinement factor.
- Extensive totals and area-weighted intensive integrals survive restriction.
- Restricted fine-cell areas match directly generated parent-cell areas.
- D4 halo edges exactly match canonical cross-face neighbor values.
- Native generation remains a coarse call and does not compile during import.

## Migration Boundary

The canonical tectonics, world-age, and geology kernels consume global IDs,
exact spherical areas, and topology-owned D4 neighbors. The legacy whole-grid
erosion stage remains on the provisional two-dimensional grid and explicitly
rejects cubed-sphere input. Canonical sparse fluvial processing instead uses
cubed-sphere child IDs and spherical edge lengths in `basin_erosion`. Neither
path may flatten the six faces, because that would create false adjacency
between face rows.
