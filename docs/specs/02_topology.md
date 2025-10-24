# Stage 1 – Geometry & Topology Layer Specification

## Responsibilities
- Provide grid coordinate systems for sphere, cylinder, torus.
- Supply neighbor iteration, distances, and cell areas for later stages.
- Maintain multi-resolution hierarchies with consistent aggregation.

## Interfaces & Data Layout
- `class Topology` (abstract):
  - `shape`: `(height, width)`.
  - `cell_area(i, j) -> float` (pulls from precomputed area buffer).
  - `neighbors(i, j, mode="D8") -> NeighborView`: returns struct with up to 8 neighbor indices and weights stored in contiguous arrays for SIMD/GPU friendliness.
  - `distance(coord_a, coord_b) -> float` using precomputed trig tables where possible.
  - `wrap(coord) -> (i, j)` enforcing topology without branches (bitmask operations for torus).
  - `to_xyz(i, j) -> np.ndarray`: 3D unit vector (for sphere, uses cached sin/cos lat).
  - `child_mapping(level)`: mapping to downsampled grids.
  - All rasters are row-major float32 with 64-byte alignment; vector fields use SoA (separate buffers per component).

### Implementations
1. **SphereTopology**
   - Equirectangular projection (lon-lat).
   - Wrap around longitude, clamp at poles.
   - Cell area precomputed using `R^2 * Δλ * (sin φ2 - sin φ1)`; store arrays of `sinφ`, `cosφ`, `Δλ` to avoid recomputation.
   - D8 neighbor offsets stored per row to account for meridian convergence; accessible as small fixed-size tables.
2. **CylinderTopology**
   - Wrap x, open y.
   - Area constant per cell.
3. **TorusTopology**
   - Wrap in both axes.
   - True regular grid; distance computed via wrap-around.

## Multi-resolution
- Precompute pyramid levels using area-weighted averaging (summed-area tables or wavelet pyramids for O(1) queries).
- Provide `ResolutionSet` storing:
  - `levels`: list of `GridInfo(height, width, scale_factor)`.
  - `project(level_from, level_to, data)` functions (both downsample and upsample) operating on contiguous buffers without copies.

## Data Products
- `TopologyMetadata`:
  - type, parameters, resolutions, area totals, coordinate bounds.
- `CoordinateArrays`:
  - For native resolution store `lon`, `lat`, `xyz`.

## Performance
- Topology precomputation runs in Rust with SIMD (fallback NumPy) to avoid Python loops.
- Neighbor tables exported as small constant buffers friendly to GPU kernels.
- Precomputation must complete <0.5 s for 8 k grid.
- Provide optional acceleration via Rust `pyo3` for neighbor generation when hydrology requires repeated calls.
- Alignment checks ensure buffers suitable for SIMD/GPU access (≥64-byte boundaries).

## Logging
- Upon initialization log: topology type, grid size, resolution count, total area, approx mean cell area.
- For each stage request (e.g., `neighbors`), instrumentation counts calls to profile hotspots.

## Testing
- Neighbor tests on sample coordinates verifying wrap behavior and counts.
- Distance symmetry checks (`distance(a,b) == distance(b,a)` within tolerance).
- Area sum matches expected surface (e.g., 4πR² for sphere within <0.1%).
- Multi-resolution projection conserves total quantity (sum weighted by area).
