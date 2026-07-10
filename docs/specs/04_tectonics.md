# Stage 3: Kinematic Plates And Crustal Provinces

## Status

V5 implements the first canonical cubed-sphere tectonic snapshot. It is the
initial state for the later event-driven geological history in Decision 009,
not yet a simulation of hundreds of millions of years of plate evolution.
The rectangular V4 path remains available only for downstream migration and
legacy comparisons.

## Inputs

The stage depends on persisted Stage 1 geometry:

- Unit XYZ cell centers.
- Exact spherical cell areas.
- Canonical D4 global neighbor IDs.
- Deterministic run seed and tectonic configuration.

No face-local wrap flags are accepted by the cubed-sphere kernel. Every
boundary calculation follows global neighbor IDs, including links between
cube faces.

`time_steps`, `time_step`, `wrap_x`, and `wrap_y` are rectangular-reference
controls and are rejected when explicitly supplied to a cubed-sphere run.
Plate count is limited to at most one plate per four cells and 4096 plates so
every requested plate can own resolvable state.

## Current Spherical Model

1. Sample approximately separated plate seeds uniformly on the sphere.
2. Apply a deterministic smooth spherical coordinate warp so Voronoi
   boundaries are curved without introducing projection seams.
3. Run area-weighted spherical Lloyd relaxation, preserve each plate's largest
   component, and deterministically absorb disconnected raster fragments into
   adjacent plates so every plate remains connected.
4. Generate continental and oceanic crustal provinces from an independent
   multi-scale spherical field. Continents are not plates.
5. Assign a rigid angular velocity to each plate, apply configured drift bias
   as a latitude-correlated rotation tendency, and remove the area-weighted
   mean rotation.
6. Convert angular velocity to a global XYZ tangent velocity at every cell.
7. Resolve convergence, divergence, and shear on every unlike-plate D4 edge
   using great-circle tangent directions.
8. Estimate local ocean-continent subduction potential and condition boundary
   fields through topology-aware graph diffusion.

## Outputs

`PlateField` has shape `(6, n, n, 7)` on the canonical topology:

```text
0  plate_id
1  continental_crust_flag
2  crust_thickness_km
3  crust_density_g_cm3
4  velocity_x_global
5  velocity_y_global
6  velocity_z_global
```

Scalar `(6, n, n)` outputs are convergence, divergence, shear, subduction
potential, and hotspot potential. Metadata records the kinematic model,
velocity basis, plate count, area-weighted continental fraction, and summary
statistics.

The global velocity vector is tangent to the sphere. Downstream face-local
solvers must explicitly transform it into their local basis; they must not
interpret XYZ components as face rows and columns.

## Acceptance

- Fixed seed and configuration reproduce byte-identical fields.
- Every requested plate owns cells and is D4-connected.
- Plate velocities are tangent to their cell centers within float tolerance.
- Area-weighted continental fraction matches the requested target within one
  coarse cell of area.
- Continental provinces cross plate interiors and do not copy plate IDs.
- Boundary activity crosses all cube faces without a face-edge spike or gap.
- All output fields are finite; subduction and hotspot potential stay in
  `[0, 1]`.

## Deferred History

The current snapshot does not yet implement plate advection, rifting, ocean
opening and closure, persistent boundary segments, microplate birth/death,
terranes, sutures, basins, stratigraphy, or orogenic state. Those require the
history objects and checkpoint/replay loop from Decisions 009-011. The V5
fields are suitable initial conditions and geometry validation, not a claim of
complete tectonic realism.
