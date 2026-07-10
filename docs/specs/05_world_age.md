# Stage 4: Age-Conditioned Crust State

## Status and purpose

This stage initializes a plausible present-day crust state from the canonical
cubed-sphere tectonic fields. It is not yet a geological history simulation.
It does not advance plates through billions of years, conserve mantle heat, or
solve flexure. Its purpose is narrower: provide deterministic, globally closed
inputs for the first elevation and surface-process stages while preserving the
causal direction from crust type and tectonic setting to terrain potential.

The rectangular implementation remains available for compatibility. The
cubed-sphere path is the canonical implementation.

## Inputs

- `PlateField`: plate ID, crust class, base thickness, density, and global XYZ
  tangent velocity.
- `BoundaryConvergence`, `BoundaryDivergence`, `BoundarySubduction`, and
  `BoundaryShear`.
- `HotspotMap`.
- Exact spherical cell areas and global D4 neighbor IDs from the geometry stage.
- `world_age` in Ga, `thermal_decay_half_life`, `hotspot_scale`,
  `isostasy_factor`, and `radiogenic_heat_scale`.

## Canonical approximation

1. Compute residual primordial heat as `exp(-ln(2) * age / half_life)` and combine it
   with the radiogenic scale to obtain an age-conditioned convective heat proxy.
   Older worlds therefore have less residual heat and, all else equal, fewer
   thermal-anomaly events. This stage does not feed the proxy back into plate velocities.
2. Adjust continental and oceanic base crust thickness separately. Oceanic crust
   becomes thinner as the heat proxy declines; cooled lithosphere becomes
   stiffer.
3. Compute an Airy-like buoyancy potential from thickness and density. Remove
   its exact spherical-area-weighted global mean so `IsostaticOffset` closes to
   zero over the planet.
4. Combine convergence, divergence, subduction, shear, and hotspot intensity
   into bounded uplift, subsidence, compression, extension, and stiffness
   proxies. D4 diffusion uses global neighbor IDs, crosses cube-face seams, and
   derives local diffusion weights from spherical cell area so its approximate
   angular width does not shrink with increasing resolution.
5. Measure area-derived angular graph distance from continental/oceanic crust transitions. The
   compatibility artifact `CoastalExposure` is currently this continental
   margin proximity, not exposure to a solved coastline.
6. Draw a global, resolution-independent Poisson count of tectonic thermal-anomaly
   events. Sample
   locations by hotspot intensity times exact cell area and store canonical
   global cell IDs plus decoded face, row, and column coordinates.

## Outputs and semantics

- `CrustThickness`: age-adjusted crust thickness proxy.
- `IsostaticOffset`: globally closed buoyancy/elevation potential.
- `UpliftRate` and `SubsidenceRate`: relative process-rate proxies.
- `TectonicCompression`, `TectonicExtension`, and `ShearMagnitude`.
- `LithosphereStiffness`: bounded relative stiffness proxy.
- `CoastalExposure`: compatibility name for continental-margin proximity.
- `BaseOceanMask`: compatibility name for oceanic-crust candidates. It is not a
  water mask; sea level and connected ocean basins remain unsolved.
- `HotspotEvents`: compatibility name for tectonic thermal-anomaly events, with
  global cell ID, face, row, column, strength, and `plume_factor`. Neither the
  event name nor `plume_factor` establishes a modeled mantle plume.
- `WorldAgeMetadata`: parameters, area-weighted diagnostics, model identifier,
  and explicit semantic labels for compatibility artifacts. Canonical metadata
  exposes `proto_ocean_area_fraction`; `water_fraction` survives only inside the
  shared native ABI and is not published by the spherical stage.

## Determinism and storage

Identical topology, tectonic artifacts, configuration, RNG seed, and native ABI
produce identical arrays and events. All large fields remain arena-backed and
are persisted by the normal stage dataset writer. Event coordinates are stable
within a fixed cubed-sphere resolution.

## Validation

- All spherical output fields have shape `(6, n, n)`, finite `float32` values,
  and deterministic contents.
- `IsostaticOffset` has near-zero spherical-area-weighted mean.
- Continental-margin distance and shear diffusion cross face boundaries through
  global D4 neighbors.
- Hotspot event IDs decode to valid face, row, and column coordinates.
- Event expectation depends on global area averages, not cell count, so changing
  resolution does not multiply event density.
- Older canonical runs have lower residual heat and convective-vigor diagnostics,
  thinner oceanic crust, greater mean stiffness, and no more thermal-anomaly events for a
  fixed deterministic random stream.
- Python validates dtype, shape, alignment, writability, and non-overlap before
  entering the native FFI.

## Known limitations

- This is structural initialization, not time-stepped crust production,
  subduction recycling, rifting, terrane accretion, or sediment history.
- `UpliftRate` and `SubsidenceRate` are relative model fields without calibrated
  physical units.
- Margin proximity is not a coastline, and oceanic crust is not necessarily
  submerged.
- Glaciation, true flexural isostasy, sea level, erosion, and sediment loading
  belong to later stages.
- The kernel is scalar Rust today. No SIMD or Rayon performance claim is made.
