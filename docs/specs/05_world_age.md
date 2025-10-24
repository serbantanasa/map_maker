# Stage 4 – World Age & Thermal Adjustments

## Purpose
- Modify tectonic outputs based on planetary age to influence crust thickness, elevation baseline, and hotspot frequency.

## Inputs
- `PlateField`, `BoundaryField`, `HotspotMap`.
- Config: `world_age` (in Ga), `thermal_decay_half_life`, `hotspot_scale`, `isostasy_factor`, `radiogenic_heat_scale` (proxy for uranium/thorium abundance).

## Procedure (Rust SIMD implementation)
1. **Thermal Decay & Heat Budget**
   - Compute normalized age factor `f = exp(-world_age / half_life)` using vectorized math over SoA buffers.
   - Scale convective vigor by `radiogenic_heat_scale` (higher heat → faster plate speeds, more hotspots) and feed that back to plate velocities.
   - Update crust thickness in-place for continental vs oceanic plates using cache-friendly strided loops.
2. **Isostatic Compensation**
   - Apply Airy isostasy model: thicker crust -> higher elevation baseline.
   - Generate `IsostaticOffset` field in the shared arena (float32).
3. **Mantle Plumes & Hotspots**
   - Rescale hotspot density by `hotspot_scale * (1 - f) * radiogenic_heat_scale`.
   - Generate stochastic hotspot events (Poisson process) entirely in Rust with deterministic RNG; store events in Arrow table (SoA) for zero-copy transfer.
4. **Dynamic Uplift/Subsidence**
   - Combine boundary convergence and thermal contraction to produce `UpliftRate` and `SubsidenceRate` fields; keep SoA layout for vectorization.
5. **Outputs**
   - `CrustThickness`, `IsostaticOffset`, `HotspotEvents`, `UpliftRate`, `SubsidenceRate`.

## Performance
- Entire stage runs in Rust with SIMD intrinsics and Rayon parallelism; target runtime <0.2 s for 8 k grid.
- Python orchestrator only sees handles; no NumPy large-array manipulation.
- Hotspot sampling uses Rust RNG; Python parity tests call into the same library.

## Logging
- Average crust thickness, uplift/subsidence mean/std (logged via staging API) plus convective vigor scalar.
- Number of hotspots, total uplift mass, heat budget parameters.
 - Produce debug CSV/JSON for events if profiling enabled.

## Testing
- Verify older worlds have thinner oceanic crust and more hotspots.
- Ensure isostatic offset integrates to zero mean when expected (precision tests using area weights).
- Check deterministic hotspot generation under fixed seed across Rust/Python boundaries.
