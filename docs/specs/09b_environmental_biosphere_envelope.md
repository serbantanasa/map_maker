# Environmental And Biosphere Resource Envelope

## Status

Implemented with provisional, uncalibrated biological response controls. The atmosphere substage precedes
climate; the biosphere-resource substage follows surface materials and initial
soils. This milestone publishes forcing and opportunity, not vegetation,
organisms, or biome labels.

## Principle

Earth is the first calibration profile, not a hard operating boundary.

Hard validation gates apply to physical and accounting invariants: finite and
nonnegative gas quantities, valid composition sums, pressure decreasing with
height in a hydrostatic atmosphere, bounded opportunity fractions, mutually
consistent annual/monthly aggregates, and no biological opportunity over open
ocean where the terrestrial output is claimed.

Earth-relative ranges are versioned diagnostics. Snowball, archipelago,
hothouse, high-productivity, and custom profiles may violate Earth diagnostics
without being rejected. A profile is a declared interpretation and benchmark
set, not an alternate physical law.

## Atmosphere Inputs And Outputs

Configured atmosphere state:

- mean sea-level pressure,
- dry oxygen, carbon-dioxide, and methane mixing ratios,
- mean atmospheric molar mass and reference temperature,
- reference greenhouse composition and CO2 doubling sensitivity,
- active validation profile.

Canonical outputs:

- `SurfacePressureKPa`,
- `OxygenPartialPressureKPa`,
- `CO2PartialPressurePa`,
- `MethanePartialPressurePa`,
- `AtmosphericCompositionCatalog`,
- `AtmosphereMetadata`.

V1 pressure follows a hydrostatic exponential approximation with scale height
derived from temperature, molar mass, and configured surface gravity. Ocean
floor elevation does not reduce atmospheric pressure below sea-level pressure.
The approximation is adequate for a first dense-atmosphere conditioning field;
vertical temperature structure and weather-scale pressure remain deferred.

CO2 greenhouse forcing uses a configurable logarithmic doubling response
relative to a declared reference concentration. Climate also retains its
explicit non-compositional offset for paleoclimate and experiment controls. The
two contributions are recorded separately in climate metadata.

## Biosphere Resource Inputs And Outputs

The post-soil resource envelope consumes monthly top-of-atmosphere insolation,
surface temperature, soil saturation and water fluxes, surface-material
support, active ice, and atmospheric state.

Canonical raw and derived outputs:

- `MonthlySurfacePARMJm2`,
- `MonthlyLiquidWaterOpportunity`,
- `MonthlyThermalOpportunity`,
- `MonthlyTerrestrialPrimaryEnergyPotentialMJm2`,
- `AnnualSurfacePARMJm2`,
- `AnnualTerrestrialPrimaryEnergyPotentialMJm2`,
- `CarbonSubstrateRelativeToReference`,
- `AerobicOxygenRelativeToReference`,
- `TerrestrialSurfaceSupportFraction`,
- `NutrientSupportIndex`,
- `EnvironmentalStressIndex`,
- `BiosphereEnvelopeConfidence`,
- `BiosphereEnvelopeMetadata`.

The raw light, water, temperature, gas, and surface fields are the durable
contract. The combined energy potential is a provisional Earth-photosynthesis
diagnostic and training target, not a universal habitability score. It may be
replaced or reinterpreted by later trait models without rerunning geology,
climate, hydrology, or soils.

Carbon-substrate support may exceed one relative to Earth. The combined energy
proxy remains bounded by PAR and the configured conversion efficiency. Aerobic support is
separate and is not multiplied into primary photosynthetic energy. High CO2
alone does not imply abundant or giant life.

## Trait-First Downstream Contract

Milestone 15b1 will map the raw envelope into continuous organism and ecosystem
traits. Earth plant functional types and familiar biome names are derived
views. They are not canonical physical state.

Future surrogate datasets must include a conditioning vector containing at
least planetary gravity, rotation, stellar forcing, atmospheric pressure and
composition, validation profile/version, and the biological response controls
used to derive targets.

## Known Capability Gaps

- Current planet controls still enforce an Earthlike dense-water-world range.
- Climate currently uses fixed albedo controls and a bounded temperature
  response rather than composition-dependent radiative transfer.
- Sea ice and continental ice sheets are not yet coupled climate components.
- L2 land/ocean state is atomic, limiting small-island and fractional-coast
  fidelity.
- The resource envelope is terrestrial and photosynthesis-oriented; oceanic
  productivity and chemosynthesis are deferred.
- Non-Earth profiles initially emit diagnostics and test the software contract;
  they are not claims of calibrated realism.

## Scientific Approximation References

- [NASA Planetary Spectrum Generator atmosphere documentation](https://psg.gsfc.nasa.gov/helpatm.php):
  hydrostatic exponential pressure and scale-height approximation.
- [IPCC Third Assessment Report, Chapter 6](https://www.ipcc.ch/site/assets/uploads/2018/03/TAR-06.pdf):
  logarithmic CO2 radiative-forcing approximation.
