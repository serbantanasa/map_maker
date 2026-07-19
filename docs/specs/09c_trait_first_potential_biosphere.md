# Trait-First Potential Biosphere

## Status

Implemented with an Earth-amplitude calibration that passes the coarse
six-seed screen and canonical face-128 check. This stage follows the
environmental resource envelope and precedes Earth plant functional types,
vegetation mixtures, biome labels, and vegetation feedback.

## Interpretation

The stage estimates the equilibrium potential of a terrestrial, carbon-based,
photosynthetic producer community if suitable life has colonized a cell. It
does not simulate abiogenesis, evolutionary history, species, competition,
succession, disturbance history, consumers, or actual present-day vegetation.

The distinction is required for non-Earth scenarios. Similar environmental
pressures need not produce identical organisms. Continuous outputs are
conditioning evidence for later evolutionary or functional models, not a claim
of universal biological convergence.

## Inputs

- Monthly and annual bounded terrestrial primary-energy potential from 15b0.
- Monthly thermal and liquid-water opportunity.
- Monthly surface temperature and soil saturation.
- Terrestrial surface and nutrient support.
- Soil depth, regolith depth, salinity, hydric fraction, and confidence.
- Atmospheric/environmental profile confidence.
- Canonical land/ocean state and cell area.

## Canonical Outputs

- `MonthlyPotentialNPPKgCM2`.
- `AnnualPotentialNPPKgCM2`.
- `PotentialVegetationCoverFraction`.
- `PotentialStandingBiomassKgCM2`.
- `GrowingSeasonFraction`.
- `ProductivitySeasonalityIndex`.
- `DroughtAdaptationPressure`.
- `ColdAdaptationPressure`.
- `HeatAdaptationPressure`.
- `WaterloggingAdaptationPressure`.
- `SalinityAdaptationPressure`.
- `PotentialWoodyAllocationTrait`.
- `PotentialResourceConservativeTrait`.
- `PotentialRootingDepthM`.
- `PotentialCanopyHeightM`.
- `PotentialLeafAreaIndex`.
- `PotentialFuelContinuityIndex`.
- `PotentialBiosphereConfidence`.
- `PotentialBiosphereMetadata`.

NPP is derived from the already bounded 15b0 chemical-energy proxy through a
configured energy-per-carbon conversion. This makes NPP auditable and prevents
15b1 from creating productivity absent from 15b0. The Earthlike configuration
retains `39.9 MJ/kg C` rather than treating chemical energy as a calibration
knob; amplitude is calibrated through the upstream photosynthetic efficiency.

Standing biomass is annual NPP times a bounded residence time. The V2 response
combines woody structure, resource-conservative strategy, and slower turnover
under low productivity, then applies a nonzero residence baseline. These terms
remain explicit configuration controls. They reduce seed-to-seed amplification
without creating biomass where NPP is zero or exceeding the configured standing-
biomass ceiling.

## Pressure And Trait Semantics

Adaptation-pressure fields describe environmental selection requirements:
drought, cold, heat, waterlogging, and salinity. They are not organism traits.

Potential community traits describe one provisional structurally plausible
response:

- woody allocation increases with sustained productivity, cover, and growing
  season and decreases with strong drought, salinity, and waterlogging;
- resource conservation increases under drought, cold, salinity, and nutrient
  scarcity;
- roots remain bounded by regolith and a configured physiological maximum;
- canopy, leaf area, biomass, and fuel continuity remain bounded by available
  productivity and supported land fraction.

These relationships are causal priors for world generation. Calibration should
use distributions and trait-environment relationships rather than direct biome
lookup tables. The [TRY Plant Trait Database](https://www.try-db.org/) is a
future calibration source. The
[Community Terrestrial Systems Model technical note](https://escomp.github.io/CTSM/tech_note/Introduction/CLM50_Tech_Note_Introduction.html)
is a reference for keeping phenology, allocation, turnover, and plant hydraulics
as distinct processes.

The current Earthlike controls use a `0.043` peak PAR-to-chemical conversion
efficiency in 15b0 and `0.5-50 year` biomass residence bounds in 15b1. The
residence response uses a `0.10` baseline and weights woody structure,
resource-conservative strategy, and low productivity by `0.60`, `0.40`, and
`2.50`. These are profile-calibrated defaults, not universal biological laws;
non-Earth profiles must be validated independently.

## Hard Gates

- Monthly NPP is nonnegative and cannot exceed the 15b0 energy-to-carbon bound.
- Annual NPP reproduces the monthly sum.
- Fractions, normalized traits, pressures, seasonality, fuel continuity, and
  confidence remain in `[0, 1]`.
- Rooting depth is nonnegative and no greater than regolith depth or the
  configured maximum.
- Canopy height, leaf area, and biomass remain within configured maxima.
- Terrestrial cover, biomass, NPP, morphology, and fuel are zero over ocean.

## Deferred

- Species, populations, evolution, migration, and ecological succession.
- Functional-type fractions and familiar biome labels (15b2).
- Fire occurrence, grazing, consumers, and trophic structure.
- Ocean productivity and chemosynthesis.
- Actual vegetation history and the one bounded soil/hydrology feedback pass.
- Calibrated snowball, hothouse, high-productivity, and alien-biochemistry
  responses.
