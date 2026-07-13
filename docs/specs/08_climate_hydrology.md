# Seasonal Climate And Hydrology Boundary

## Status

Canonical seasonal climate V1 is implemented on the cubed sphere. Depression-aware
hydrology remains the next stage. Climate supplies monthly runoff potential but
does not route water, fill lakes, create rivers, or erode channels.

## Scientific Intent

The climate pass must make temperature, prevailing wind, precipitation, snow,
and runoff respond causally to orbital forcing, latitude, land/ocean thermal
contrast, elevation, circulation, and terrain. It targets structurally plausible
monthly climatology, not research-grade atmospheric dynamics or daily weather.

## Inputs

- `planet`: monthly top-of-atmosphere insolation and solar declination.
- `elevation`: pre-erosion bedrock and broad relief.
- `world_age`: provisional land/ocean mask.
- `geometry`: consumed through the dependency graph as canonical cell area,
  latitude, XYZ unit vectors, and D4 neighbors.

The stage creates `ClimateOrographyM` by smoothing only within land or ocean
domains over five neighbor passes. Ocean bathymetry becomes zero atmospheric
surface height; land orography is nonnegative. This prevents abyssal depth from
becoming a false coastal atmospheric wall while preserving broad mountain belts.

## Computational Approximation

### Temperature

1. Linearized outgoing-longwave energy balance consumes monthly insolation,
   land/ocean albedo, and a zero-integral meridional heat-transport tendency.
2. Land and ocean use separate thermal response rates, producing greater land
   seasonality and ocean lag.
3. Neighbor exchange represents unresolved atmospheric heat transport.
4. Positive climate orography applies a configurable environmental lapse rate.
5. Multiple climatological years are integrated to a periodic seasonal state.

### Wind

1. Smooth latitude-dependent Hadley, Ferrel, and polar surface tendencies provide
   trade winds, mid-latitude westerlies, and polar easterlies.
2. Circulation bands migrate with solar declination.
3. Monthly thermal gradients add pressure-gradient and Coriolis-deflected flow.
4. Broad orography steers upslope flow.
5. Wind is stored as a global XYZ tangent vector plus speed, avoiding cube-face
   component discontinuities.

### Moisture And Orography

1. Warm ocean cells evaporate into a persistent atmospheric moisture column.
2. Each transport step combines wind-directed advection, lateral synoptic mixing,
   and retained local moisture.
   The configured step count is defined at face-128 and scales with face
   resolution so transport distance does not change merely because detail changes.
3. Condensation responds to column loading, broad convergence, upwind ascent, and
   leeward descent. Rainout is rate-limited so a single coarse coastal cell cannot
   remove an entire air mass.
4. Ocean and land condensation thresholds differ so moisture can be exported from
   oceans and penetrate continents.
5. A conservative graph mixing pass turns transport-cell events into a monthly
   climatology footprint while preserving area-weighted precipitation totals.

### Snow, Evaporation, And Runoff Potential

- Precipitation partitions smoothly into rain and snow from monthly temperature.
- A bounded multi-year snow store produces snowfall, melt, and snow-water
  equivalent fields; explicit glaciers and ice sheets are not yet modeled.
- Ocean evaporation and provisional land evaporation feed the moisture cycle.
- Runoff potential removes bounded evaporation and applies relief/storm runoff
  tendencies. It is a hydrology input, not discharge.

## Outputs

Monthly `(12, 6, n, n)` float32 fields:

- `MonthlySurfaceTemperatureC`
- `MonthlyWindSpeedMps`
- `MonthlyPrecipitationMm`
- `MonthlyRelativeHumidity`
- `MonthlySnowfallMm`
- `MonthlySnowmeltMm`
- `MonthlySnowWaterEquivalentMm`
- `MonthlyEvaporationMm`
- `MonthlyRunoffPotentialMm`

`MonthlyWindVectorXYZMps` adds a final length-three component. Annual fields are
`AnnualMeanTemperatureC`, `AnnualPrecipitationMm`, and `AnnualAridityIndex`.
`ClimateOrographyM` and `ClimateMetadata` preserve the effective driver and model
contract for inspection and surrogate training.

## Validation Gates

- Monthly means/sums exactly reconstruct annual temperature and precipitation.
- Wind vectors remain tangent to the sphere and their norms match stored speed.
- Northern and southern mid-latitude temperature peaks are about six months apart.
- Comparable mid-latitude oceans have lower seasonal range than land.
- Temperature residual after latitude correction decreases with orography.
- Precipitation changes when orography changes and remains continuous across cube
  faces.
- Humidity, precipitation, snow, evaporation, and runoff remain finite and
  physically bounded; runoff cannot exceed rain plus available melt.
- Deterministic inputs produce byte-identical, cacheable artifacts.
- Earth-like defaults remain inside broad temperature and water-cycle plausibility
  bounds. These are realism gates, not Earth calibration claims.

## Current Limits

- Prescribed circulation tendencies replace a primitive-equation atmosphere.
- Ocean currents, sea ice, clouds, vegetation feedback, glaciers, monsoonal land
  pressure, and transient storms are simplified or absent.
- Land evaporation precedes explicit soils and is therefore provisional.
- Monthly precipitation mixing is a deliberate unresolved-weather approximation.
- All monthly fields are currently resident in memory during the stage. Face-128
  is small, but face-1024 requires several gigabytes; month/tile streaming and
  chunked persistence are required before very large global runs.

## Hydrology Handoff

The next canonical stage consumes monthly runoff potential, precipitation,
snowmelt, evaporation, temperature, climate orography, final terrain, lithology,
and topology. It must implement the depression/lake/spill/breach and vector river
contracts in Decisions 004, 005, and 014. Climate does not pre-draw river lines.
