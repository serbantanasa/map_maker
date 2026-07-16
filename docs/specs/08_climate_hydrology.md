# Seasonal Climate And Hydrology Boundary

## Status

Canonical seasonal climate V1 and the first depression-aware modern hydrology
pass are implemented on the cubed sphere. Climate supplies monthly runoff
potential but does not pre-route water or draw rivers. Hydrology consumes that
handoff and writes persistent lakes, breaches, drainage topology, discharge,
basins, and vector river reaches.

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

## Hydrology Pass 1

### Inputs

- Climate: monthly runoff potential and evaporation plus annual aridity index.
- Elevation: pre-erosion bedrock elevation and terrain relief.
- Geology: rock strength and sediment accommodation.
- World age: provisional ocean mask.
- Planet and geometry: physical radius, cell areas, XYZ unit vectors, and global
  cubed-sphere D4 neighbors.

### Depression And Lake Model

1. A deterministic global priority flood begins from every ocean cell. It writes
   the minimum spill elevation and an ocean-directed parent for every land cell.
2. Connected land below its spill surface becomes a registered depression rather
   than being silently flattened.
3. The whole upstream catchment contributes monthly runoff. Lake evaporation and
   geology-modulated seepage are subtracted to estimate storage balance and fill
   time.
4. Systems with shallow subgrid area-weighted mean depth may remain wetlands.
   Water-limited systems remain closed; sustained positive balance produces an
   open outlet. Open systems retain a visible lake while passing accumulated
   discharge downstream.
5. Depression nodes are evaluated upstream-to-downstream. Net outflow from an
   upstream open or breached basin contributes to the next depression's monthly
   inflow and can turn a previously terminal lake into a fill-spill lake chain.
6. A coarse cell is a container, not an atomic land-cover label. `TerrainReliefM`
   defines a uniform unresolved elevation distribution around the coarse bedrock
   elevation. Intersecting a lake level with that distribution produces continuous
   water area and integrated volume. A connected-basin fraction prevents all
   hypsometrically low subgrid terrain from being assigned to one water body.
7. Open lakes use the spill elevation. Closed lakes solve a deterministic
   area-versus-level curve until evaporation and seepage balance inflow. Each
   participating cell receives a `LakeFraction` or `WetlandFraction` in `[0, 1]`;
   the whole depression remains part of the drainage basin.
8. Sustained overflow, available head, discharge, weak rock, and low accommodation
   produce a breach score. Accepted breaches carve a short coarse outlet path,
   record gorge incision and a sediment pulse, and trigger a second priority flood.

### Drainage And Rivers

- Preserved closed water bodies become explicit registered sinks. Open lakes use
  the earliest valid downstream edge in the final flood order, preventing lake
  rewiring from introducing cycles.
- A topological sort is a hard gate. The receiver graph is rejected if any land
  cell is not covered or if an accidental cycle exists.
- Cell area and all twelve monthly runoff fields accumulate once through that DAG.
  Registered open-water evaporation and seepage are removed at lake outlets
  before discharge continues downstream. Every land cell receives a basin ID and
  propagated sink type.
- River cells are selected from physical discharge and contributing area.
  Provisional BFS routing inside a preserved depression is never emitted as river
  geometry, including in mixed shoreline cells. Discharge still traverses the
  drainage graph; refined inlet, lake-crossing, and outlet geometry is deferred.
- Junction-to-junction reaches form the canonical river graph. Exact cubed-sphere
  cell paths are retained for refinement and a two-pass spherical corner-cutting
  generalization writes smooth unit-XYZ polylines for rendering.
- Zero-width hydrologic connector reaches carry routed topology and flux across
  preserved depression or waterbody support cells where the coarse pass cannot
  justify open-channel geometry. They have no physical channel dimensions,
  corridor support, or incision and must be replaced by resolved local geometry
  before any channel process is applied there. Ordinary below-threshold land
  paths remain unresolved and fail the source-to-sink readiness gate.
- Reach attributes include monthly discharge and velocity, slope, stream power,
  Strahler order, estimated width/depth, valley and floodplain width, meandering,
  braiding, incision, sediment load, bed material, and morphology class.

### Persistent Outputs

Raster support fields include depression/lake IDs and classes, fractional lake
and wetland coverage, potential fill depth, hydrologic elevation, breach incision,
receiver IDs, tangent flow vectors, slope, contributing area, monthly and mean
discharge, velocity, stream power, basin and sink IDs, river corridor, and
floodplain potential.

Arrow products are `DepressionCatalog`, separate `LakeCatalog` and
`WetlandCatalog` products, `WaterBodyCellCatalog`, `BreachCatalog`, `BasinCatalog`,
`DrainageGraph`, and `RiverReachCatalog`. `WaterBodyCellCatalog` is the sparse
cell-to-waterbody relation and records covered area, class, and both fractions.
`HydrologyMetadata` records physical controls, counts, area-weighted diagnostics,
conservation error, and artifact semantics.

### Hard Gates

- The land receiver graph is acyclic and covers every land cell.
- Every terminal drains to ocean or a registered closed water body.
- Contributing area is nondecreasing downstream. Monthly discharge is
  nondecreasing except at registered open-water loss nodes.
- Source runoff agrees with terminal discharge/inflow plus registered open-water
  losses within floating-point tolerance.
- Flow vectors are tangent unit vectors on routed cells.
- Reach IDs and downstream references are valid and the reach graph is acyclic.
- Every terminal reach ends at ocean or a registered lake, wetland, or
  endorheic sink; unresolved terminals block downstream processing.
- Hydrologic connectors have zero channel width, depth, local velocity, stream
  power, and incision.
- Exact and smoothed reach geometries share endpoints and remain on the unit sphere.
- Lake and wetland fractions are finite, bounded by `[0, 1]`, mutually exclusive,
  and reproduce catalog water area when weighted by physical cell area.
- Identical inputs produce byte-identical arrays and Arrow tables.

### Current Hydrology Limits

- The support receiver graph uses D4 neighbors. Smoothed vectors remove rendering
  stair steps, but a future L2 routing upgrade should evaluate diagonal or
  facet-based flow without breaking cube-face topology.
- Priority flood identifies spill-connected depression components, but an explicit
  nested fill-spill-merge hierarchy and time-ordered lake-chain events are not yet
  implemented.
- Land evaporation is provisional and not a dedicated open-water potential
  evaporation field. Lake extent and endorheic statistics therefore remain
  calibration targets.
- The connected-basin fraction is a coarse subgrid approximation. Hierarchical
  refinement must replace it with resolved local hypsometry and approach atomic
  coverage only where the refined cell is genuinely uniform.
- Depression discovery still begins from coarse mean elevation, and the current
  sparse membership product has at most one resolved waterbody per cell. It does
  not yet discover several independent minor lakes inside one coarse cell.
  Refinement may instantiate those from explicitly unresolved fractional water
  while preserving parent area and storage.
- Fractional area is now Earth-plausible in the face-128 audit, but lake depth
  and total volume remain high. Basin-age sediment infill and explicit
  bathymetric refinement are required before lake volume is calibrated.
- Groundwater, infiltration, losing/intermittent reaches, glaciers, engineered
  channels, and explicit delta distributary growth are absent or represented only
  by coarse reach attributes.
- Breach erosion is a basin-scale coarse incision event. The sparse selected-basin
  erosion and Hydrology Pass 2 now stabilize one refined basin, but detailed
  gorge evolution, global sediment feedback, and monthly local lake balance
  remain future work.
- Current statistical thresholds are provisional. Multi-seed conditional
  validation against basin, lake, and river distributions is required before the
  Earth-like default is considered calibrated.
- Resolution stability is not yet achieved: face-128 audits produce materially
  more closed-drainage land than face-64. Fine hydrology must inherit accepted
  coarse trunk connectivity and flux before very high-resolution worlds are
  considered trustworthy.
