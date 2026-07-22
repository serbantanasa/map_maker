# Seasonal Snow, Land Ice, And Sea Ice

## Status

Implemented, provisional calibration. The Rust-backed `cryosphere` stage runs
after atmospheric climate and before hydrology.

## Scientific Boundary

The model targets **visible persistent ice** for maps and water storage:

1. **Mountain glaciers** — cold upper slopes from `TerrainReliefM` peak cooling
   inside warmer coarse cells.
2. **Polar / near-permanent ice caps** — whole-cell accumulation when
   cell-mean surface temperature is cold enough, without requiring multi-km
   mean elevation on ~72 km tiles.
3. **Seasonal sea ice** — an ocean-temperature-driven thickness reservoir with
   winter growth, summer melt, and retained perennial concentration.

It does not solve glacier stress, full ice-sheet dynamics, ice streams, or
glacial landforms. Sea ice is thermodynamic only: it does not yet drift, raft,
form leads, or feed back into atmospheric heat and moisture transport.

Each coarse land cell contains separate seasonal-snow and glacier-ice water
equivalent. **All ice mass comes from climate precipitation** (as snow). There
is no synthetic precipitation floor and no free ice source. Snow age is
tracked as a mass-weighted age moment. Snow that survives long enough converts
to firn/ice; cold climates convert stored snow faster and may route a share of
**real** new snowfall directly into ice (partitioned from the same snowfall,
not added twice). Warm-season melt removes seasonal snow first and glacier ice
separately. Glacier melt participates in canonical runoff potential.

Hard audits reject (1) ice-reservoir mass-balance residuals and (2) firn/direct
ice inputs that exceed land snowfall.

Ocean cells maintain a separate sea-ice thickness state. Growth uses freezing
degree-months below the configurable seawater freezing point and slows as the
insulating ice thickens. Melt uses warm degree-months above a configurable
threshold. Fractional cover is derived continuously from thickness, avoiding a
binary coarse-cell ice mask. This state neither creates freshwater runoff nor
enters the land glacier budget.

`TerrainReliefM` is an unresolved relief prior, not summit elevation. A bounded
mountain fraction samples peak-cooled climate for alpine glaciers. Separately,
an ice-cap fraction grows when cell-mean temperature is well below freezing so
cold low-relief polar land can hold permanent ice.

## Ice Transfer

Ice above the configured activation storage transfers monthly toward the
lowest neighboring bedrock cell. Transfer conserves area-weighted water volume
across unequal cubed-sphere cells. Transfer into ocean is recorded as calving.
This provides bounded downslope spreading while leaving stress-driven flow for
a later model.

## Outputs

Monthly fields include canonical snowfall, seasonal snowmelt, snow-water
equivalent, firn conversion, glacier melt, glacier ice storage, and melt-aware
runoff potential, plus sea-ice concentration and thickness. Annual fields
include glacier mass balance, flow export and import, calving, glacier
sublimation, and fractional land-ice cover. Metadata reports minimum, maximum,
mean, perennial, and seasonal sea-ice ocean-area fractions.

## Validation

- Every storage and flux is finite and nonnegative where applicable.
- Per-cell ice storage change reconstructs firn input, ice melt, sublimation,
  flow export, and area-adjusted flow import.
- Global inland flow export and import conserve physical water volume.
- Snowmelt and glacier melt remain separate outputs and surrogate targets.
- Sea ice is zero on land, bounded to `[0, 1]` over ocean, and its thickness is
  finite, nonnegative, and capped.
- Identical inputs and controls produce byte-identical artifacts.

The `earth_cryosphere_v1` diagnostics use broad combined-hemisphere ocean-area
concentration envelopes: monthly minimum `0.02–0.08`, monthly maximum
`0.05–0.12`, and annual mean `0.035–0.09`. They are Earthlike-profile checks,
not universal limits; snowball, ice-free, and other climate profiles may lie
outside them intentionally. The reference is the NSIDC Sea Ice Index seasonal
climatology, with an explicit caveat that its standard extent metric uses a 15%
concentration threshold while these simulator KPIs are area-equivalent
concentrations.

## Deferred

- Stress-driven ice velocity, crevassing, surging, and basal sliding.
- Ice-sheet isostasy, shelves, sea-level feedback, and calibrated calving.
- Sea-ice advection, ridging, leads, brine rejection, albedo coupling, and
  ocean heat transport.
- Glacier erosion, U-shaped valleys, cirques, moraines, and outwash plains.
- Earth-reference calibration by latitude, elevation, equilibrium-line
  altitude, and glacier inventory threshold.
