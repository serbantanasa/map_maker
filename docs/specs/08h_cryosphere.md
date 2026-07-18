# Seasonal Snow, Firn, And Glacier Reservoir

## Status

Implemented, provisional calibration. The Rust-backed `cryosphere` stage runs
after atmospheric climate and before hydrology.

## Scientific Boundary

The V1 model targets persistent mountain ice and its water-storage effect. It
does not solve glacier stress, full ice-sheet dynamics, or glacial landforms.

Each coarse land cell contains separate seasonal-snow and glacier-ice water
equivalent. Snow age is tracked as a mass-weighted age moment. Snow that
survives long enough converts gradually to firn/ice; warm-season melt removes
seasonal snow first and glacier ice separately. Glacier melt then participates
in canonical runoff potential.

`TerrainReliefM` is an unresolved relief prior, not summit elevation. A bounded
highland fraction samples an upper-terrain climate using a configurable relief
multiplier and the environmental lapse rate. This permits cold upper slopes in
a warm coarse cell without painting ice solely by latitude or altitude.

## Ice Transfer

Ice above the configured activation storage transfers monthly toward the
lowest neighboring bedrock cell. Transfer conserves area-weighted water volume
across unequal cubed-sphere cells. Transfer into ocean is recorded as calving.
This provides bounded downslope spreading while leaving stress-driven flow for
a later model.

## Outputs

Monthly fields include canonical snowfall, seasonal snowmelt, snow-water
equivalent, firn conversion, glacier melt, glacier ice storage, and melt-aware
runoff potential. Annual fields include glacier mass balance, flow export and
import, calving, glacier sublimation, and fractional ice cover.

## Validation

- Every storage and flux is finite and nonnegative where applicable.
- Per-cell ice storage change reconstructs firn input, ice melt, sublimation,
  flow export, and area-adjusted flow import.
- Global inland flow export and import conserve physical water volume.
- Snowmelt and glacier melt remain separate outputs and surrogate targets.
- Identical inputs and controls produce byte-identical artifacts.

## Deferred

- Stress-driven ice velocity, crevassing, surging, and basal sliding.
- Ice-sheet isostasy, shelves, sea-level feedback, and calibrated calving.
- Glacier erosion, U-shaped valleys, cirques, moraines, and outwash plains.
- Earth-reference calibration by latitude, elevation, equilibrium-line
  altitude, and glacier inventory threshold.
