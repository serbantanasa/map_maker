# Surface Materials And Initial Soils

## Status

Implemented, provisional calibration. The Rust-backed `surface_materials` stage
runs only after final surface water and hydrology validation pass their hard
readiness gates.

## Purpose

Convert geological substrate, modern erosion/deposition evidence, climate, and
accepted hydrology into fractional L2 surface materials and physical initial-
soil properties. This stage supplies map units for later L3 catenas; it does not
pretend a coarse cell contains one homogeneous soil.

## Inputs

- Geological province class, crust age, rock strength, sediment accommodation,
  and province confidence.
- Relief, flow slope, river corridor, floodplain potential, depression fill,
  and global lake/wetland fractions.
- Fine selected-basin final lake/wetland coverage, hydroperiod, salinity, eroded
  volume, and deposited volume restricted to L2 parents.
- Annual and monthly climate plus canonical snow and glacier melt provenance.
- Active glacier fraction and upstream readiness metadata.

## Surface-Material Outputs

Seven `float32` component fractions share shape `(6, n, n)`:

- `BedrockSurfaceFraction`
- `ResidualRegolithFraction`
- `ColluviumFraction`
- `AlluviumFraction`
- `LacustrineSedimentFraction`
- `GlacialDepositFraction`
- `VolcaniclasticFraction`

They sum to one on land and zero on ocean. `DominantSurfaceMaterialCode` is a
derived rendering/index label; downstream physical models should consume the
fractions.

## Initial-Soil Outputs

- Soil-bearing fraction, regolith depth, and soil depth.
- Fine-earth sand, silt, and clay fractions plus separate coarse fragments.
- Bulk density, potential organic carbon, pH, carbonate, and salinity.
- Drainage, available water capacity, nutrient/fertility potential, erodibility,
  surface-reset age, hydric-soil fraction, and confidence.

Potential organic carbon and fertility are explicitly pre-vegetation fields.
They become canonical only after functional vegetation and the bounded feedback
pass.

The stage also persists the exact L2 water state used by the soil kernel:

- `EffectiveLakeFraction`,
- `EffectiveWetlandFraction`,
- `EffectiveSurfaceWaterHydroperiod`,
- `RefinedSurfaceWaterMask`.

Outside the selected refined basin these reproduce coarse hydrology. Inside it,
fine-cell water area is conservatively restricted using the summed physical
child area in square kilometres. Downstream stages consume these effective
fields rather than independently choosing between coarse and refined water.

## Monthly Soil Water

The kernel spins up an independent L2 soil bucket and persists:

- liquid input after snow/rain partitioning,
- end-of-month soil water and saturation,
- actual evapotranspiration,
- quick runoff,
- deep drainage,
- annual storage change.

All water quantities are whole-cell-equivalent millimetres over non-open land.
Deep drainage is a recharge proxy, not routed groundwater or guaranteed river
baseflow.

## Hard Acceptance

- Material fractions and fine-earth texture close within configured tolerance.
- All fractions and normalized indices remain in `[0, 1]`.
- Soil depth is nonnegative and no greater than regolith depth.
- Monthly water input equals evapotranspiration, runoff, drainage, and storage
  change within configured tolerance.
- Ocean outputs remain zero and identical inputs reproduce identical outputs.
- Refined lake/wetland area and erosion/deposition volume reconstruct from L2
  parent fractions and depths within `1e-6` relative error.
- The stage refuses unconverged surface water or failed hydrology hard gates.

## Known Limitations

- Parent chemistry is an evidence-conditioned province prior because a
  stratigraphic/lithology ledger does not yet exist.
- Global modern erosion/deposition remains coarse outside the selected refined
  basin.
- No lateral groundwater, permafrost, explicit soil horizons, soil order,
  vegetation feedback, or ecological wetland confirmation exists yet.
- Calibration against global soil, regolith, and surface-material inventories
  remains required before Earth-like area fractions are accepted.
