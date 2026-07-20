# Functional Vegetation Mixtures

## Status

Milestone 15b2a is implemented and calibrated by the 15b2b
`earth_functional_vegetation_v1` profile. This stage follows the passing
`earth_biosphere_v1` calibration and converts continuous potential-biosphere
traits into mixed functional vegetation. Familiar biome labels remain a
derived view and are not part of this milestone.

## Interpretation

Every functional fraction is a physical fraction of the full coarse cell, not
a probability and not a fraction conditional on vegetation. On land, the eight
functional strategies plus five non-vegetated classes sum to one. Ocean cells
carry zero terrestrial fractions and use the reserved dominant-cover code.

The functional strategies are:

1. cold-adapted woody,
2. warm evergreen woody,
3. seasonal woody,
4. xeric shrub,
5. cool-season herbaceous,
6. warm-season herbaceous,
7. hydrophytic,
8. low-stature resource-conservative.

These are continuous producer-community strategies. They do not assert a
species, leaf morphology, Earth taxon, evolutionary history, or single
categorical biome.

The non-vegetated partition is:

1. bare ground,
2. saline barren,
3. glacier or persistent ice,
4. inland open water,
5. unsupported or effectively soil-free surface.

`Unsupported` is an area fraction, not a declaration that whole cells are
uninhabitable. It is the non-open ground not covered by the upstream
`SoilBearingFraction`, currently dominated by exposed or very shallow bedrock.
Cliffs, rock outcrops, talus, and sparse crevice vegetation can be represented
when a region is refined; V1 does not pretend they carry continuous mineral
soil at L2.

Hydrologic wetland support can host hydrophytic vegetation. It is not painted
as a biome and does not imply that every seasonally saturated surface is a
swamp.

## Inputs

- Potential cover, NPP, biomass, growing season, seasonality, adaptation
  pressures, woody allocation, resource-conservative strategy, fuel
  continuity, and confidence from 15b1.
- Annual temperature from climate.
- Soil fertility, depth, drainage, salinity, soil-bearing support, and
  confidence.
- Glacier fraction, the effective coarse/refined inland-water and wetland
  fractions used by soils, terrain relief, and canonical land/ocean state.

## Outputs

- `FunctionalTypeFractions`, with a fixed eight-entry strategy axis.
- `NonVegetatedFractions`, with a fixed five-entry cover axis.
- `FunctionalResourcePotentials`: fire tendency, grazing potential, forest
  resource potential, pasture potential, and crop potential.
- `FunctionalVegetationConfidence`.
- `DominantFunctionalCoverCode`, a derived query/rendering code.
- `FunctionalVegetationCatalog` and `FunctionalVegetationMetadata`.

Potential pasture and crop layers describe physical suitability before land
use, domestication, management, irrigation, or social choice. They are not
actual farms or pastures.

The dominant code uses a hierarchical comparison. Aggregate vegetation
competes with each individual nonvegetated class; if vegetation wins, the
largest functional strategy supplies the code. It is not a thirteenth
independent state variable and does not alter the physical fractions.

## Hard Gates

- Every fraction and potential is finite and in `[0, 1]`.
- PFT fractions sum to the functional vegetated fraction and never exceed the
  upstream potential cover or ice/water-free support.
- PFT and non-vegetated fractions sum to one on every land cell within the
  configured tolerance.
- Ocean cells contain no terrestrial fraction or resource potential.
- Dominant codes exist in the persisted catalog and agree with the independently
  recomputed hierarchical comparison.
- Fixed input and configuration produce byte-identical outputs.

## Canonical Diagnostics

The face-128 seed-42 world passes every hard gate. Land cover is `44.35%`
functional vegetation, `36.38%` bare ground, `2.34%` saline barren, `3.17%`
inland open water, approximately zero persistent ice, and `13.77%` unsupported
surface. Maximum per-cell partition error is `4.30e-8`.

Within the vegetated fraction, woody strategies account for `34.69%`,
herbaceous strategies for `37.74%`, hydrophytic vegetation for `4.16%`, and
the xeric-shrub plus low-stature strategies for the remaining `23.41%`.

Area-weighted land means for fire, grazing, forest-resource, pasture, and crop
potential are `0.101`, `0.234`, `0.131`, `0.126`, and `0.294`. Their respective
area-weighted 90th percentiles are `0.153`, `0.337`, `0.325`, `0.186`, and
`0.408`. The 15b2b profile validates broad amplitude, climate response, and
multi-seed stability without interpreting these suitability indices as actual
land use.

## Deferred

- Familiar biome and vegetation-zone labels.
- Earth-observation calibration of global and climate-stratum PFT mixtures.
- Species, competition, succession, migration, disturbance events, consumers,
  and actual land use.
- Vegetation-to-soil, hydrology, albedo, or erosion feedback.
- Independent functional taxonomies for non-Earth evolutionary histories.
