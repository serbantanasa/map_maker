# Derived Familiar Biomes

## Status

Milestone 15b2c is implemented and passes the canonical face-128 world plus the
fixed six-seed face-64 `earth_biomes_v1` profile. Biome names are derived
interpretations over conserved physical and functional state. They are not
canonical simulation inputs.

## Purpose

Expose familiar ecological labels for rendering, game queries, regional
refinement priors, and surrogate targets without painting climate rectangles or
discarding ecotones. Every classified land cell retains a 13-component biome
mixture; primary and secondary codes are compact queries over that mixture.

## Inputs

- Upstream annual temperature and precipitation.
- Growing season, productivity seasonality, drought pressure, and waterlogging
  pressure from the trait-first potential biosphere.
- Eight conserved functional-vegetation fractions and five nonvegetated surface
  fractions.
- Effective wetland support, surface elevation, and unresolved terrain relief.
- Fire tendency and upstream biosphere/functional classification confidence.

No biome output feeds back into these drivers during this stage.

## Taxonomy

The mixture axis contains tropical rainforest, tropical seasonal forest,
savanna, hot desert, xeric shrubland, temperate forest, temperate grassland,
steppe, boreal forest, tundra, cold desert, alpine, and wetland. Inland open
water and persistent ice remain separate physical landscape codes. Ocean is
outside the terrestrial partition.

This is a broad game-facing V1 taxonomy. It does not claim that the 13 classes
are an exhaustive ecoregion system. Mediterranean woodland, mangrove, flooded
grassland, montane forest, and other finer labels can later be derived from the
persisted mixtures and physical context without changing canonical state.

## Computational Contract

1. Smooth thermal, moisture, seasonality, highland, wet-support, bare-ground,
   and functional-strategy responses produce nonnegative evidence for each
   familiar biome.
2. Evidence is normalized only within the ecological-ground fraction. Biome
   fractions therefore sum exactly to `1 - persistent ice - inland open water`
   on land.
3. A deterministic causal fallback is used only when every ordinary score is
   zero. It depends on wet support, highland state, temperature, and drought.
4. Dominant and secondary biome codes are the two largest conditional mixture
   components where ecological-ground support is material.
5. Dominant landscape first compares aggregate ecological ground with open
   water and persistent ice. A mixed vegetated cell is not declared barren just
   because its ecological fraction is distributed among several labels.
6. Classification confidence combines both upstream confidence fields and is
   reduced by mixture entropy and weak ground support. Dominance margin and
   normalized entropy remain separate inspectable outputs.

## Outputs

- `BiomeFractions`: 13 full-cell ecological area fractions.
- `BiomeClassificationConfidence`.
- `BiomeDominanceMargin`.
- `BiomeTransitionIndex`: normalized mixture entropy.
- `DominantBiomeCode` and `SecondaryBiomeCode`.
- `DominantLandscapeCode`: ocean, biome, inland water, or persistent ice.
- `BiomeCatalog` and `DerivedBiomeMetadata`.

The stage writes both cube-net and equirectangular dominant, mixture, and
transition renders. Equirectangular products are for inspection and atlas
prototyping; physical calculations remain on the cubed sphere.

## Earth Profile

`derived_biomes_validation` independently reconstructs every global biome mean
from exclusive upstream-climate strata. Hard gates enforce finite bounded
mixtures, exact partition closure, valid codes, dominant-code reproduction, and
distinct secondary codes.

The `earth_biomes_v1` diagnostics use broad potential-natural mixture ranges
and causal relationships rather than modern land use. They require, where the
upstream climate stratum is sufficiently present:

- rainforest and forest concentration in warm-humid land;
- savanna and dryland concentration in warm-dry land;
- temperate forest in cool-moist land and open vegetation in cool-dry land;
- boreal forest in cold land and cold-open mixtures in polar land;
- more forest in moist than dry counterpart climates;
- more savanna and open cover in their dry counterpart climates;
- alpine concentration in high terrain; and
- wetland concentration where hydrologic and soil wet support exists.

No generation field is clamped to these ranges. Non-Earth profiles retain the
same outputs with Earth comparisons marked not applicable.

## Current Baseline

The canonical face-128 seed passes all 36 checks. Its land-area mixture is
`26.63%` forest, `5.41%` warm open, `32.72%` temperate open, `21.49%` core
dryland, `4.67%` tundra, `2.24%` alpine, `3.67%` wetland, and `3.17%` inland
open water. Mean transition index is `0.561` and mean classification confidence
is `0.351`; all 13 biomes occur as nontrivial dominant classes.

Across the six face-64 worlds, forest spans `27.53-35.26%`, warm open
`4.31-10.51%`, temperate open `21.84-32.81%`, core dryland `18.77-21.10%`,
tundra `4.62-9.20%`, alpine `1.84-3.51%`, and wetland `4.22-6.10%`. Every
diagnostic passes in at least five of six seeds, every stability gate passes,
and all 13 dominant classes remain nontrivial in every world.

The dedicated command persists `report.json`, `ensemble_kpis.parquet`, and
`biome_gallery.png` under `out/biosphere_validation`. Human gallery review
remains required even when numerical profiles pass.

## Deferred

- The one bounded vegetation-to-soil, hydrology, albedo, and erosion feedback.
- Disturbance history, succession, migration, competition, and species.
- Marine biomes and ocean productivity.
- Finer ecoregions and selected-region biome realization.
- Independently calibrated non-Earth functional and biome taxonomies.
