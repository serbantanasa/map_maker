# Earth Functional Vegetation Validation

## Status

Milestone 15b2b implemented and passing. This is the final nested 15b milestone;
subsequent work returns to the numbered pipeline stages.

## Contract

`earth_functional_vegetation_v1` validates functional cover after the
independent `earth_biosphere_v1` carbon gate. It does not classify named biomes
and does not fit modern agriculture, deforestation, fire suppression, or other
human land use.

The profile uses three kinds of evidence:

1. broad global fractional-cover ranges informed by vegetation continuous
   fields and potential-natural-vegetation studies;
2. directional and bounded climate-stratum responses; and
3. nondegenerate, directionally plausible resource-suitability distributions.

Reference ranges remain diagnostics during ordinary generation. Named
non-Earth profiles can execute outside them without being clamped toward Earth.

## Global Earthlike Ranges

| Land-area metric | Accepted range |
| --- | ---: |
| Functional vegetation | `35-75%` |
| Woody functional cover | `13-40%` |
| Herbaceous functional cover | `12-40%` |
| Xeric shrub plus low-stature cover | `5-30%` |
| Hydrophytic functional cover | `0.5-10%` |
| Bare, saline, and unsupported ground | `20-60%` |
| Inland open water | `0.5-6%` |

These intentionally broad ranges compare fractional cover with potential
ecosystem structure. They are not assertions that ecosystem-class area and
projected plant cover are identical quantities.

## Climate Structure

The profile requires, when the relevant upstream climate strata are present:

- at least `15%` woody cover in cool-moist land;
- at least `30%` woody cover and at most `20%` hydrophytic cover in warm-humid
  land;
- at least `8%` xeric plus low-stature cover in warm-dry land;
- at most `5%` woody cover in polar land;
- more woody cover in moist than dry climates;
- more xeric/low-stature cover in warm-dry than warm-humid climates;
- more hydrophytic cover in warm-humid than warm-dry climates;
- seasonal warm land more fire-prone than warm-humid land;
- warm-humid land more suitable for forest resources than warm-dry land; and
- warm-dry land more suitable for grazing than warm-humid land.

Resource p90 gates ensure usable dynamic range. Their stronger scientific
meaning comes from directional comparisons because the outputs describe
physical suitability, not observed use.

## Ensemble Contract

The existing six-seed face-64 command now runs through
`functional_vegetation_validation`. It retains the predeclared `80%`
per-diagnostic pass rule and adds:

- global functional-fraction coefficient of variation at most `0.30`;
- global resource-p90 coefficient of variation at most `0.35`; and
- reportable climate-stratum functional means with coefficient of variation at
  most `0.60`.

All six worlds pass every functional diagnostic. Ensemble global means and
ranges are:

| Metric | Mean | Seed range | CV |
| --- | ---: | ---: | ---: |
| Functional vegetation | `51.96%` | `50.01-54.54%` | `0.030` |
| Woody cover | `19.20%` | `17.29-21.58%` | `0.079` |
| Herbaceous cover | `18.53%` | `17.96-19.04%` | `0.019` |
| Xeric plus low-stature cover | `12.49%` | `12.12-12.78%` | `0.018` |
| Hydrophytic cover | `1.75%` | `1.45-2.08%` | `0.118` |
| Inland open water | `1.14%` | `0.73-1.72%` | `0.261` |

Canonical face-128 seed 42 has `47.52%` functional vegetation, `15.71%` woody
cover, `17.86%` herbaceous cover, `12.70%` xeric plus low-stature cover, and
`1.25%` hydrophytic cover. Cool-moist woody cover is `23.0%`; warm-humid woody
and hydrophytic cover are `51.2%` and `10.3%`. Every hard and Earth-profile gate
passes.

## References

- [MODIS Vegetation Continuous Fields Collection 6.1](https://modis-land.gsfc.nasa.gov/vcc.html)
- [Global mapping of potential natural vegetation](https://doi.org/10.1111/geb.12759)
- [Potential global ecosystem extents](https://doi.org/10.1038/s41561-025-01742-z)
- [Global Lakes and Wetlands Database version 2](https://doi.org/10.5194/essd-17-2277-2025)
- [GFED4 burned-area analysis](https://doi.org/10.1002/jgrg.20042)

## Deferred

- Named biome and vegetation-zone labels.
- Actual land use and domestication.
- Disturbance history, succession, competition, and migration.
- The bounded vegetation feedback pass.
