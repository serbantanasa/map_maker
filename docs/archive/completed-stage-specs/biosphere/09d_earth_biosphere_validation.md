# Earth Biosphere Validation Profile

## Status

Implemented and passing as `earth_biosphere_v1`. The six-seed coarse screen and
canonical face-128 check pass hard invariants, Earthlike diagnostics, and
ensemble tolerances.

This profile validates the potential natural terrestrial biosphere emitted by
15b1. It does not validate actual modern vegetation, human land use, ocean
productivity, consumers, disturbance history, or 15b2 functional types.

## Separation Of Concerns

The profile has three independent outcomes:

1. Hard invariants: finite state, complete climate-stratum partitioning, and
   exact reconstruction of global productivity and biomass from strata.
2. Earth comparison: versioned global totals and directional climate-response
   relationships. These diagnose calibration and do not stop non-Earth worlds.
3. Ensemble stability: tolerances across fixed seeds. A mean result may not
   hide a failed seed or excessive seed-to-seed variance.

The standalone `validate-biosphere` command requires all three to pass. The
ordinary generation DAG only treats hard invariants as validity gates.

## Global Earth Profile

| KPI | Provisional accepted range |
| --- | ---: |
| Land surface fraction | `0.18-0.36` |
| Potential terrestrial NPP | `50-75 Pg C/year` |
| Land-mean potential NPP | `0.28-0.55 kg C/m2/year` |
| Potential vegetation biomass | `771-1,107 Pg C` |
| Land-mean potential biomass | `4.2-8.1 kg C/m2` |

The NPP envelope includes contemporary global estimates around `53-56 Pg
C/year` and leaves room for potential natural vegetation without human land
use. The biomass range is the published six-estimate potential-vegetation
range under current Earth climate. Scalar cover, LAI, canopy, rooting, and
latent trait means are reported but do not yet have defensible global gates.

References:

- [NASA facts about Earth's surface](https://science.nasa.gov/earth/facts/)
- [Contemporary global terrestrial NPP](https://pmc.ncbi.nlm.nih.gov/articles/PMC4234638/)
- [Potential and actual global vegetation biomass](https://doi.org/10.1038/nature25138)
- [ORNL DAAC multi-biome NPP observations](https://doi.org/10.3334/ORNLDAAC/1352)
- [MODIS Vegetation Continuous Fields](https://modis-land.gsfc.nasa.gov/ValStatus.php?ProductID=MOD44)
- [TRY plant trait database](https://www.try-db.org/)

## Climate Distributions

Evaluation strata use only upstream annual temperature and precipitation. They
are exclusive response bins, not biomes or a substitute for Koppen classes.

| Stratum | Definition |
| --- | --- |
| Polar | mean temperature below `-5 C` |
| Cold | `-5 C` to below `5 C` |
| Cool dry | `5-18 C`, precipitation below `500 mm/year` |
| Cool moist | `5-18 C`, precipitation at least `500 mm/year` |
| Warm dry | at least `18 C`, precipitation below `500 mm/year` |
| Warm seasonal | at least `18 C`, precipitation `500-1,500 mm/year` |
| Warm humid | at least `18 C`, precipitation at least `1,500 mm/year` |

For every stratum, the stage persists physical area and area-weighted mean,
P10, P50, and P90 for temperature, precipitation, NPP, biomass, cover, growing
season, woody allocation, canopy height, LAI, rooting depth, and confidence.
Directional Earth checks currently require moist cool land to outproduce cool
dry land and warm humid land to substantially outproduce polar and warm dry
land. A relationship is not scored if either stratum covers less than `0.5%`
of generated land.

## Multi-Seed Contract

The coarse screen uses six fixed face-64 seeds. It requires at least five
successful worlds, at least `80%` within-reference results for every applicable
Earth diagnostic, and the following dispersion bounds:

| Ensemble metric | Maximum dispersion |
| --- | ---: |
| Land fraction | absolute range `0.08` |
| Global NPP | coefficient of variation `0.35` |
| Global biomass | coefficient of variation `0.40` |
| Mean cover | coefficient of variation `0.25` |
| Vegetated land fraction | coefficient of variation `0.20` |
| Climate-stratum area | absolute range `0.22` |
| Stratum mean NPP, when sufficiently present | coefficient of variation `0.60` |

Coarse screening must be followed by the canonical face-128 seed. Acceptance
of a model change requires both scales; neither threshold set may be tuned to a
single generated world.

## Artifacts And Command

The per-world `biosphere_validation` stage writes:

- `BiosphereKpiCatalog`,
- `BiosphereClimateDistributionCatalog`,
- `BiosphereValidationMetadata`.

The ensemble writes `report.json` and `ensemble_kpis.parquet`:

```bash
uv run map-maker validate-biosphere --config configs/biosphere_validation.yaml
```

## Current Baseline

All six fixed face-64 worlds now complete. Hard invariants, climate-response
directional pass-fraction requirements, and every ensemble-dispersion gate
pass.

| Coarse-screen KPI | Current result |
| --- | ---: |
| Land fraction | `35.00%` in every seed, all six pass |
| Potential terrestrial NPP | `54.78-73.02 Pg C/year`, all six pass |
| Land-mean potential NPP | `0.307-0.409 kg C/m2/year`, all six pass |
| Potential vegetation biomass | `888.33-1,061.11 Pg C`, all six pass |
| Land-mean potential biomass | `4.98-5.94 kg C/m2`, all six pass |
| NPP coefficient of variation | `0.094` |
| Biomass coefficient of variation | `0.062` |

All hard invariants, Earth diagnostics, directional relationships, and
dispersion gates pass in every seed after the connected-ocean geography repair.

The canonical face-128 seed passes the complete Earth profile:

| Canonical KPI | Current result |
| --- | ---: |
| Land fraction | `35.00%` |
| Potential terrestrial NPP | `50.28 Pg C/year` |
| Land-mean potential NPP | `0.282 kg C/m2/year` |
| Potential vegetation biomass | `844.54 Pg C` |
| Land-mean potential biomass | `4.73 kg C/m2` |

The canonical world has `712.90 mm/year` mean land precipitation and `0.541`
mean liquid-water opportunity, compared with `711.20 mm/year` and `0.556` for
face-64 seed 42. Resolution-aware moisture transport remains stable after the
surface-geography repair and fractional coastal climate mixing. The approved
generated-Earthlike land band is `18-36%`; observed Earth's approximately
`29%` remains the reference point. The canonical `35%` setpoint is inside that
band, not a required exclusive fraction.
