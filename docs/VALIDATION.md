# Validation

`map_maker` uses validity gates before subjective or statistical quality
assessment. The current harness is an integration baseline for the executable
prototype. Its thresholds are provisional safety envelopes, not calibrated
Earth-realism criteria.

## Running The Gallery

```bash
uv run map-maker validate --config configs/validation.yaml
```

The command generates the fixed seeds in the validation configuration and
writes:

- `out/validation/report.json`: metrics, thresholds, checksums, and gate results.
- `out/validation/gallery.png`: labeled side-by-side physical previews.
- `out/validation/runs/`: persisted datasets and manifests for every seed.

The command exits nonzero when an automated gate fails. A passing exit code
still requires human review of `gallery.png`.

## Global Gates

`cold_determinism`
: Regenerates the first seed with an independent empty cache. Every persisted
  artifact checksum and the final preview checksum must match.

`unique_seed_outputs`
: Every configured seed must produce a distinct preview checksum. This catches
  seed omissions in cache keys and RNG plumbing.

`cache_replay`
: An immediate replay of every seed must restore every stage from cache while
  preserving every artifact checksum and the preview checksum. Artifact content
  is verified while hydrating cache entries; corrupt entries are invalidated and
  recomputed rather than returned as hits.

## Per-World Gates

`finite_fields`
: Every persisted numeric array must contain only finite values.

`land_fraction`
: Land fraction must remain inside the configured prototype envelope.

`land_components`
: Counts wrap-connected land components above a resolution-relative minimum
  size. This rejects single-feature failures without treating pixel noise as
  islands.

`largest_landmass_fraction`
: Limits the fraction of all land owned by the largest connected component.
  The current permissive ceiling allows supercontinents because assembly state
  is not yet represented causally.

`mixed_plate_fraction`
: Measures the fraction of mechanical plates carrying both continental and
  oceanic crust. This catches regression to the rejected one-plate/one-crust
  model.

`longitude_seam_ratio`
: Compares the mean elevation jump across the longitude seam with ordinary
  adjacent-cell elevation differences.

`plate_boundary_relief_ratio`
: Compares mean terrain gradient on mechanical plate boundaries with background
  terrain gradient. It catches elevation that directly redraws the plate mosaic.

## Human Review

Review the fixed gallery for failures not represented by current metrics:

- Repeated continent silhouettes or obvious generator motifs.
- Straight or uniformly narrow mountain streaks.
- Polygonal plate-cell impressions.
- Projection or wrap seams.
- Implausible coast stair-stepping, radial islands, or bathymetric rays.
- Seeds that are numerically distinct but visually near-identical.

Human review is recorded outside the report for now. A later review artifact
should record reviewer, software version, decision, and notes without allowing a
subjective pass to override failed automated gates.

## Hydrology Pass 1 Audit

Hydrology has stage-local automated gates even though its metrics are not yet
integrated into `map-maker validate`:

- The land receiver graph must be complete and acyclic.
- Every terminal must be ocean-connected or a registered closed water body.
- Contributing area must be nondecreasing downstream. Monthly discharge may
  decrease only at registered open-water evaporation/seepage nodes.
- Integrated source runoff must equal terminal discharge/inflow plus registered
  open-water evaporation and seepage within `1e-6` relative error.
- River-reach references and smooth spherical geometries must be valid and
  deterministic.
- Synthetic native scenarios require the same depression to remain terminal
  under arid low inflow, overflow under sustained wet inflow, and breach only
  when the configured erosion criterion is met.

A fractional-water six-seed face-64 audit produced:

| Metric | Observed range |
| --- | ---: |
| Permanent lakes | 34-58 |
| Wetlands | 0-1 |
| Open lake outlets | 28-53 |
| Fractional lake area / land | 0.55-1.76% |
| Fractional wetland area / land | 0-0.007% |
| Closed-drainage land | 0.94-6.15% |

Three face-128 seeds produced `2.16-2.70%` permanent lake area, `0.009-0.067%`
shallow-water wetland area, and `8.8-24.7%` closed-drainage land. The canonical
seed reports `2.70%` lake area and passes the provisional Earth-like `1.5-4.0%` lake-area
band. The band covers published natural-lake estimates of approximately 1.8% of
all land in [HydroLAKES](https://doi.org/10.1038/ncomms13603) and 3.7% of
nonglaciated land when much smaller lakes are included in the
[high-resolution inventory](https://doi.org/10.1002/2014GL060641). It is a
validation diagnostic, not a global-area clamp.

The previous whole-cell lake statistic is retired as an area estimate. Coarse
support cells now carry `LakeFraction` and `WetlandFraction`; area diagnostics
weight those fractions by physical cubed-sphere cell area. These ranges remain
observations, not proof of calibrated lake morphology or bathymetry.

Lake depth remains provisional. The canonical face-128 seed stores about
`458,000 km3` in permanent lakes with an area-weighted mean depth near `95 m`.
That is high relative to Earth and is not accepted as calibrated; sedimentary
infilling, explicit basin age, glacial excavation, and resolved bathymetric
hypsometry are still absent.

The shallow-water wetland class remains sparse after classification was corrected
to use equilibrium subgrid mean depth instead of coarse-cell depth. Ecological
wetlands require hydroperiod, water table, soils, and vegetation and are not
calibrated by this lake-depression pass.

Resolution stability still fails. Fractional area removes whole-cell inflation,
but face-128 worlds retain materially more lakes and closed drainage than the
face-64 sweep. Finer elevation exposes additional local depressions that can
capture drainage which the coarse pass sent onward. Do not tune this away with a
resolution-specific lake count. The required fix is hierarchical hydrology:
preserve accepted coarse trunk connectivity and flux while refining local basins,
tributaries, wetlands, and channels. Until that constraint exists, closed-drainage
and lake statistics are provisional and Pass 1 is not calibrated for arbitrary
resolution.

## Selected-Basin Refinement

The canonical sparse refinement selects ocean-draining basin 226. Its 1,448
coarse basin cells plus one inherited ocean boundary cell produce 370,944
face-2048 child records at factor 16 without allocating a global face-2048
raster. Twenty-one inherited reaches produce 11,253 km of D4-contiguous fine
reach paths and 2,613 sparse reach-cell memberships.

Observed conservation and topology diagnostics:

| Metric | Canonical result |
| --- | ---: |
| Maximum parent area relative error | `8.1e-12` |
| Maximum restricted elevation error | `< 1e-5 m` |
| D4 path topology | Pass |
| Inherited parent path | Pass |
| Downstream-path junction merges | Pass |
| Reverse directed edges | `0` |
| Combined path DAG | Pass |
| Inherited discharge error | `0` |

Requested physical channel area is `1,366.835 km2`; centerline-cell fractions
represent `1,366.835 km2`, or roughly 0.02% of the 6.98 million km2 basin. This
demonstrates why river width must remain a fractional/vector property even at
the refined level. Approximately `157.3 km3` of potential incision is recorded
as reach volume and has not been applied to cell-wide elevation.

Broad corridors do not yet fit entirely in their centerline cells. Requested
valley area is `39,917 km2`, of which `30,347 km2` (76.0%) is currently
represented; requested floodplain area is `38,043 km2`, of which `29,499 km2`
(77.5%) is represented. The missing support must be spread laterally into
neighboring fine cells before erosion rather than hidden by capped fractions.

Source-to-sink readiness currently fails. Of 13 terminal registered reaches,
one reaches the ocean and 12 end at land cells where Hydrology Pass 1 fell below
the river extraction threshold. The refinement stage classifies and reports
these inherited gaps instead of inventing connections. Conservative incision
and sediment routing remain blocked until the gaps close. These are contract and
conservation results, not accepted erosion calibration.

## Calibration Rule

Do not tighten or loosen a threshold solely to make the current gallery pass.
Changes require at least one of:

- A synthetic failed case demonstrating metric sensitivity.
- A deterministic geological benchmark scenario.
- A distribution derived from an accepted reference dataset.
- A documented expansion of allowed world-history states.

No composite realism score is reported yet. The prototype lacks enough causal
subsystems and calibrated dimensions for such a number to be meaningful.
