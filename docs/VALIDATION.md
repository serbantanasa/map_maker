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

The canonical face-128 reach graph contains 1,259 physical channel reaches and
220 zero-width hydrologic connectors. All 369 terminal reaches resolve to 351
ocean outlets or 18 registered inland sinks; unresolved terminals are zero.
Connector paths cover 885 coarse reach cells while publishing zero channel
width, depth, local velocity, stream power, valley/floodplain support, and
incision. They preserve routed flux and graph continuity without inventing a
physical river through unresolved waterbody support.

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
resolution-specific lake count. The selected-basin prototype now preserves
accepted coarse trunk connectivity and flux, but the contract is not yet
generalized to local basins, tributaries, wetlands, and channels at arbitrary
resolution. Closed-drainage and lake statistics therefore remain provisional.

## Selected-Basin Refinement

The canonical sparse refinement selects ocean-draining basin 226. Its 1,448
coarse basin cells plus one inherited ocean boundary cell produce 370,944
face-2048 child records at factor 16 without allocating a global face-2048
raster. Twenty-one physical channel reaches and 16 zero-width hydrologic
connectors produce 14,068 km of D4-contiguous topological paths, of which
10,530 km is physical channel. The physical support catalog contains 3,152
sparse memberships: 2,442 centerline records and 710 lateral corridor records.

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
| Source-to-sink readiness | Pass |
| Unresolved reach terminals | `0` |
| Inherited discharge error | `0` |
| Per-cell corridor capacity | Pass |
| Nested channel/floodplain/valley support | Pass |
| Preserved-depression process exclusion | Pass (`127` parents) |

Requested physical channel area is `1,297.241 km2`; centerline-cell fractions
represent `1,297.241 km2`, or roughly 0.02% of the 6.98 million km2 basin. This
demonstrates why river width must remain a fractional/vector property even at
the refined level. Approximately `150.20 km3` of potential incision is recorded
as reach volume and has not been applied to cell-wide elevation.

Requested valley area is `38,010.699 km2` and represented area is
`38,010.698 km2`. Requested floodplain area is `36,230.158 km2` and represented
area is `36,230.157 km2`. The remaining differences are floating-point
allocation error below `2e-8` relative. Per-cell summed support never exceeds
one for channel, valley, or floodplain, including cells shared by multiple
reaches, and every membership preserves channel <= floodplain <= valley.

The basin has one terminal reach, which reaches the ocean. Sixteen zero-width
connectors preserve the accepted drainage path through coarse depression
support without pretending those cells contain physical channels. They carry no
physical memberships. Physical channel reaches also stop their process support
at shared connector endpoints, so no channel, valley, floodplain, or incision
area enters a connector-owned or preserved-depression parent. The basin
therefore passes the topology and corridor input gates for conservative
incision and sediment routing.

## Selected-Basin Erosion

The first conservative fluvial pass builds 2,434 physical bed nodes and 2,421
physical edges in 13 components. The components are separated where zero-width
connectors cross process-excluded depression support. All 2,442 physical
centerline memberships receive bed and incision records; connectors receive
none.

Observed profile and sediment diagnostics:

| Metric | Canonical result |
| --- | ---: |
| Shared-junction bed error | `0 m` |
| Minimum realized downstream slope | `1.0e-5` |
| Maximum conditioning incision | `537.11 m` |
| Potential incision diagnostic | `150.20 km3` |
| Actual grade-conditioning incision | `145.53 km3` |
| Floodplain deposition | `95.96 km3` |
| Inland terminal deposition | `0 km3` |
| Ocean sediment export | `49.57 km3` |
| Sediment conservation residual | `< 7.7e-6 m3` |
| Maximum full-cell mean change | `12.41 m` |
| Maximum coarse-parent mean change | `0.95 m` |
| Preserved-depression process exclusion | Pass |
| Connector physical-process exclusion | Pass |
| Post-serialization float64 grade audit | Pass |
| Cross-catalog reach/cell/parent budget audit | Pass |

The actual-to-potential incision ratio is approximately `0.969`, but this
agreement is not a calibration target. Actual incision is the volume required
to make the unresolved fine terrain prior downstream-graded; potential incision
was inherited from coarse reach relief. Their proximity in one basin and seed
may be coincidental. The maximum cut shows why the next validation phase must
inspect longitudinal profiles and multi-seed distributions before accepting
morphology.

Floodplain deposition is allocated only inside the previously conserved
floodplain footprint. Every remaining solid volume exits through the registered
ocean terminal. Cell means change by net volume divided by physical cell area,
not by applying channel depth to the entire cell. These are contract and
conservation results, not accepted erosion or sediment calibration.

The stage recomputes grade from persisted float64 bed records and cell
coordinates. It also reconciles profile erosion with reach-local and cell-local
erosion, reach deposition with cell deposition, every reach's available and
outgoing volume, downstream inputs, parent restrictions, and native totals. A
face-16 seed-22 regression fixture contains both a connector and an excluded
parent so the corresponding tests are non-vacuous.

## Sparse Hydrology Pass 2

The canonical pass routes 370,944 sparse face-2048 children without allocating
a global face-2048 raster. Of these, 338,176 selected-basin cells contribute
source area, 2,434 are fixed physical trunk anchors, 32,512 remain inherited
preserved-depression handoffs, and 256 belong to outside-basin terminal support.

Observed stabilization diagnostics:

| Metric | Canonical result |
| --- | ---: |
| Physical trunk edges preserved | `2,421 / 2,421` |
| Cyclic or uncovered cells | `0` |
| Receiver-change cells | `2,870` (`0.849%`) |
| Receiver-change area | `53,917 km2` (`0.847%`) |
| Baseline depression candidates | `9,023` |
| Stabilized depression candidates | `8,944` |
| Newly affected depression-candidate area | `37.14 km2` (`0.000583%`) |
| Wholly new depression components | `0` |
| Removed depression-candidate area | `13,831 km2` |
| Maximum stabilized priority-fill depth | `214.14 m` |
| Independent terminal-area residual | `0 km2` |
| New candidates intersecting corridor support | `0` |
| Additional erosion correction required | No |

The 8,944 stabilized candidates cover about 1.46 million km2 and have roughly
35,663 km3 of potential topographic fill volume. Those numbers are deliberately
not reported as lake area or lake volume. They include unresolved local storage
positions in noisy five-kilometre terrain and require runoff, evaporation,
seepage, hydroperiod, soil, and vegetation tests before any waterbody label.

Pass 2 applies one bounded local receiver correction while preserving every
physical bed edge, junction, reach identity, and zero-width connector handoff.
The canonical change is well below the provisional 15% count and area bounds;
the bounds are rejection guards, not calibration targets. Independent audits
recomputed from the emitted cell catalog agree with the native correction
counts and areas.

A preliminary face-16 sweep collected 12 seeds whose upstream basin refinement
passed. Pass-2 receiver-change area ranged from `1.45-9.93%`, receiver-change
cell count ranged from `1.50-9.93%`, no wholly new depression component was
created, and no seed requested another erosion correction. Twenty-three other
attempted seeds failed the existing refinement routing gate before Pass 2
executed. That high upstream rejection rate remains a separate refinement
defect and is not counted as Hydrology Pass-2 instability.

## Refined Seasonal Surface Water

The canonical surface-water pass consumes all 8,944 Pass-2 candidates and
77,525 candidate-support children. It assigns 320,751 active source children to
disjoint first-downstream-candidate catchments and solves the periodic monthly
state of the acyclic candidate graph upstream-to-downstream.

| Metric | Canonical result |
| --- | ---: |
| Represented parent runoff volume | `346.298 km3/year` |
| Candidate-network direct inflow | `281.000 km3/year` |
| Parent-to-child runoff inheritance error | `0` |
| Maximum fixed-point iterations | `19` |
| Potential connected water area at all spills | `365,195 km2` |
| Pre-adjustment permanent-lake candidates | `8,936` |
| Outlet-erosion feedback candidates | `3,125` |
| Pre-incision transient-storage area | `242,636 km2` |
| Accepted permanent-lake candidates | `5,811` |
| Hydrologic-wetland candidates | `8` |
| Accepted standing-water mean area | `120,063 km2` (`1.885%`) |
| Independent annual water-balance error | `2.49e-15` |
| Maximum cell-to-candidate area reconstruction error | `5.77e-8` relative |

The first unconstrained audit classified nearly every local candidate as a
permanent lake and produced about 909,000 km2 of mean water area. That failed
the realism review. Two structural corrections were made rather than loosening
a validation bound: local subcell low terrain now has a connected-basin cap,
and sustained overflow produces explicit outlet-erosion feedback based on head,
rock strength, sediment accommodation, and discharge.

The 242,636 km2 transient figure is the conservative pre-incision fill/spill
state, not accepted standing water. `surface_water_ready_for_soils` remains
false until a bounded outlet-incision and local reroute pass consumes all 3,125
feedback records. The current class and area distribution is provisional; it
has not passed a multi-seed Earth-derived lake and wetland calibration.

## Bounded Outlet Incision And Final Surface Water

The canonical correction follows the accepted receiver graph with narrow
subgrid beds. It changes cell-mean terrain only by reconstructed eroded volume,
keeps physical channel anchors unchanged, and retains corrected ordinary cells
through explicit receiver constraints. The first correction pass reports:

| Metric | Canonical result |
| --- | ---: |
| Requested candidates | `3,125` |
| Applied / bounded-accepted | `2,942 / 183` |
| Corrected cells | `8,885` |
| Corrected active area | `167,290 km2` (`2.627%`) |
| Subgrid eroded volume | `13.918 km3` |
| Maximum bed incision | `160.65 m` |
| Maximum cell-mean lowering | `4.04 m` |
| Receiver-change area | `359,457 km2` (`5.644%`) |
| Cycle repair | `1` round, `1` ordinary cell |
| Physical trunk anchors preserved | `2,434 / 2,434` |

Monthly balance and correction then converge in seven rounds. Residual feedback
counts are `2,181, 671, 109, 22, 3, 1, 0`; cumulative outlet erosion is about
`16.491 km3`. The final graph retains 10,857 ordinary routing constraints and
publishes `surface_water_ready_for_soils = 1`. Annual candidate-network water
balance closes to `2.8e-16` relative error.

Final accepted standing-water mean area is `238,049 km2`, or `3.74%` of the
selected basin's active area. The area fraction is inside the earlier
provisional Earth-like lake-area envelope, but morphology is not calibrated:
10,062 candidates are permanent lakes, seven are hydrologic wetlands, and none
are seasonal. Multi-seed and multi-resolution comparison must test candidate
count, size distribution, hydroperiod, and bathymetry before these labels are
treated as Earth-like.

The original Python depression-catalog builder scaled as cells times
candidates. Grouped aggregation preserves byte-identical canonical output and
measured `4.77 s` before versus `0.17 s` after on the development machine
(approximately 28x for that operation). This is a local benchmark, not a
cross-platform performance guarantee.

## Calibration Rule

Do not tighten or loosen a threshold solely to make the current gallery pass.
Changes require at least one of:

- A synthetic failed case demonstrating metric sensitivity.
- A deterministic geological benchmark scenario.
- A distribution derived from an accepted reference dataset.
- A documented expansion of allowed world-history states.

No composite realism score is reported yet. The prototype lacks enough causal
subsystems and calibrated dimensions for such a number to be meaningful.

## Executable Hydrology KPI Profile

The `hydrology_validation` stage converts the hydrology review into persistent,
machine-readable rows. Hard topology and conservation failures are kept
separate from provisional Earth comparisons and from known missing capability.
The Earth profile currently references HydroLAKES and the high-resolution lake
inventory for lake area, MERIT-Plus for endorheic area, observed continental
discharge for runoff depth, and separate wetland and floodplain inventories.

Seasonal snow and spring melt are implemented. A separate Rust cryosphere
kernel now spins age-tracked snow and firn/ice reservoirs, transfers excess ice
downslope with area-weighted conservation, and feeds separately reported snow
and glacier melt into runoff potential. The KPI profile distinguishes seasonal
snow, perennial snow, and fractional glacier ice. Stress-driven ice flow,
ice-sheet dynamics, and glacial erosion remain outside V1.

The canonical KPI report contains 44 rows. All hard invariants pass. It finds
51 monthly reach-to-reach discharge decreases after final lake coupling; all
51 are attributed to registered coarse or refined surface-water storage, while
unaccounted decreases are zero. The global lake-area diagnostic is within its
reference envelope. Closed-drainage land (`24.7%`) is above the provisional
Earth envelope, and generated runoff depth (`144 mm/year`) is below it. These
are calibration findings, not permission to tune one seed directly.

The selected basin receives about 31% of its pre-soil liquid input from
snowmelt, with peak melt and runoff in month 11 for its generated seasonal
phase; it currently contains no glacier melt. Globally, `57.5%` of land exceeds
10 mm snow-water equivalent in at least one month and `0.074%` retains it
through every month. Fractional glacier ice appears on the upper terrain of the
tallest tropical range and a smaller midlatitude highland rather than from a
latitude mask.

The final lake coupling profiles 683 terminal candidate networks into 8,196
monthly adjustment records. It conserves network water to machine precision
and keeps all reach entry and exit discharge nonnegative. Scale mismatch moves
373 monthly losses downstream to the first inherited reach that actually
contains the corresponding fine tributary flow; each remap remains persisted
for later calibration.

## Surface Materials And Initial Soils

The canonical `surface_materials` run consumes the passing hydrology report and
restricts selected-basin final water, erosion, and deposition onto 1,449 L2
parents. Material and fine-earth texture mixtures close to better than `5e-8`;
the persisted monthly soil-water budget closes to `7.6e-9` relative error.

| Metric | Canonical result |
| --- | ---: |
| Exposed bedrock fraction of land | `18.09%` |
| Residual regolith fraction | `64.32%` |
| Colluvium fraction | `6.44%` |
| Alluvium fraction | `6.03%` |
| Lacustrine sediment fraction | `4.90%` |
| Glacial deposit fraction | `0.109%` |
| Volcaniclastic fraction | `0.105%` |
| Soil-bearing fraction of land | `79.27%` |
| Hydric-soil fraction of land | `5.15%` |
| Mean regolith / soil depth | `1.37 m / 0.74 m` |
| Mean available water capacity | `93.89 mm` |
| Mean potential organic carbon | `1.61 kg C/m2` |
| Mean initial soil pH | `5.69` |
| Annual modeled soil liquid input | `436.64 mm` |
| Actual evapotranspiration | `150.73 mm` |
| Quick soil runoff | `275.23 mm` |
| Deep drainage proxy | `10.69 mm` |

These are one-seed structural diagnostics, not accepted Earth calibration.
Glacial deposits are especially incomplete because there is no paleoglacial
history, and parent chemistry remains a geological-province prior without a
stratigraphic lithology ledger. The soil runoff partition is persisted for the
future bounded feedback pass but does not modify accepted river hydrographs.
Hydric soil is saturation evidence; ecological wetlands still require
functional vegetation and groundwater interpretation.

## Atmosphere And Biosphere Resource Envelope

Milestone 15b0 separates hard physical gates from profile diagnostics. The
`earthlike` atmosphere profile supplies provisional reference diagnostics; it
does not claim that the downstream biological response has been calibrated.
Named non-Earth profiles remain executable when they miss Earth ranges.

Hard atmosphere gates require finite nonnegative gas quantities, a dry
composition no greater than one, positive pressure, and a valid hydrostatic
scale height. The climate metadata records configured and composition-derived
greenhouse offsets separately.

Hard biosphere-envelope gates require:

- monthly water and thermal opportunity in `[0, 1]`,
- annual fields that reproduce monthly aggregates,
- primary-energy potential no greater than PAR times the configured conversion
  efficiency,
- zero terrestrial primary-energy potential over ocean,
- finite nonnegative carbon and oxygen support fields.

The combined energy product is a provisional Earth-photosynthesis diagnostic,
not a universal habitability or biomass score. PAR fraction, mean atmospheric
transmission, response temperatures, water and CO2 half-saturation controls,
and photosynthetic conversion efficiency still require calibration against
accepted reference datasets. Ocean productivity, chemosynthesis, vegetation,
biome labels, and atmosphere-biosphere feedback are not implemented.

The canonical face-128 seed currently reports:

| Metric | Canonical result |
| --- | ---: |
| Area-mean surface pressure | `99.70 kPa` |
| Minimum high-terrain pressure | `69.91 kPa` |
| Atmospheric scale height | `8.43 km` |
| Composition-derived climate offset | `0 C` at the reference `280 ppm CO2` |
| Mean terrestrial surface PAR | `3,037.94 MJ/m2/year` |
| Mean terrestrial primary-energy proxy | `2.85 MJ/m2/year` |
| Mean thermal opportunity | `0.698` |
| Mean liquid-water opportunity | `0.466` |
| Mean carbon-substrate support | `0.980` relative to reference |
| Mean aerobic-oxygen support | `0.954` relative to reference |
| Land above provisional `5 MJ/m2/year` threshold | `15.56%` |
| Annual aggregation error | `5.40e-8` relative |

The low energy proxy and productive-area fraction are calibration findings.
They currently reflect restrictive initial nutrient support and uncalibrated
conversion controls; they are not accepted estimates of Earth net primary
productivity. The values must not be tuned from this one seed alone.
