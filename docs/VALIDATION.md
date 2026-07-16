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

## Calibration Rule

Do not tighten or loosen a threshold solely to make the current gallery pass.
Changes require at least one of:

- A synthetic failed case demonstrating metric sensitivity.
- A deterministic geological benchmark scenario.
- A distribution derived from an accepted reference dataset.
- A documented expansion of allowed world-history states.

No composite realism score is reported yet. The prototype lacks enough causal
subsystems and calibrated dimensions for such a number to be meaningful.
