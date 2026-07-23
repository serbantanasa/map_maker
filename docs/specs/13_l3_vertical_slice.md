# L3 Regional Vertical Slice

Status: base terrain, hydrology V0, physical channel geometry, and initial
surface materials/soils implemented; adaptive channel refinement pending

## Purpose

L3 turns one bounded L2 catchment into terrain and hydrology usable by later
game-map generation. It is the first scale that may discover tributaries and
apply fluvial terrain change. It is not a globally dense replay.

The canonical first target is `temperate-highland-catchment`, selected from the
seed-42 basin-395 handoff at outlet parent cell `80324`. It covers approximately
`102,000 km2`, has one coarse hydrologic outlet, and is configured in
`configs/l3_vertical_slice.yaml`. Selection is reproduced with:

```bash
uv run map-maker l3-target
```

The first physical L3 stage is reproduced with:

```bash
uv run map-maker l3-terrain
```

Route its monthly water, depressions, and vector river network with:

```bash
uv run map-maker l3-hydrology
```

Derive smooth physical centerlines and ecology-facing corridor support with:

```bash
uv run map-maker l3-channel-geometry
```

Realize surface materials, initial mineral soils, and monthly soil water with:

```bash
uv run map-maker l3-surface-materials
```

It writes approximately `6.04 million` cubed-sphere cells at about `198 m` under
`out/cubed-sphere-crust-state-42/l3/temperate-highland-catchment/base-terrain/`.
The stored terrain is a continuous rectangular routing window. Its inner
`5.20 million`-cell display rectangle has complete relief and hydrology; an
outer `836,352`-cell band is hidden process context. There are no internal
no-data or unprocessed display holes.

## Resolution And Budget

- The base terrain grid is nominally `200 m`; the canonical spherical cells
  have an area-equivalent width of `196.7 m`. This is not a `100 m` raster.
- River corridors may refine deterministically to `25-50 m` where lateral
  morphology must be resolved.
- Channels narrower than the active cell remain vectors with fractional raster
  support. A finer raster never replaces canonical river identity.
- The inherited catchment envelope remains approximately `2.60 million` cells.
  The `5,203,968`-cell display covers `201,376.96 km2`; routing uses all
  `6,040,320` stored cells under a seven-million-cell and `24 GB` process-memory
  cap, leaving capacity for the operating system on a 32 GB machine.
- Arrays are chunked Zarr datasets. Variable-length graphs, vectors, and event
  records are Parquet. No stage may require a dense global L3 allocation.

## Inputs

True L2 inputs are child geometry, area, conditional terrain, surface
occupancy, hydraulic controls, and inherited major-reach geometry. L0 climate,
geology, materials, soils, biosphere, and biome fields remain parent priors.
They may constrain an L3 process but cannot be relabeled as recomputed L3
physics.

L2 already owns coherent regional morphology from approximately `5-72 km`.
L3 consumes that shape and adds local residual terrain at roughly `4.5 km` and
below. Increasing L3 wavelengths to compensate for a smooth L2 source is a
contract violation; the L2 realization must be repaired instead.

The target package contains a complete coarse upstream catchment, a continuous
L2 terrain window extending four L2 cells (about `18 km`) beyond its bounding
box, and one further L2 source-context ring. It records the sole outlet, L2
child indexes, source checksums, expected raster cost, and mutually exclusive
core, inherited process-halo, and outside roles. Those masks select the target
and conserve forcing but no longer limit routing. Separate display and hidden-
halo masks partition the full stored rectangle. Hydrology selects the natural
D8 basin with maximum inherited-target overlap and measures acceptance on that
routed core.

## V0 Process Order

1. **Implemented:** generate seamless deterministic base terrain conditioned on
   L2 means, relief, lithology, orogenic direction, and fixed boundary values.
2. **Implemented:** downscale monthly precipitation, snowmelt, and runoff as conservative
   forcing fields; do not run a new atmospheric model in V0.
3. **Implemented:** perform full-window depression-aware routing with explicit
   lake storage and spill controls; retain the inherited outlet as an alignment
   anchor rather than an artificial one-cell terminal.
4. **Implemented:** discover tributary vectors from accumulated flow while preserving inherited
   major-trunk identity and monthly hydrographs.
5. **Implemented:** derive physical width, depth, velocity, slope, stream
   power, sediment load, and preliminary channel/floodplain/valley fractional
   support.
6. **Implemented:** smooth physical channel reaches inside their raw D8
   corridors, retain exact graph endpoints, and publish distance,
   flow-persistence, channel, riparian, floodplain, and valley support.
7. **Implemented:** realize initial surface-material mixtures, mineral-soil
   properties, and monthly soil water from inherited geology/climate plus L3
   terrain, water, and channel support.
8. Apply bounded fluvial incision and deposition only where the active terrain
   resolution can represent the process support.
9. Adaptively refine selected channel corridors before resolving banks,
   meanders, crossings, or categorical water surfaces.

Every stage writes resumable intermediate data and exposes supervised input and
output pairs suitable for later surrogate training.

## Implemented Base Terrain

The base hierarchy subdivides every terrain-window L2 cell by `22 x 22`; the
implied global face resolution is `45056`, but only the regional rectangle
exists in storage.
Cell IDs are `uint64`. Arrays are parent-major Zarr chunks containing 64 L2
parents apiece, with separate raw-generation and conditioning completion
markers. Completion markers follow durable array and statistics writes, and a
cache hit rechecks the published Zarr tree checksum. The V0 bilinear conditioner
requires the bounded window to remain inside
one cubed-sphere face; a future cross-face target must add reciprocal D8 corner
transport rather than treating face-edge neighbors as absent.

The native residual field is deterministic in global spherical coordinates and
uses wavelengths no broader than approximately one L2 cell. L2 context,
unresolved relief, rock strength, and orogenic orientation control its shape.
A second pass interpolates bounded L2-center corrections over a shared D8
bilinear lattice. This is deliberately a soft constraint: each final L2 mean
must fall within `15 m` and `5%` of inherited relief, rather than receiving an
exact repeated correction stamp. Decision 052 records the rejected exact
approach and the governing semantics.

The artifact retains raw terrain for idempotent resume and surrogate examples,
plus final elevation, offset from L2, unresolved relief, spherical geometry,
parent conditioning, chunk records, checksums, a validation report, and a
native-face terrain diagnostics: clean physical relief in `terrain.png` and
explicit inherited-target boundaries in `terrain_domain.png`. Both are cropped
to the common fully processable display after hillshade is computed with the
hidden halo, and both include legends and an approximate kilometre scale bar.
The terrain stage itself does not route water or create river cells.

## Implemented Hydrology V0

The hydrology artifact routes the complete `6,040,320`-cell stored rectangle
with a native D8 graph, then crops diagnostics to the inner `5,203,968` cells.
It conservatively redistributes inherited monthly forcing,
realizes physical ocean separately from terrain elevation, iterates
fill/spill/breach relaxation, and publishes lakes, prospective breach events,
generated reaches, inherited reaches, and inherited-reach alignment. Base
terrain is not mutated.

The outer four L2 cells on each side, equal to `88` L3 cells or about `17.3 km`,
are a hidden routing halo. No displayed cell may carry a process-boundary
terminal or non-finite hydrology state. The older core/halo/outside masks remain
selection provenance and do not create partial-map sentinels.

The inherited outlet is an inland coarse handoff, not necessarily a sea mouth
and not a literal L3 terminal. Hydrology selects the natural fine basin with
the greatest inherited-envelope overlap. Hard gates cover its dominance over
the second candidate, overlap in both directions, area ratio, outer-halo
contact, inherited hydrograph agreement, forcing and runoff conservation,
receiver topology, open-water fraction, and unexplained downstream discharge
loss.

Rivers remain vector reach paths with stable IDs. `reported_reach_support`
rasterizes those paths only for diagnostics; `waterbody_flow_connector` marks
the zero-width parts that carry graph continuity through lakes or unresolved
hydraulic controls. Neither field claims a cell-wide channel. Graph validation
uses complete reach support so flow cannot stop and restart at each lake. The
map keeps connectors through small ponds but suppresses their stroke beneath
lakes at least `50 km2`, where the lake polygon itself shows the continuous
water path. Physical width and depth remain reach attributes.

The canonical seed-42 result contains `6.04 million` routed cells, a displayed
area of `201,376.96 km2`, and a refined target basin of `89,851.80 km2`. That
basin contains `12,005` reported reaches and `2,539` lakes; its roughly
`1,084 m3/s` outlet differs from the inherited monthly hydrograph by `19.4%`.
It retains `70.13%` of inherited area, has `79.35%` of its own area inside the
coarse target, and leads the second candidate by `6.11:1`. Open water covers
`8.68%` of core land and no material discharge loss is unexplained. This inland
window contains no physical ocean, 14 explicit closed sinks, and no endorheic
depressions. Sparse arrowheads show downstream direction. Both hydrology maps
include a legend and approximate `100 km` scale bar.

## Implemented Channel Geometry V0

The channel-geometry artifact consumes accepted terrain and hydrology without
mutating either one. Endpoint-anchored Chaikin smoothing converts physical
channel reaches from raw D8 cell paths into continuous local and spherical
polylines. Lake and unresolved-hydraulic connectors retain graph continuity
but do not become physical channel beds.

The canonical result preserves all `33,181` source reach records and smooths
`17,095` physical channels. Every graph endpoint is exact, no centerline
self-intersects, the largest displacement from a raw path is `17.4 m`, and the
smallest smoothed/raw length ratio is `0.858`. Mean interior turn angle falls
from `31.6` to `5.2` degrees. Cold generation uses about `2.0 GB` peak RSS,
writes about `63 MB`, and a cache replay verifies every output checksum.

The raster product covers all `6.04 million` stored cells and retains the same
`5.20 million`-cell display. It publishes distance to physical channel,
distance to a declared reliably flowing subset, nearest stable reach identity,
flow-persistence fraction, and exactly nested fractional channel, riparian,
floodplain, and valley support. Reliability currently means at least six
months at or above `0.15 m3/s`; all generated channels still have at least one
zero-flow month. This is not called perennial flow because groundwater and
baseflow are not yet modeled.

The diagnostic is
`channel-geometry-v0/channel_geometry.png`. It includes complete terrain,
water, smoothed channels, ecology support, a legend, and a labelled
`100 km` scale. Adaptive `25-50 m` bank-resolution corridor meshes remain a
separate later stage.

## Implemented Surface Materials V0

The surface-material artifact reuses the Rust property-first materials and
initial-soil kernel over 47 resumable L3 chunks. Continuous geology and monthly
climate priors are bilinearly conditioned across L0 boundaries, geological
province identity remains categorical, temperature receives a bounded
elevation lapse correction, and accepted L3 precipitation, melt, lakes,
wetlands, channels, floodplains, and relief drive the fine result. The monthly
soil-water bucket receives a 24-year periodic spin-up.

A completion marker is trusted only when its durable chunk-stat record exists
and every output-chunk checksum still matches. A missing or damaged completed
chunk is marked incomplete and regenerated before validation.

Active river width is not used as a proxy for alluvial history. The inherited
coarse `AlluviumFraction` is a soft cumulative depositional prior, localized by
L3 valley support, channel distance, and slope and persisted independently as
`AlluvialLegacyFraction`. It can place abandoned floodplain, terrace, fan, and
older valley deposits without widening current channel water. It is not an
applied sediment-transport solve and must be replaced or updated when a later
L3 erosion/deposition history exists.

The canonical result covers all `6,040,320` stored cells and the common
`5,203,968`-cell display. Land is `71.4%` soil-bearing with `2.6%` hydric-soil
support, `1.03 m` mean regolith, `0.77 m` mean soil depth, pH `5.56`, and
fertility potential `0.21`. Area-weighted materials are `28.7%` exposed
bedrock, `48.2%` residual regolith, `7.3%` colluvium, `9.2%` alluvium, `6.6%`
lacustrine sediment, and less than `0.1%` glacial deposit. The parent-material
mixture L1 difference p95 is `0.72`; it is bounded but deliberately not exact.

All material and texture sums, monthly water budgets, finite/bounded outputs,
soil support, source/output checksums, memory, and storage gates pass. Peak RSS
is about `2.2 GB`; the artifact is about `1.6 GB`. Its diagnostic,
`surface-materials-v0/surface_materials.png`, includes the complete display,
legend, and labelled `100 km` scale. These are initial mineral soils: no
groundwater/baseflow, vegetation feedback, soil taxonomy, dynamic disturbance,
or mineral-resource deposits are claimed.

## Next Contract: L3 Ecology V0

L3 ecology must replay the established causal stack in regional chunks:

1. reconstruct monthly insolation from the inherited orbital field, monthly
   temperature from the shared parent climate plus persisted L3 lapse
   adjustment, and hydrostatic CO2/oxygen partial pressure from the source
   atmosphere controls plus L3 elevation;
2. run the existing Rust biosphere-envelope kernel with L3 monthly soil liquid
   input, saturation, soil support, nutrients, fertility, salinity, and
   confidence;
3. run the existing Rust potential-biosphere kernel from that fine resource
   envelope;
4. run the existing Rust functional-vegetation kernel with L3 soil, water,
   glacier, wetland, and relief state; and
5. run the existing Rust derived-biome kernel only after the functional
   partition passes.

The stage will persist monthly envelope and productivity state, potential
traits, eight functional vegetation fractions, five nonvegetated fractions,
resource potentials, 13 familiar biome fractions, confidence/transition state,
and reproducible query codes. Arrays remain cell-first in the regional Zarr
artifact; native kernels receive month- or class-first contiguous chunks.

Inherited potential-biosphere, functional-vegetation, and biome fields are
comparison priors and surrogate-training context. They are not copied labels,
hard parent quotas, or permission to create L0-aligned ecological blocks. Fine
soil water and terrain may move a child away from its parent mixture, while
represented-parent divergence and parent-boundary motif metrics remain bounded.

V0 reuses the source world's orbital, atmosphere, and calibrated ecology
controls. It does not run a regional atmosphere, model vegetation feedback,
simulate succession/fire/grazing events, place species, or realize human land
use. Channel/riparian effects enter through accepted fractional floodplain,
hydric-soil, wetland, and soil-water state; a narrow river does not make a whole
200 m cell wetland.

Acceptance requires finite and bounded outputs, zero terrestrial state over
physical ocean, exact functional/nonvegetated and biome/ice/water partitions,
code reconstruction, monthly-to-annual productivity closure, rooting depth
within regolith, checksum-audited resume, and a storage ceiling initially set
at `6 GB`. Relational diagnostics must show wetter fine soils favoring
hydrophytic/wetland mixtures, cold/high terrain favoring cold or alpine
mixtures, and deeper fertile valley soils improving productivity and resource
potential relative to comparable adjacent slopes. A complete diagnostic must
include a legend and labelled kilometre scale. Earthlike global abundance
ranges are context, not hard regional quotas.

## Required Outputs

- base terrain elevation and unresolved relief;
- depression, lake, spill, receiver, and contributing-area state;
- monthly runoff, discharge, snowmelt contribution, and water storage;
- canonical reach and tributary graph with stable IDs and vector geometry;
- fractional channel, floodplain, valley, lake, and wetland support;
- prospective and applied erosion/deposition kept as separate fields;
- sediment flux and deposited material class;
- mutually exclusive surface-material mixtures, initial mineral-soil
  properties, and monthly soil-water state;
- context/boundary state and exact source provenance;
- diagnostic terrain, basin, river-hierarchy, and process-budget renders.

## Acceptance Gates

- deterministic cold run and checksum-identical replay;
- no tile, L2-parent, or cubed-sphere seam signal above the declared threshold;
- no repeated parent correction motif above the declared correlation threshold;
- every L2 terrain mean remains inside its absolute and relief-relative
  conditioning tolerance;
- exact target extent, unique stable IDs, and bounded peak memory/storage;
- a continuous terrain rectangle with exhaustive, mutually exclusive core,
  process-halo, and outside masks;
- exhaustive display/hidden-halo masks, full routing of every stored cell, and
  no process-boundary terminal or invalid hydrology in the display rectangle;
- hydrological validity and conservation acceptance measured on the fine routed core;
- inherited/routed catchment overlap, area-ratio, and process-boundary gates;
- the natural target basin is dominant by inherited-area overlap and every
  non-lake flow reaches an explicit physical or outer-halo terminal;
- no accidental sinks and no receiver cycles;
- no material downstream discharge decrease without explicit waterbody loss;
- inherited major trunks remain connected and discharge-conservative;
- precipitation/runoff, lake storage, erosion, deposition, and sediment budgets
  close within their declared tolerances;
- material and soil-texture mixtures close, soil depth never exceeds regolith,
  monthly soil water closes, and every persisted soil state is finite;
- channel, floodplain, and valley fractions are nested and never exceed cell
  capacity;
- applied terrain change has physically resolved support and never represents a
  narrow river as whole-cell excavation;
- human review rejects repeated parent/tile motifs, implausible tributary
  density, rectilinear drainage, broken river hierarchy, and incoherent lakes.
- every new diagnostic map includes a legend and labelled physical scale where
  that projection has a meaningful local scale.

## Non-Goals

- full dynamic regional atmosphere or ocean circulation;
- final taxonomic or vegetation-conditioned soils, ecosystems, minerals,
  settlements, or game tiles;
- resolving every narrow channel at the `200 m` base scale;
- refining the entire basin-395 L2 package to L3;
- training or deploying the neural surrogate.
