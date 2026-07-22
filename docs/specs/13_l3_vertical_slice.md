# L3 Regional Vertical Slice

Status: base terrain and hydrology V0 implemented; channel geometry pending

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

It writes approximately `6.04 million` cubed-sphere cells at about `198 m` under
`out/cubed-sphere-crust-state-42/l3/temperate-highland-catchment/base-terrain/`.
The stored terrain is a continuous rectangular window. The irregular catchment,
its process halo, and terrain outside the process domain are explicit masks;
there are no internal no-data holes.

## Resolution And Budget

- The base terrain grid uses approximately `200 m` cells.
- River corridors may refine deterministically to `25-50 m` where lateral
  morphology must be resolved.
- Channels narrower than the active cell remain vectors with fractional raster
  support. A finer raster never replaces canonical river identity.
- The catchment core remains approximately `2.60 million` cells. Its complete
  working window is capped at seven million base cells and a `24 GB` process
  memory budget, leaving capacity for the operating system on a 32 GB machine.
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
core, process-halo, and outside roles. The inherited catchment core selects the
target and conserves forcing. Hydrology refines its actual D8 outlet watershed
inside the halo and measures process acceptance on that routed core.

## V0 Process Order

1. **Implemented:** generate seamless deterministic base terrain conditioned on
   L2 means, relief, lithology, orogenic direction, and fixed boundary values.
2. **Implemented:** downscale monthly precipitation, snowmelt, and runoff as conservative
   forcing fields; do not run a new atmospheric model in V0.
3. **Implemented:** perform depression-aware routing with explicit lake storage, spill controls,
   and one registered regional outlet.
4. **Implemented:** discover tributary vectors from accumulated flow while preserving inherited
   major-trunk identity and monthly hydrographs.
5. **Implemented:** derive physical width, depth, velocity, slope, stream power, sediment load,
   and channel/floodplain/valley fractional support.
6. Apply bounded fluvial incision and deposition only where the active terrain
   resolution can represent the process support.
7. Adaptively refine selected channel corridors before resolving banks,
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
explicit domain boundaries in `terrain_domain.png`. Both include legends and
an approximate kilometre scale bar. It does not route water or create river
cells.

## Implemented Hydrology V0

The hydrology artifact routes the full process domain with a native D8 graph.
It conservatively redistributes inherited monthly forcing, realizes physical
ocean separately from terrain elevation, iterates fill/spill/breach relaxation,
and publishes lakes, prospective breach events, generated reaches, inherited
reaches, and inherited-reach alignment. Base terrain is not mutated.

The registered outlet is a regional handoff, not necessarily a sea mouth.
Hydrology derives a fine routed catchment around that outlet rather than using
the inherited L0 envelope as an artificial wall. Both masks are durable. Hard
gates cover overlap in both directions, area ratio, outer-halo contact, outlet
hydrograph agreement, forcing and runoff conservation, receiver topology,
open-water fraction, and unexplained downstream discharge loss.

Rivers remain vector reach paths with stable IDs. `reported_reach_support`
rasterizes those paths only for diagnostics; `waterbody_flow_connector` marks
the zero-width parts that carry graph continuity through lakes or unresolved
hydraulic controls. Neither field claims a cell-wide channel. Graph validation
uses complete reach support so flow cannot stop and restart at each lake. The
map keeps connectors through small ponds but suppresses their stroke beneath
lakes at least `50 km2`, where the lake polygon itself shows the continuous
water path. Physical width and depth remain reach attributes.

The canonical seed-42 result contains `6.04 million` cells, a routed core of
about `96,560 km2`, `18,555` reported reaches, `2,819` core lakes, a
Strahler-order-6 network, and a roughly `1,190 m3/s` outlet. Its outlet monthly
hydrograph differs from the inherited target by about `11.5%`, open water
covers about `8.61%` of core land, and no material discharge loss is
unexplained. This selected inland window contains no physical ocean, no closed
routing sinks, and no endorheic depressions. Sparse arrowheads in the diagnostic
show downstream direction so headwater starts are not mistaken for river ends.
Both hydrology maps include a legend and approximate `100 km` scale bar.

## Required Outputs

- base terrain elevation and unresolved relief;
- depression, lake, spill, receiver, and contributing-area state;
- monthly runoff, discharge, snowmelt contribution, and water storage;
- canonical reach and tributary graph with stable IDs and vector geometry;
- fractional channel, floodplain, valley, lake, and wetland support;
- prospective and applied erosion/deposition kept as separate fields;
- sediment flux and deposited material class;
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
- hydrological validity and conservation acceptance measured on the fine routed core;
- inherited/routed catchment overlap, area-ratio, and process-boundary gates;
- all non-lake flow reaches the registered outlet or another explicit terminal;
- no accidental sinks and no receiver cycles;
- no material downstream discharge decrease without explicit waterbody loss;
- inherited major trunks remain connected and discharge-conservative;
- precipitation/runoff, lake storage, erosion, deposition, and sediment budgets
  close within their declared tolerances;
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
- final calibrated soils, ecosystems, minerals, settlements, or game tiles;
- resolving every narrow channel at the `200 m` base scale;
- refining the entire basin-395 L2 package to L3;
- training or deploying the neural surrogate.
