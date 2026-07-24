# Validation Contract

Status: current. Detailed July 2026 measurements moved to the
[validation archive](archive/validation/2026-07-canonical-validation-baseline.md).

`map_maker` separates hard validity gates, profile diagnostics, and human visual
review. A profile miss must not be reported as a conservation failure, and a
passing numerical report must not override failed visual review.

## Commands And Evidence

Canonical multi-seed validation:

```bash
uv run map-maker validate-biosphere --config configs/biosphere_validation.yaml
```

This command owns the current cubed-sphere numerical profiles and writes the
surface-geography and biome galleries under `out/biosphere_validation/`.

Framework and legacy-path integration validation:

```bash
uv run map-maker validate --config configs/validation.yaml
```

This command still gates deterministic execution, cache behavior, and the
legacy rectangular gallery. Its gallery is not evidence for canonical
cubed-sphere geography.

Physical-atlas evidence is generated with:

```bash
uv run map-maker atlas
```

The atlas manifest must identify source artifacts, checksums, projection, and
style version. Human acceptance remains part of the current roadmap gate.

The selected-basin L2 handoff is generated and self-validated with:

```bash
uv run map-maker regional-handoff
```

Its validation report gates parent area conservation, bounded absolute and
relief-relative terrain-mean conditioning, bounded correction amplitude,
sea-level-relative surface-mean conditioning, exact parent ocean/wetland and
stable-`LakeID` basin-area reconstruction, bounded child occupancy, stable
child/parent identity, and complete refined river-path references. It also
gates unresolved hydraulic depth and each parent's absolute and relief-relative
child-elevation span, rejects a p95 parent-boundary terrain residual larger than
the configured ratio to ordinary interior edge jumps, and independently rejects
a repeated L0-parent terrain motif. Its
manifest records source, config, software, and native-library
fingerprints plus checksums for every output.

The first L3 target selection is validated with:

```bash
uv run map-maker l3-target
```

The target must be a complete upstream closure with one outlet, a continuous
terrain rectangle, a complete source-context ring, mutually exclusive domain
roles, unique referenced L2 children, a passing source handoff, and a base-grid
estimate inside the configured workstation budget.

The first physical L3 terrain artifact is generated and self-validated with:

```bash
uv run map-maker l3-terrain
```

It gates stable `uint64` identity, cell count and physical scale, complete L2
context, spherical area conservation, absolute and relief-relative L2 mean
conditioning and explicit solver convergence, alternate chunk-partition replay,
L2/chunk boundary residuals, the rejected parent-bubble motif, continuous
coverage and domain masks, finite terrain, storage, and observed-or-estimated
workstation memory. Resume markers follow durable writes, and a cache hit
rechecks the published Zarr checksum. The terrain diagnostic still requires
human review; a numerical seam pass cannot waive a visible repeated tile
pattern or an unclear map without legend and scale.

The first L3 hydrology artifact is generated and self-validated with:

```bash
uv run map-maker l3-hydrology
```

It gates exact represented-parent monthly forcing conservation, native runoff
conservation and acyclic receiver topology, complete stored-window routing,
finite hydrology and no process terminal in the display, natural target-basin
dominance, inherited/fine catchment overlap in both directions, area ratio,
outer-halo contact, inherited hydrograph agreement, bounded open-water area,
cumulative prospective breach incision, river-network presence, inherited
trunk support, and zero material downstream discharge losses not explained by
lake or wetland losses. The published fine routed core is replayed from the
receiver graph during validation. Exact display and hidden-halo counts are hard
state; no visible cell may be terrain-only context. Cache hits recheck the
complete output tree.

The coarse causal mineral inventory is generated and self-validated with:

```bash
uv run map-maker-pipeline --stage mineral_systems_validation \
  --config configs/cubed_sphere_crust_state.yaml
```

Its hard gates cover finite bounded causal fields, reconstruction of each
family potential from its six persisted supports, terrestrial support over
open ocean, catalog identity and references, topology-aware candidate local
maxima, and independent directional enrichment against upstream geological
drivers. The canonical seed-42 face-128 artifact passes all ten families and
publishes 385 regional systems and 996 coarse candidate hypotheses. These
counts are diagnostic, not Earth abundance quotas. The command exits nonzero
when the per-world hard gate is red, and a red artifact cannot enter a regional
handoff.

The fixed mineral-system ensemble is validated with:

```bash
uv run map-maker validate-minerals \
  --config configs/mineral_systems_validation.yaml
```

The 2026-07-24 baseline passes seeds `42`, `101`, `202`, `303`, `404`, and
`505` at face resolution 64. Every family passes its directional and
non-collapse gates on all six seeds, and all six combined state checksums are
distinct. The screening config relaxes only named
coarse L2 terrain-acceptance metrics needed to execute the reduced-resolution
screen; it does not alter generated terrain or mineral state. The report
records those overrides with `acceptance_only_no_terrain_state_change`
semantics. Petroleum, measured reserves, economic viability, and L3 deposit
geometry remain explicitly unsupported.

## Gate Classes

`hard_invariant`
: A physical, accounting, topology, schema, or determinism requirement. Any
  failure rejects the artifact.

`profile_diagnostic`
: A comparison with a named, versioned scenario profile such as Earthlike.
  Ordinary generation may complete outside the range while recording the miss.

`human_review`
: A dated disposition of artifacts whose important defects are not adequately
  represented by current metrics. It cannot waive a hard failure.

## Active Hard Contract

| Domain | Required properties |
| --- | --- |
| Execution and storage | Cold determinism, distinct seeded outputs, checksum-verified cache replay, finite persisted fields, compatible artifact schemas, and recorded native fingerprints. |
| Geometry and surface geography | Reciprocal spherical topology, area conservation, connected-ocean semantics, bounded fractional coast/water state, and no projection seam in canonical state. |
| Hydrology | Complete acyclic receiver and reach graphs, registered terminals, nonnegative discharge, explicit storage/loss attribution, closed water budgets, and source-to-terminal identity across hydraulic connectors. |
| River resolution | River vectors and the reach graph remain canonical at every resolution. Raster layers carry fractional channel-water, floodplain, wetland, and valley effects; they do not replace the graph with categorical river cells or whole-cell excavation. Prospective and applied process fields remain distinct. |
| Materials and ecology | Finite nonnegative state, bounded fractions, exact mixture/partition closure within declared tolerance, and no terrestrial state over unsupported ocean area. |
| Mineral systems | Each coarse prospectivity field has persisted source, process, transport, trap, timing, and preservation support; potentials reconstruct within declared tolerance; terrestrial systems vanish over open ocean; catalogs have stable IDs and valid local-maxima references; and every supported family passes directional causal-enrichment and multi-seed non-collapse gates. |
| Atlas | Rendering consumes immutable truth artifacts, records provenance, and never feeds cartographic width, projection, or style into simulation state. |
| L2 handoff | Sparse child terrain and surface occupancy conserve L0 parents; inherited parent priors are not presented as L2 physics; every vector path references packaged children; package publication follows validation. |
| L3 target | The selected coarse catchment has one outlet, complete upstream ownership, a continuous terrain window and source ring, explicit domain roles, a seam-valid L2 source, stable indexes, and bounded estimated base-grid cost. |
| L3 base terrain | Continuous regional 200 m-class terrain has stable 64-bit identity, complete context, explicit core/halo/outside masks, bounded L2 conditioning error, no parent/chunk seam or repeated correction motif, checksum-verified replay, and bounded memory/storage. Hydrology gates apply only to the core. |
| L3 hydrology | Every stored cell is routed, every displayed cell has finite hydrology and no outer-boundary terminal, the hidden halo is explicit, the refined natural basin is dominant by inherited-target overlap, and core topology, hydrograph, water, reach, and loss budgets pass independently. |

For coarse and sparse fluvial products, applied trunk erosion, deposition, and
cell-mean terrain change remain zero. The separately bounded outlet-spill
correction must either converge under its own volume gates or publish an
explicit regional-refinement deferral. See
[Decision 048](DECISIONS.md#decision-048-terrain-resolution-owns-physical-river-morphology)
and the active [erosion contract](specs/06_erosion.md).

## Profile Diagnostics

Earth-reference checks are versioned diagnostics for generated Earthlike
worlds, not universal clamps. They cover surface geography, climate and
cryosphere, lake/wetland/endorheic area, runoff, biosphere carbon amplitude,
functional cover, derived biome mixtures, and ensemble dispersion. Named
non-Earth profiles may intentionally lie outside Earth ranges while still
passing hard invariants.

River diagnostics distinguish total hydrologic path length from physical
channel length. They also report connector share and unique drainage-basin
counts above `3000`, `4000`, and `5000 km`; counting source branches as separate
great rivers is not accepted. The Earthlike longest-path comparison is a broad
`4000-8000 km` profile diagnostic. See
[Decision 049](DECISIONS.md#decision-049-fractional-water-does-not-break-river-identity).
Sparse refinement must also keep prospective incision below its configured
hard bound after applying registered water surfaces and submerged-approach
backwater controls; a buried coarse basin floor is not a valid river bed.

Thresholds belong in versioned validation configuration and emitted report
metadata. Do not copy transient observed values into this contract.

## Human Review

Review canonical galleries and projected maps for repeated motifs, plate-cell
imprints, straight mountain belts, implausible coasts or islands, projection
seams, illegible shelves and margins, incorrect river hierarchy, and styles
that hide the underlying physical state. Record the date, artifact checksums,
reviewer, disposition, and notes.

The July 2026 surface-geography review is evidence for the broad morphology
screen, but it is not final acceptance of the global physical map.

## Change Control

Do not change a threshold solely to make a current seed or gallery pass. A
change requires at least one of:

- a synthetic failed case demonstrating metric sensitivity;
- a deterministic process benchmark;
- a distribution derived from an accepted reference dataset;
- a documented expansion or correction of allowed scenario states.

Preserve old measurements and superseded thresholds as dated archive reports.
The current contract should describe what must be evaluated, not accumulate
implementation diaries.
