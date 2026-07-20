# map_maker

`map_maker` is a deterministic, multi-resolution planetary history and map
generation project. The next-generation engine uses Rust for computational
state and simulation loops, with Python orchestration for stages, persisted
artifacts, experiments, diagnostics, and cartography.

The existing image generator is retained under `map_maker.legacy` while the
new pipeline is developed and validated.

## Development Setup

Python 3.12 or 3.13 and a Rust toolchain are required. With `uv`:

```bash
uv venv --python 3.12
uv sync --extra dev
uv run map-maker-build-native
uv run pytest
cargo test --workspace
```

The native build is explicit. Importing `map_maker` never invokes Cargo or a
C++ compiler. To load native libraries built elsewhere, set
`MAP_MAKER_NATIVE_LIB_DIR` to their containing directory.

Simulation native libraries expose explicit per-library ABI versions.
`map-maker doctor` verifies those contracts and reports each binary SHA-256
fingerprint. Simulation-library fingerprints
are included in stage cache keys and run manifests, so replacing a native binary
cannot silently reuse outputs from different code.

The equivalent `pip` workflow is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
map-maker-build-native
pytest
cargo test --workspace
```

## Commands

Check that Python, Cargo, and all native libraries are ready:

```bash
uv run map-maker doctor
```

Generate the default configured world stack:

```bash
uv run map-maker generate --config configs/pipeline.yaml
```

The result is written under `out/<run-id>/`:

- `world.png`: physical-map preview.
- `run.json`: complete run manifest, statistics, cache state, and artifact paths.
- `datasets/`: persisted NumPy, Arrow, and JSON stage artifacts.
- `visuals/`: per-stage diagnostic renders.

The generator can also be configured directly:

```bash
uv run map-maker generate --width 1024 --height 512 --seed 8675309
```

Rerunning the same configuration reuses the stage cache. The current executable
stack includes tectonics, crust/world-age fields, erosion/sedimentation, dataset
persistence, diagnostics, and final cartography. The canonical cubed-sphere path
now reaches connected geological provinces, boundary segments, causal
pre-erosion elevation/orogenic morphology, monthly orbital forcing, and seasonal
climate/orographic precipitation. The first depression-aware hydrology pass now
writes fractional lake and wetland coverage, spill outlets, breaches, conservative
drainage basins, monthly discharge, sparse waterbody membership, and vector river
reaches. Explicit geological event history, spherical global erosion and
sediment feedback, ecosystem dynamics, and complete regional terrain remain
implementation milestones. The canonical path now also
publishes fractional L2 surface materials, initial mineral-soil properties, and
a conservative monthly soil-water partition. Atmospheric composition and
hydrostatic pressure are now explicit pre-climate artifacts, and the post-soil
`biosphere_envelope` stage publishes raw light, thermal, water, carbon, oxygen,
nutrient, and surface-support fields without assigning vegetation or biomes.
The Rust-backed `potential_biosphere` stage converts that envelope into bounded
potential NPP, cover, biomass, growing season, adaptation pressure, rooting,
canopy, leaf-area, and fuel-continuity fields while deliberately withholding
species, functional types, and biome labels. A first sparse basin-refinement
stage now realizes one inherited trunk network at approximately 5 km scale,
preserves parent terrain means and convergent reach junctions, and stores physical
channel, valley, and floodplain fractions without carving whole cells. It also
closes coarse extraction gaps with zero-width hydrologic connectors, verifies
source-to-sink readiness, and conserves broad valley and floodplain support in
nearby fine cells. The downstream sparse erosion pass now solves shared-junction
bed profiles, applies volume-based subgrid incision, routes newly eroded sediment
through connectors, deposits only on allocated floodplain support, and restricts
terrain-volume feedback to coarse parents. Its morphology and retention
parameters remain provisional. The bounded sparse Hydrology Pass 2 now consumes
the volume-adjusted cell means and float64 channel beds, preserves the accepted
trunk and connector identities, reroutes local child drainage, and persists
before/after depression candidates without labeling them all as lakes. The
refined seasonal surface-water stage now conserves inherited monthly runoff,
solves fractional fill/spill states, and separates accepted standing water from
systems requiring outlet incision. The bounded outlet stage cuts narrow subgrid
beds by physical volume, preserves ordinary-cell and physical-trunk identities,
reruns routing in Rust, and repeats monthly balance to a zero-feedback gate.
Final lake coupling keeps losses on their owning branch and records unresolved
coarse/fine discharge mismatch instead of borrowing flow from a sibling
tributary. The current output is a functional prototype rather than an atlas-
grade world.

Current feature development is frozen after derived biomes until the global
map-export milestone is accepted. Hydrology and biosphere work is bugfix-only;
active product work is surface geography, elevation and orogeny, continental
margins, cartography, and their multi-seed visual gates.

The passing `earth_biosphere_v1` profile now gates a Rust-backed
`functional_vegetation` stage. It conservatively partitions each land cell among
eight continuous producer-community strategies and five nonvegetated classes,
then publishes fire, grazing, forest-resource, pasture, and crop suitability.
The six-seed `earth_functional_vegetation_v1` profile gates broad global cover,
climate-stratum response, resource-potential shape, and ensemble stability. The
dominant cover code is a hierarchical rendering/query artifact. The downstream
`derived_biomes` stage preserves a 13-component full-cell mixture while exposing
familiar primary, secondary, and landscape codes. Its `earth_biomes_v1` profile
gates broad global abundance, causal climate response, and six-seed stability.
Biome names remain derived views over canonical physical and functional state;
actual land use is not implemented.

Run the legacy rectangular fixed-seed integration gallery and framework gates:

```bash
uv run map-maker validate --config configs/validation.yaml
```

This writes `out/validation/report.json` and `out/validation/gallery.png`. A
passing command establishes deterministic artifacts, distinct seed outputs,
cache correctness, finite fields, acceptable prototype landmass morphology,
bounded longitude seams, and bounded plate-boundary relief. It does not replace
the required human gallery review or claim calibrated geological realism. This
command does not assess the canonical cubed-sphere geography.

Generate the canonical cubed-sphere geometry diagnostic:

```bash
uv run map-maker topology --face-resolution 96 --output-dir out/topology
```

This writes a globally continuous XYZ-colored cube net and a geometry report.
The canonical tectonic snapshot, age-conditioned crust state, connected
geological provinces, and initial elevation now run directly on cubed-sphere
neighbor IDs. The legacy whole-grid `erosion` stage remains on the provisional
two-dimensional path and explicitly rejects six-face fields; canonical sparse
cubed-sphere fluvial processing runs through `basin_erosion`.

Run the previous procedural generator for comparison:

```bash
uv run map-maker-legacy --width 512 --height 512 --seed 42 --out out/legacy.png
```

Render an individual pipeline diagnostic:

```bash
uv run map-maker-pipeline --stage tectonics --width 256 --height 128

# Canonical six-face tectonic snapshot
uv run map-maker-pipeline --stage tectonics --config configs/cubed_sphere_tectonics.yaml

# Canonical age-conditioned crust state
uv run map-maker-pipeline --stage world_age --config configs/cubed_sphere_crust_state.yaml

# Canonical connected geological provinces and boundary segments
uv run map-maker-pipeline --stage geology --config configs/cubed_sphere_crust_state.yaml

# Canonical pre-erosion bedrock elevation and orogenic morphology
uv run map-maker-pipeline --stage elevation --config configs/cubed_sphere_crust_state.yaml

# Canonical planetary boundary conditions and monthly orbital forcing
uv run map-maker-pipeline --stage planet --config configs/cubed_sphere_crust_state.yaml

# Atmospheric composition, hydrostatic pressure, and CO2 forcing
uv run map-maker-pipeline --stage atmosphere --config configs/cubed_sphere_crust_state.yaml

# Canonical seasonal temperature, wind, precipitation, and evaporation
uv run map-maker-pipeline --stage climate --config configs/cubed_sphere_crust_state.yaml

# Seasonal snow, firn/ice mass balance, glacier melt, and canonical runoff
uv run map-maker-pipeline --stage cryosphere --config configs/cubed_sphere_crust_state.yaml

# Canonical lakes, breaches, drainage graph, basins, and vector river reaches
uv run map-maker-pipeline --stage hydrology --config configs/cubed_sphere_crust_state.yaml

# One sparse face-2048 basin with connected trunks and conserved corridor support
uv run map-maker-pipeline --stage basin_refinement --config configs/cubed_sphere_crust_state.yaml

# Junction-consistent subgrid incision and conservative sediment routing
uv run map-maker-pipeline --stage basin_erosion --config configs/cubed_sphere_crust_state.yaml

# Bounded local rerouting and depression stability after erosion
uv run map-maker-pipeline --stage hydrology_pass2 --config configs/cubed_sphere_crust_state.yaml

# Monthly fractional lakes, wetlands, transient storage, and outlet feedback
uv run map-maker-pipeline --stage surface_water --config configs/cubed_sphere_crust_state.yaml

# One bounded outlet-incision and local-reroute pass
uv run map-maker-pipeline --stage outlet_incision --config configs/cubed_sphere_crust_state.yaml

# Iterative incision plus final zero-feedback monthly surface-water balance
uv run map-maker-pipeline --stage surface_water_final --config configs/cubed_sphere_crust_state.yaml

# Final lake storage and overflow coupled into monthly reach hydrographs
uv run map-maker-pipeline --stage lake_hydrographs --config configs/cubed_sphere_crust_state.yaml

# Hard hydrology invariants plus scale-aware Earth-reference diagnostics
uv run map-maker-pipeline --stage hydrology_validation --config configs/cubed_sphere_crust_state.yaml

# Fractional L2 surface materials, initial soils, and monthly soil water
uv run map-maker-pipeline --stage surface_materials --config configs/cubed_sphere_crust_state.yaml

# Raw environmental resources for later trait-first vegetation and ecosystems
uv run map-maker-pipeline --stage biosphere_envelope --config configs/cubed_sphere_crust_state.yaml

# Continuous potential producer-community traits without biome labels
uv run map-maker-pipeline --stage potential_biosphere --config configs/cubed_sphere_crust_state.yaml

# Per-world global totals and climate-stratum Earth diagnostics
uv run map-maker-pipeline --stage biosphere_validation --config configs/cubed_sphere_crust_state.yaml

# Conservative functional vegetation and physical resource suitability
uv run map-maker-pipeline --stage functional_vegetation --config configs/cubed_sphere_crust_state.yaml

# Per-world functional cover, climate-response, and resource-potential profile
uv run map-maker-pipeline --stage functional_vegetation_validation --config configs/cubed_sphere_crust_state.yaml

# Familiar biome mixtures derived from causal functional vegetation
uv run map-maker-pipeline --stage derived_biomes --config configs/cubed_sphere_crust_state.yaml

# Per-world Earth biome-mixture and causal-response diagnostics
uv run map-maker-pipeline --stage derived_biomes_validation --config configs/cubed_sphere_crust_state.yaml

# Canonical fixed-seed profiles plus surface-geography and biome galleries
uv run map-maker validate-biosphere --config configs/biosphere_validation.yaml
```

The canonical command writes `out/biosphere_validation/report.json`,
`surface_geography_gallery.png`, and `biome_gallery.png`. Both galleries require
human review even when every numerical gate passes.

Built wheels currently contain the Python orchestration package only. Until native
wheel bundling is implemented, run the pipeline from a source checkout after the
explicit native build, or point `MAP_MAKER_NATIVE_LIB_DIR` at compatible libraries.

## Design

- [Planet engine specification](docs/PLANET_ENGINE_SPEC.md)
- [Decision log](docs/DECISIONS.md)
- [Validation gates](docs/VALIDATION.md)
- [Working notes](docs/NOTES.md)
