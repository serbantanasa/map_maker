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

Every native library currently exposes ABI version `2`. `map-maker doctor`
verifies that ABI and reports the binary SHA-256 fingerprint. Simulation-library fingerprints
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

Generate the implemented world stack through erosion:

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
reaches. Explicit geological event history, spherical
erosion and sediment feedback, hydrology pass 2, soils, biomes, and regional
refinement remain implementation milestones; the current output is a functional
prototype rather than an atlas-grade world.

Run the fixed six-seed integration gallery and provisional hard gates:

```bash
uv run map-maker validate --config configs/validation.yaml
```

This writes `out/validation/report.json` and `out/validation/gallery.png`. A
passing command establishes deterministic artifacts, distinct seed outputs,
cache correctness, finite fields, acceptable prototype landmass morphology,
bounded longitude seams, and bounded plate-boundary relief. It does not replace
the required human gallery review or claim calibrated geological realism.

Generate the canonical cubed-sphere geometry diagnostic:

```bash
uv run map-maker topology --face-resolution 96 --output-dir out/topology
```

This writes a globally continuous XYZ-colored cube net and a geometry report.
The canonical tectonic snapshot, age-conditioned crust state, connected
geological provinces, and initial elevation now run directly on cubed-sphere
neighbor IDs. Erosion remains on the provisional two-dimensional path and
explicitly rejects six-face fields.

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

# Canonical seasonal temperature, wind, precipitation, snow, and runoff potential
uv run map-maker-pipeline --stage climate --config configs/cubed_sphere_crust_state.yaml

# Canonical lakes, breaches, drainage graph, basins, and vector river reaches
uv run map-maker-pipeline --stage hydrology --config configs/cubed_sphere_crust_state.yaml
```

Built wheels currently contain the Python orchestration package only. Until native
wheel bundling is implemented, run the pipeline from a source checkout after the
explicit native build, or point `MAP_MAKER_NATIVE_LIB_DIR` at compatible libraries.

## Design

- [Planet engine specification](docs/PLANET_ENGINE_SPEC.md)
- [Decision log](docs/DECISIONS.md)
- [Validation gates](docs/VALIDATION.md)
- [Working notes](docs/NOTES.md)
