# Stage 0 – Pipeline Framework & Instrumentation

## Responsibilities
- Central orchestration of stage execution.
- Dependency resolution and caching.
- Timing, memory, and parameter logging.
- Common data structures and configuration schema.

## Data Structures & Memory Ownership
- `PipelineConfig`: parsed from YAML/JSON; includes topology choice, resolutions, run ID, per-stage overrides.
- `MemoryArena`: single run-scoped allocator managed in Rust. Provides aligned, contiguous buffers (row-major float32/float64) with 64-byte alignment for SIMD.
- `GridHandle` / `VectorHandle`: thin structs (pointer, shape, stride, dtype) referencing arena buffers; immutable once handed off.
- `PipelineContext`:
  - `topology`: instance implementing `Topology` interface.
  - `resolutions`: list of `GridInfo` objects (native + downsampled).
  - `rng_pool`: deterministic seed stream (SplitMix64 keyed by stage name + run seed).
  - `arena`: reference to `MemoryArena` for buffer allocation.
  - `cache_dir`, `log_dir`.
- `StageResult`:
  - `artifacts`: mapping of names → handles/metadata (Arrow tables, raster handles).
  - `stats`: structured metrics (timing, memory, checksum).
  - `dependencies`: names of stages consumed.

## Components
1. **Topology Loader**
   - Factory mapping config to concrete topology (sphere/cylinder/torus).
   - Precomputes wrap policies and coordinate transforms.

2. **Stage Registry**
   - Decorator `@stage(name, inputs, outputs)` registers callable (Python convenience) but stores lightweight descriptors pointing to Rust/C++ function pointers when available.
   - Validates DAG (no cycles, missing inputs).

3. **Execution Engine**
   - Builds DAG, topologically sorts, caches completed stages (hash based on inputs + config).
   - Supports dependency-based scheduling to multiple worker threads / GPU queues (submit when inputs ready).
   - Emits timing logs per stage and per subroutine (context manager `with timed("subtask")`).
   - Captures peak RSS via platform counters (psutil, Mach task info) if available.

4. **Logging**
   - JSON Lines log per run: `logs/run_<timestamp>.jsonl` (written via buffered queue, periodic flush to avoid blocking).
   - Each entry: stage, start/end, duration (ns precision using platform high-res timer), memory, cache status.
   - Optional platform integration (Apple signposts, Linux perf markers) for external profilers.
   - Companion human-readable summary (`.md`) enumerating metrics.

5. **Dataset Writer**
   - Persists rasters (COG GeoTIFF or NPZ), vector/graph data (GeoParquet/Arrow).
   - Metadata file bundling run config, stage stats, and artifact paths.

## Performance Targets
- Framework overhead <3% of total runtime.
- DAG resolution and cache lookup O(number_of_stages).
- Logging asynchronous (write queue) to avoid blocking.

## Testing Strategy
- Unit tests for:
  - DAG cycle detection, cache hits/misses, deterministic seed allocation.
  - Timing context increments log entries.
  - Topology loader returns correct class.
- Integration tests:
  - Run mock pipeline with 3 dummy stages, assert ordering and cached rerun skip.
