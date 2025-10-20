# Code Style (Python-first; Rust optional later)

## Language & Versions
- Python 3.11+ (required)
- Optional Rust 1.78+ for future perf modules (via `pyo3`)

## Formatting & Linting (strict)
- `black` (line length 100)
- `ruff` (enable: F,E,I,UP,ANN,SIM,PL,NPY)
- `mypy` strict (`disallow_untyped_defs = True`, `warn_unused_ignores = True`)
- `pytest` for tests; `pytest -q`

## Conventions
- Type hints everywhere; no implicit `Any`.
- Pure functions preferred in `core/`; side-effects isolated to `io/` and `cli/`.
- Functions ≤ 50 LOC; split if bigger.
- RNG usage: pass `np.random.Generator` explicitly, seeded in one place.
- Naming:
  - arrays: `elev`, `temp`, `moist`, `flow_acc`
  - constants: `UPPER_SNAKE`
  - functions: `verb_noun`
- Docstrings: Google style, with argument/return types.
- Logging: `logging` (no prints in library code). CLI may print succinct run stats.

## Dependencies (v0.1)
- `numpy`, `Pillow`, `noise`, `pyyaml`, `pytest`, `black`, `ruff`, `mypy`
- Keep the graph small; propose new deps via PR.

## File Layout
src/map_maker/
init.py
cli.py
core/
elevation.py
climate.py
moisture.py
biomes.py
rivers.py
io/
render.py
save.py
tests/


## Comments & TODOs
- Use `# TODO(vX.Y): reason` with short, actionable intent.

## Commit Messages (Conventional Commits)
- `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`
- Example: `feat: add D8 flow accumulation and basic rivers`