# Canonical Elevation And Orogenic Morphology

## Status

V1 implementation target. This stage creates pre-erosion bedrock elevation on
the canonical cubed sphere. It does not finalize sea level, coastlines,
drainage, sediment, or eroded landforms.

## Purpose

Convert crustal state and geological process evidence into a continuous,
seam-free initial topographic surface. The result must express broad crustal
buoyancy, ocean-basin depth, sedimentary accommodation, and localized tectonic
relief without turning geological labels into fixed elevation classes.

## Inputs

- Cubed-sphere geometry, cell areas, and reciprocal D4 neighbors.
- Plate crust type, thickness, density, and motion.
- Crust thickness, isostatic offset, uplift, subsidence, compression,
  extension, shear, stiffness, and proto-ocean mask.
- Persisted spherical hotspot events. The noisy tectonic thermal-potential
  raster is not interpreted directly as volcanic topography.
- Crust age, rock strength, sediment accommodation, province confidence,
  boundary regime, boundary confidence, and boundary segment identity.
- A deterministic stage seed and versioned morphology parameters.

## Scientific Model

Elevation is the sum of independently inspectable causal components:

1. **Crustal buoyancy** supplies broad continental freeboard and age-dependent
   ocean-basin depth from crust thickness, density, and isostatic state.
2. **Basin subsidence** lowers extensional, sediment-accommodating, and old
   oceanic crust without forcing all cells in a named basin to one height.
3. **Orogenic morphology** localizes relief around active process corridors.
   Collision, subduction, spreading, rifting, and transform regimes use
   different cross-strike profiles.
4. **Correlated structural variation** modulates relief coherently along and
   across belts. It may not introduce cube-face seams or independent per-cell
   noise.

Geological province classes are evidence and modifiers. They may adjust
strength, accommodation, confidence, and plausible morphology, but they are
not elevation bins.

## Boundary Morphology

- Continental collision creates broad, high, asymmetric belts whose amplitude
  follows compression and whose width responds to lithosphere strength.
- Continental subduction creates a narrow oceanward trench and an inland
  volcanic/magmatic arc. The oceanic and continental sides are identified from
  crust state rather than arbitrary edge ordering.
- Intra-oceanic subduction creates a trench plus a displaced island-arc ridge.
- Spreading ridges create broad positive oceanic relief centered on the
  boundary and age/depth gradients away from it.
- Continental rifts create an axial depression with lower shoulders and
  elevated roughness, not a mountain wall.
- Transform boundaries create narrow, low-amplitude structural relief and may
  orient later drainage, but do not receive collision-scale uplift.
- Hotspot events create isolated finite-width volcanic centers on their host
  plates; event strength and plume proxy modulate amplitude.

Boundary influence is propagated by angular distance over the canonical
neighbor graph. Profiles therefore cross cube-face seams continuously and use
physical angular widths rather than face-local pixel widths.

### Corridor Realization

The plate boundary graph is a causal skeleton, not the literal centerline of
every resulting landform. V2 realizes each active boundary as a process
corridor:

- Low-amplitude, higher-frequency spherical warp terms bend long plate edges
  without changing plate connectivity.
- An independent, spatially correlated activity field strengthens and weakens
  sections along a boundary while retaining a nonzero causal floor.
- Collision, arc, ridge, trench, and rift profiles receive coherent lateral
  offsets within their allowed corridors.
- Angular distance and inherited activity are smoothed only within each plate
  before profile evaluation to suppress D4 distance quantization without
  leaking evidence across unrelated boundaries.

This realization must produce connected regional systems rather than uniform
walls or disconnected decorative mountain patches.

## Outputs

- `BedrockElevationM`: signed pre-erosion bedrock elevation relative to the
  provisional zero datum.
- `CrustalElevationM`: broad buoyancy and ocean-basin component.
- `OrogenicElevationM`: collision, arc, ridge, rift-shoulder, and transform
  contribution.
- `BasinDepressionM`: positive magnitude subtracted for trenches, rifts, and
  sediment-accommodating basins.
- `TerrainReliefM`: unresolved local-relief prior for later refinement and
  erosion; it is not added wholesale to cell elevation.
- `ElevationConfidence`: confidence inherited from geological evidence and
  proximity to constrained structures.
- `ElevationMetadata`: parameters, component statistics, and model semantics.

## Required Invariants

- Every numeric output is finite and deterministic for identical inputs,
  seed, software version, and configuration.
- Reciprocal graph topology produces no cube-face discontinuity.
- Proto-continental and proto-oceanic area classifications remain distinguishable
  before final sea-level selection; isolated islands are not deleted here.
- Collision uplift is statistically concentrated near collision corridors.
- Subduction trenches occur on oceanic sides and arcs are displaced away from
  the boundary rather than painted on top of the trench.
- Stable continental interiors retain substantial elevation variation and may
  contain basins; continental-sized flat plateaus are a hard visual failure.
- Orogenic amplitude varies along long boundary segments; uniform walls are a
  hard visual failure.
- Exact boundary spines may not dominate their process-corridor flanks by an
  unbounded ratio, and long active segments must retain measurable amplitude
  variation.
- A geology class alone is insufficient to determine elevation.

## Deferred Work

- Geological-time integration of evolving elevation and sediment load.
- Flexural/isostatic response to erosion, deposition, ice, and water loading.
- Drainage-aware erosion, basin filling and overflow, sediment routing, and
  coastal reworking.
- Final sea-level solution, shoreline topology, and atlas cartography.
- Conditional higher-resolution terrain realization and restriction checks.

## Validation

- Analytic synthetic cases for collision, subduction polarity, spreading,
  rifting, and inactive boundaries.
- Multi-seed global distributions for continental freeboard, ocean depth,
  maximum relief, hypsometry, and boundary localization.
- Seam checks over every cross-face neighbor edge.
- Correlation checks between causal component fields and their source evidence.
- Fixed truth-render gallery reviewed for plateaus, polygonal walls, concentric
  halos, streaks, seam artifacts, and lost islands.
