# Planet Engine Spec

Status: canonical product and scientific scope. The [decision log](DECISIONS.md)
owns approved implementation choices, the [roadmap](ROADMAP.md) owns current
sequencing, and live stage contracts remain under `specs/`. Completed and
superseded contracts are indexed in the [archive](archive/README.md).

## Project Definition

The project is a causally plausible, beautiful, multi-resolution world stack
generator for a long-running civilization simulation game.

The generator should produce geography that can explain later human outcomes:
bands, migration, warlord states, trade routes, fertile river valleys, ports,
mountain passes, chokepoints, mineral provinces, coal basins, and harsh
frontier zones.

The terrain is not just painted output. It is the latest visible frame of a
plausible planetary and surface-process history.

## Product Goal

Create a beautiful world that can be used as feedstock for a simulation game.
The game mechanics are out of scope for this work, but world generation should
produce the causal layers those mechanics will eventually consume.

Core world-stack outputs include:
- Elevation, slope, bathymetry, shelves, coastlines.
- Oceans, lakes, wetlands, rivers, deltas, drainage basins.
- Climate, seasonality, rainfall, temperature, aridity.
- Erosion, sedimentation, floodplains, incision, basin history.
- Soils, fertility, biome tendencies, terrain constraints.
- Lithology, crustal context, mineral systems, coal/energy potential.
- Regional hierarchy: continents, watersheds, valleys, coastlands, uplands,
  deserts, passes, and strategic corridors.

## Philosophy

The engine pursues structural realism, not numerical geophysics.

Structural realism means:
- Correct causal relationships.
- Plausible geological and climatic history.
- Believable morphology.
- Believable statistics.
- Useful downstream simulation layers.

It does not mean research-grade PDE solvers or exact Earth-system modeling.

Examples:
- Do not place a mountain range directly. Generate tectonic convergence,
  uplift, erosion, and resulting mountain morphology.
- Do not paint a desert directly. Generate insolation, winds, rain shadow,
  continentality, aridity, and resulting biome/soil stress.
- Do not draw rivers decoratively. Generate basin water balance, fill-spill
  behavior, overflow carving, drainage stabilization, and discharge-aware
  river geometry.

## Initial Scope

V1 is calibrated first against Earth-like dense rocky water worlds:
- Earth-like radius, gravity, and density.
- Liquid oceans.
- Active tectonics.
- Silicate crust.
- An Earth-like atmosphere as the default profile.
- Stable main-sequence star in a conservative habitable range.
- Configurable but bounded obliquity, eccentricity, rotation, and moon
  parameters.

Earth is a calibration profile, not the engine's permanent operating boundary.
The causal state and artifact contracts must remain usable for snowball worlds,
ice-cap-free hothouses, oceanic archipelagos, high-CO2 worlds, and other dense
rocky water-world experiments. Earth-relative ranges are versioned diagnostics;
they are not physical validity gates. Physical/accounting invariants remain
hard gates.

That distinction does not imply that every such world is already simulated
credibly. Current implementation limits include Earthlike planet-config bounds,
a bounded climate-temperature kernel, atomic L2 land/ocean cells,
thermodynamic-only sea ice, incomplete ice-sheet dynamics, and no calibrated
non-Earth scenario suite.
These are capability gaps to expose and remove, not assumptions to embed in
downstream artifacts.

V1 non-goals:
- Arbitrary planet chemistry, gas giants, and unconstrained exotic physics.
- Human/civilization simulation.
- Local human-scale tactical maps as part of the core planet engine.
- Research-grade physical simulation.

## Approved Decisions

The [Decision Log](DECISIONS.md) is the only current list of numbered decisions,
their status, and supersession. Do not duplicate that growing index here.
Decisions 010 and 011 remain tentative prototype hypotheses; later decisions
explicitly supersede narrower parts of earlier contracts where stated.

## Geological Synthesis Model

The engine uses a hybrid causal model:

```text
coarse geological history
  -> crust, terranes, events, basins, and geological provinces
  -> process constraints and provenance
  -> specialized terrain morphology
  -> climate, hydrology, erosion, and sedimentation
  -> realism validation and causal replay
```

The geological simulation is responsible for explaining why features exist
and constraining their location, age, scale, material, and broad morphology.
It is not responsible for deriving every detailed ridge or valley directly
from approximate geodynamics.

Continental collision is stateful and spatially segmented. Neighboring parts
of one plate boundary may simultaneously contain mature continental collision,
active oceanic subduction, remnant seas, rotating microplates, and back-arc
extension. Sedimentary basins and stratigraphic packages persist through
deposition, burial, inversion, uplift, and erosion.

## Design Questionnaire Progress

Current substantive-design completion estimate: approximately 90 percent.

Tentative prototype hypotheses do not count as fully settled until their
failure modes have been exercised in code.

Settled at the architectural level:
- Product goal and game-feedstock scope.
- Structural-realism target and process-constrained synthesis.
- Earth-like planetary boundary conditions.
- Cubed-sphere canonical topology.
- Multi-resolution strategy.
- Persistent immutable artifacts and stage contracts.
- Basin-aware hydrology and graph/vector rivers.
- V1 world stack and first vertical-slice scope.
- Terrane-, event-, and basin-aware geological history model.
- Seasonal intermediate-complexity climate model.
- Basin-scale historical erosion, sedimentation, and isostatic feedback.
- Topological, physical, statistical, scenario, and visual hydrology gates.
- Property-first soils, terrain catenas, and functional vegetation mixtures.
- Causal mineral/energy systems, prospectivity, and major deposit catalogs.
- Probabilistic subgrid priors and constrained deterministic refinement.
- Separate diagnostic truth renders and versioned physical atlas rendering.
- Hard realism gates plus separate subsystem score dimensions and benchmarks.
- Rust-owned simulation loops with Python-owned orchestration and analysis.

Still requiring substantive decisions:
- Explicit geological-history duration and inherited initial crust.
- Implementation milestones and the first executable geological scenario.

## Neural Surrogate Direction

The simulator should be useful without neural networks.

However, every stage should preserve enough structured inputs and outputs that
future surrogate models can be trained from generated runs.

Likely future surrogates:
- Erosion surrogate.
- Hydrology surrogate.
- Soil/biome surrogate.
- Preview/downscaling surrogate.
- Eventually, a full-stack near-instant generator if evidence supports it.

Implication:
Stage outputs should be deterministic, versioned, inspectable, and stored in
training-friendly arrays/tables with config, seed, and software version
metadata.

Any surrogate that crosses planetary scenarios must also receive the planetary
and environmental conditioning vector used by the simulator. A model trained
only on Earth-default outputs is an Earth-profile emulator, not a general world
generator.

## Mineral Systems And Energy Resources

Resource generation is a first-class objective.

The engine should not randomly place "iron" or "coal". It should produce
geological histories that make deposits plausible:
- Magmatic deposits.
- Hydrothermal systems.
- Sedimentary deposits.
- Weathering/residual deposits.
- Placer deposits.
- Supergene enrichment.
- Coal basins.
- Evaporites and salt basins.

Outputs should include resource potential and deposit catalogs rather than only
painted resource points.

Likely layers:
- LithologyMap.
- CrustAge.
- StructuralLineaments.
- MetallogenicProvinceMap.
- OrePotentialByCommodity.
- CoalBasinPotential.
- EvaporiteBasinPotential.
- PlacerPotential.
- DepositCatalog.
- ResourceConfidence.
- DepthOrExposure.
- AccessibilityHints.

## First Vertical Slice

Target: small global L0-L1 plus one selected L3 region.

The vertical slice should prove:
- Cubed-sphere topology and wraparound.
- Persistent run directory.
- Deterministic config/seed behavior.
- Tectonics and crust/world-age layers.
- Elevation and early erosion.
- First-pass climate.
- Depression-aware hydrology.
- Inspectable layer stack.
- Projected map export.
- One refined region suitable for later game use.

## Open Topics

Topics still needing discussion:
- Explicit geological-history duration and inherited initial crust model.
- Zarr chunk shape and compression defaults.
- Exact cubed-sphere indexing and projection math.
- Artifact metadata schema details.
- First vertical-slice acceptance tests.
