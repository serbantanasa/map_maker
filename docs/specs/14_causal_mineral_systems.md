# Causal Mineral Systems V0

Status: implemented and validated V0, 2026-07-24.

Authority: Decisions 016 and 061. This document defines the bounded V0
deliverable; it does not authorize regional orebody geometry.

## Purpose

Create a deterministic, inspectable global/L2 inventory of geological systems
that can plausibly concentrate useful materials. The output is both simulation
state and future surrogate-training data. It must answer why a location is
prospective, not merely assign a commodity color.

## Supported Families

The stage models arc magmatic-hydrothermal, orogenic/shear,
mafic-ultramafic, volcanogenic seafloor, sediment-hosted basin, ancient
iron/cratonic, weathering/residual/supergene, placer/heavy-mineral,
evaporite/chemical-sediment, and coal-basin systems.

V0 volcanogenic-seafloor support is restricted to current eligible marine
tectonic settings. Preserved or accreted VMS on land requires explicit
paleo-seafloor lineage and remains deferred; present-day arc or orogen
membership is not a substitute for that history.

Petroleum is deferred until burial-temperature, source, maturation, reservoir,
seal, trap, migration, timing, and preservation history exists. V0 must report
that absence; it must not substitute sediment-basin membership for a petroleum
system.

## Causal Contract

Each family emits six normalized support fields:

- `MineralSourceSupport`;
- `MineralProcessSupport`;
- `MineralTransportSupport`;
- `MineralTrapSupport`;
- `MineralTimingSupport`; and
- `MineralPreservationSupport`.

`MineralSystemPotential` combines all six supports. A missing causal link must
materially reduce potential. `MineralSystemConfidence` combines geological,
process, and observation confidence independently from potential. A bounded
seeded perturbation may represent unresolved subcell variability only after
the causal score is formed. Its spatial frequencies scale with the parent face
resolution, and its potential modifier is limited to approximately one percent
around the causal score so it cannot determine regional system identity.

The first commodity axis contains copper, gold, silver, lead-zinc,
nickel-cobalt, chromium-PGE, iron, uranium, phosphate, bauxite,
tin-tungsten, heavy-mineral sands, salt, potash, and coal. These fields are
relative prospectivity, not reserves or prices.

## Scale And Persistence

The global cubed-sphere cell remains a coarse regional support volume. Persist
all causal supports, potentials, confidence, dominant-system codes, commodity
prospectivity, model metadata, a mineral-system catalog, and a major-deposit
candidate catalog.

System IDs are stable combinations of family and geological province.
Candidate IDs are stable combinations of family and host cell. Candidate
selection uses deterministic topology-aware local maxima and declared spacing,
potential, and confidence thresholds. Size, grade, exposure, and depth are
relative probabilistic descriptors. They do not claim a cell-sized orebody.

L3 later conditions on these fields and IDs to realize subcell veins, seams,
lenses, placers, alteration halos, and weathering profiles. L3 may reject or
split a coarse candidate, but must preserve its lineage.

## Inputs

V0 consumes:

- geological province identity/class, crust age, rock strength, sediment
  accommodation, and confidence;
- convergence, divergence, shear, subduction, hotspot, uplift, subsidence,
  compression, extension, and lithosphere stiffness;
- physical ocean/shelf state, elevation, relief, and topology-aware
  geomorphic slope;
- climate temperature, precipitation, and aridity;
- drainage area, stream power, river, floodplain, lake, and wetland support;
- bedrock, alluvial, lacustrine, volcaniclastic, regolith, soil, salinity,
  drainage, hydric, and confidence fields; and
- potential productivity, biomass, vegetation cover, and biosphere confidence.

## Hard Acceptance

The stage fails unless:

- every array is finite and bounded in `[0, 1]`, except categorical codes;
- system and commodity axes match their versioned catalogs;
- terrestrial systems are zero over open ocean and seafloor systems remain
  tied to eligible marine tectonic settings;
- potential is reproducible from persisted causal supports and bounded
  unresolved variability;
- dominant codes reconstruct exactly from potential;
- catalog IDs are unique and stable, candidates are valid local maxima, and
  every candidate references a persisted system;
- each supported family has nonzero eligible support and passes its declared
  directional enrichment comparison;
- replay is checksum-identical; and
- a fixed multi-seed screen passes execution, causality, and non-collapse
  gates without enforcing commodity abundance quotas.

The canonical pipeline command must return failure when the per-world hard
gate is red. A regional handoff must reject the same state before publication.

## Diagnostics

Publish:

- a dominant-system world map with named legend;
- a commodity-prospectivity atlas with named panels;
- a major-candidate map keyed by family; and
- tabular causal enrichment and abundance KPIs.

All projected maps identify the projection and include a physical scale tied
to the configured planet radius.

## Explicit Deferrals

V0 does not provide petroleum, mine economics, extraction technology, exact
tonnage, economic cutoff grade, legal ownership, L3 orebody geometry, thermal
coal rank, diagenetic fluid simulation, or fully simulated deep-time
metallogenic events. These require later causal stages rather than additional
random fields.
