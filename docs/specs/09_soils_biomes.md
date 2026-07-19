# Soils And Functional Biomes

## Status

Split into bounded milestones. Fractional surface materials and initial soils
are implemented in `09a_surface_materials_initial_soils.md`. The environmental
and biosphere resource envelope is implemented in
`09b_environmental_biosphere_envelope.md`. The trait-first potential biosphere
is implemented in `09c_trait_first_potential_biosphere.md`. Functional vegetation
mixtures, biome labels, and the optional one-pass feedback remain later work.
The pre-15b2 Earth calibration contract is implemented in
`09d_earth_biosphere_validation.md`.

## Implemented Foundation

The `surface_materials` stage publishes:

- fractional L2 surface-material map units,
- initial mineral-soil depth, texture, chemistry, and water capacity,
- monthly soil storage, saturation, evapotranspiration, runoff, and drainage,
- hydric-soil evidence without declaring ecological wetlands.

## Milestone 15b0: Environmental Envelope

Before assigning plant functional types, the pipeline publishes the raw
monthly light, temperature, liquid-water, atmospheric-substrate, oxygen, and
land-surface support fields. This prevents an Earth biome classifier from
becoming canonical state and supplies explicit conditioning inputs for future
surrogates.

## Milestone 15b1: Trait-First Potential Biosphere

The pipeline converts physical opportunity into continuous potential
producer-community state: NPP, cover, biomass, growing season, adaptation
pressures, allocation strategy, rooting depth, canopy height, leaf area, fuel
continuity, and confidence. These are equilibrium potentials under an explicit
photosynthetic-colonization assumption, not proof that evolution produced the
same organisms on every world.

## Milestone 15b2: Functional Vegetation

15b2 calibration must not begin until the `earth_biosphere_v1` per-world and
multi-seed reports exist. Those reports now pass hard invariants, ensemble
stability, and the global and land-mean NPP and biomass ranges at face 64 and
face 128. The complete profile also accepts the approved `35%` generated-
Earthlike landmass. A later model that separates continental crust, shelf, sea
level, and emerged land must rerun the profile rather than inherit this result.

Canonical vegetation will use mixtures of plant functional types rather than a
single painted biome code. Required outputs are:

- plant functional-type fractions,
- potential biomass and net primary productivity,
- canopy cover and rooting depth,
- growing-season length and seasonality,
- fire and grazing tendency,
- forest, pasture, and crop potential,
- bare, saline, wetland, ice, and other non-vegetated fractions.

Familiar biome and climate labels are derived products for rendering and game
queries. They are not primary simulation state.

High carbon dioxide is likewise not a giant-life or high-productivity switch.
Potential productivity and organism scale remain constrained by energy,
oxygen, pressure, gravity, temperature, water, nutrients, and organism traits.

## Bounded Feedback

After initial vegetation, V1 may update organic carbon, infiltration, soil
moisture, and erosion resistance once. Any change to runoff must produce a new
hydrology artifact and rerun the relevant conservation and topology gates. The
pipeline may not silently mutate the accepted hydrology products.

## Deferred

- Individual species and ecological succession.
- Human land use.
- Detailed wildfire, pest, and disturbance history.
- A GPU backend before the CPU model and scientific contract are calibrated.
