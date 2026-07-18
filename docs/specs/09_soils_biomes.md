# Soils And Functional Biomes

## Status

Split into bounded milestones. Fractional surface materials and initial soils
are implemented in `09a_surface_materials_initial_soils.md`. The environmental
and biosphere resource envelope is specified in
`09b_environmental_biosphere_envelope.md`. Functional vegetation, biome labels,
and the optional one-pass feedback remain planned.

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

## Next Milestone: Functional Vegetation

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
