# Erosion And Sedimentation

## Status

The canonical cubed-sphere path implements a conservative prospective fluvial
constraint pass in `basin_erosion` for one sparse refined basin. It does not
apply physical terrain erosion. The older `erosion` stage remains a rectangular
compatibility prototype and is not the scientific implementation described
here.

## Scientific Intent

Erosion changes subgrid landforms and moves solid material from source to sink.
An ordinary river does not lower an entire regional cell by its channel depth.
The sparse pass therefore solves a candidate channel-bed constraint and routes
the prospective channel-prism volume through the registered reach graph, while
leaving applied terrain erosion and deposition at zero. A future regional pass
at roughly `100-250 m` owns physical valley realization.

This is structural realism rather than a geological-time landscape evolution
model. It establishes correct topology and budgets before adding lithology,
weathering, hillslope diffusion, mass wasting, lateral migration, delta growth,
compaction, flexure, or repeated hydrology feedback.

## Inputs

- Sparse refined child terrain and physical cell areas.
- Source-to-sink-ready physical reaches and zero-width connectors.
- Ordered centerline memberships with physical width and in-cell length.
- Conserved valley and floodplain support fractions.
- Reach slope, terminal class, and downstream reach identity.
- Coarse potential incision and sediment-load fields as diagnostics. They are
  not silently converted into calibrated fine-scale process history.

## Bed Profiles

Physical centerline memberships create a directed node graph. A node shared by
several reaches at a confluence has one bed elevation. Starting from the refined
terrain prior, the native solver constructs the greatest downstream-graded
surface satisfying:

```text
bed_downstream <= bed_upstream - minimum_bed_slope * edge_length
bed <= terrain_prior
```

This is the least-incision downstream envelope. It removes only the material
required to condition the unresolved terrain prior into a routable bed. It does
not force the inherited coarse potential-incision volume into the fine terrain.
That potential remains an explicit comparison metric.

Connectors have no physical bed. A channel component on either side of an
unresolved lake or depression is graded independently until local waterbody and
outlet geometry replaces the connector.

## Candidate Incision And Terrain Ownership

For every physical centerline membership:

```text
candidate_depth = terrain_prior - candidate_channel_bed
prospective_volume = channel_width * in_cell_reach_length * candidate_depth
```

The candidate bed is a vector constraint, not a raster elevation. At the
canonical factor-16 resolution, children are still about `4.5 km` across, so
even a volume-correct cell mean would attach channel morphology to the wrong
spatial support. This stage therefore enforces:

```text
applied_terrain_erosion = 0
applied_terrain_deposition = 0
cell_mean_delta = 0
terrain_after = terrain_prior
```

Hydrology retains the inherited trunk receiver graph. Regional refinement must
discover smaller tributaries and realize channel, bank, valley, and floodplain
morphology against a finer terrain and material state.

## Sediment Routing

Prospective solid volume is routed upstream-to-downstream through every reach.
Physical reaches may retain a bounded fraction on their allocated floodplain
support. The provisional retention fraction increases with floodplain-to-valley
support and decreases with inherited reach slope; deposition is capped by a
configured maximum support depth. Connectors transfer the complete incoming
volume without erosion or deposition.

At a registered inland sink, remaining volume enters a terminal sediment
inventory. At an ocean terminal it is exported to the future delta/shelf model.
The following identity is a hard gate:

```text
prospective_excavation = prospective_floodplain_deposition
                         + prospective_terminal_deposition
                         + prospective_ocean_export
```

The inherited `sediment_load` field is an instantaneous flux diagnostic. It is
preserved on the reach catalog but is not mixed dimensionally with the newly
eroded historical volume.

## Outputs

- `ChannelBedProfileCatalog`: junction-consistent physical bed elevations,
  incision depths, lengths, and eroded volumes.
- `FluvialRiverReachCatalog`: inherited reach data plus local, incoming,
  deposited, transferred, terminal, and exported sediment budgets.
- `ErodedBasinCellCatalog`: prospective process budgets, explicit zero applied
  process fields, and unchanged terrain for every refined child.
- `BasinErosionParentCatalog`: prospective child budgets restricted to each
  coarse parent, plus zero applied parent feedback.
- `BasinErosionMetadata`: profile, connector, exclusion, and conservation gates
  plus canonical totals.

## Hard Gates

- The inherited reach graph is source-to-sink ready and acyclic.
- Every shared physical junction has one bed elevation.
- Every physical edge meets the configured downstream grade.
- Bed elevation never exceeds the terrain prior in this incision-only pass.
- Connectors have no bed, incision, support deposition, or local sediment.
- No process volume enters a preserved-depression parent.
- Membership volumes equal width times length times incision depth.
- Applied cell-mean and parent-restricted changes remain exactly zero.
- All prospective sediment is deposited at a registered support/sink or
  exported through a registered ocean terminal.

## Current Limits

- The least-incision envelope can expose hundreds of metres of conditioning
  where the unresolved terrain prior rises along an inherited trunk. Such cuts
  are budget-correct but not accepted calibration.
- Floodplain retention is a bounded structural approximation, not a calibrated
  settling, competence, or grain-size model.
- Sediment provenance, lithology, suspended/bed-load partition, transport time,
  avulsion, meander migration, deltas, shelves, and compaction are absent.
- Sparse Hydrology Pass 2 uses unchanged terrain plus fixed vector trunk
  receivers. Surface materials consume only explicitly applied process fields.
- The bounded lake-outlet correction is a separate hydraulic repair and does
  not authorize coarse trunk or bank erosion.
- Repeated geological-time erosion, uplift, sediment loading, flexure, and
  isostatic response remain future passes.
