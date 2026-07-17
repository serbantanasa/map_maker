# Erosion And Sedimentation

## Status

The canonical cubed-sphere path implements a first conservative fluvial pass in
`basin_erosion` for one sparse refined basin. The older `erosion` stage remains
a rectangular compatibility prototype and is not the scientific implementation
described here.

## Scientific Intent

Erosion changes subgrid landforms and moves solid material from source to sink.
An ordinary river does not lower an entire regional cell by its channel depth.
The first pass therefore solves a channel-bed constraint, converts only the
physical channel cut into sediment volume, routes that volume through the
registered reach graph, and applies overbank deposition only to allocated
floodplain support.

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

## Incision And Terrain Feedback

For every physical centerline membership:

```text
incision_depth = terrain_prior - channel_bed
eroded_volume = channel_width * in_cell_reach_length * incision_depth
```

The channel bed is a subgrid feature. The full cell mean changes only by net
physical volume divided by full cell area:

```text
cell_mean_delta = (deposited_volume - eroded_volume) / cell_area
```

This preserves both the narrow channel morphology and a restriction-compatible
terrain response without creating cell-wide trenches.

## Sediment Routing

Newly eroded solid volume is routed upstream-to-downstream through every reach.
Physical reaches may retain a bounded fraction on their allocated floodplain
support. The provisional retention fraction increases with floodplain-to-valley
support and decreases with inherited reach slope; deposition is capped by a
configured maximum support depth. Connectors transfer the complete incoming
volume without erosion or deposition.

At a registered inland sink, remaining volume enters a terminal sediment
inventory. At an ocean terminal it is exported to the future delta/shelf model.
The following identity is a hard gate:

```text
eroded = floodplain_deposited + terminal_deposited + ocean_exported
```

The inherited `sediment_load` field is an instantaneous flux diagnostic. It is
preserved on the reach catalog but is not mixed dimensionally with the newly
eroded historical volume.

## Outputs

- `ChannelBedProfileCatalog`: junction-consistent physical bed elevations,
  incision depths, lengths, and eroded volumes.
- `FluvialRiverReachCatalog`: inherited reach data plus local, incoming,
  deposited, transferred, terminal, and exported sediment budgets.
- `ErodedBasinCellCatalog`: sparse process volumes and volume-derived terrain
  mean feedback for every refined child.
- `BasinErosionParentCatalog`: child process volumes restricted to each coarse
  parent.
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
- Cell-mean and parent-restricted changes reproduce physical volumes.
- All newly eroded sediment is deposited at a registered support/sink or
  exported through a registered ocean terminal.

## Current Limits

- The least-incision envelope can expose hundreds of metres of conditioning
  where the unresolved terrain prior rises along an inherited trunk. Such cuts
  are budget-correct but not accepted calibration.
- Floodplain retention is a bounded structural approximation, not a calibrated
  settling, competence, or grain-size model.
- Sediment provenance, lithology, suspended/bed-load partition, transport time,
  avulsion, meander migration, deltas, shelves, and compaction are absent.
- The sparse Hydrology Pass 2 consumes the altered terrain distribution and
  solved channel beds, and the surface-water stage now solves monthly local
  balance. Global refined hydrology and the requested local outlet-incision
  feedback remain future work.
- Repeated geological-time erosion, uplift, sediment loading, flexure, and
  isostatic response remain future passes.
