# L3 Ecology Readiness: 2026-07-23

Status: in progress

## Objective

Turn the accepted nominal-200 m L3 terrain and hydrology into a stable substrate
for soils and biomes without changing the routed water budget, accepted terrain,
or canonical river graph.

The required outcome today is durable physical river geometry plus
ecology-facing channel, riparian, and floodplain support. L3 surface materials
and soils are the first stretch outcome. Mineral generation is not in today's
implementation scope because it requires a causal upstream mineral-system
model before regional deposit realization.

## Work Order

1. **Record and protect the baseline**
   - Confirm the current terrain and hydrology artifacts and tests are green.
   - Treat base terrain, receivers, discharge, lakes, and reach identities as
     immutable inputs.
2. **Physical river centerlines**
   - Add a derived, resumable L3 channel-geometry artifact.
   - Smooth selected raw D8 reach paths into continuous physical polylines.
   - Preserve stable reach IDs, graph endpoints, downstream order, lake
     connectors, and inherited-trunk identity.
   - Keep sub-cell channels as vectors; do not paint whole 200 m cells as river.
3. **Ecology support fields**
   - Publish distance to reported channel and distance to perennial channel.
   - Publish nested channel, riparian, floodplain, and valley influence fields
     from physical reach dimensions and local terrain.
   - Keep these as fractional/probabilistic support rather than categorical
     land-cover replacement.
4. **Validation and diagnostic**
   - Add deterministic unit and miniature end-to-end coverage.
   - Reject disconnected paths, endpoint drift, non-finite geometry,
     non-monotone path distance, broken support nesting, or checksum mismatch.
   - Render a complete regional diagnostic with a legend and labelled
     kilometre scale.
5. **Stretch: L3 surface materials and initial soils**
   - Consume inherited L2 geology/material/soil priors, L3 terrain, lakes,
     wetlands, and the new channel-support fields.
   - Realize bedrock exposure, residual regolith, alluvium, lacustrine
     sediment, soil depth, drainage class, hydric tendency, pH, and salinity.
   - Preserve provenance: inherited priors remain distinct from recomputed L3
     state.
6. **Stretch: biome-stage contract**
   - Specify how L2 climate and vegetation priors combine with L3 soils,
     hydrology, elevation, and disturbance support.
   - Keep biome output fractional and reserve mineral deposits for a separate
     causal geology/resource milestone.

## Required Acceptance

- Existing L3 terrain and hydrology tests remain green.
- No accepted terrain, receiver, discharge, lake, or reach-graph value changes.
- Smoothed polylines are deterministic, finite, downstream ordered, and tied
  to the original stable reach IDs.
- Every centerline remains within its declared raw-path corridor; endpoints
  remain anchored at graph junctions and waterbody transitions.
- Ecology support is finite over the full stored routing window and acceptance
  is reported separately for the displayed routed core.
- Channel support is no broader than riparian support; riparian support is no
  broader than floodplain/valley support under the declared semantics.
- Cold generation and cache replay verify output checksums.
- Every new map has a legend and a labelled physical scale.
- Peak memory and storage remain within the existing 24 GB and 4 GB hydrology
  budgets unless a separate budget is documented.

## Stop Conditions

Stop and record a blocker instead of weakening gates when:

- smoothing changes river topology or creates crossings that cannot be
  attributed to an existing confluence;
- a required distance field would force an unbounded global allocation;
- support widths cannot be reconciled with physical reach attributes;
- completing a stretch task would require inventing absent L2 climate,
  lithology, or mineral history.

## End-Of-Day Evidence

- Commits for each completed milestone.
- Test and lint results.
- Paths to the canonical diagnostic images and manifests.
- A short status update here marking completed, deferred, and blocked items.
