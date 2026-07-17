# Refined Seasonal Surface-Water Balance

## Status

Implemented provisional stage following Hydrology Pass 2.

## Objective

Convert resolved local topographic storage candidates into seasonal
surface-water states. The stage answers whether a candidate is dry, transient,
seasonal, permanently inundated, or hydrologically wetland-like while retaining
fractional cell coverage and a complete monthly water budget.

It does not re-route terrain, alter the accepted river trunk, regenerate
inherited coarse waterbodies, or claim ecological wetland certainty.

## Inputs

- `hydrology_pass2`: stabilized receiver DAG, local candidate membership,
  spill elevations, child areas, terrain, and unresolved parent relief.
- `climate`: monthly runoff potential and provisional evaporation.
- `geology`: sediment accommodation used to modulate seepage.
- `basin_refinement`: refinement factor and represented-parent contract.

Monthly parent climate depths are inherited by every child. Multiplication by
conserved child area preserves represented parent volume exactly.

## Catchments And Candidate Graph

The stabilized cell graph is traversed downstream-to-upstream. A source belongs
to the first local candidate encountered on its downstream path. This produces
disjoint direct catchments and excludes water that never encounters a local
candidate.

Each candidate's spill receiver is followed to the next downstream candidate.
The resulting candidate DAG is processed upstream-to-downstream each month so
overflow can fill, spill, or seasonally sustain a downstream system.

## Subcell Hypsometry

Candidate cells use the Pass-2 spill elevation as their maximum water surface.
Parent relief is reduced by the square root of the refinement factor to retain
unresolved child-scale relief. A bounded uniform subcell elevation distribution
then supplies area and volume as functions of water level. A connected-basin
fraction caps the low subcell area assigned to one unresolved local waterbody;
the cap does not apply to inherited large lakes, which this stage excludes.

The native kernel precomputes a monotone hypsometric lookup for every candidate.
Monthly storage is inverted through that curve, producing water level, area,
depth, and per-cell inundation fractions without turning a five-kilometre child
into an all-water pixel.

## Monthly Balance

For every climatological month:

1. Add direct catchment runoff and upstream candidate overflow.
2. Estimate open-water evaporation and geology-modulated seepage over the
   current inundated area.
3. Apply losses no greater than available storage.
4. Retain water up to spill capacity and route excess downstream.
5. Persist end-of-month storage, area, losses, overflow, and cell fractions.

Because the candidate graph is acyclic, each candidate's climatological annual
cycle is solved upstream-to-downstream as a bounded periodic-storage fixed
point. This removes arbitrary transient spin-up years while retaining a strict
year-boundary convergence check against candidate capacity.

## Outlet-Erosion Feedback

The first monthly solve retains the Pass-2 sill. Candidates with sustained
overflow are then scored from available head, rock weakness, low sediment
accommodation, and overflow discharge. A lake-like candidate above the
configured score and discharge thresholds is published as `transient_storage`
with a recommended outlet incision. Its monthly fields remain the conservative
pre-incision fill/spill state; they are not accepted standing-water coverage.

This stage does not silently alter terrain or reroute the accepted graph. A
future bounded outlet-incision pass consumes the feedback and reruns local
stabilization.

## Persistent Outputs

- `SurfaceWaterCandidateCatalog`: one row per Pass-2 candidate, pre-adjustment
  and accepted class, outlet-erosion feedback,
  catchment, capacity, hydroperiod, annual budget, convergence, and monthly
  fixed-size-list fields.
- `SeasonalSurfaceWaterCellCatalog`: one row per candidate child with potential,
  minimum, mean, maximum, and twelve monthly inundation fractions.
- `SurfaceWaterMonthlyStateCatalog`: normalized candidate-month rows for direct
  inspection and surrogate training.
- `SurfaceWaterMetadata`: class counts and areas, topology, convergence,
  fractional bounds, climate inheritance, and conservation diagnostics.

## Hard Gates

- Candidate and receiver identities are valid and both cell and candidate
  graphs are acyclic.
- Direct catchments are disjoint and include only active source area.
- Every monthly fraction is finite and in `[0, 1]`; storage is finite and inside
  candidate capacity.
- Parent-to-child climate inheritance conserves represented runoff volume.
- Final-year direct inflow equals evaporation plus seepage plus terminal
  overflow plus storage change within configured tolerance.
- Repeated inputs produce byte-identical catalogs and cache keys.

## Current Limits

- Monthly runoff and evaporation are inherited from the coarse parent without
  local storm, aspect, groundwater, or rain-shadow downscaling.
- Climate evaporation remains a provisional proxy for open-water potential
  evaporation and is exposed through a configurable multiplier.
- Uniform subcell relief is a structural approximation, not surveyed
  bathymetry.
- Seepage uses sediment accommodation; explicit soil permeability and aquifers
  do not yet exist.
- Hydrologic wetland output is an input to future soils and biomes, not their
  final classification.
