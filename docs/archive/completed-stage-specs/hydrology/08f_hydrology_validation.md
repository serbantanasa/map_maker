# Hydrology Validation Profile

## Status

Implemented, provisional. The `hydrology_validation` stage runs after final
surface-water balance and writes `HydrologyKpiCatalog`,
`HydrologyReachLossCatalog`, and `HydrologyValidationMetadata`.

## Evaluation Model

Each KPI is one of:

- `hard_invariant`: topology, terminal continuity, discharge continuity, or
  conservation. Failures remain visible in the output instead of aborting the
  reporting stage.
- `earth_diagnostic`: a global generated quantity compared with a versioned
  Earth reference envelope.
- `diagnostic`: a reported quantity with no valid reference distribution yet.
- `capability`: an explicit implemented/not-implemented boundary.

Earth comparisons are scale-aware. Lake counts depend on minimum mapped size;
wetland area depends on ecological and inundation definitions; drainage density
depends on channel-extraction threshold; and a single basin requires analogs
matched by climate, relief, geology, and glacial history. These quantities are
not tuned against one unconditional global mean.

## Current KPI Families

- Candidate graph validity, water balance, spill fraction, throughflow, losses,
  and maximum linked spill-chain length.
- Reach terminal resolution, allowed terminal paths, preserved trunks, dead
  ends, and mean/monthly downstream discharge regressions.
- Reach/month discharge decreases with their responsible registered depression
  and waterbody IDs. Only decreases attributed to explicit storage pass.
- Global and selected-basin lake/wetland area, closed drainage, runoff depth,
  and seasonal hydrograph concentration.
- Seasonal snow-affected area, genuinely perennial snow area, glacierized area,
  snowmelt and glacier-melt shares, and peak melt/runoff month.
- Implemented capability markers for glacier mass balance and lake-to-reach
  hydrographs; explicit gaps for monthly floodplain inundation and ecological
  wetland confirmation.

## Earth Reference Profile V1

- Messager et al. (2016), HydroLAKES:
  <https://doi.org/10.1038/ncomms13603>
- Verpoorter et al. (2014), high-resolution global lake inventory:
  <https://doi.org/10.1002/2014GL060641>
- Prusevich et al. (2024), MERIT-Plus endorheic basins:
  <https://doi.org/10.1038/s41597-023-02875-9>
- Dai and Trenberth (2002), continental freshwater discharge:
  <https://doi.org/10.1175/1525-7541(2002)003%3C0660:EOFDFC%3E2.0.CO;2>
- Davidson et al. (2018), global wetland extent:
  <https://doi.org/10.1071/MF17019>
- Nardi et al. (2019), GFPLAIN250m:
  <https://doi.org/10.1038/sdata.2018.309>

## Current Boundary

The Rust cryosphere stage now owns canonical seasonal snow, multi-year firn/ice
mass balance, parameterized downslope transfer, and separate snow- and
ice-melt runoff fluxes. It is not yet a calibrated stress-driven glacier or
ice-sheet model. Floodplain support remains geomorphic rather than a monthly
flood simulation. Refined hydrologic wetlands are not ecological wetland
labels.
