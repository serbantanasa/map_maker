# Stage 4B: Geological Province Initialization

## Status and purpose

This stage converts the canonical kinematic plate snapshot and age-conditioned
crust state into persistent, connected geological process objects. It exists so
initial elevation does not infer terrain directly from plate IDs or from a
binary continental/oceanic mask.

Hard downstream rule: province class is not an elevation lookup. Initial
elevation must combine crustal buoyancy, boundary/event morphology, province
state, roughness distributions, and later loading/erosion. A shield is not a
plateau, and an orogen is not a fixed-width raised stripe.

V1 is an initializer, not the event-driven geological history approved in
Decision 009. Its outputs state current structural evidence and confidence.
Later history sweeps may replace classifications while retaining the artifact
and catalog contracts.

The stage currently requires the cubed sphere. The rectangular erosion stack
does not consume these artifacts.

## Inputs

- Exact spherical cell areas and global D4 neighbor IDs.
- `PlateField`, subduction potential, and tectonic thermal-anomaly potential.
- Isostatic potential, uplift, subsidence, compression, extension, shear,
  continental-margin proximity, lithosphere stiffness, and proto-oceanic crust.
- Planet age from `WorldAgeMetadata`.

## Province model

Every cell receives one evidence class:

- Shield.
- Stable platform.
- Sedimentary basin.
- Orogen.
- Continental rift.
- Continental arc.
- Shelf or passive margin.
- Abyssal basin.
- Oceanic ridge.
- Intra-oceanic arc.
- Volcanic province (reserved; not emitted by this initializer).

Class evidence is ordered so active tectonic settings take precedence over
stable interiors. Sedimentary basins require tectonically quiet accommodation;
low-amplitude boundary subsidence is not promoted to a basin. The volcanic
province code is reserved because the current noisy thermal-anomaly proxy cannot
support a coherent province object or establish a modeled mantle plume.

After evidence classification, components below a class-specific physical
spherical-area threshold merge into the longest adjacent same-crust class.
This suppresses resolution-scale threshold fragments without deleting resolved
linear rifts, ridges, shelves, arcs, or orogens. An undersized component with no
same-crust neighbor is an isolated crust domain, such as a coarse island; it is
preserved under Decision 011 and its confidence is capped at `0.5` rather than
being merged across the land/ocean boundary.

Global D4 connected-component labeling assigns stable dense province IDs. A
province catalog records class, cell count, exact spherical area, parent plate
when unambiguous, area-weighted crust age, rock strength, sediment
accommodation, and confidence. Provinces may cross plates; plates and geological
provinces are deliberately different object systems.

## Boundary segments

Every undirected D4 edge whose endpoints belong to different plates receives a
deterministic unordered plate pair and one current kinematic regime:

- Inactive boundary.
- Continental collision.
- Subduction margin.
- Intra-oceanic subduction.
- Continental rift.
- Spreading ridge.
- Transform.

Each physical edge is classified once from endpoint-averaged evidence, then
written identically to both reciprocal directed edge slots. Confidence-weighted
along-boundary smoothing and a minimum angular segment length suppress
one-edge regime flips. Edges with the same plate pair and regime are labeled
into globally connected segments using face-crossing D4 boundary adjacency.
The segment catalog stores the plate pair, regime, edge count, approximate
angular length, mean process evidence, and confidence. The current area-derived
length is diagnostic, not a survey-grade boundary polyline.

## Dense outputs

- `GeologicalProvinceID` (`int32`).
- `GeologicalProvinceClass` (`uint8`).
- `CrustAgeGa` (`float32`).
- `RockStrength` (`float32`, `[0, 1]`).
- `SedimentAccommodation` (`float32`, `[0, 1]`).
- `ProvinceConfidence` (`float32`, `[0, 1]`).
- `BoundarySegmentID` (`int32`, shape `(6, n, n, 4)`, `-1` on nonboundary
  directed D4 edge slots).
- `BoundaryRegime` (`uint8`, shape `(6, n, n, 4)`, zero on nonboundary slots).
- `BoundaryConfidence` (`float32`, shape `(6, n, n, 4)`, zero on nonboundary
  slots).

## Catalog and metadata outputs

- `GeologicalProvinceCatalog` (Arrow).
- `BoundarySegmentCatalog` (Arrow).
- `GeologyMetadata` (JSON), including explicit
  `initialization_not_simulated_deep_time` history semantics.

## Hard acceptance

- Every raster province ID maps to exactly one catalog row and one globally D4
  connected component.
- Every nonnegative boundary edge slot has an identical reciprocal slot. Every
  segment ID maps to one catalog row and one globally connected boundary
  component with a stable unordered plate pair and regime.
- Province cell counts equal the planet cell count and province areas sum to
  `4*pi` within floating-point tolerance.
- Oceanic crust age remains at or below 250 Ma in this Earth-like initializer;
  continental mean crust age exceeds oceanic mean crust age.
- IDs and catalogs are deterministic for identical dependency artifacts.
- Classes, strengths, accommodation, and confidence remain in declared ranges.
- Truth renders expose province class, boundary regime, and crust age on a cube
  net without flattening or face-local wrap assumptions.

## Known limitations

- No terrane ledger, stratigraphic packages, basin fill history, sutures,
  inherited structures, or explicit event chronology exists yet.
- Crust age is an evidence-conditioned prior, not an integrated age-of-formation
  solution. Oceanic age does not yet derive from tracked ridge production and
  subduction consumption.
- The D4 edge graph preserves every raster-resolved incident plate pair at
  triple junctions, but exact subcell vector geometry remains deferred.
- Process thresholds are provisional and require deterministic Earth-analog
  scenarios before calibration.
