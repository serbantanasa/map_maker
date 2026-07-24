# Documentation

Start here for the current product and engineering documentation. The root
[README](../readme.md) owns setup, commands, and the executable feature summary;
this index owns document authority and lifecycle.

## Active Documents

| Document | Owns |
| --- | --- |
| [Planet Engine Spec](PLANET_ENGINE_SPEC.md) | Canonical product goal, scientific scope, and long-term system boundaries. |
| [Decision Log](DECISIONS.md) | Accepted technical decisions, status, and explicit supersession. |
| [Current Roadmap](ROADMAP.md) | Current work, sequencing, freeze boundary, and milestone exit criteria. |
| [Validation Contract](VALIDATION.md) | Live hard gates, diagnostics, human review, and evidence policy. |
| [Connected Sea Level](specs/03_sea_level.md) | Active surface-geography, coast, shelf, and sea-level contract. |
| [Erosion And Sedimentation](specs/06_erosion.md) | Prospective fluvial constraints and the unresolved regional morphology contract. |
| [Elevation And Orogeny](specs/07_elevation.md) | Active bedrock elevation and tectonic-morphology contract. |
| [Physical Atlas Export](specs/11_physical_atlas_export.md) | Current atlas composition, provenance, export, and visual-acceptance contract. |
| [L2 Regional Handoff](specs/12_l2_regional_handoff.md) | Selected-basin L2 package, conservative child surfaces, inherited priors, and L3 boundary contract. |
| [L3 Regional Vertical Slice](specs/13_l3_vertical_slice.md) | Selected catchment, 200 m terrain and hydrology ownership, adaptive river corridors, budgets, and acceptance gates. |
| [Causal Mineral Systems V0](specs/14_causal_mineral_systems.md) | Coarse causal resource systems, persisted training features, validation, and explicit deposit/petroleum deferrals. |
| [Code Standards](standards.md) | Repository process and contribution rules. |

The product spec owns intent, the decision log owns approved choices, and the
roadmap owns priority. Stage documents must not redefine those concerns.

## Archive

The [archive index](archive/README.md) records every moved document and its old
path. Historical stage contracts retain their filenames under:

- `archive/completed-stage-specs/foundation/`
- `archive/completed-stage-specs/hydrology/`
- `archive/completed-stage-specs/biosphere/`
- `archive/superseded-stage-plans/`

Historical status notes and dated validation results live under
`archive/status/` and `archive/validation/`.

Archive rather than delete a document when its milestone is complete, its plan
is superseded, or its measurements are a dated snapshot. Archived material
remains useful for provenance but is not current authority. Repair links and
factual transcription errors in place; record a new current decision or spec
instead of silently modernizing an archived contract.
