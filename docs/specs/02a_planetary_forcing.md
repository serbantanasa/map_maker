# Planetary Boundary Conditions And Orbital Forcing

## Status

V1 canonical implementation. This stage turns the approved Earth-like planet,
star, orbit, rotation, and moon controls into deterministic forcing fields for
modern climate and paleoclimate proxies. It does not simulate stellar evolution,
orbital migration, atmospheric composition, tides, or obliquity evolution.

## Inputs

- Canonical cell area and latitude from `geometry`.
- Dense rocky planet controls: radius, surface gravity, and mean density.
- Star luminosity and effective temperature.
- Semi-major axis, eccentricity, obliquity, rotation period, orbital period,
  perihelion date, and northern vernal equinox date.
- Moon mass and orbital distance.

The currently executable parameter bounds deliberately cover only plausible
Earth-like water-world experiments. Mean incident flux is restricted to
`0.65-1.5` times the Earth baseline. These bounds describe the implemented and
tested envelope, not the permanent architecture. Atmospheric habitability is
not implied by passing them.

Atmospheric composition and pressure belong to the separate `atmosphere`
stage. Orbital forcing remains top-of-atmosphere energy and does not infer a
surface climate or biological suitability.

## Approximation

1. Solve Kepler's equation for the midpoint of twelve equal-duration orbital
   intervals.
2. Derive orbital distance and solar longitude relative to the configured
   northern vernal equinox.
3. Derive solar declination from obliquity.
4. Integrate top-of-atmosphere flux over one rotation at every cubed-sphere cell,
   including polar day and night.
5. Area-weight diagnostics using canonical cubed-sphere steradian cell areas.
6. Derive a normalized lunar tide-strength index proportional to `mass / distance^3`
   and a bounded V1 obliquity-stability proxy.

Months are climatological equal-time bins, not Gregorian calendar months.
`MonthlyInsolationWm2` is daily-mean top-of-atmosphere shortwave forcing at each
month midpoint. Climate must model albedo, atmospheric absorption, heat storage,
and transport separately.

## Outputs

- `MonthlyInsolationWm2`: `(12, 6, n, n)` float32.
- `MonthlyDaylightHours`: `(12, 6, n, n)` float32.
- `AnnualMeanInsolationWm2`: `(6, n, n)` float32.
- `InsolationSeasonalityWm2`: monthly maximum minus minimum.
- `PolarLightExtremeFraction`: fraction of sampled months in polar day or night.
- `OrbitalDistanceAU`: twelve sampled distances.
- `SolarDeclinationRad`: twelve sampled declinations.
- `PlanetMetadata`: parameters, model semantics, and area-weighted diagnostics.

## Validation Gates

- Earth defaults produce global mean top-of-atmosphere forcing near `340 W/m2`.
- Area-weighted equatorial annual forcing exceeds polar forcing.
- Northern and southern mid-latitude seasonal maxima are approximately six
  months apart.
- Circular orbits have constant orbital distance.
- Polar day/night occurs only at high latitude for Earth-like obliquity.
- Monthly means reproduce the persisted annual mean and seasonality fields.
- Results are deterministic and cacheable without an RNG seed.

## Downstream Contract

Modern climate consumes the twelve monthly forcing fields directly. Paleoclimate
may consume annual mean and seasonality plus a configured geological warm/cold
offset. All forcing arrays remain persisted so climate training examples can be
reconstructed without rerunning orbital calculations.

## Open-Envelope Boundary

Decision 032 requires later work to widen or profile the current radius,
gravity, stellar-flux, rotation, and temperature bounds. Widening a config
range is not sufficient by itself: each new region needs an explicit scenario,
diagnostic expectations, and a statement of which missing physics limit its
interpretation.
