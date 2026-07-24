"""Causal global/L2 mineral systems and coarse deposit candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from PIL import Image, ImageDraw

from .._mineral_systems_native import COMMODITY_COUNT, SYSTEM_COUNT, run_mineral_systems
from ..cubed_sphere import CubedSphereGrid
from ..models import StageResult
from ..registry import stage
from ..visualization import VisualizationRequest, VisualizationResult
from .geology import PROVINCE_CLASSES
from .sea_level import _equirectangular_rgb
from .surface_materials import EARTH_MEAN_RADIUS_M

if TYPE_CHECKING:
    from ..execution import PipelineContext

SYSTEM_NAMES = (
    "arc_magmatic_hydrothermal",
    "orogenic_shear",
    "mafic_ultramafic",
    "volcanogenic_seafloor",
    "sediment_hosted_basin",
    "ancient_iron_cratonic",
    "weathering_supergene",
    "placer_heavy_mineral",
    "evaporite_chemical_sediment",
    "coal_basin",
)

COMMODITY_NAMES = (
    "copper",
    "gold",
    "silver",
    "lead_zinc",
    "nickel_cobalt",
    "chromium_pge",
    "iron",
    "uranium",
    "phosphate",
    "bauxite",
    "tin_tungsten",
    "heavy_mineral_sands",
    "salt",
    "potash",
    "coal",
)

PRIMARY_COMMODITY = (
    "copper",
    "gold",
    "nickel_cobalt",
    "lead_zinc",
    "lead_zinc",
    "iron",
    "bauxite",
    "heavy_mineral_sands",
    "salt",
    "coal",
)

BYPRODUCTS = (
    ("gold", "silver", "tin_tungsten"),
    ("gold", "silver", "tin_tungsten"),
    ("chromium_pge", "nickel_cobalt"),
    ("copper", "silver", "lead_zinc"),
    ("silver", "uranium", "phosphate"),
    ("uranium", "iron"),
    ("nickel_cobalt", "iron"),
    ("gold", "chromium_pge", "tin_tungsten"),
    ("potash", "phosphate"),
    (),
)

CAUSAL_OUTPUTS = (
    "MineralSourceSupport",
    "MineralProcessSupport",
    "MineralTransportSupport",
    "MineralTrapSupport",
    "MineralTimingSupport",
    "MineralPreservationSupport",
)

SYSTEM_COLORS = np.asarray(
    [
        (201, 82, 61),
        (153, 83, 142),
        (74, 117, 90),
        (46, 145, 173),
        (204, 155, 67),
        (158, 64, 54),
        (117, 158, 72),
        (221, 188, 72),
        (220, 198, 151),
        (87, 75, 67),
    ],
    dtype=np.uint8,
)

OUTPUT_NAMES = {
    "source_out": "MineralSourceSupport",
    "process_out": "MineralProcessSupport",
    "transport_out": "MineralTransportSupport",
    "trap_out": "MineralTrapSupport",
    "timing_out": "MineralTimingSupport",
    "preservation_out": "MineralPreservationSupport",
    "unresolved_out": "MineralUnresolvedSupport",
    "potential_out": "MineralSystemPotential",
    "confidence_out": "MineralSystemConfidence",
    "commodity_out": "CommodityProspectivity",
    "dominant_system_out": "DominantMineralSystemCode",
}


@dataclass(frozen=True)
class MineralSystemsConfig:
    minimum_dominant_potential: float = 0.14
    system_catalog_potential_threshold: float = 0.22
    deposit_minimum_potential: float = 0.25
    deposit_minimum_confidence: float = 0.20
    maximum_deposits_per_system: int = 128
    minimum_deposit_spacing_steps: int = 2
    maximum_potential_reconstruction_error: float = 2e-6
    maximum_open_ocean_terrestrial_support: float = 1e-7

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "MineralSystemsConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown mineral-system controls: {', '.join(sorted(unknown))}")
        values: dict[str, int | float] = {}
        integer_fields = {"maximum_deposits_per_system", "minimum_deposit_spacing_steps"}
        for name, field in cls.__dataclass_fields__.items():
            raw = mapping.get(name, field.default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be numeric")
            values[name] = int(raw) if name in integer_fields else float(raw)
        config = cls(**values)  # type: ignore[arg-type]
        unit_fields = (
            "minimum_dominant_potential",
            "system_catalog_potential_threshold",
            "deposit_minimum_potential",
            "deposit_minimum_confidence",
        )
        for name in unit_fields:
            value = getattr(config, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if config.deposit_minimum_potential < config.system_catalog_potential_threshold:
            raise ValueError(
                "deposit_minimum_potential must be no lower than "
                "system_catalog_potential_threshold"
            )
        if not 1 <= config.maximum_deposits_per_system <= 10_000:
            raise ValueError("maximum_deposits_per_system must be in [1, 10000]")
        if not 0 <= config.minimum_deposit_spacing_steps <= 16:
            raise ValueError("minimum_deposit_spacing_steps must be in [0, 16]")
        if not 1e-12 <= config.maximum_potential_reconstruction_error <= 1e-3:
            raise ValueError("maximum_potential_reconstruction_error must be in [1e-12, 1e-3]")
        if not 0.0 <= config.maximum_open_ocean_terrestrial_support <= 1e-3:
            raise ValueError("maximum_open_ocean_terrestrial_support must be in [0, 1e-3]")
        return config


def _artifact_array(result: StageResult, name: str) -> np.ndarray:
    record = result.artifact_records.get(name)
    if record is None or record.value is None:
        raise KeyError(f"Missing dependency artifact '{name}'")
    value = record.value
    return np.asarray(value.array() if hasattr(value, "array") else value)


def _artifact_mapping(result: StageResult, name: str) -> Mapping[str, object]:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, Mapping):
        raise KeyError(f"Missing dependency metadata '{name}'")
    return cast(Mapping[str, object], record.value)


def _artifact_table(result: StageResult, name: str) -> pa.Table:
    record = result.artifact_records.get(name)
    if record is None or record.value is None or not isinstance(record.value, pa.Table):
        raise KeyError(f"Missing dependency table '{name}'")
    return cast(pa.Table, record.value)


def _system_id(system_code: int, province_id: int) -> int:
    return system_code * 1_000_000_000 + province_id + 1


def _blocked_within_steps(
    cell_id: int,
    accepted: set[int],
    neighbors: np.ndarray,
    steps: int,
) -> bool:
    if cell_id in accepted:
        return True
    frontier = {cell_id}
    visited = {cell_id}
    for _ in range(steps):
        next_frontier: set[int] = set()
        for current in frontier:
            for neighbor in neighbors[current]:
                adjacent = int(neighbor)
                if adjacent in accepted:
                    return True
                if adjacent not in visited:
                    visited.add(adjacent)
                    next_frontier.add(adjacent)
        frontier = next_frontier
    return False


def _system_catalog(
    topology: CubedSphereGrid,
    potential: np.ndarray,
    confidence: np.ndarray,
    province_id: np.ndarray,
    province_class: np.ndarray,
    crust_age: np.ndarray,
    *,
    planet_radius_m: float,
    threshold: float,
) -> pa.Table:
    flat_potential = potential.reshape(SYSTEM_COUNT, -1)
    flat_confidence = confidence.reshape(SYSTEM_COUNT, -1)
    flat_province_id = province_id.reshape(-1)
    flat_province_class = province_class.reshape(-1)
    flat_age = crust_age.reshape(-1)
    area_km2 = topology.cell_areas.reshape(-1) * planet_radius_m**2 / 1e6
    rows: list[dict[str, object]] = []
    for system_index, system_name in enumerate(SYSTEM_NAMES):
        active = flat_potential[system_index] >= threshold
        for host_province in np.unique(flat_province_id[active]):
            members = np.flatnonzero(active & (flat_province_id == host_province))
            if members.size == 0:
                continue
            representative = int(members[np.argmax(flat_potential[system_index, members])])
            class_code = int(flat_province_class[representative])
            weights = area_km2[members]
            rows.append(
                {
                    "mineral_system_id": _system_id(system_index + 1, int(host_province)),
                    "system_code": system_index + 1,
                    "system_name": system_name,
                    "province_id": int(host_province),
                    "province_class_code": class_code,
                    "province_class_name": PROVINCE_CLASSES.get(class_code, "unknown"),
                    "represented_cell_count": int(members.size),
                    "represented_area_km2": float(np.sum(weights)),
                    "area_weighted_mean_potential": float(
                        np.sum(flat_potential[system_index, members] * weights) / np.sum(weights)
                    ),
                    "maximum_potential": float(np.max(flat_potential[system_index, members])),
                    "area_weighted_mean_confidence": float(
                        np.sum(flat_confidence[system_index, members] * weights) / np.sum(weights)
                    ),
                    "representative_cell_id": representative,
                    "area_weighted_host_crust_age_ga": float(
                        np.sum(flat_age[members] * weights) / np.sum(weights)
                    ),
                    "primary_commodity": PRIMARY_COMMODITY[system_index],
                    "byproduct_commodities": list(BYPRODUCTS[system_index]),
                }
            )
    return pa.Table.from_pylist(rows)


def _formation_age_proxy(system_index: int, crust_age_ga: float, unresolved: float) -> float:
    recent_scale = (0.06, 0.35, 0.25, 0.18, 0.70, 0.88, 0.03, 0.01, 0.18, 0.12)
    if system_index == 5:
        return max(0.0, crust_age_ga * (0.72 + 0.20 * unresolved))
    return max(
        0.0,
        min(crust_age_ga, recent_scale[system_index] * (0.65 + 0.70 * unresolved)),
    )


def _size_class(score: float) -> str:
    if score >= 0.82:
        return "very_large_relative"
    if score >= 0.62:
        return "large_relative"
    if score >= 0.40:
        return "medium_relative"
    return "small_relative"


def _deposit_catalog(
    topology: CubedSphereGrid,
    potential: np.ndarray,
    confidence: np.ndarray,
    causal: Mapping[str, np.ndarray],
    unresolved: np.ndarray,
    commodity: np.ndarray,
    province_id: np.ndarray,
    province_class: np.ndarray,
    crust_age: np.ndarray,
    bedrock: np.ndarray,
    residual_regolith: np.ndarray,
    *,
    planet_radius_m: float,
    config: MineralSystemsConfig,
) -> pa.Table:
    total = topology.cell_count
    flat_potential = potential.reshape(SYSTEM_COUNT, total)
    flat_confidence = confidence.reshape(SYSTEM_COUNT, total)
    flat_unresolved = unresolved.reshape(SYSTEM_COUNT, total)
    flat_commodity = commodity.reshape(COMMODITY_COUNT, total)
    flat_causal = {name: values.reshape(SYSTEM_COUNT, total) for name, values in causal.items()}
    neighbors = topology.neighbor_indices.reshape(total, 4)
    area_km2 = topology.cell_areas.reshape(-1) * planet_radius_m**2 / 1e6
    flat_province_id = province_id.reshape(-1)
    flat_province_class = province_class.reshape(-1)
    flat_age = crust_age.reshape(-1)
    exposure = np.clip(bedrock.reshape(-1) + 0.25 * residual_regolith.reshape(-1), 0.0, 1.0)
    lon = topology.longitude.reshape(-1)
    lat = topology.latitude.reshape(-1)
    resolution = topology.face_resolution
    face_size = resolution * resolution
    rows: list[dict[str, object]] = []
    causal_columns = {
        "MineralSourceSupport": "source_support",
        "MineralProcessSupport": "process_support",
        "MineralTransportSupport": "transport_support",
        "MineralTrapSupport": "trap_support",
        "MineralTimingSupport": "timing_support",
        "MineralPreservationSupport": "preservation_support",
    }
    for system_index, system_name in enumerate(SYSTEM_NAMES):
        values = flat_potential[system_index]
        local_maximum = values >= np.max(values[neighbors], axis=1)
        candidates = np.flatnonzero(
            local_maximum
            & (values >= config.deposit_minimum_potential)
            & (flat_confidence[system_index] >= config.deposit_minimum_confidence)
        )
        order = candidates[
            np.lexsort(
                (candidates, -flat_confidence[system_index, candidates], -values[candidates])
            )
        ]
        accepted: set[int] = set()
        for cell_id_value in order:
            cell_id = int(cell_id_value)
            if _blocked_within_steps(
                cell_id,
                accepted,
                neighbors,
                config.minimum_deposit_spacing_steps,
            ):
                continue
            accepted.add(cell_id)
            if len(accepted) > config.maximum_deposits_per_system:
                break
            host_province = int(flat_province_id[cell_id])
            unmodeled = float(flat_unresolved[system_index, cell_id])
            score = float(values[cell_id])
            conf = float(flat_confidence[system_index, cell_id])
            size_score = float(np.clip(0.70 * score + 0.30 * unmodeled, 0.0, 1.0))
            grade_score = float(
                np.clip(
                    0.58 * score
                    + 0.24 * flat_causal["MineralSourceSupport"][system_index, cell_id]
                    + 0.18 * unmodeled,
                    0.0,
                    1.0,
                )
            )
            footprint_fraction = 0.0005 + 0.025 * size_score**2
            face, within_face = divmod(cell_id, face_size)
            row, col = divmod(within_face, resolution)
            primary_index = COMMODITY_NAMES.index(PRIMARY_COMMODITY[system_index])
            record: dict[str, object] = {
                "deposit_candidate_id": (system_index + 1) * 1_000_000_000 + cell_id,
                "mineral_system_id": _system_id(system_index + 1, host_province),
                "system_code": system_index + 1,
                "system_name": system_name,
                "primary_commodity": PRIMARY_COMMODITY[system_index],
                "byproduct_commodities": list(BYPRODUCTS[system_index]),
                "host_cell_id": cell_id,
                "face": face,
                "row": row,
                "column": col,
                "longitude_deg": float(np.rad2deg(lon[cell_id])),
                "latitude_deg": float(np.rad2deg(lat[cell_id])),
                "province_id": host_province,
                "province_class_code": int(flat_province_class[cell_id]),
                "province_class_name": PROVINCE_CLASSES.get(
                    int(flat_province_class[cell_id]), "unknown"
                ),
                "system_potential": score,
                "system_confidence": conf,
                "primary_commodity_prospectivity": float(flat_commodity[primary_index, cell_id]),
                "host_crust_age_ga": float(flat_age[cell_id]),
                "formation_age_ga_proxy": _formation_age_proxy(
                    system_index, float(flat_age[cell_id]), unmodeled
                ),
                "formation_age_semantics": "structural_proxy_not_event_history",
                "surface_exposure_fraction_proxy": float(exposure[cell_id]),
                "depth_to_top_m_proxy": float(
                    (1.0 - exposure[cell_id]) * (15.0 + 420.0 * (1.0 - score))
                ),
                "footprint_area_km2_proxy": float(area_km2[cell_id] * footprint_fraction),
                "relative_size_score": size_score,
                "relative_size_class": _size_class(size_score),
                "relative_grade_mean": grade_score,
                "relative_grade_log_sigma": float(0.18 + 0.42 * (1.0 - conf)),
                "preservation_confidence": float(
                    flat_causal["MineralPreservationSupport"][system_index, cell_id] * conf
                ),
                "unresolved_subcell_support": unmodeled,
                "economic_viability_modeled": False,
                "measured_reserve_modeled": False,
            }
            for artifact_name, column_name in causal_columns.items():
                record[column_name] = float(flat_causal[artifact_name][system_index, cell_id])
            rows.append(record)
    return pa.Table.from_pylist(rows)


def _potential_reconstruction(
    causal: Mapping[str, np.ndarray],
    unresolved: np.ndarray,
) -> np.ndarray:
    stacked = np.stack([np.asarray(causal[name], dtype=np.float64) for name in CAUSAL_OUTPUTS])
    positive = np.all(stacked > 0.0, axis=0)
    geometric = np.prod(stacked, axis=0) ** (1.0 / len(CAUSAL_OUTPUTS))
    bottleneck = np.min(stacked, axis=0)
    reconstructed = (0.68 * geometric + 0.32 * bottleneck) * (
        0.99 + 0.02 * np.asarray(unresolved, dtype=np.float64)
    )
    return np.where(positive, np.clip(reconstructed, 0.0, 1.0), 0.0)


def _equatorial_scale_bar(
    equatorial_circumference_km: float,
    map_width_pixels: int,
    *,
    target_pixels: int = 56,
) -> tuple[float, int]:
    if (
        not np.isfinite(equatorial_circumference_km)
        or equatorial_circumference_km <= 0.0
        or map_width_pixels <= 0
        or target_pixels <= 0
    ):
        raise ValueError("scale-bar geometry must be finite and positive")
    target_km = equatorial_circumference_km * target_pixels / map_width_pixels
    magnitude = 10.0 ** np.floor(np.log10(target_km))
    normalized = target_km / magnitude
    multiplier = 5.0 if normalized >= 5.0 else 2.0 if normalized >= 2.0 else 1.0
    distance_km = float(multiplier * magnitude)
    pixels = max(1, round(map_width_pixels * distance_km / equatorial_circumference_km))
    return distance_km, pixels


def _distance_label(distance_km: float) -> str:
    if distance_km >= 1.0:
        return f"{distance_km:,.0f}"
    return f"{distance_km:.2g}"


def _annotated_map(
    rgb_faces: np.ndarray,
    *,
    title: str,
    legend: list[tuple[str, tuple[int, int, int]]],
    equatorial_circumference_km: float,
) -> tuple[Image.Image, float]:
    projected = _equirectangular_rgb(rgb_faces)
    legend_width = 270
    header_height = 34
    canvas = Image.new(
        "RGB",
        (projected.shape[1] + legend_width, projected.shape[0] + header_height),
        (242, 242, 238),
    )
    canvas.paste(Image.fromarray(projected, mode="RGB"), (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{title} | equirectangular projection", fill=(25, 27, 28))
    y = header_height + 10
    for label, color in legend:
        draw.rectangle((projected.shape[1] + 12, y, projected.shape[1] + 28, y + 12), fill=color)
        draw.text((projected.shape[1] + 36, y), label, fill=(25, 27, 28))
        y += 22
    bar_km, bar_pixels = _equatorial_scale_bar(
        equatorial_circumference_km,
        projected.shape[1],
    )
    bar_y = projected.shape[0] + header_height - 26
    draw.line((18, bar_y, 18 + bar_pixels, bar_y), fill=(245, 245, 240), width=4)
    draw.line((18, bar_y - 4, 18, bar_y + 4), fill=(245, 245, 240), width=2)
    draw.line(
        (18 + bar_pixels, bar_y - 4, 18 + bar_pixels, bar_y + 4),
        fill=(245, 245, 240),
        width=2,
    )
    draw.text(
        (18, bar_y - 18),
        f"{_distance_label(bar_km)} km at equator",
        fill=(245, 245, 240),
    )
    return canvas, bar_km


def _visualizer(result: StageResult, request: VisualizationRequest) -> list[VisualizationResult]:
    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _artifact_mapping(result, "MineralSystemsMetadata")
    circumference_km = float(metadata["equatorial_circumference_km"])
    dominant = _artifact_array(result, "DominantMineralSystemCode")
    system_rgb = np.full((*dominant.shape, 3), (28, 48, 61), dtype=np.uint8)
    for code, color in enumerate(SYSTEM_COLORS, start=1):
        system_rgb[dominant == code] = color
    system_path = output_dir / "dominant_mineral_systems.png"
    system_map, system_bar_km = _annotated_map(
        system_rgb,
        title="Dominant causal mineral system",
        legend=[
            ("none / open ocean", (28, 48, 61)),
            *[
                (name.replace("_", " "), tuple(int(value) for value in color))
                for name, color in zip(SYSTEM_NAMES, SYSTEM_COLORS, strict=True)
            ],
        ],
        equatorial_circumference_km=circumference_km,
    )
    system_map.save(system_path)

    commodity = _artifact_array(result, "CommodityProspectivity")
    panels: list[Image.Image] = []
    commodity_projection_width = 0
    for index, name in enumerate(COMMODITY_NAMES):
        values = np.clip(np.asarray(commodity[index], dtype=np.float64), 0.0, 1.0)
        strength = np.sqrt(values)[..., None]
        low = np.asarray((26.0, 43.0, 54.0))
        high = np.asarray((241.0, 189.0, 72.0))
        rgb = np.asarray(low * (1.0 - strength) + high * strength, dtype=np.uint8)
        projection = Image.fromarray(_equirectangular_rgb(rgb), mode="RGB")
        projection.thumbnail((320, 160))
        commodity_projection_width = projection.width
        panel = Image.new("RGB", (330, 188), (238, 238, 234))
        panel.paste(projection, ((330 - projection.width) // 2, 22))
        ImageDraw.Draw(panel).text((8, 5), name.replace("_", " "), fill=(25, 27, 28))
        panels.append(panel)
    atlas_header_height = 50
    atlas = Image.new(
        "RGB",
        (5 * 330, 3 * 188 + atlas_header_height),
        (238, 238, 234),
    )
    draw = ImageDraw.Draw(atlas)
    draw.text(
        (8, 3),
        "Relative commodity prospectivity | equirectangular | dark=low, gold=high",
        fill=(25, 27, 28),
    )
    atlas_bar_km, atlas_bar_pixels = _equatorial_scale_bar(
        circumference_km,
        commodity_projection_width,
    )
    atlas_bar_y = 34
    draw.line((8, atlas_bar_y, 8 + atlas_bar_pixels, atlas_bar_y), fill=(25, 27, 28), width=3)
    draw.line((8, atlas_bar_y - 3, 8, atlas_bar_y + 3), fill=(25, 27, 28), width=2)
    draw.line(
        (
            8 + atlas_bar_pixels,
            atlas_bar_y - 3,
            8 + atlas_bar_pixels,
            atlas_bar_y + 3,
        ),
        fill=(25, 27, 28),
        width=2,
    )
    draw.text(
        (16 + atlas_bar_pixels, atlas_bar_y - 8),
        f"{_distance_label(atlas_bar_km)} km at equator",
        fill=(25, 27, 28),
    )
    for index, panel in enumerate(panels):
        atlas.paste(
            panel,
            ((index % 5) * 330, atlas_header_height + (index // 5) * 188),
        )
    commodity_path = output_dir / "commodity_prospectivity_atlas.png"
    atlas.save(commodity_path)

    candidate_catalog = _artifact_table(result, "MajorDepositCandidateCatalog")
    muted = np.asarray(0.45 * system_rgb + 0.55 * np.asarray((55, 62, 62)), dtype=np.uint8)
    candidate_map, candidate_bar_km = _annotated_map(
        muted,
        title="Major deposit candidates (coarse subgrid hypotheses)",
        legend=[
            (
                name.replace("_", " "),
                tuple(int(value) for value in color),
            )
            for name, color in zip(SYSTEM_NAMES, SYSTEM_COLORS, strict=True)
        ],
        equatorial_circumference_km=circumference_km,
    )
    candidate_draw = ImageDraw.Draw(candidate_map)
    projection_width = dominant.shape[2] * 8
    projection_height = dominant.shape[1] * 4
    for record in candidate_catalog.to_pylist():
        code = int(record["system_code"])
        x = int(
            np.clip(
                (float(record["longitude_deg"]) + 180.0) / 360.0 * projection_width,
                0,
                projection_width - 1,
            )
        )
        y = 34 + int(
            np.clip(
                (90.0 - float(record["latitude_deg"])) / 180.0 * projection_height,
                0,
                projection_height - 1,
            )
        )
        color = tuple(int(value) for value in SYSTEM_COLORS[code - 1])
        candidate_draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=(245, 245, 238))
    candidate_path = output_dir / "major_deposit_candidates.png"
    candidate_map.save(candidate_path)

    return [
        VisualizationResult(
            system_path,
            "DominantMineralSystemCode",
            {
                "projection": "equirectangular",
                "equatorial_scale_bar_km": system_bar_km,
            },
        ),
        VisualizationResult(
            commodity_path,
            "CommodityProspectivity",
            {
                "projection": "equirectangular",
                "commodity_count": COMMODITY_COUNT,
                "equatorial_scale_bar_km": atlas_bar_km,
            },
        ),
        VisualizationResult(
            candidate_path,
            "MajorDepositCandidateCatalog",
            {
                "projection": "equirectangular",
                "equatorial_scale_bar_km": candidate_bar_km,
                "candidate_count": candidate_catalog.num_rows,
            },
        ),
    ]


@stage(
    "mineral_systems",
    inputs=(
        "geology",
        "tectonics",
        "world_age",
        "sea_level",
        "elevation",
        "climate",
        "hydrology",
        "surface_materials",
        "potential_biosphere",
        "biosphere_validation",
        "planet",
    ),
    outputs=(
        *CAUSAL_OUTPUTS,
        "MineralUnresolvedSupport",
        "MineralSystemPotential",
        "MineralSystemConfidence",
        "CommodityProspectivity",
        "DominantMineralSystemCode",
        "MineralSystemCatalog",
        "MajorDepositCandidateCatalog",
        "MineralSystemsMetadata",
    ),
    version="v2",
    native_libraries=("mineral_systems_native",),
    visualizer=_visualizer,
)
def mineral_systems_stage(
    context: PipelineContext,
    deps: Mapping[str, StageResult],
    config_mapping: Mapping[str, object],
) -> Mapping[str, object]:
    config = MineralSystemsConfig.from_mapping(config_mapping)
    if not isinstance(context.topology, CubedSphereGrid):
        raise NotImplementedError("causal mineral systems require topology: cubed_sphere")
    biosphere_validation = _artifact_mapping(
        deps["biosphere_validation"], "BiosphereValidationMetadata"
    )
    if biosphere_validation.get("hard_gate_pass") != 1:
        raise RuntimeError("mineral systems require passing biosphere hard gates")

    shape = context.topology.face_shape
    system_shape = (SYSTEM_COUNT, *shape)
    commodity_shape = (COMMODITY_COUNT, *shape)
    output_shapes = {
        **{name: system_shape for name in CAUSAL_OUTPUTS},
        "MineralUnresolvedSupport": system_shape,
        "MineralSystemPotential": system_shape,
        "MineralSystemConfidence": system_shape,
        "CommodityProspectivity": commodity_shape,
        "DominantMineralSystemCode": shape,
    }
    handles = {
        name: context.arena.allocate_array(
            f"mineral_systems_{name.lower()}",
            output_shape,
            np.uint8 if name == "DominantMineralSystemCode" else np.float32,
        )
        for name, output_shape in output_shapes.items()
    }
    views = {name: handle.mutable_view() for name, handle in handles.items()}

    geology = deps["geology"]
    tectonics = deps["tectonics"]
    world_age = deps["world_age"]
    sea_level = deps["sea_level"]
    elevation = deps["elevation"]
    climate = deps["climate"]
    hydrology = deps["hydrology"]
    materials = deps["surface_materials"]
    biosphere = deps["potential_biosphere"]

    def f32(result: StageResult, name: str) -> np.ndarray:
        return np.ascontiguousarray(_artifact_array(result, name), dtype=np.float32)

    native_outputs = {
        native_name: views[artifact_name] for native_name, artifact_name in OUTPUT_NAMES.items()
    }
    with context.timed("causal_mineral_systems_kernel"):
        run_mineral_systems(
            face_resolution=context.topology.face_resolution,
            seed=int(context.config.rng_seed),
            minimum_dominant_potential=config.minimum_dominant_potential,
            xyz=np.ascontiguousarray(context.topology.xyz, dtype=np.float32),
            ocean=f32(sea_level, "SurfaceOceanMask"),
            shelf=f32(sea_level, "ContinentalShelfFraction"),
            relief=f32(elevation, "TerrainReliefM"),
            elevation=f32(sea_level, "SurfaceElevationM"),
            terrain_slope=f32(materials, "SurfaceGeomorphicSlope"),
            province_class=np.ascontiguousarray(
                _artifact_array(geology, "GeologicalProvinceClass"), dtype=np.uint8
            ),
            crust_age=f32(geology, "CrustAgeGa"),
            rock_strength=f32(geology, "RockStrength"),
            accommodation=f32(geology, "SedimentAccommodation"),
            province_confidence=f32(geology, "ProvinceConfidence"),
            elevation_confidence=f32(elevation, "ElevationConfidence"),
            convergence=f32(tectonics, "BoundaryConvergence"),
            divergence=f32(tectonics, "BoundaryDivergence"),
            shear=f32(tectonics, "BoundaryShear"),
            subduction=f32(tectonics, "BoundarySubduction"),
            hotspot=f32(tectonics, "HotspotMap"),
            uplift=f32(world_age, "UpliftRate"),
            subsidence=f32(world_age, "SubsidenceRate"),
            compression=f32(world_age, "TectonicCompression"),
            extension=f32(world_age, "TectonicExtension"),
            stiffness=f32(world_age, "LithosphereStiffness"),
            temperature=f32(climate, "AnnualMeanTemperatureC"),
            precipitation=f32(climate, "AnnualPrecipitationMm"),
            aridity=f32(climate, "AnnualAridityIndex"),
            contributing_area=np.ascontiguousarray(
                _artifact_array(hydrology, "ContributingAreaKm2"), dtype=np.float64
            ),
            stream_power=f32(hydrology, "StreamPowerW"),
            river=f32(hydrology, "RiverCorridor"),
            floodplain=f32(hydrology, "FloodplainPotential"),
            lake=f32(materials, "EffectiveLakeFraction"),
            wetland=f32(materials, "EffectiveWetlandFraction"),
            bedrock=f32(materials, "BedrockSurfaceFraction"),
            residual_regolith=f32(materials, "ResidualRegolithFraction"),
            alluvium=f32(materials, "AlluviumFraction"),
            lacustrine=f32(materials, "LacustrineSedimentFraction"),
            volcaniclastic=f32(materials, "VolcaniclasticFraction"),
            soil_depth=f32(materials, "SoilDepthM"),
            salinity=f32(materials, "SoilSalinityIndex"),
            drainage=f32(materials, "SoilDrainageIndex"),
            hydric_soil=f32(materials, "HydricSoilFraction"),
            soil_confidence=f32(materials, "SoilConfidence"),
            annual_npp=f32(biosphere, "AnnualPotentialNPPKgCM2"),
            standing_biomass=f32(biosphere, "PotentialStandingBiomassKgCM2"),
            vegetation_cover=f32(biosphere, "PotentialVegetationCoverFraction"),
            biosphere_confidence=f32(biosphere, "PotentialBiosphereConfidence"),
            **native_outputs,
        )

    causal = {name: np.asarray(views[name]) for name in CAUSAL_OUTPUTS}
    unresolved = np.asarray(views["MineralUnresolvedSupport"])
    potential = np.asarray(views["MineralSystemPotential"])
    confidence = np.asarray(views["MineralSystemConfidence"])
    commodity = np.asarray(views["CommodityProspectivity"])
    for name, values in {
        **causal,
        "MineralUnresolvedSupport": unresolved,
        "MineralSystemPotential": potential,
        "MineralSystemConfidence": confidence,
        "CommodityProspectivity": commodity,
    }.items():
        if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"{name} contains non-finite or out-of-range values")
    reconstructed = _potential_reconstruction(causal, unresolved)
    reconstruction_error = float(
        np.max(np.abs(reconstructed - np.asarray(potential, dtype=np.float64)))
    )
    if reconstruction_error > config.maximum_potential_reconstruction_error:
        raise RuntimeError("mineral-system potential reconstruction audit failed")
    expected_dominant = np.argmax(potential, axis=0).astype(np.uint8) + 1
    maximum_potential = np.max(potential, axis=0)
    expected_dominant[maximum_potential < config.minimum_dominant_potential] = 0
    dominant = np.asarray(views["DominantMineralSystemCode"])
    if not np.array_equal(dominant, expected_dominant):
        raise RuntimeError("dominant mineral-system codes do not reconstruct")
    open_ocean = _artifact_array(sea_level, "SurfaceOceanMask") >= 0.5
    terrestrial_indices = [index for index in range(SYSTEM_COUNT) if index != 3]
    ocean_leakage = float(
        np.max(potential[np.asarray(terrestrial_indices)][:, open_ocean], initial=0.0)
    )
    if ocean_leakage > config.maximum_open_ocean_terrestrial_support:
        raise RuntimeError("terrestrial mineral-system support leaks into open ocean")

    planet_metadata = _artifact_mapping(deps["planet"], "PlanetMetadata")
    planet_radius_m = float(planet_metadata["planet_radius_earth"]) * EARTH_MEAN_RADIUS_M
    province_id = _artifact_array(geology, "GeologicalProvinceID")
    province_class = _artifact_array(geology, "GeologicalProvinceClass")
    crust_age = _artifact_array(geology, "CrustAgeGa")
    system_catalog = _system_catalog(
        context.topology,
        potential,
        confidence,
        province_id,
        province_class,
        crust_age,
        planet_radius_m=planet_radius_m,
        threshold=config.system_catalog_potential_threshold,
    )
    deposit_catalog = _deposit_catalog(
        context.topology,
        potential,
        confidence,
        causal,
        unresolved,
        commodity,
        province_id,
        province_class,
        crust_age,
        _artifact_array(materials, "BedrockSurfaceFraction"),
        _artifact_array(materials, "ResidualRegolithFraction"),
        planet_radius_m=planet_radius_m,
        config=config,
    )
    if system_catalog.num_rows == 0 or deposit_catalog.num_rows == 0:
        raise RuntimeError("mineral-system catalogs collapsed to zero records")
    system_ids = set(system_catalog["mineral_system_id"].to_pylist())
    if not set(deposit_catalog["mineral_system_id"].to_pylist()).issubset(system_ids):
        raise RuntimeError("deposit candidate references an absent mineral system")
    deposit_ids = deposit_catalog["deposit_candidate_id"].to_pylist()
    if len(deposit_ids) != len(set(deposit_ids)):
        raise RuntimeError("deposit candidate IDs are not unique")

    areas = np.asarray(context.topology.cell_areas, dtype=np.float64)
    metadata: dict[str, object] = {
        **asdict(config),
        "model": "causal_mineral_systems_v0",
        "topology": "cubed_sphere",
        "face_resolution": context.topology.face_resolution,
        "planet_radius_m": planet_radius_m,
        "equatorial_circumference_km": 2.0 * np.pi * planet_radius_m / 1e3,
        "system_axis": list(SYSTEM_NAMES),
        "commodity_axis": list(COMMODITY_NAMES),
        "causal_axis": [
            name.removeprefix("Mineral").removesuffix("Support").lower() for name in CAUSAL_OUTPUTS
        ],
        "system_count": SYSTEM_COUNT,
        "commodity_count": COMMODITY_COUNT,
        "mineral_system_catalog_count": system_catalog.num_rows,
        "major_deposit_candidate_count": deposit_catalog.num_rows,
        "potential_reconstruction_maximum_absolute_error": reconstruction_error,
        "open_ocean_terrestrial_maximum_potential": ocean_leakage,
        "family_area_weighted_mean_potential": {
            name: float(np.sum(potential[index] * areas) / np.sum(areas))
            for index, name in enumerate(SYSTEM_NAMES)
        },
        "family_maximum_potential": {
            name: float(np.max(potential[index])) for index, name in enumerate(SYSTEM_NAMES)
        },
        "family_candidate_count": {
            name: int(np.count_nonzero(np.asarray(deposit_catalog["system_code"]) == index + 1))
            for index, name in enumerate(SYSTEM_NAMES)
        },
        "prospectivity_semantics": "relative_structural_prospectivity_not_reserves",
        "confidence_semantics": (
            "geological_process_and_observation_confidence_independent_from_potential"
        ),
        "unresolved_support_semantics": (
            "bounded_seeded_spherical_subcell_prior_scaled_to_parent_face_resolution"
        ),
        "candidate_geometry_semantics": "subgrid_hypothesis_not_cell_sized_orebody",
        "formation_age_semantics": "structural_proxy_not_simulated_event_history",
        "stable_system_id_semantics": "system_code_times_1e9_plus_province_id_plus_one",
        "stable_candidate_id_semantics": "system_code_times_1e9_plus_host_cell_id",
        "petroleum_supported": 0,
        "petroleum_deferral_reason": (
            "missing_source_burial_maturity_reservoir_seal_trap_migration_timing_history"
        ),
        "measured_reserves_supported": 0,
        "economic_viability_supported": 0,
        "l3_deposit_geometry_supported": 0,
        "causal_mineral_systems_ready_for_validation": 1,
    }
    for handle in handles.values():
        handle.seal()
    context.logger.log_event(
        {"type": "mineral_systems_summary", "stage": "mineral_systems", **metadata}
    )
    return {
        **handles,
        "MineralSystemCatalog": system_catalog,
        "MajorDepositCandidateCatalog": deposit_catalog,
        "MineralSystemsMetadata": metadata,
    }


__all__ = [
    "CAUSAL_OUTPUTS",
    "COMMODITY_NAMES",
    "MineralSystemsConfig",
    "SYSTEM_NAMES",
    "mineral_systems_stage",
]
