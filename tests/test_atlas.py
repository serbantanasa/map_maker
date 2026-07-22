from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image
import rasterio

from map_maker.cli import main
from map_maker.pipeline.atlas import (
    AtlasExportConfig,
    AtlasExportResult,
    AtlasStyleConfig,
    DrainageBasinAtlasResult,
    _basin_boundary_masks,
    _basin_color_indices,
    _draw_rivers,
    _project_equal_earth,
    _project_equal_earth_categorical,
    _roll_to_central,
    _select_atlas_river_rows,
    _smooth_polyline,
    _write_geotiff,
    choose_central_meridian,
    compose_physical_texture,
    cubed_sphere_to_equirectangular,
    cubed_sphere_to_equirectangular_nearest,
)
from map_maker.pipeline.cubed_sphere import CubedSphereGrid


def _synthetic_layers(height: int = 24, width: int = 48) -> dict[str, np.ndarray]:
    longitude = np.linspace(-1.0, 1.0, width)[None, :]
    latitude = np.linspace(1.0, -1.0, height)[:, None]
    elevation = 1_400.0 * np.sin(np.pi * longitude) * np.cos(np.pi * latitude / 2.0)
    ocean = (longitude < -0.25).astype(np.float64) * np.ones((height, 1))
    biomes = np.zeros((height, width, 13), dtype=np.float64)
    biomes[..., 1] = 1.0 - ocean
    return {
        "elevation_m": elevation,
        "terrain_relief_m": np.maximum(elevation, 0.0) * 0.35,
        "ocean_fraction": ocean,
        "ocean_depth_m": np.maximum(-elevation + 600.0, 0.0) * ocean,
        "shelf_fraction": ocean * (longitude > -0.75),
        "biome_fractions": biomes,
        "bedrock_fraction": np.clip(elevation / 3_000.0, 0.0, 1.0),
        "soil_salinity": np.zeros((height, width)),
        "wetland_fraction": np.zeros((height, width)),
        "lake_fraction": np.zeros((height, width)),
        "snow_persistence": np.clip((np.abs(latitude) - 0.65) * 3.0, 0.0, 1.0)
        * np.ones((1, width)),
        "glacier_fraction": np.zeros((height, width)),
        "mean_sea_ice_fraction": ocean * np.clip((np.abs(latitude) - 0.65) * 3.0, 0.0, 1.0),
        "perennial_sea_ice_fraction": ocean * np.clip((np.abs(latitude) - 0.82) * 5.0, 0.0, 1.0),
    }


def test_atlas_config_resolves_paths_and_validates_style(tmp_path: Path):
    config_path = tmp_path / "atlas.yaml"
    config_path.write_text(
        """\
format_version: 1
world_config: world.yaml
output_dir: products
style:
  width_px: 800
  central_meridian_deg: auto
  draw_rivers: true
""",
        encoding="utf8",
    )

    config = AtlasExportConfig.from_file(config_path, width_px=900, draw_rivers=False)

    assert config.world_config == tmp_path / "world.yaml"
    assert config.output_dir == tmp_path / "products"
    assert config.style.width_px == 900
    assert config.style.central_meridian_deg is None
    assert not config.style.draw_rivers
    with pytest.raises(ValueError, match="width_px"):
        AtlasStyleConfig.from_mapping({"width_px": 200})


def test_cubed_sphere_sampling_and_ocean_seam_are_deterministic():
    faces = np.full((6, 4, 4), 7.0, dtype=np.float32)
    sampled = cubed_sphere_to_equirectangular(faces)
    categorical = cubed_sphere_to_equirectangular_nearest(faces.astype(np.int32))

    assert sampled.shape == (16, 32)
    assert np.allclose(sampled, 7.0)
    assert categorical.shape == (16, 32)
    assert categorical.dtype == np.int32
    assert np.all(categorical == 7)

    ocean = np.zeros((12, 360), dtype=np.float64)
    ocean[:, 88:93] = 1.0
    central = choose_central_meridian(ocean)
    seam = ((central - 180.0 + 180.0) % 360.0) - 180.0
    assert seam == pytest.approx(-89.5, abs=2.0)
    assert np.array_equal(_roll_to_central(sampled, 0.0), sampled)
    assert np.array_equal(_roll_to_central(sampled, 180.0), np.roll(sampled, -16, axis=1))
    smoothed = _smooth_polyline([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
    assert smoothed[0] == (0.0, 0.0)
    assert smoothed[-1] == (2.0, 0.0)


def test_cubed_sphere_sampling_interpolates_continuous_fields_across_face_edges():
    grid = CubedSphereGrid.create(16)
    field = grid.xyz[..., 0] + 2.0 * grid.xyz[..., 1] - 0.5 * grid.xyz[..., 2]

    sampled = cubed_sphere_to_equirectangular(field)
    height, width = sampled.shape
    longitude = (np.arange(width) + 0.5) / width * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    expected = np.cos(lat) * np.cos(lon) + 2.0 * np.cos(lat) * np.sin(lon) - 0.5 * np.sin(lat)

    error = np.abs(sampled - expected)
    assert float(np.mean(error)) < 0.01
    assert float(np.max(error)) < 0.05


def test_texture_projection_and_geotiff_are_nonblank(tmp_path: Path):
    style = AtlasStyleConfig(width_px=512)
    layers = _synthetic_layers()
    texture = compose_physical_texture(layers, style)
    projected, coverage, projected_ocean, transform = _project_equal_earth(
        texture,
        layers["ocean_fraction"],
        width_px=style.width_px,
        border_fraction=style.border_fraction,
    )

    assert texture.shape == (24, 48, 3)
    assert texture.dtype == np.uint8
    assert projected.shape[1] == 512
    assert projected.std() > 10.0
    assert coverage.max() == 255
    assert coverage[0, 0] == 0
    assert np.isfinite(projected_ocean[coverage > 0]).all()

    path = tmp_path / "atlas.tif"
    _write_geotiff(path, projected, coverage, transform, 37.0)
    with rasterio.open(path) as dataset:
        assert dataset.count == 4
        assert dataset.width == 512
        assert dataset.crs is not None
        assert dataset.colorinterp[-1].name == "alpha"


def test_categorical_basin_projection_and_adjacency_coloring():
    labels = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 2, 2, 1],
            [3, 3, 2, -1],
        ],
        dtype=np.int32,
    )
    catalog = pa.table(
        {
            "basin_id": pa.array([0, 1, 2, 3], type=pa.int32()),
            "area_km2": pa.array([300_000.0, 200_000.0, 100_000.0, 50_000.0]),
        }
    )

    colors = _basin_color_indices(labels, catalog)
    for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3)):
        assert colors[first] != colors[second]

    projected, coverage, _ = _project_equal_earth_categorical(
        labels,
        width_px=512,
        border_fraction=0.025,
    )
    assert projected.shape == coverage.shape
    assert set(np.unique(projected)).issubset({-2, -1, 0, 1, 2, 3})
    assert np.all(projected[coverage == 0] == -2)

    basin_edge, major_edge, inland_edge, coast = _basin_boundary_masks(
        labels,
        np.asarray([True, False, False, False]),
        np.asarray([False, False, True, False]),
    )
    assert np.any(basin_edge)
    assert np.any(major_edge)
    assert np.any(inland_edge)
    assert np.any(coast)


def test_discharge_scaled_river_overlay_and_cli(tmp_path: Path, monkeypatch, capsys):
    texture = compose_physical_texture(_synthetic_layers(), AtlasStyleConfig())
    projected, coverage, _, transform = _project_equal_earth(
        texture,
        _synthetic_layers()["ocean_fraction"],
        width_px=512,
        border_fraction=0.025,
    )
    longitudes = np.radians(np.asarray((-35.0, 0.0, 35.0)))
    reaches = pa.Table.from_pylist(
        [
            {
                "reach_kind": "channel",
                "discharge_mean": 8_000.0,
                "polyline_on_cubed_sphere": [
                    [float(np.cos(lon)), float(np.sin(lon)), 0.0] for lon in longitudes
                ],
            }
        ]
    )
    rendered, count = _draw_rivers(
        Image.fromarray(projected), reaches, transform, AtlasStyleConfig(), 0.0
    )
    assert count == 1
    assert np.any(np.asarray(rendered) != projected)

    config_path = tmp_path / "atlas.yaml"
    config_path.write_text("world_config: world.yaml\n", encoding="utf8")
    result = AtlasExportResult(
        png_path=tmp_path / "map.png",
        geotiff_path=tmp_path / "map.tif",
        metadata_path=tmp_path / "map.json",
        width_px=512,
        height_px=256,
        central_meridian_deg=12.5,
        rendered_river_count=1,
    )
    monkeypatch.setattr("map_maker.pipeline.atlas.export_physical_atlas", lambda config: result)

    assert main(["atlas", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Physical atlas" in output
    assert "12.50 degrees" in output


def test_drainage_basin_atlas_cli(tmp_path: Path, monkeypatch, capsys):
    config_path = tmp_path / "atlas.yaml"
    config_path.write_text("world_config: world.yaml\n", encoding="utf8")
    result = DrainageBasinAtlasResult(
        png_path=tmp_path / "basins.png",
        basin_id_geotiff_path=tmp_path / "basins.tif",
        metadata_path=tmp_path / "basins.json",
        width_px=512,
        height_px=256,
        central_meridian_deg=12.5,
        basin_count=27,
        major_basin_count=8,
        inland_terminal_basin_count=2,
        rendered_river_count=12,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.atlas.export_drainage_basin_atlas", lambda config: result
    )

    assert main(["basin-atlas", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Drainage-basin atlas" in output
    assert "27 basins" in output


def test_atlas_completes_major_river_trunks_through_connectors():
    geometry = [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]]
    reaches = pa.Table.from_pylist(
        [
            {
                "reach_id": 0,
                "downstream_reach_id": 4,
                "reach_kind": "channel",
                "discharge_mean": 900.0,
                "strahler_order": 2,
                "polyline_on_cubed_sphere": geometry,
            },
            {
                "reach_id": 1,
                "downstream_reach_id": 2,
                "reach_kind": "channel",
                "discharge_mean": 5_000.0,
                "strahler_order": 3,
                "polyline_on_cubed_sphere": geometry,
            },
            {
                "reach_id": 2,
                "downstream_reach_id": -1,
                "reach_kind": "channel",
                "discharge_mean": 8_000.0,
                "strahler_order": 4,
                "polyline_on_cubed_sphere": geometry,
            },
            {
                "reach_id": 3,
                "downstream_reach_id": -1,
                "reach_kind": "channel",
                "discharge_mean": 1_500.0,
                "strahler_order": 2,
                "polyline_on_cubed_sphere": geometry,
            },
            {
                "reach_id": 4,
                "downstream_reach_id": 1,
                "reach_kind": "connector",
                "discharge_mean": 4_700.0,
                "strahler_order": 3,
                "polyline_on_cubed_sphere": geometry,
            },
            {
                "reach_id": 5,
                "downstream_reach_id": 4,
                "reach_kind": "channel",
                "discharge_mean": 700.0,
                "strahler_order": 1,
                "polyline_on_cubed_sphere": geometry,
            },
        ]
    )

    selected = _select_atlas_river_rows(reaches, 4_000.0)

    assert {row["reach_id"] for row in selected} == {0, 1, 2, 4}
