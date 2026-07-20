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
    _draw_rivers,
    _project_equal_earth,
    _roll_to_central,
    _smooth_polyline,
    _write_geotiff,
    choose_central_meridian,
    compose_physical_texture,
    cubed_sphere_to_equirectangular,
)


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
    faces = np.arange(6, dtype=np.float32)[:, None, None] * np.ones((6, 4, 4))
    sampled = cubed_sphere_to_equirectangular(faces)

    assert sampled.shape == (16, 32)
    assert set(np.unique(sampled)) == set(range(6))

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
