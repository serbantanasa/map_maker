from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from map_maker.cli import main
import map_maker.pipeline.l3_channel_geometry as channel_geometry_module
from map_maker.pipeline.l3_channel_geometry import (
    L3ChannelGeometryConfig,
    L3ChannelGeometryResult,
    _ChannelSources,
    _build_centerline_table,
    _chaikin_polyline,
    _cross_section_fraction,
    _distance_and_nearest,
    _support_fields,
    generate_l3_channel_geometry,
)


def test_config_resolves_paths_and_rejects_invalid_reliability(tmp_path: Path) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
terrain_output_dir: terrain
hydrology_output_dir: hydrology
channel_geometry_output_dir: channels
channel_geometry:
  smoothing_iterations: 3
  reliable_flow_minimum_active_months: 7
limits:
  maximum_channel_geometry_storage_gb: 1
""",
        encoding="utf8",
    )
    config = L3ChannelGeometryConfig.from_file(config_path)
    assert config.terrain_dir == tmp_path / "terrain"
    assert config.hydrology_dir == tmp_path / "hydrology"
    assert config.output_dir == tmp_path / "channels"
    assert config.smoothing_iterations == 3
    assert config.reliable_flow_minimum_active_months == 7

    invalid = L3ChannelGeometryConfig(
        terrain_dir=tmp_path,
        hydrology_dir=tmp_path,
        output_dir=tmp_path,
        reliable_flow_minimum_active_months=13,
    )
    try:
        invalid.validate()
    except ValueError as exc:
        assert "reliable_flow_minimum_active_months" in str(exc)
    else:
        raise AssertionError("invalid active-month threshold was accepted")


def test_chaikin_preserves_endpoints_and_rounds_a_grid_corner() -> None:
    raw = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    smooth = _chaikin_polyline(raw, 2)
    np.testing.assert_array_equal(smooth[0], raw[0])
    np.testing.assert_array_equal(smooth[-1], raw[-1])
    assert len(smooth) > len(raw)
    raw_length = np.sum(np.linalg.norm(np.diff(raw, axis=0), axis=1))
    smooth_length = np.sum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))
    assert np.sqrt(2.0) < smooth_length < raw_length


def test_cross_section_fraction_handles_subcell_and_wide_corridors() -> None:
    distance = np.asarray([0.0, 100.0, 200.0, 400.0])
    narrow = _cross_section_fraction(distance, np.full(4, 20.0), 200.0)
    wide = _cross_section_fraction(distance, np.full(4, 400.0), 200.0)
    np.testing.assert_allclose(narrow[0], 0.1)
    assert narrow[1] < narrow[0]
    assert narrow[-1] == 0.0
    np.testing.assert_allclose(wide[:2], [1.0, 1.0])
    np.testing.assert_allclose(wide[2], 0.5)
    assert wide[-1] == 0.0


def test_distance_transform_returns_nearest_stable_label() -> None:
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1, 1] = 4
    labels[3, 4] = 9
    distance, nearest = _distance_and_nearest(labels, 200.0)
    assert distance[1, 1] == 0.0
    assert distance[3, 4] == 0.0
    assert nearest[1, 1] == 3
    assert nearest[3, 4] == 8
    assert np.all(np.isfinite(distance))
    assert set(np.unique(nearest)) == {3, 8}


def _tiny_sources() -> _ChannelSources:
    height = width = 5
    count = height * width
    rows, columns = np.indices((height, width), dtype=np.int32)
    cell_id = np.arange(10_000, 10_000 + count, dtype=np.uint64)
    path_rows = np.asarray([6, 7, 13, 18], dtype=np.int32)
    xyz = np.stack(
        (
            np.ones(len(path_rows), dtype=np.float64),
            columns.reshape(-1)[path_rows] * 1e-4,
            rows.reshape(-1)[path_rows] * 1e-4,
        ),
        axis=1,
    )
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    reaches = pa.table(
        {
            "reach_id": pa.array([3], type=pa.int32()),
            "from_cell_id": pa.array([int(cell_id[path_rows[0]])], type=pa.int64()),
            "to_cell_id": pa.array([int(cell_id[path_rows[-1]])], type=pa.int64()),
            "upstream_reach_ids": pa.array([[]], type=pa.list_(pa.int32())),
            "downstream_reach_id": pa.array([-1], type=pa.int32()),
            "cell_path": pa.array(
                [cell_id[path_rows].astype(np.int64).tolist()],
                type=pa.list_(pa.int64()),
            ),
            "reach_kind": pa.array(["channel"]),
            "discharge_mean": pa.array([8.0], type=pa.float32()),
            "discharge_seasonal": pa.array(
                [
                    [1.0] * 8 + [0.0] * 4,
                ],
                type=pa.list_(pa.float32(), 12),
            ),
            "channel_width_m": pa.array([20.0], type=pa.float32()),
            "floodplain_width_m": pa.array([300.0], type=pa.float32()),
            "valley_width_m": pa.array([600.0], type=pa.float32()),
            "polyline_on_cubed_sphere": pa.array(
                [xyz.astype(np.float32).tolist()],
                type=pa.list_(pa.list_(pa.float32(), 3)),
            ),
        }
    )
    order = np.arange(count, dtype=np.int32)
    return _ChannelSources(
        target_id="tiny",
        terrain_manifest={},
        hydrology_manifest={},
        reaches=reaches,
        actual_cell_size_m=200.0,
        cell_id=cell_id,
        row=rows.reshape(-1),
        column=columns.reshape(-1),
        elevation_m=np.linspace(100.0, 0.0, count, dtype=np.float32),
        inside_display=np.ones(count, dtype=bool),
        inside_core=np.ones(count, dtype=bool),
        lake_fraction=np.zeros(count, dtype=np.float32),
        wetland_fraction=np.zeros(count, dtype=np.float32),
        inherited_floodplain_fraction=np.zeros(count, dtype=np.float32),
        spatial_order=order,
        inverse_spatial_order=order,
        height=height,
        width=width,
    )


def test_centerline_table_and_support_are_anchored_finite_and_nested(tmp_path: Path) -> None:
    sources = _tiny_sources()
    config = L3ChannelGeometryConfig(
        terrain_dir=tmp_path,
        hydrology_dir=tmp_path,
        output_dir=tmp_path,
    )
    table, metrics = _build_centerline_table(sources, config)
    assert table.num_rows == 1
    assert metrics["channel_reach_count"] == 1
    assert metrics["reliable_flow_reach_count"] == 1
    assert metrics["maximum_endpoint_drift_m"] == 0.0
    assert metrics["maximum_centerline_offset_m"] < 150.0
    assert table["reliable_flow"][0].as_py()

    fields = _support_fields(table, sources, config)
    for values in fields.values():
        assert np.all(np.isfinite(values))
    channel = fields["channel_fraction"]
    riparian = fields["riparian_fraction"]
    floodplain = fields["floodplain_fraction"]
    valley = fields["valley_fraction"]
    assert np.all(channel <= riparian)
    assert np.all(riparian <= floodplain)
    assert np.all(floodplain <= valley)
    assert np.count_nonzero(channel) > 0
    assert np.count_nonzero(valley) > np.count_nonzero(channel)


def test_cli_reports_channel_geometry_result(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
terrain_output_dir: terrain
hydrology_output_dir: hydrology
channel_geometry_output_dir: channels
""",
        encoding="utf8",
    )
    result = L3ChannelGeometryResult(
        output_dir=tmp_path / "channels",
        manifest_path=tmp_path / "channels/manifest.json",
        validation_path=tmp_path / "channels/validation.json",
        zarr_path=tmp_path / "channels/channel_geometry.zarr",
        preview_path=tmp_path / "channels/channel_geometry.png",
        target_id="tiny",
        display_cell_count=25,
        channel_reach_count=7,
        reliable_flow_reach_count=5,
        validation_passed=True,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.l3_channel_geometry.generate_l3_channel_geometry",
        lambda _config: result,
    )
    assert main(["l3-channel-geometry", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "Smoothed 7 physical channel reaches" in output
    assert "25 displayed cells" in output


def test_generate_replays_and_rejects_corrupt_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sources = _tiny_sources()
    config = L3ChannelGeometryConfig(
        terrain_dir=tmp_path / "terrain",
        hydrology_dir=tmp_path / "hydrology",
        output_dir=tmp_path / "channels",
    )
    monkeypatch.setattr(
        channel_geometry_module,
        "_load_sources",
        lambda _config: sources,
    )
    monkeypatch.setattr(
        channel_geometry_module,
        "_fingerprint",
        lambda _config, _sources: (
            "tiny-channel-fingerprint",
            {"test_fixture": "tiny"},
        ),
    )

    result = generate_l3_channel_geometry(config)
    assert result.validation_passed
    assert result.channel_reach_count == 1
    assert result.display_cell_count == 25
    first_manifest = result.manifest_path.read_bytes()

    cached = generate_l3_channel_geometry(config)
    assert cached.manifest_path.read_bytes() == first_manifest

    cached.preview_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        generate_l3_channel_geometry(config)
