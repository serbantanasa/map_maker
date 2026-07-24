from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from map_maker.cli import main
from map_maker.pipeline import l3_surface_materials
from map_maker.pipeline._surface_materials_native import run_surface_materials
from map_maker.pipeline.l3_surface_materials import (
    L3SurfaceMaterialsConfig,
    L3SurfaceMaterialsResult,
    NATIVE_OUTPUT_NAMES,
    REQUIRED_PARENT_PRIORS,
    _MaterialSources,
    _alluvial_legacy_fraction,
    _allocate_outputs,
    _chunk_inputs,
    _completed_chunk_is_valid,
    _file_checksum,
    _generate_chunks,
    _initialize_partial,
    _interpolate_prior,
    _interpolation_stencil,
    _kernel_controls,
    _local_relief,
    _local_terrain_slope,
    _require_disjoint_output,
    _require_source_manifest_checksum,
    _validate,
)
from map_maker.pipeline.stages.surface_materials import MATERIAL_OUTPUTS


def test_config_resolves_paths_and_validates_soil_controls(tmp_path: Path) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
terrain_output_dir: terrain
hydrology_output_dir: hydrology
channel_geometry_output_dir: channels
surface_materials_output_dir: soils
l3_surface_materials:
  chunk_rows: 32768
  local_relief_radius_cells: 8
  terrain_slope_smoothing_radius_cells: 6
  spinup_years: 12
limits:
  maximum_surface_materials_storage_gb: 1
""",
        encoding="utf8",
    )
    config = L3SurfaceMaterialsConfig.from_file(config_path)
    assert config.terrain_dir == tmp_path / "terrain"
    assert config.hydrology_dir == tmp_path / "hydrology"
    assert config.channel_geometry_dir == tmp_path / "channels"
    assert config.output_dir == tmp_path / "soils"
    assert config.chunk_rows == 32_768
    assert config.local_relief_radius_cells == 8
    assert config.terrain_slope_smoothing_radius_cells == 6
    assert config.spinup_years == 12


def test_local_relief_uses_continuous_spatial_window() -> None:
    elevation = np.zeros(25, dtype=np.float32)
    elevation[12] = 100.0
    order = np.arange(25, dtype=np.int32)
    relief = _local_relief(elevation, order, 5, 5, 1).reshape(5, 5)
    assert relief[2, 2] == 100.0
    assert relief[1, 1] == 100.0
    assert relief[0, 0] == 0.0


def test_local_terrain_slope_uses_physical_elevation_gradient() -> None:
    rows, columns = np.indices((5, 5), dtype=np.float32)
    elevation = (8.0 * rows + 6.0 * columns).reshape(-1)
    order = np.arange(25, dtype=np.int32)
    slope = _local_terrain_slope(elevation, order, 5, 5, 2.0, 1).reshape(5, 5)
    assert slope[2, 2] == pytest.approx(5.0, abs=1e-6)
    assert np.isfinite(slope).all()
    assert np.all(slope >= 0.0)


def test_alluvial_legacy_localizes_parent_prior_to_low_valley_ground(
    tmp_path: Path,
) -> None:
    config = L3SurfaceMaterialsConfig(
        terrain_dir=tmp_path,
        hydrology_dir=tmp_path,
        channel_geometry_dir=tmp_path,
        output_dir=tmp_path,
        chunk_rows=16_384,
    )
    legacy = _alluvial_legacy_fraction(
        np.full(4, 0.2, dtype=np.float32),
        np.asarray([0.0, 500.0, 5_000.0, 500.0], dtype=np.float32),
        np.asarray([1.0, 0.6, 0.0, 0.6], dtype=np.float32),
        np.asarray([0.001, 0.003, 0.003, 0.08], dtype=np.float32),
        config,
    )
    assert legacy[0] > legacy[1] > legacy[2]
    assert legacy[1] > legacy[3]
    assert np.all((legacy >= 0.0) & (legacy <= config.maximum_alluvial_fraction))


def test_source_lineage_requires_exact_manifest_checksum() -> None:
    manifest = {"source": {"terrain_manifest_sha256": "expected"}}
    _require_source_manifest_checksum(
        manifest,
        "terrain_manifest_sha256",
        "expected",
        "hydrology",
    )
    with pytest.raises(RuntimeError, match="source lineage mismatch"):
        _require_source_manifest_checksum(
            manifest,
            "terrain_manifest_sha256",
            "different",
            "hydrology",
        )


def test_output_path_must_not_overlap_upstream_artifacts(tmp_path: Path) -> None:
    terrain = tmp_path / "terrain"
    with pytest.raises(ValueError, match="overlaps source artifact"):
        _require_disjoint_output(terrain, (terrain,))
    with pytest.raises(ValueError, match="overlaps source artifact"):
        _require_disjoint_output(terrain / "soils", (terrain,))
    with pytest.raises(ValueError, match="overlaps source artifact"):
        _require_disjoint_output(tmp_path, (terrain,))


def _tiny_sources() -> _MaterialSources:
    height = width = 4
    count = height * width
    rows, columns = np.indices((height, width), dtype=np.int32)
    parent_ids = np.arange(4, dtype=np.int32)
    l0_parent = (rows // 2 * 2 + columns // 2).reshape(-1).astype(np.int32)
    parent_priors: dict[str, np.ndarray] = {}
    for path in REQUIRED_PARENT_PRIORS:
        if path.endswith("GeologicalProvinceClass"):
            values = np.asarray([2, 3, 6, 7], dtype=np.uint8)
        elif path.endswith("MonthlySurfaceTemperatureC"):
            values = np.asarray(
                [[8.0 + parent + month * 0.5 for month in range(12)] for parent in range(4)],
                dtype=np.float32,
            )
        elif path.endswith("MonthlyPrecipitationMm"):
            values = np.full((4, 12), 80.0, dtype=np.float32)
        elif path.endswith("MonthlyEvaporationMm"):
            values = np.full((4, 12), 45.0, dtype=np.float32)
        elif path.endswith("MonthlySnowfallMm"):
            values = np.full((4, 12), 8.0, dtype=np.float32)
        elif any(path.endswith(name) for name in MATERIAL_OUTPUTS):
            material_prior = {
                "BedrockSurfaceFraction": 0.10,
                "ResidualRegolithFraction": 0.55,
                "ColluviumFraction": 0.05,
                "AlluviumFraction": 0.18,
                "LacustrineSedimentFraction": 0.07,
                "GlacialDepositFraction": 0.00,
                "VolcaniclasticFraction": 0.05,
            }
            name = next(name for name in MATERIAL_OUTPUTS if path.endswith(name))
            values = np.full(4, material_prior[name], dtype=np.float32)
        elif path.endswith("ClimateOrographyM"):
            values = np.full(4, 100.0, dtype=np.float32)
        elif path.endswith("CrustAgeGa"):
            values = np.linspace(1.0, 3.0, 4, dtype=np.float32)
        elif path.endswith("GlacierIceFraction"):
            values = np.zeros(4, dtype=np.float32)
        elif path.endswith("SoilDepthM"):
            values = np.full(4, 1.2, dtype=np.float32)
        elif path.endswith("SoilPH"):
            values = np.full(4, 6.5, dtype=np.float32)
        else:
            values = np.full(4, 0.7, dtype=np.float32)
        parent_priors[path] = values

    monthly_precipitation = np.full((12, count), 90.0, dtype=np.float32)
    hydrology = {
        "forcing/monthly_precipitation_mm": monthly_precipitation,
        "forcing/monthly_snowmelt_mm": np.full((12, count), 4.0, dtype=np.float32),
        "forcing/monthly_glacier_melt_mm": np.zeros((12, count), dtype=np.float32),
        "surface/physical_ocean_fraction": np.zeros(count, dtype=np.float32),
        "surface/lake_fraction": np.zeros(count, dtype=np.float32),
        "surface/wetland_fraction": np.zeros(count, dtype=np.float32),
        "routing/flow_slope": np.full(count, 0.003, dtype=np.float32),
        "routing/depression_fill_depth_m": np.zeros(count, dtype=np.float32),
    }
    hydrology["surface/lake_fraction"][10] = 0.6
    hydrology["surface/wetland_fraction"][9] = 0.5
    channels = {
        "support/channel_fraction": np.zeros(count, dtype=np.float32),
        "support/floodplain_fraction": np.zeros(count, dtype=np.float32),
        "support/valley_fraction": np.zeros(count, dtype=np.float32),
        "support/distance_to_channel_m": np.full(count, 5_000.0, dtype=np.float32),
        "support/centerline_seed": np.zeros(count, dtype=bool),
    }
    channels["support/channel_fraction"][5:8] = 0.15
    channels["support/floodplain_fraction"][4:8] = 0.8
    channels["support/valley_fraction"][4:8] = 0.9
    channels["support/distance_to_channel_m"][4:8] = 100.0
    channels["support/centerline_seed"][5:8] = True
    elevation = np.linspace(0.0, 300.0, count, dtype=np.float32)
    order = np.arange(count, dtype=np.int32)
    return _MaterialSources(
        target_id="tiny",
        terrain_manifest={},
        hydrology_manifest={},
        channel_manifest={},
        handoff_manifest={},
        handoff_dir=Path("."),
        terrain={},
        hydrology=hydrology,
        channels=channels,
        parent_ids=parent_ids,
        parent_priors=parent_priors,
        cell_id=np.arange(1000, 1000 + count, dtype=np.uint64),
        face=np.zeros(count, dtype=np.uint8),
        row=rows.reshape(-1),
        column=columns.reshape(-1),
        l0_parent_id=l0_parent,
        area_km2=np.full(count, 0.04, dtype=np.float64),
        elevation_m=elevation,
        local_relief_m=_local_relief(elevation, order, height, width, 1),
        local_terrain_slope=_local_terrain_slope(
            elevation,
            order,
            height,
            width,
            200.0,
            1,
        ),
        inside_display=np.ones(count, dtype=bool),
        inside_core=np.ones(count, dtype=bool),
        spatial_order=order,
        height=height,
        width=width,
        parent_face_resolution=2,
        child_face_resolution=4,
        actual_cell_size_m=200.0,
    )


def test_parent_interpolation_and_native_chunk_are_continuous_and_conservative(
    tmp_path: Path,
) -> None:
    sources = _tiny_sources()
    config = L3SurfaceMaterialsConfig(
        terrain_dir=tmp_path,
        hydrology_dir=tmp_path,
        channel_geometry_dir=tmp_path,
        output_dir=tmp_path,
        chunk_rows=16_384,
        spinup_years=12,
    )
    rows, weights, exact = _interpolation_stencil(sources, 0, 16)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert np.array_equal(exact, sources.l0_parent_id)
    interpolated = _interpolate_prior(
        np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32),
        rows,
        weights,
    )
    assert np.all(np.isfinite(interpolated))
    assert np.unique(interpolated).size > np.unique(exact).size

    inputs, drivers, _ = _chunk_inputs(sources, config, 0, 16)
    assert inputs["monthly_temperature"].shape == (12, 16)
    assert inputs["monthly_precipitation"].shape == (12, 16)
    assert np.all(inputs["monthly_snowfall"] <= inputs["monthly_precipitation"])
    assert np.ptp(drivers["TemperatureAdjustmentC"]) > 0.0
    assert drivers["AlluvialLegacyFraction"][5] > drivers["AlluvialLegacyFraction"][0]
    np.testing.assert_array_equal(drivers["LocalTerrainSlope"], inputs["terrain_slope"])
    assert not np.array_equal(
        inputs["terrain_slope"],
        sources.hydrology["routing/flow_slope"],
    )
    outputs = _allocate_outputs(16)
    metadata = run_surface_materials(
        **_kernel_controls(config),
        **inputs,
        **outputs,
    )
    artifact_outputs = {
        artifact_name: outputs[native_name]
        for native_name, artifact_name in NATIVE_OUTPUT_NAMES.items()
    }
    material_sum = sum(artifact_outputs[name] for name in MATERIAL_OUTPUTS)
    np.testing.assert_allclose(material_sum, 1.0, atol=1e-6)
    assert artifact_outputs["AlluviumFraction"][5] > artifact_outputs["AlluviumFraction"][0]
    assert artifact_outputs["LacustrineSedimentFraction"][10] > 0.3
    assert artifact_outputs["MonthlySoilWaterMm"].shape == (12, 16)
    assert metadata["water_balance_relative_error"] < 1e-9


def test_cli_reports_surface_material_result(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
terrain_output_dir: terrain
hydrology_output_dir: hydrology
channel_geometry_output_dir: channels
surface_materials_output_dir: soils
""",
        encoding="utf8",
    )
    result = L3SurfaceMaterialsResult(
        output_dir=tmp_path / "soils",
        manifest_path=tmp_path / "soils/manifest.json",
        validation_path=tmp_path / "soils/validation.json",
        zarr_path=tmp_path / "soils/surface_materials.zarr",
        preview_path=tmp_path / "soils/surface_materials.png",
        target_id="tiny",
        display_cell_count=16,
        chunk_count=1,
        validation_passed=True,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.l3_surface_materials.generate_l3_surface_materials",
        lambda _config: result,
    )
    assert main(["l3-surface-materials", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "16 displayed cells" in output
    assert "1 chunks passed" in output


def test_generate_cache_and_corruption_detection(tmp_path: Path, monkeypatch) -> None:
    sources = _tiny_sources()
    output_dir = tmp_path / "surface-materials"
    config = L3SurfaceMaterialsConfig(
        terrain_dir=tmp_path / "terrain",
        hydrology_dir=tmp_path / "hydrology",
        channel_geometry_dir=tmp_path / "channels",
        output_dir=output_dir,
        chunk_rows=16_384,
        spinup_years=4,
        maximum_storage_gb=0.5,
    )
    monkeypatch.setattr(l3_surface_materials, "_load_sources", lambda _config: sources)
    monkeypatch.setattr(
        l3_surface_materials,
        "_fingerprint",
        lambda _config, _sources: ("tiny-fingerprint", {"test": True}),
    )

    first = l3_surface_materials.generate_l3_surface_materials(config)
    assert first.validation_passed
    assert first.zarr_path.exists()
    first_manifest_mtime = first.manifest_path.stat().st_mtime_ns

    cached = l3_surface_materials.generate_l3_surface_materials(config)
    assert cached.manifest_path.stat().st_mtime_ns == first_manifest_mtime

    chunk = first.zarr_path / "materials" / "BedrockSurfaceFraction" / "0"
    chunk.write_bytes(chunk.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="Cached tree checksum mismatch"):
        l3_surface_materials.generate_l3_surface_materials(config)


def test_resume_regenerates_a_corrupt_completed_chunk(tmp_path: Path) -> None:
    sources = _tiny_sources()
    config = L3SurfaceMaterialsConfig(
        terrain_dir=tmp_path / "terrain",
        hydrology_dir=tmp_path / "hydrology",
        channel_geometry_dir=tmp_path / "channels",
        output_dir=tmp_path / "result",
        chunk_rows=8,
        spinup_years=2,
    )
    partial = tmp_path / ".result.partial"
    root = _initialize_partial(partial, config, sources, "resume-test")
    resumed, chunk_count = _generate_chunks(root, partial, config, sources)
    assert resumed == 0
    assert chunk_count == 2
    output_paths = tuple(l3_surface_materials.OUTPUT_PATHS.values())
    assert _completed_chunk_is_valid(root, partial, 0, 0, 8, output_paths)
    assert _completed_chunk_is_valid(root, partial, 1, 8, 16, output_paths)

    chunk = partial / "surface_materials.zarr" / "materials" / "BedrockSurfaceFraction" / "0"
    expected_checksum = _file_checksum(chunk)
    chunk.write_bytes(chunk.read_bytes() + b"corrupt")
    resumed, chunk_count = _generate_chunks(root, partial, config, sources)
    assert resumed == 1
    assert chunk_count == 2
    assert _file_checksum(chunk) == expected_checksum


def test_validation_rejects_unbounded_negative_and_ocean_state(tmp_path: Path) -> None:
    sources = _tiny_sources()
    sources.hydrology["surface/physical_ocean_fraction"][0] = 1.0
    config = L3SurfaceMaterialsConfig(
        terrain_dir=tmp_path / "terrain",
        hydrology_dir=tmp_path / "hydrology",
        channel_geometry_dir=tmp_path / "channels",
        output_dir=tmp_path / "result",
        chunk_rows=8,
        spinup_years=2,
    )
    partial = tmp_path / ".result.partial"
    root = _initialize_partial(partial, config, sources, "validation-test")
    _generate_chunks(root, partial, config, sources)
    root["soil/SoilFertilityPotential"][1] = 2.0
    root["soil/RegolithDepthM"][2] = -1.0
    root["monthly/MonthlySoilWaterMm"][0, 0] = 1.0

    validation = _validate(root, config, sources)
    assert not validation["gates"]["bounded_fraction_outputs"]
    assert not validation["gates"]["nonnegative_static_outputs"]
    assert not validation["gates"]["ocean_outputs_zero"]
