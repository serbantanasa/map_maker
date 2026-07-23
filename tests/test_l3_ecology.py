from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from map_maker.cli import main
from map_maker.pipeline import l3_ecology
from map_maker.pipeline.l3_ecology import (
    L3EcologyConfig,
    L3EcologyResult,
    REQUIRED_PARENT_PRIORS,
    _EcologySources,
    _generate_chunks,
    _initialize_partial,
    _interpolate_prior,
    _interpolation_stencil,
    _open_partial,
    _run_chunk,
    _validate,
)
from map_maker.pipeline.stages.atmosphere import AtmosphereConfig
from map_maker.pipeline.stages.biosphere_envelope import BiosphereEnvelopeConfig
from map_maker.pipeline.stages.derived_biomes import DerivedBiomeConfig
from map_maker.pipeline.stages.functional_vegetation import FunctionalVegetationConfig
from map_maker.pipeline.stages.planet import PlanetConfig
from map_maker.pipeline.stages.potential_biosphere import PotentialBiosphereConfig


def test_config_resolves_paths_and_keeps_ecology_controls_regional(tmp_path: Path) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
terrain_output_dir: terrain
hydrology_output_dir: hydrology
channel_geometry_output_dir: channels
surface_materials_output_dir: soils
ecology_output_dir: ecology
l3_ecology:
  chunk_rows: 32768
  minimum_wet_ecology_response: -0.1
limits:
  maximum_ecology_storage_gb: 2
""",
        encoding="utf8",
    )
    config = L3EcologyConfig.from_file(config_path)
    assert config.terrain_dir == tmp_path / "terrain"
    assert config.hydrology_dir == tmp_path / "hydrology"
    assert config.channel_geometry_dir == tmp_path / "channels"
    assert config.surface_materials_dir == tmp_path / "soils"
    assert config.output_dir == tmp_path / "ecology"
    assert config.chunk_rows == 32_768
    assert config.minimum_wet_ecology_response == -0.1
    assert config.maximum_storage_gb == 2.0


def _tiny_sources() -> _EcologySources:
    height = width = 64
    count = height * width
    rows, columns = np.indices((height, width), dtype=np.int32)
    cell_id = np.arange(1_000_000, 1_000_000 + count, dtype=np.uint64)
    area = np.full(count, 0.04, dtype=np.float64)
    elevation = (200.0 + rows.reshape(-1) * 12.0 + 45.0 * np.sin(columns.reshape(-1) / 5.0)).astype(
        np.float32
    )
    relief = np.full(count, 350.0, dtype=np.float32)
    valley = columns.reshape(-1) % (width // 2) < width // 4
    wetness = np.where(valley, 0.75, 0.30).astype(np.float32)
    l0_parent = (rows // (height // 2) * 2 + columns // (width // 2)).reshape(-1)
    parent_ids = np.arange(4, dtype=np.int32)
    parent_priors: dict[str, np.ndarray] = {}
    for path in REQUIRED_PARENT_PRIORS:
        if path.endswith("MonthlyInsolationWm2"):
            value = np.asarray(
                [[220.0 + 80.0 * np.sin((month + 0.5) * np.pi / 6.0) for month in range(12)]],
                dtype=np.float32,
            )
        elif path.endswith("MonthlySurfaceTemperatureC"):
            value = np.asarray(
                [[8.0 + 10.0 * np.sin((month + 0.5) * np.pi / 6.0) for month in range(12)]],
                dtype=np.float32,
            )
        elif path.endswith("MonthlyPrecipitationMm"):
            value = np.full((1, 12), 55.0, dtype=np.float32)
        elif path.endswith("GlacierIceFraction"):
            value = np.zeros(1, dtype=np.float32)
        elif path.endswith("AnnualPotentialNPPKgCM2"):
            value = np.full(1, 0.20, dtype=np.float32)
        elif path.endswith("FunctionalTypeFractions"):
            value = np.asarray(
                [[0.20, 0.05, 0.20, 0.05, 0.20, 0.10, 0.05, 0.05]],
                dtype=np.float32,
            )
        elif path.endswith("BiomeFractions"):
            value = np.full((1, 13), 1.0 / 13.0, dtype=np.float32)
        else:
            raise AssertionError(path)
        if value.shape[0] == 1:
            value = np.repeat(value, len(parent_ids), axis=0)
        parent_priors[path] = value

    monthly_precipitation = np.full((12, count), 85.0, dtype=np.float32)
    hydrology = {
        "forcing/monthly_precipitation_mm": monthly_precipitation,
        "surface/physical_ocean_fraction": np.zeros(count, dtype=np.float32),
        "surface/lake_fraction": np.zeros(count, dtype=np.float32),
        "surface/wetland_fraction": np.where(valley, 0.08, 0.0).astype(np.float32),
    }
    channels = {
        "support/valley_fraction": np.where(valley, 0.85, 0.05).astype(np.float32),
        "support/centerline_seed": (
            (columns.reshape(-1) == width // 4) & (rows.reshape(-1) % 3 == 0)
        ),
    }
    monthly_saturation = np.repeat(wetness[:, None], 12, axis=1)
    monthly_liquid = np.full((count, 12), 80.0, dtype=np.float32)
    surface = {
        "drivers/TemperatureAdjustmentC": (-0.002 * elevation).astype(np.float32),
        "monthly/MonthlySoilLiquidInputMm": monthly_liquid,
        "monthly/MonthlySoilSaturationFraction": monthly_saturation,
        "soil/SoilBearingFraction": np.full(count, 0.90, dtype=np.float32),
        "soil/SoilNutrientPotential": np.where(valley, 0.80, 0.55).astype(np.float32),
        "soil/SoilFertilityPotential": np.where(valley, 0.80, 0.50).astype(np.float32),
        "soil/SoilSalinityIndex": np.zeros(count, dtype=np.float32),
        "soil/SoilConfidence": np.full(count, 0.90, dtype=np.float32),
        "soil/SoilDepthM": np.where(valley, 1.4, 0.8).astype(np.float32),
        "soil/RegolithDepthM": np.where(valley, 2.5, 1.5).astype(np.float32),
        "soil/HydricSoilFraction": np.where(valley, 0.20, 0.01).astype(np.float32),
        "soil/SoilDrainageIndex": np.where(valley, 0.45, 0.75).astype(np.float32),
    }
    order = np.arange(count, dtype=np.int32)
    return _EcologySources(
        target_id="tiny",
        terrain_manifest={},
        hydrology_manifest={},
        channel_manifest={},
        surface_manifest={},
        handoff_manifest={},
        handoff_dir=Path("."),
        world_config_path=Path("world.yaml"),
        terrain={},
        hydrology=hydrology,
        channels=channels,
        surface=surface,
        parent_ids=parent_ids,
        parent_area_km2=np.asarray(
            [float(np.sum(area[l0_parent == parent])) for parent in parent_ids],
            dtype=np.float64,
        ),
        parent_priors=parent_priors,
        cell_id=cell_id,
        face=np.zeros(count, dtype=np.uint8),
        row=rows.reshape(-1),
        column=columns.reshape(-1),
        l0_parent_id=np.ascontiguousarray(l0_parent, dtype=np.int32),
        area_km2=area,
        elevation_m=elevation,
        local_relief_m=relief,
        inside_display=np.ones(count, dtype=bool),
        inside_core=np.ones(count, dtype=bool),
        spatial_order=order,
        height=height,
        width=width,
        parent_face_resolution=2,
        child_face_resolution=64,
        actual_cell_size_m=200.0,
        planet_config=PlanetConfig(),
        atmosphere_config=AtmosphereConfig(),
        envelope_config=BiosphereEnvelopeConfig(),
        potential_config=PotentialBiosphereConfig(),
        functional_config=FunctionalVegetationConfig(),
        biome_config=DerivedBiomeConfig(),
    )


def _tiny_config(tmp_path: Path) -> L3EcologyConfig:
    return L3EcologyConfig(
        terrain_dir=tmp_path / "terrain",
        hydrology_dir=tmp_path / "hydrology",
        channel_geometry_dir=tmp_path / "channels",
        surface_materials_dir=tmp_path / "soils",
        output_dir=tmp_path / "ecology",
        chunk_rows=16_384,
        maximum_parent_npp_relative_difference_p95=100.0,
        maximum_parent_functional_l1_difference_p95=10.0,
        maximum_parent_biome_l1_difference_p95=10.0,
        maximum_parent_boundary_p95_ratio=1_000.0,
        maximum_parent_boundary_absolute_difference_p95=1.0,
        maximum_repeated_parent_motif_correlation_p95=1.0,
        minimum_wet_ecology_response=-1.0,
        minimum_cold_highland_response=-1.0,
        minimum_valley_productivity_response=-1.0,
        minimum_valley_resource_response=-1.0,
        maximum_storage_gb=1.0,
    )


def test_flat_regional_chunk_replays_all_four_native_kernels() -> None:
    sources = _tiny_sources()
    rows, weights, exact = _interpolation_stencil(sources, 0, len(sources.cell_id))
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert np.array_equal(exact, sources.l0_parent_id)
    assert set(np.unique(exact)) == {0, 1, 2, 3}
    interpolated = _interpolate_prior(
        np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32),
        rows,
        weights,
    )
    assert np.unique(interpolated).size > 4
    drivers, envelope, potential, functional, biome, stats = _run_chunk(
        sources,
        0,
        len(sources.cell_id),
    )
    assert drivers["MonthlyTemperatureC"].shape == (12, len(sources.cell_id))
    np.testing.assert_allclose(drivers["AnnualPrecipitationMm"], 12.0 * 55.0)
    assert envelope["monthly_primary_energy_out"].shape == (12, len(sources.cell_id))
    assert potential["monthly_npp_out"].shape == (12, len(sources.cell_id))
    assert functional["functional_type_fractions_out"].shape == (
        8,
        len(sources.cell_id),
    )
    assert biome["biome_fractions_out"].shape == (13, len(sources.cell_id))
    np.testing.assert_allclose(
        np.sum(functional["functional_type_fractions_out"], axis=0)
        + np.sum(functional["nonvegetated_fractions_out"], axis=0),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.sum(biome["biome_fractions_out"], axis=0)
        + functional["nonvegetated_fractions_out"][2]
        + functional["nonvegetated_fractions_out"][3],
        1.0,
        atol=1e-6,
    )
    assert stats["functional_maximum_partition_absolute_error"] < 1e-5


def test_fine_soil_water_changes_productivity_under_matched_climate() -> None:
    sources = _tiny_sources()
    count = len(sources.cell_id)
    wet = np.arange(count) < count // 2
    surface = {name: np.array(values, copy=True) for name, values in sources.surface.items()}
    surface["drivers/TemperatureAdjustmentC"][:] = 0.0
    surface["monthly/MonthlySoilLiquidInputMm"][:] = np.where(
        wet[:, None],
        110.0,
        10.0,
    )
    surface["monthly/MonthlySoilSaturationFraction"][:] = np.where(
        wet[:, None],
        0.80,
        0.20,
    )
    for name, value in (
        ("soil/SoilBearingFraction", 0.90),
        ("soil/SoilNutrientPotential", 0.70),
        ("soil/SoilFertilityPotential", 0.70),
        ("soil/SoilSalinityIndex", 0.0),
        ("soil/SoilConfidence", 0.90),
        ("soil/SoilDepthM", 1.0),
        ("soil/RegolithDepthM", 2.0),
        ("soil/HydricSoilFraction", 0.05),
        ("soil/SoilDrainageIndex", 0.60),
    ):
        surface[name][:] = value
    priors = {name: np.array(values, copy=True) for name, values in sources.parent_priors.items()}
    temperature = priors["parent_priors/climate/MonthlySurfaceTemperatureC"]
    temperature[:] = temperature[0]
    matched = replace(
        sources,
        surface=surface,
        parent_priors=priors,
        elevation_m=np.full(count, 500.0, dtype=np.float32),
        local_relief_m=np.full(count, 100.0, dtype=np.float32),
    )
    _, _, potential, _, _, _ = _run_chunk(matched, 0, count)
    annual_npp = potential["annual_npp_out"]
    assert float(np.mean(annual_npp[wet])) > float(np.mean(annual_npp[~wet]))


def test_cli_reports_ecology_result(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "l3.yaml"
    config_path.write_text(
        """\
terrain_output_dir: terrain
hydrology_output_dir: hydrology
channel_geometry_output_dir: channels
surface_materials_output_dir: soils
ecology_output_dir: ecology
""",
        encoding="utf8",
    )
    result = L3EcologyResult(
        output_dir=tmp_path / "ecology",
        manifest_path=tmp_path / "ecology/manifest.json",
        validation_path=tmp_path / "ecology/validation.json",
        zarr_path=tmp_path / "ecology/ecology.zarr",
        preview_path=tmp_path / "ecology/ecology.png",
        target_id="tiny",
        display_cell_count=4_096,
        chunk_count=1,
        validation_passed=True,
    )
    monkeypatch.setattr(
        "map_maker.pipeline.l3_ecology.generate_l3_ecology",
        lambda _config: result,
    )
    assert main(["l3-ecology", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "4096 displayed cells" in output
    assert "1 chunks passed" in output


def test_generate_cache_replay_and_corruption_detection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sources = _tiny_sources()
    config = _tiny_config(tmp_path)
    monkeypatch.setattr(l3_ecology, "_load_sources", lambda _config: sources)
    monkeypatch.setattr(
        l3_ecology,
        "_fingerprint",
        lambda _config, _sources: ("tiny-fingerprint", {"test": True}),
    )
    first = l3_ecology.generate_l3_ecology(config)
    assert first.validation_passed
    first_manifest_mtime = first.manifest_path.stat().st_mtime_ns
    cached = l3_ecology.generate_l3_ecology(config)
    assert cached.manifest_path.stat().st_mtime_ns == first_manifest_mtime

    chunk = first.zarr_path / "biomes" / "BiomeFractions" / "0.0"
    chunk.write_bytes(chunk.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="Cached tree checksum mismatch"):
        l3_ecology.generate_l3_ecology(config)


def test_partial_resume_repairs_corrupt_geometry_and_output_chunks(tmp_path: Path) -> None:
    sources = _tiny_sources()
    config = replace(_tiny_config(tmp_path), chunk_rows=1_024)
    partial = tmp_path / ".ecology.partial"
    root = _initialize_partial(partial, config, sources, "resume-integrity")
    resumed_count, chunk_count = _generate_chunks(root, partial, config, sources)
    assert resumed_count == 0
    assert chunk_count == 4

    geometry_chunk = partial / "ecology.zarr" / "geometry" / "cell_id" / "0"
    output_chunk = partial / "ecology.zarr" / "biomes" / "BiomeFractions" / "1.0"
    geometry_chunk.write_bytes(b"corrupt geometry")
    output_chunk.write_bytes(b"corrupt output")
    del root

    root, resumed = _open_partial(partial, config, sources, "resume-integrity")
    assert resumed
    resumed_count, chunk_count = _generate_chunks(root, partial, config, sources)
    assert resumed_count == 2
    assert chunk_count == 4
    np.testing.assert_array_equal(root["geometry/cell_id"][:], sources.cell_id)
    validation = _validate(root, config, sources)
    assert validation["geometry_mismatch_count"] == 0
    assert validation["gates"]["geometry_matches_sources"]


def test_validation_reconstructs_query_codes(tmp_path: Path) -> None:
    sources = _tiny_sources()
    config = _tiny_config(tmp_path)
    partial = tmp_path / ".ecology.partial"
    root = _initialize_partial(partial, config, sources, "validation")
    _generate_chunks(root, partial, config, sources)
    assert _validate(root, config, sources)["gates"]["biome_codes_reconstructed"]
    root["geometry/cell_id"][0] = sources.cell_id[0] + np.uint64(1)
    validation = _validate(root, config, sources)
    assert not validation["gates"]["geometry_matches_sources"]
    root["geometry/cell_id"][0] = sources.cell_id[0]
    root["biomes/DominantBiomeCode"][0] = np.uint8(13)
    validation = _validate(root, config, sources)
    assert not validation["gates"]["biome_codes_reconstructed"]
