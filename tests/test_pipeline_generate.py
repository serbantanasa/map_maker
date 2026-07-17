from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from map_maker.cli import main
from map_maker.pipeline import PipelineConfig, registry
from map_maker.pipeline.generate import generate_world, render_world


def _config(tmp_path: Path, run_id: str, *, seed: int = 17) -> PipelineConfig:
    return PipelineConfig.from_mapping(
        {
            "topology": "sphere",
            "resolutions": [{"height": 32, "width": 64}],
            "rng_seed": seed,
            "run_id": run_id,
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs"),
            "stage_overrides": {
                "tectonics": {"num_plates": 8, "time_steps": 3},
                "erosion": {"steps": 2},
            },
        }
    )


def test_render_world_produces_rgb_image():
    elevation = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(8, 16)
    ocean = elevation < 0.0
    incision = np.maximum(elevation - 0.5, 0.0)
    image = render_world(elevation, ocean, incision)

    assert image.mode == "RGB"
    assert image.size == (16, 8)
    assert np.asarray(image).std() > 0.0


def test_generate_world_writes_preview_manifest_and_reuses_cache(tmp_path: Path):
    registry().clear()
    first = generate_world(_config(tmp_path, "first"), generate_stage_visuals=False)
    second = generate_world(_config(tmp_path, "second"), generate_stage_visuals=False)

    assert first.image_path.exists()
    assert first.manifest_path.exists()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf8"))
    assert manifest["status"] == "complete"
    assert list(manifest["stages"]) == ["erosion", "geometry", "tectonics", "world_age"]
    assert manifest["statistics"]["land_fraction"] > 0.0
    assert set(manifest["native_libraries"]) == {
        "climate_native",
        "elevation_native",
        "erosion_native",
        "fluvial_native",
        "geology_native",
        "hydrology_native",
        "hydrology_pass2_native",
        "planet_native",
        "refinement_native",
        "surface_water_native",
        "tectonics_native",
        "topology_native",
        "world_age_native",
    }
    assert all(len(library["sha256"]) == 64 for library in manifest["native_libraries"].values())
    assert (first.run_dir / "datasets" / "erosion" / "ElevationRaw.npy").exists()
    assert not any(stage.stats.cache_hit for stage in first.stages.values() if stage.stats)
    assert all(stage.stats.cache_hit for stage in second.stages.values() if stage.stats)


def test_changing_seed_invalidates_cache_and_changes_world(tmp_path: Path):
    first = generate_world(_config(tmp_path, "seed-a", seed=17), generate_stage_visuals=False)
    second = generate_world(_config(tmp_path, "seed-b", seed=18), generate_stage_visuals=False)

    assert not any(stage.stats.cache_hit for stage in second.stages.values() if stage.stats)
    assert first.image_path.read_bytes() != second.image_path.read_bytes()


def test_primary_cli_generates_world(tmp_path: Path):
    exit_code = main(
        [
            "generate",
            "--width",
            "64",
            "--height",
            "32",
            "--seed",
            "91",
            "--run-id",
            "cli-test",
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--tectonic-steps",
            "2",
            "--erosion-steps",
            "2",
            "--no-stage-visuals",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "out" / "cli-test" / "world.png").exists()
    assert (tmp_path / "out" / "cli-test" / "run.json").exists()


def test_primary_generate_rejects_unmigrated_cubed_sphere(tmp_path: Path, capsys):
    config_path = tmp_path / "cubed.json"
    config_path.write_text(
        json.dumps(
            {
                "topology": "cubed_sphere",
                "resolutions": [{"face_resolution": 16}],
                "run_id": "cubed-generate",
                "output_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf8",
    )
    assert main(["generate", "--config", str(config_path)]) == 2
    assert "has not migrated to cubed_sphere" in capsys.readouterr().err


def test_doctor_reports_unbuilt_native_libraries(monkeypatch, capsys):
    monkeypatch.setenv("MAP_MAKER_NATIVE_PROFILE", "definitely-missing")

    assert main(["doctor"]) == 1
    output = capsys.readouterr().out
    assert "Native tectonics_native: MISSING" in output
    assert "Run: map-maker-build-native" in output


def test_doctor_accepts_external_libraries_without_cargo_or_workspace(monkeypatch, capsys):
    import map_maker.cli as cli

    monkeypatch.setattr(cli, "workspace_root", lambda: None)
    monkeypatch.setattr(cli.shutil, "which", lambda command: None)
    monkeypatch.setattr(
        cli,
        "native_library_info",
        lambda name: {
            "path": f"/external/{name}",
            "abi_version": 1,
            "sha256": "a" * 64,
        },
    )

    assert main(["doctor"]) == 0
    assert "Ready to generate" in capsys.readouterr().out
