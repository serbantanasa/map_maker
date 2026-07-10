from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from map_maker.cli import main
from map_maker.pipeline.validation import (
    ValidationConfig,
    _longitude_seam_ratio,
    _major_component_sizes,
    _plate_boundary_relief_ratio,
    validate_gallery,
)


def _write_validation_config(tmp_path: Path) -> Path:
    base_config = tmp_path / "pipeline.yaml"
    base_config.write_text(
        """
topology: sphere
resolutions:
  - width: 64
    height: 32
rng_seed: 0
run_id: validation-test
output_dir: unused
cache_dir: unused-cache
log_dir: unused-logs
stage_overrides:
  tectonics:
    num_plates: 8
    time_steps: 2
  erosion:
    steps: 2
""".strip(),
        encoding="utf8",
    )
    validation_config = tmp_path / "validation.yaml"
    validation_config.write_text(
        """
base_config: pipeline.yaml
output_dir: validation-output
seeds: [11, 12]
provisional_thresholds:
  min_land_fraction: 0.0
  max_land_fraction: 1.0
  min_land_components: 1
  max_largest_landmass_fraction: 1.0
  min_mixed_plate_fraction: 0.0
  max_longitude_seam_ratio: 100.0
  max_plate_boundary_relief_ratio: 100.0
""".strip(),
        encoding="utf8",
    )
    return validation_config


def test_component_measurement_wraps_longitude():
    land = np.zeros((8, 16), dtype=bool)
    land[:, 0] = True
    land[:, -1] = True
    land[2:6, 7] = True

    sizes = _major_component_sizes(land)

    assert sizes == [16, 4]


def test_seam_and_plate_relief_metrics_detect_synthetic_failures():
    elevation = np.zeros((16, 32), dtype=np.float32)
    elevation[:, 0] = 100.0
    assert _longitude_seam_ratio(elevation) > 2.0

    plate_ids = np.zeros((16, 32), dtype=np.int32)
    plate_ids[:, 16:] = 1
    elevation[:, 16:] = 100.0
    assert _plate_boundary_relief_ratio(plate_ids, elevation) > 3.0


def test_validation_gallery_reports_determinism_cache_and_unique_seeds(tmp_path: Path):
    config_path = _write_validation_config(tmp_path)
    config = ValidationConfig.from_file(config_path)

    result = validate_gallery(config)

    assert result.passed
    assert result.gallery_path.exists()
    assert result.report_path.exists()
    report = json.loads(result.report_path.read_text(encoding="utf8"))
    assert report["status"] == "pass"
    assert report["human_gallery_review_required"] is True
    assert {gate["name"] for gate in report["global_gates"]} == {
        "cache_replay",
        "cold_determinism",
        "unique_seed_outputs",
    }
    assert all(world["cache_replay_passed"] for world in report["worlds"])


def test_validation_cli_returns_success(tmp_path: Path):
    config_path = _write_validation_config(tmp_path)

    assert main(["validate", "--config", str(config_path)]) == 0
