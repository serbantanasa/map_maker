from __future__ import annotations

from pathlib import Path

import pytest

from map_maker.pipeline.mineral_systems_ensemble import (
    MineralEnsembleConfig,
    MineralEnsembleThresholds,
    MineralSeedReport,
    evaluate_mineral_ensemble,
)
from map_maker.pipeline.stages.mineral_systems import SYSTEM_NAMES


def _report(seed: int, *, failed_family: int | None = None) -> MineralSeedReport:
    rows = tuple(
        {
            "system_code": index + 1,
            "system_name": name,
            "peak_potential": 0.40 + 0.01 * index + 0.001 * seed,
            "passed": index != failed_family,
        }
        for index, name in enumerate(SYSTEM_NAMES)
    )
    return MineralSeedReport(
        seed=seed,
        hard_gate_pass=failed_family is None,
        hard_failures=(
            ()
            if failed_family is None
            else (f"{SYSTEM_NAMES[failed_family]}_directional_or_noncollapse",)
        ),
        family_rows=rows,
        candidate_count=500 + seed,
        system_count=200 + seed,
        state_checksum=f"state-{seed}",
    )


def test_ensemble_accepts_stable_directional_seed_reports():
    reports = [_report(seed) for seed in (1, 2, 3, 4, 5)]
    evaluation = evaluate_mineral_ensemble(reports, MineralEnsembleThresholds())
    assert evaluation.hard_gate_pass
    assert evaluation.hard_failures == ()
    assert all(evaluation.family_catalog["passed"].to_pylist())


def test_ensemble_rejects_per_seed_and_family_collapse():
    reports = [_report(seed, failed_family=2 if seed < 4 else None) for seed in (1, 2, 3, 4, 5)]
    evaluation = evaluate_mineral_ensemble(reports, MineralEnsembleThresholds())
    assert not evaluation.hard_gate_pass
    assert "mafic_ultramafic_ensemble" in evaluation.hard_failures


def test_ensemble_never_tolerates_catalog_integrity_failure():
    reports = [_report(seed) for seed in (1, 2, 3, 4, 5)]
    broken = reports[0]
    reports[0] = MineralSeedReport(
        seed=broken.seed,
        hard_gate_pass=False,
        hard_failures=("candidate_stable_id_mismatch",),
        family_rows=broken.family_rows,
        candidate_count=broken.candidate_count,
        system_count=broken.system_count,
        state_checksum=broken.state_checksum,
    )
    evaluation = evaluate_mineral_ensemble(reports, MineralEnsembleThresholds())
    assert not evaluation.hard_gate_pass
    assert "per_seed_integrity_gate" in evaluation.hard_failures


def test_ensemble_rejects_duplicate_seed_state():
    reports = [_report(seed) for seed in (1, 2, 3, 4, 5)]
    duplicate = reports[0].state_checksum
    reports[1] = MineralSeedReport(
        seed=reports[1].seed,
        hard_gate_pass=True,
        hard_failures=(),
        family_rows=reports[1].family_rows,
        candidate_count=reports[1].candidate_count,
        system_count=reports[1].system_count,
        state_checksum=duplicate,
    )
    evaluation = evaluate_mineral_ensemble(reports, MineralEnsembleThresholds())
    assert not evaluation.hard_gate_pass
    assert "distinct_seed_state" in evaluation.hard_failures


def test_ensemble_config_requires_unique_minimum_seed_set(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("topology: cubed_sphere\n", encoding="utf8")
    config_path = tmp_path / "minerals.yaml"
    config_path.write_text(
        """
format_version: 1
base_config: base.yaml
seeds: [1, 1, 2]
ensemble_tolerances:
  minimum_seed_count: 3
""".strip() + "\n",
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="unique"):
        MineralEnsembleConfig.from_file(config_path)

    config_path.write_text(
        """
format_version: 1
base_config: base.yaml
seeds: [1, 2]
ensemble_tolerances:
  minimum_seed_count: 3
""".strip() + "\n",
        encoding="utf8",
    )
    with pytest.raises(ValueError, match="at least 3"):
        MineralEnsembleConfig.from_file(config_path)
