"""Primary command-line interface for the planetary pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ._native import (
    NativeLibraryAbiError,
    NativeLibraryNotBuiltError,
    native_library_info,
    workspace_root,
)
from .native_build import NATIVE_LIBRARIES

if TYPE_CHECKING:
    from .pipeline.config import PipelineConfig


def _generate_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("generate", help="Run the physical pipeline through erosion.")
    parser.add_argument("--config", type=Path, help="Pipeline YAML or JSON configuration.")
    parser.add_argument("--width", type=int, help="Output grid width (default: 512).")
    parser.add_argument("--height", type=int, help="Output grid height (default: 256).")
    parser.add_argument("--seed", type=int, help="Deterministic world seed (default: 42).")
    parser.add_argument("--run-id", help="Stable output run identifier.")
    parser.add_argument("--output-dir", type=Path, help="Output root (default: out).")
    parser.add_argument("--cache-dir", type=Path, help="Cache root (default: <output>/cache).")
    parser.add_argument("--log-dir", type=Path, help="Log root (default: <output>/logs).")
    parser.add_argument("--plates", type=int, help="Override tectonic plate count.")
    parser.add_argument("--tectonic-steps", type=int, help="Override tectonic time steps.")
    parser.add_argument("--erosion-steps", type=int, help="Override erosion steps.")
    parser.add_argument(
        "--no-stage-visuals",
        action="store_true",
        help="Only render the final physical map.",
    )
    return parser


def _validation_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate", help="Run the fixed-seed integration gallery and hard gates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/validation.yaml"),
        help="Validation YAML configuration (default: configs/validation.yaml).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override validation output root.")
    return parser


def _biosphere_validation_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate-biosphere",
        help="Run the Earth biosphere, functional-vegetation, and biome multi-seed profiles.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/biosphere_validation.yaml"),
        help="Biosphere ensemble YAML (default: configs/biosphere_validation.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override biosphere validation output root."
    )
    return parser


def _atlas_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "atlas", help="Export the canonical projected physical world map."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/physical_atlas.yaml"),
        help="Atlas YAML configuration (default: configs/physical_atlas.yaml).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override atlas output directory.")
    parser.add_argument("--width", type=int, help="Override output width in pixels.")
    parser.add_argument(
        "--central-meridian",
        type=float,
        help="Override automatic ocean-seam placement, in degrees longitude.",
    )
    parser.add_argument(
        "--no-rivers",
        action="store_true",
        help="Omit vector river channels from this export.",
    )
    return parser


def _topology_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "topology", help="Generate the canonical cubed-sphere topology diagnostic."
    )
    parser.add_argument(
        "--face-resolution",
        type=int,
        default=64,
        help="Cells along each face edge (default: 64).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/topology"),
        help="Diagnostic output directory (default: out/topology).",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> "PipelineConfig":
    from .pipeline.config import GridInfo, PipelineConfig, ResolutionSet

    if args.config:
        config = PipelineConfig.from_file(args.config)
    else:
        width = args.width or 512
        height = args.height or 256
        seed = 42 if args.seed is None else args.seed
        output_dir = (args.output_dir or Path("out")).expanduser().resolve()
        run_id = args.run_id or f"world-{seed}-{width}x{height}"
        config = PipelineConfig.from_mapping(
            {
                "topology": "sphere",
                "resolutions": [{"height": height, "width": width}],
                "rng_seed": seed,
                "run_id": run_id,
                "output_dir": str(output_dir),
                "cache_dir": str(args.cache_dir or output_dir / "cache"),
                "log_dir": str(args.log_dir or output_dir / "logs"),
            }
        )

    if config.topology.lower() == "cubed_sphere":
        raise ValueError(
            "generate through erosion has not migrated to cubed_sphere; use "
            "`map-maker-pipeline --stage tectonics --config <config>`"
        )
    else:
        width = args.width or config.resolution_set.native.width
        height = args.height or config.resolution_set.native.height
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        config.resolution_set = ResolutionSet((GridInfo(height=height, width=width),))
    if args.seed is not None:
        config.rng_seed = args.seed
    if args.run_id:
        config.run_id = args.run_id
    if args.output_dir:
        config.output_dir = args.output_dir.expanduser().resolve()
    if args.cache_dir:
        config.cache_dir = args.cache_dir.expanduser().resolve()
    if args.log_dir:
        config.log_dir = args.log_dir.expanduser().resolve()

    overrides = {name: dict(values) for name, values in config.stage_overrides.items()}
    if args.plates is not None:
        overrides.setdefault("tectonics", {})["num_plates"] = args.plates
    if args.tectonic_steps is not None:
        overrides.setdefault("tectonics", {})["time_steps"] = args.tectonic_steps
    if args.erosion_steps is not None:
        overrides.setdefault("erosion", {})["steps"] = args.erosion_steps
    config.stage_overrides = overrides
    return config


def _doctor() -> int:
    failures: list[str] = []
    missing_libraries = False
    root = workspace_root()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Cargo: {shutil.which('cargo') or 'MISSING'}")
    print(f"Workspace: {root or 'MISSING'}")
    for library in NATIVE_LIBRARIES:
        try:
            info = native_library_info(library)
        except NativeLibraryNotBuiltError:
            print(f"Native {library}: MISSING")
            failures.append(f"native library {library} has not been built")
            missing_libraries = True
        except NativeLibraryAbiError as exc:
            print(f"Native {library}: INCOMPATIBLE")
            failures.append(str(exc))
        else:
            print(
                f"Native {library}: {info['path']} "
                f"(ABI {info['abi_version']}, sha256 {info['sha256'][:12]})"
            )
    if missing_libraries and shutil.which("cargo") is None:
        failures.append("Cargo is required to build missing native libraries")
    if missing_libraries and root is None:
        failures.append("a source workspace is required to build missing native libraries")
    if failures:
        print("\nNot ready:")
        for failure in failures:
            print(f"  - {failure}")
        if missing_libraries and root is not None and shutil.which("cargo") is not None:
            print("\nRun: map-maker-build-native")
        return 1
    print("\nReady to generate.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "legacy":
        from .legacy.cli import main as legacy_main

        return legacy_main(raw_args[1:])

    parser = argparse.ArgumentParser(prog="map-maker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _generate_parser(subparsers)
    _validation_parser(subparsers)
    _biosphere_validation_parser(subparsers)
    _atlas_parser(subparsers)
    _topology_parser(subparsers)
    subparsers.add_parser("doctor", help="Check that the native pipeline is runnable.")
    subparsers.add_parser("legacy", help="Run the previous procedural generator.")
    args = parser.parse_args(raw_args)

    if args.command == "doctor":
        return _doctor()

    if args.command == "validate":
        try:
            from .pipeline.validation import ValidationConfig, validate_gallery

            validation_config = ValidationConfig.from_file(args.config, output_dir=args.output_dir)
            validation = validate_gallery(validation_config)
        except (
            NativeLibraryAbiError,
            NativeLibraryNotBuiltError,
            FileNotFoundError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"map-maker: {exc}", file=sys.stderr)
            return 2
        print(f"Validation report: {validation.report_path}")
        print(f"Seed gallery: {validation.gallery_path}")
        if validation.passed:
            print(f"PASS: {len(validation.worlds)} worlds passed provisional gates.")
            return 0
        print("FAIL: one or more provisional gates failed.")
        for gate in validation.global_gates:
            if not gate.passed:
                print(f"  global.{gate.name}: {gate.value} (expected {gate.expectation})")
        for world in validation.worlds:
            if not world.cache_replay_passed:
                print(f"  seed {world.seed}.cache_replay: failed")
            for gate in world.gates:
                if not gate.passed:
                    print(
                        f"  seed {world.seed}.{gate.name}: {gate.value} "
                        f"(expected {gate.expectation})"
                    )
        return 1

    if args.command == "validate-biosphere":
        try:
            from .pipeline.biosphere_ensemble import (
                BiosphereEnsembleConfig,
                run_biosphere_ensemble,
            )

            ensemble_config = BiosphereEnsembleConfig.from_file(
                args.config, output_dir=args.output_dir
            )
            biosphere_validation = run_biosphere_ensemble(ensemble_config)
        except (
            NativeLibraryAbiError,
            NativeLibraryNotBuiltError,
            FileNotFoundError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"map-maker: {exc}", file=sys.stderr)
            return 2
        print(f"Biosphere validation report: {biosphere_validation.report_path}")
        print(f"Ensemble KPI catalog: {biosphere_validation.metric_catalog_path}")
        if biosphere_validation.biome_gallery_path is not None:
            print(f"Biome gallery: {biosphere_validation.biome_gallery_path}")
        if biosphere_validation.surface_geography_gallery_path is not None:
            print(
                f"Surface-geography gallery: {biosphere_validation.surface_geography_gallery_path}"
            )
        if biosphere_validation.passed:
            print(
                f"PASS: {biosphere_validation.seed_count} worlds satisfy earth_biosphere_v1 "
                "and earth_functional_vegetation_v1, plus earth_biomes_v1, "
                "with ensemble tolerances."
            )
            return 0
        if biosphere_validation.execution_valid:
            print(
                "OUTSIDE REFERENCE: simulation invariants and ensemble tolerances pass, "
                "but Earth calibration does not."
            )
        else:
            print("FAIL: a hard invariant or ensemble stability tolerance failed.")
        for gate in biosphere_validation.gates:
            if not gate.passed:
                print(f"  {gate.name}: {gate.value} (expected {gate.expectation})")
        return 1

    if args.command == "atlas":
        try:
            from .pipeline.atlas import AtlasExportConfig, export_physical_atlas

            atlas_config = AtlasExportConfig.from_file(
                args.config,
                output_dir=args.output_dir,
                width_px=args.width,
                central_meridian_deg=args.central_meridian,
                draw_rivers=False if args.no_rivers else None,
            )
            atlas = export_physical_atlas(atlas_config)
        except (
            NativeLibraryAbiError,
            NativeLibraryNotBuiltError,
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"map-maker: {exc}", file=sys.stderr)
            return 2
        print(f"Physical atlas: {atlas.png_path}")
        print(f"Projected GeoTIFF: {atlas.geotiff_path}")
        print(f"Atlas metadata: {atlas.metadata_path}")
        print(
            f"Rendered {atlas.width_px}x{atlas.height_px} Equal Earth map at "
            f"{atlas.central_meridian_deg:.2f} degrees central longitude "
            f"with {atlas.rendered_river_count} river reaches."
        )
        return 0

    if args.command == "topology":
        try:
            from .pipeline.cubed_sphere import run_cubed_sphere_diagnostic

            net_path, report_path = run_cubed_sphere_diagnostic(
                face_resolution=args.face_resolution,
                output_dir=args.output_dir,
            )
        except (
            NativeLibraryAbiError,
            NativeLibraryNotBuiltError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(f"map-maker: {exc}", file=sys.stderr)
            return 2
        print(f"Cube net: {net_path}")
        print(f"Topology report: {report_path}")
        return 0

    try:
        from .pipeline.generate import generate_world

        config = _config_from_args(args)
        result = generate_world(config, generate_stage_visuals=not args.no_stage_visuals)
    except (
        NativeLibraryAbiError,
        NativeLibraryNotBuiltError,
        FileNotFoundError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"map-maker: {exc}", file=sys.stderr)
        return 2

    cache_hits = sum(
        bool(stage.stats and stage.stats.cache_hit) for stage in result.stages.values()
    )
    print(f"World map: {result.image_path}")
    print(f"Run manifest: {result.manifest_path}")
    print(
        f"Completed {len(result.stages)} stages in {result.elapsed_seconds:.2f}s "
        f"({cache_hits} cache hits)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
