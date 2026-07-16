"""Explicit native workspace build command."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence

from ._native import library_filename, workspace_root

NATIVE_LIBRARIES = (
    "climate_native",
    "elevation_native",
    "erosion_native",
    "fluvial_native",
    "geology_native",
    "hydrology_native",
    "hydrology_pass2_native",
    "planet_native",
    "refinement_native",
    "tectonics_native",
    "topology_native",
    "world_age_native",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build map_maker native libraries.")
    parser.add_argument(
        "--profile",
        choices=("debug", "release"),
        default="release",
        help="Cargo profile to build (default: release).",
    )
    parser.add_argument(
        "--unlocked",
        action="store_true",
        help="Allow Cargo to update Cargo.lock.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = workspace_root()
    if root is None:
        parser.error("could not locate the map_maker Cargo workspace")

    command = ["cargo", "build", "--workspace"]
    if args.profile == "release":
        command.append("--release")
    if not args.unlocked:
        command.append("--locked")

    subprocess.run(command, cwd=root, check=True)

    target_dir = root / "target" / args.profile
    missing = [
        target_dir / library_filename(name)
        for name in NATIVE_LIBRARIES
        if not (target_dir / library_filename(name)).is_file()
    ]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Cargo completed but native libraries are missing:\n{paths}")

    print(f"Native libraries built in {target_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
