"""Utility helpers for ad-hoc pipeline commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image
import time

from .config import PipelineConfig
from .execution import ExecutionEngine
from .registry import registry, stage
from .visualization import VisualizationRequest, VisualizationResult

# Ensure built-in stages are registered.
from .stages import tectonics as _tectonics_module  # noqa: F401


def _to_array(value) -> np.ndarray:
    if value is None:
        raise ValueError("Missing artifact value for visualization")
    if hasattr(value, "array"):
        return np.array(value.array(), copy=False)
    return np.asarray(value)


def _write_png(path: Path, data: np.ndarray) -> VisualizationResult:
    image = Image.fromarray(data, mode="L" if data.ndim == 2 else "RGB")
    image.save(path)
    return VisualizationResult(path=path, artifact_name=path.stem, metadata={})


def _normalize(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * 255.0).astype(np.uint8)


def _topology_visualizer(stage_result, request: VisualizationRequest) -> Optional[VisualizationResult]:
    lon_record = stage_result.artifact_records.get("longitude")
    lat_record = stage_result.artifact_records.get("latitude")
    xyz_record = stage_result.artifact_records.get("xyz_coords")
    if not lon_record or not lat_record:
        return None

    lon = _to_array(lon_record.value)
    lat = _to_array(lat_record.value)
    lon_img = ((lon + np.pi) / (2 * np.pi)) % 1.0
    lat_img = (lat + (np.pi / 2)) / np.pi
    lon_png = (lon_img * 255.0).astype(np.uint8)
    lat_png = (lat_img * 255.0).astype(np.uint8)

    phase = 0.6 * np.pi  # ~30% offset
    checker = (((np.sin(4 * lon + phase) > 0) ^ (np.sin(4 * lat + phase) > 0)).astype(np.uint8) * 255)

    results: list[VisualizationResult] = []
    lon_path = request.output_dir / "lon_gradient.png"
    lat_path = request.output_dir / "lat_gradient.png"
    checker_path = request.output_dir / "wrap_checker.png"
    results.append(_write_png(lon_path, lon_png))
    results.append(_write_png(lat_path, lat_png))
    results.append(_write_png(checker_path, checker))

    anchor_lon = 0.63 * np.pi
    anchor_lat = -0.27 * np.pi
    delta_lon = (lon - anchor_lon + np.pi) % (2 * np.pi) - np.pi
    delta_lat = lat - anchor_lat
    wrap_signal = np.cos(delta_lon) * np.cos(delta_lat)
    wrap_gradient = _normalize(wrap_signal)
    wrap_path = request.output_dir / "wrap_gradient.png"
    results.append(_write_png(wrap_path, wrap_gradient))

    if xyz_record and xyz_record.value is not None:
        xyz = _to_array(xyz_record.value)
        value = _normalize(xyz[..., 2])
        xyz_path = request.output_dir / "z_height.png"
        results.append(_write_png(xyz_path, value))

    return results


def _register_topology_stage() -> str:
    stage_name = "topology_visualizer"
    reg = registry()
    if stage_name in reg:
        return stage_name

    @stage(stage_name, outputs=("xyz_coords", "longitude", "latitude"), visualizer=_topology_visualizer)
    def topology_stage(context, deps, config):
        topo = context.topology
        return {
            "xyz_coords": topo.xyz,
            "longitude": topo.lon,
            "latitude": topo.lat,
        }

    return stage_name


def _register_tectonics_stage() -> str:
    stage_name = "tectonics"
    if stage_name in registry():
        return stage_name
    from .stages import tectonics  # noqa: F401

    return stage_name


def run_topology_visualizer(
    topology: str = "sphere",
    *,
    width: int = 256,
    height: int = 128,
    output_dir: Path | str = Path("out"),
    cache_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    run_id: str | None = None,
    generate_visuals: bool = True,
) -> Tuple[Path, float]:
    """Execute a single topology stage and return the visuals directory path."""

    cache_dir = cache_dir or Path("cache")
    log_dir = log_dir or Path("logs")
    run_id = run_id or f"topology_{topology}_{height}x{width}"

    config = PipelineConfig.from_mapping(
        {
            "topology": topology,
            "resolutions": [{"height": height, "width": width}],
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "log_dir": str(log_dir),
            "run_id": run_id,
        }
    )

    stage_name = _register_topology_stage()
    engine = ExecutionEngine(config, generate_visuals=generate_visuals)
    start = time.perf_counter()
    engine.run([stage_name])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return config.run_visual_dir() / stage_name, elapsed_ms


def run_tectonics_visualizer(
    topology: str = "sphere",
    *,
    width: int = 256,
    height: int = 128,
    output_dir: Path | str = Path("out"),
    cache_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    run_id: str | None = None,
    num_plates: int = 20,
    continental_fraction: float = 0.35,
    lloyd_iterations: int = 4,
    velocity_scale: float = 1.0,
    drift_bias: float = 0.1,
    hotspot_density: float = 0.03,
    subduction_bias: float = 0.5,
    time_steps: int = 24,
    time_step: float = 0.5,
    wrap_x: bool = True,
    wrap_y: bool = False,
    rng_seed: int = 0,
    generate_visuals: bool = True,
) -> Tuple[Path, float]:
    cache_dir = cache_dir or Path("cache")
    log_dir = log_dir or Path("logs")
    run_id = run_id or f"tectonics_{topology}_{height}x{width}"

    config = PipelineConfig.from_mapping(
        {
            "topology": topology,
            "resolutions": [{"height": height, "width": width}],
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "log_dir": str(log_dir),
            "run_id": run_id,
            "rng_seed": rng_seed,
            "stage_overrides": {
                "tectonics": {
                    "num_plates": num_plates,
                    "continental_fraction": continental_fraction,
                    "lloyd_iterations": lloyd_iterations,
                    "velocity_scale": velocity_scale,
                    "drift_bias": drift_bias,
                    "hotspot_density": hotspot_density,
                    "subduction_bias": subduction_bias,
                    "time_steps": time_steps,
                    "time_step": time_step,
                    "wrap_x": wrap_x,
                    "wrap_y": wrap_y,
                },
            },
        }
    )

    stage_name = _register_tectonics_stage()
    engine = ExecutionEngine(config, generate_visuals=generate_visuals)
    start = time.perf_counter()
    results = engine.run([stage_name])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    stage_result = results[stage_name]
    plate_record = stage_result.artifact_records.get("PlateField")
    if plate_record and plate_record.value is not None:
        plate_data = _to_array(plate_record.value)
        continental = plate_data[..., 1] >= 0.5
        land_img = np.zeros((*continental.shape, 3), dtype=np.uint8)
        land_img[continental] = (204, 177, 111)
        land_img[~continental] = (34, 87, 182)
        land_path = (config.run_visual_dir() / stage_name) / "land_ocean.png"
        land_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(land_img, mode="RGB").save(land_path)

    return config.run_visual_dir() / stage_name, elapsed_ms


def main(args: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce pipeline stage visualization PNGs.")
    parser.add_argument("--stage", choices=["topology", "tectonics"], default="topology")
    parser.add_argument("--topology", default="sphere", choices=["sphere", "cylinder", "torus"])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--skip-visuals", action="store_true", help="Disable PNG generation (for debugging).")
    parser.add_argument("--tectonics-plates", type=int, default=24)
    parser.add_argument("--tectonics-continental-fraction", type=float, default=0.35)
    parser.add_argument("--tectonics-lloyd", type=int, default=4)
    parser.add_argument("--tectonics-velocity", type=float, default=1.0)
    parser.add_argument("--tectonics-drift", type=float, default=0.1)
    parser.add_argument("--tectonics-hotspot-density", type=float, default=0.03)
    parser.add_argument("--tectonics-subduction-bias", type=float, default=0.5)
    parser.add_argument("--tectonics-steps", type=int, default=24)
    parser.add_argument("--tectonics-dt", type=float, default=0.5)
    parser.add_argument("--tectonics-wrap-x", dest="tectonics_wrap_x", action="store_true")
    parser.add_argument("--no-tectonics-wrap-x", dest="tectonics_wrap_x", action="store_false")
    parser.add_argument("--tectonics-wrap-y", dest="tectonics_wrap_y", action="store_true")
    parser.add_argument("--no-tectonics-wrap-y", dest="tectonics_wrap_y", action="store_false")
    parser.add_argument("--tectonics-rng-seed", type=int, default=0)
    parser.set_defaults(tectonics_wrap_x=True, tectonics_wrap_y=False)
    parsed = parser.parse_args(list(args) if args is not None else None)

    if parsed.stage == "topology":
        visuals_dir, elapsed_ms = run_topology_visualizer(
            topology=parsed.topology,
            width=parsed.width,
            height=parsed.height,
            output_dir=parsed.output_dir,
            cache_dir=parsed.cache_dir,
            log_dir=parsed.log_dir,
            run_id=parsed.run_id,
            generate_visuals=not parsed.skip_visuals,
        )
        print(f"Topology visuals written to {visuals_dir} (elapsed {elapsed_ms:.2f} ms)")
    else:
        visuals_dir, elapsed_ms = run_tectonics_visualizer(
            topology=parsed.topology,
            width=parsed.width,
            height=parsed.height,
            output_dir=parsed.output_dir,
            cache_dir=parsed.cache_dir,
            log_dir=parsed.log_dir,
            run_id=parsed.run_id,
            num_plates=parsed.tectonics_plates,
            continental_fraction=parsed.tectonics_continental_fraction,
            lloyd_iterations=parsed.tectonics_lloyd,
            velocity_scale=parsed.tectonics_velocity,
            drift_bias=parsed.tectonics_drift,
            hotspot_density=parsed.tectonics_hotspot_density,
            subduction_bias=parsed.tectonics_subduction_bias,
            time_steps=parsed.tectonics_steps,
            time_step=parsed.tectonics_dt,
            wrap_x=parsed.tectonics_wrap_x,
            wrap_y=parsed.tectonics_wrap_y,
            rng_seed=parsed.tectonics_rng_seed,
            generate_visuals=not parsed.skip_visuals,
        )
        print(f"Tectonics visuals written to {visuals_dir} (elapsed {elapsed_ms:.2f} ms)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
