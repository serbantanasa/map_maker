"""Command-line entry point for map generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import yaml

from map_maker.generate import generate_world, render_png


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        msg = f"Config must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def main() -> None:
    parser = argparse.ArgumentParser("map_maker")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--out", type=str, default="out/map.png")
    args = parser.parse_args()

    config = load_config(args.config)

    t0 = time.time()
    world = generate_world(width=args.width, height=args.height, seed=args.seed, config=config)
    image = render_png(world)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    metadata = {
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "config": config,
        "stats": world["stats"],
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    elapsed = time.time() - t0
    print(f"Wrote {out_path} and {out_path.with_suffix('.json')} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
