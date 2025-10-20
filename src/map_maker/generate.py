"""Procedural world generation utilities."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from noise import pnoise2
from PIL import Image

Array = np.ndarray


def _perlin(width: int, height: int, scale: float, octaves: int, seed: int) -> Array:
    """Return Perlin noise in [-1,1]."""
    arr = np.empty((height, width), dtype=np.float32)
    freq = 1.0 / max(1.0, scale)
    for y in range(height):
        for x in range(width):
            arr[y, x] = pnoise2(
                x * freq,
                y * freq,
                octaves=octaves,
                repeatx=1 << 16,
                repeaty=1 << 16,
                base=seed,
            )
    return arr


def _normalize(a: Array, lo: float, hi: float) -> Array:
    amin, amax = float(a.min()), float(a.max())
    if amax == amin:
        return np.full_like(a, (lo + hi) / 2.0, dtype=np.float32)
    b = (a - amin) / (amax - amin)
    return (lo + b * (hi - lo)).astype(np.float32)


def _radial_falloff(width: int, height: int, power: float = 3.0) -> Array:
    """0 at center, 1 at corners; raise to power to shape."""
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    maxd = math.hypot(cx, cy)
    yy, xx = np.ogrid[:height, :width]
    d = np.hypot(xx - cx, yy - cy) / maxd
    return np.clip(d**power, 0.0, 1.0).astype(np.float32)


def make_elevation(width: int, height: int, seed: int, cfg: Dict) -> Array:
    n1 = _perlin(
        width,
        height,
        cfg.get("elev_scale", 300.0),
        cfg.get("elev_octaves", 6),
        seed,
    )
    n2 = _perlin(
        width,
        height,
        cfg.get("elev_scale2", 80.0),
        cfg.get("elev_octaves2", 3),
        seed + 101,
    )
    base = 0.7 * n1 + 0.3 * n2
    base = _normalize(base, -1.0, 1.0)
    fall = _radial_falloff(width, height, cfg.get("falloff_power", 2.6))
    elev = base - (cfg.get("falloff_strength", 0.6) * fall)
    return _normalize(elev, -1.0, 1.0)


def make_temperature(width: int, height: int, elev: Array, cfg: Dict) -> Array:
    # Latitude gradient: equator in the middle (y=H/2).
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    lat = 1.0 - np.abs(yy)  # 1 at equator, 0 at poles
    lapse = cfg.get("temp_lapse", 0.45)  # how much elevation cools
    temp = lat - (np.clip(elev, -1, 1) + 1.0) * 0.5 * lapse
    return np.clip(temp, 0.0, 1.0).astype(np.float32)


def make_moisture(width: int, height: int, seed: int, elev: Array, cfg: Dict) -> Array:
    m = _perlin(
        width,
        height,
        cfg.get("moist_scale", 220.0),
        cfg.get("moist_octaves", 4),
        seed + 202,
    )
    m = _normalize(m, 0.0, 1.0)
    # Boost moisture near coasts (cheap heuristic): blur ocean mask once.
    ocean = elev < cfg.get("ocean_level", 0.0)
    ocean_u8 = ocean.astype(np.uint8)
    # 3x3 box blur to approximate proximity
    prox = ocean_u8.astype(np.float32)
    prox = (
        prox
        + np.pad(prox[1:], ((0, 1), (0, 0)))  # up
        + np.pad(prox[:-1], ((1, 0), (0, 0)))  # down
        + np.pad(prox[:, 1:], ((0, 0), (0, 1)))  # right
        + np.pad(prox[:, :-1], ((0, 0), (1, 0)))  # left
    ) / 5.0
    boost = cfg.get("coast_moist_boost", 0.25) * prox
    moist = np.clip(m + boost, 0.0, 1.0)
    return moist


def classify_biomes(temp: Array, moist: Array) -> Array:
    """
    Simple Whittaker-like classification.
    0: ocean (fill later), 1: tundra, 2: boreal, 3: temperate grass,
    4: temperate forest, 5: savanna, 6: tropical dry, 7: tropical rainforest,
    8: desert, 9: montane
    """
    b = np.zeros_like(temp, dtype=np.uint8)
    # Start with dryness
    desert = (moist < 0.18) & (temp > 0.25)
    tundra = temp < 0.18
    boreal = (~tundra) & (temp < 0.35) & (moist > 0.35)
    temperate_forest = (temp >= 0.35) & (temp < 0.65) & (moist >= 0.45)
    temperate_grass = (temp >= 0.35) & (temp < 0.65) & (moist < 0.45) & ~desert
    savanna = (temp >= 0.65) & (moist >= 0.25) & (moist < 0.50)
    tropical_rain = (temp >= 0.65) & (moist >= 0.50)
    tropical_dry = (temp >= 0.65) & (moist >= 0.18) & (moist < 0.25)
    b[desert] = 8
    b[tundra] = 1
    b[boreal] = 2
    b[temperate_grass] = 3
    b[temperate_forest] = 4
    b[savanna] = 5
    b[tropical_dry] = 6
    b[tropical_rain] = 7
    # crude montane override (cold due to height)
    b[temp < 0.22] = 1
    return b


def d8_flow_accum(elev: Array, ocean_level: float) -> Array:
    """Compute D8 flow accumulation with descending-order DP."""
    h, w = elev.shape
    # order cells by elevation descending so that upstream adds to downstream
    idx_flat = np.argsort(elev.ravel())[::-1]
    flow = np.ones_like(elev, dtype=np.uint32)  # each cell contributes at least 1
    # Precompute neighbors offsets
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k in idx_flat:
        y = k // w
        x = k % w
        z = elev[y, x]
        # ocean or flats: if ocean, don't route further
        if z <= ocean_level:
            continue
        best = None
        bestz = z
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                nz = elev[ny, nx]
                if nz < bestz:
                    bestz = nz
                    best = (ny, nx)
        if best is not None:
            flow[best] += flow[y, x]
    return flow


def render_png(world: Dict[str, Array]) -> Image.Image:
    elev = world["elev"]
    temp = world["temp"]
    moist = world["moist"]
    biomes = world["biome"]
    flow = world["flow_acc"]
    ocean = elev < world["params"]["ocean_level"]

    h, w = elev.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Colors (RGB)
    ocean_colors = [(56, 89, 138), (38, 64, 102), (23, 43, 71)]
    biome_colors = {
        1: (200, 220, 230),  # tundra
        2: (100, 140, 110),  # boreal
        3: (180, 200, 120),  # temp grass
        4: (80, 140, 80),  # temp forest
        5: (210, 200, 120),  # savanna
        6: (190, 170, 90),  # tropical dry
        7: (40, 120, 60),  # tropical rainforest
        8: (210, 180, 120),  # desert
    }

    # ocean shading by depth
    depth = np.clip(-(elev - world["params"]["ocean_level"]), 0.0, 1.0)
    deep = depth > 0.66
    mid = (depth > 0.33) & ~deep
    shallow = (depth > 0.0) & ~deep & ~mid
    img[ocean & deep] = ocean_colors[2]
    img[ocean & mid] = ocean_colors[1]
    img[ocean & shallow] = ocean_colors[0]

    # land biomes
    land = ~ocean
    for biome_id, color in biome_colors.items():
        img[(biomes == biome_id) & land] = color

    # rivers (draw over land; thickness via accumulation)
    rivers = (flow >= world["params"]["river_acc_min"]) & land
    img[rivers] = (30, 60, 120)

    return Image.fromarray(img, mode="RGB")


def generate_world(width: int, height: int, seed: int, config: Dict) -> Dict[str, Array | Dict]:
    params = {
        "ocean_level": float(config.get("ocean_level", 0.0)),
        "river_acc_min": int(config.get("river_acc_min", max(150, (width * height) // 6000))),
    }
    elev = make_elevation(width, height, seed, config)
    temp = make_temperature(width, height, elev, config)
    moist = make_moisture(width, height, seed, elev, config)
    biome = classify_biomes(temp, moist)
    flow_acc = d8_flow_accum(elev, params["ocean_level"])

    # Mark ocean in biome = 0 and ensure every land tile has a biome
    biome = biome.copy()
    land = elev >= params["ocean_level"]
    biome[~land] = 0
    biome[(biome == 0) & land] = 3  # fallback to temperate grassland

    stats = {
        "land_ratio": float((elev >= params["ocean_level"]).mean()),
        "max_flow": int(flow_acc.max()),
        "num_river_cells": int((flow_acc >= params["river_acc_min"]).sum()),
    }

    return {
        "elev": elev,
        "temp": temp,
        "moist": moist,
        "biome": biome,
        "flow_acc": flow_acc,
        "stats": stats,
        "params": params,
    }
